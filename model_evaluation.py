import numpy as np
from trainDeepHit import compute_cindex_at_time
import matplotlib.pyplot as plt
from lifelines import AalenJohansenFitter
import pandas as pd
import statsmodels.api as sm
import shap
from pathlib import Path
from matplotlib.ticker import AutoMinorLocator


def bootstrap_cindex_at_time(
    model,
    data,
    time,
    label,
    eval_horizon,
    n_bootstrap=1000,
    seed=42
):
    """
    Bootstrap 95% confidence intervals for event-specific C-index.

    Parameters
    ----------
    model : trained DeepHit model
    data : validation features
    time : validation follow-up times
    label : validation labels
    eval_horizon : evaluation time horizon
    n_bootstrap : number of bootstrap samples
    seed : random seed

    Returns
    -------
    baseline : (num_Event,)
        Original C-index on the validation set.

    ci_lower : (num_Event,)
        Lower 95% confidence interval.

    ci_upper : (num_Event,)
        Upper 95% confidence interval.

    bootstrap_scores : (n_bootstrap, num_Event)
        All bootstrap C-index values.
    """

    rng = np.random.default_rng(seed)
    n = len(time)

    # Original estimate
    baseline = compute_cindex_at_time(
        model,
        data,
        time,
        label,
        eval_horizon
    )
    num_Event = baseline.shape[0]
    bootstrap_scores = np.zeros((n_bootstrap, num_Event))

    for b in range(n_bootstrap):

        # Sample patients with replacement
        idx = rng.choice(n, size=n, replace=True)

        data_b = data[idx]
        time_b = time[idx]
        label_b = label[idx]

        bootstrap_scores[b] = compute_cindex_at_time(
            model,
            data_b,
            time_b,
            label_b,
            eval_horizon
        )

    ci_lower = np.percentile(bootstrap_scores, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_scores, 97.5, axis=0)

    return baseline, ci_lower, ci_upper, bootstrap_scores




def permutation_feature_importance(
    model,
    x_val,
    time_val,
    label_val,
    eval_horizon,
    n_repeats=50
):
    """
    Permutation feature importance using drop in C-index.
    Returns array of shape (num_Event, num_features).
    """
    n_features = x_val.shape[1]
    baseline = compute_cindex_at_time(
        model, x_val, time_val, label_val, eval_horizon
    )
    
    importance = np.zeros((num_Event, n_features))

    for j in range(n_features):
        drops = []

        for _ in range(n_repeats):
            x_perm = x_val.copy()
            perm_idx = np.random.permutation(x_perm.shape[0])
            x_perm[:, j] = x_perm[perm_idx, j]

            cidx_perm = compute_cindex_at_time(
                model, x_perm, time_val, label_val, eval_horizon
            )

            drops.append(baseline - cidx_perm)

        importance[:, j] = np.mean(drops, axis=0)

        if j % 10 == 0:
            print(f"PFI done for feature {j}/{n_features}")

    return importance



def deep_hit_calibration(
    model,
    data,
    time,
    label,
    eval_time,
    risk_index = 0,
    plot_curve=True
):

    # ==================================================
    # 1. Predict from DeepHit model
    # ==================================================
    pred = model.call(
        data,
        training=False
    ).numpy()

    # Expected shape:
    # (patients, events, time_bins)

    # ==================================================
    # 2. Extract predicted MACE cumulative incidence
    # ==================================================

    # Sum probabilities up to 12-month horizon
    risk_1yr = np.sum(
        pred[:, risk_index, :eval_time[0]],
        axis=1
    )

    # ==================================================
    # 3. Create evaluation dataframe
    # ==================================================
    eval_df = pd.DataFrame({
        "time": time.ravel(),

        # Original competing-risk labels:
        # 0 = censored
        # 1 = MACE
        # 2 = competing event
        "event": label.ravel(),

        "risk_1yr": risk_1yr
    })

    # ==================================================
    # 4. Create predicted risk deciles
    # ==================================================
    eval_df["risk_decile"] = pd.qcut(
        eval_df["risk_1yr"],
        q=10,
        labels=False,
        duplicates="drop"
    )

    # ==================================================
    # 5. Calculate observed cumulative incidence
    #    using Aalen-Johansen estimator
    # ==================================================
    aj = AalenJohansenFitter()

    calibration_results = []

    for decile, group in eval_df.groupby("risk_decile"):

        # Mean predicted risk in this decile
        predicted = group["risk_1yr"].mean()

        # Fit competing-risk CIF
        aj.fit(
            durations=group["time"],
            event_observed=group["event"],
            event_of_interest=1
        )

        cif = aj.cumulative_density_

        # CIF at 12 months
        available_times = cif.index[
            cif.index <= eval_time[0]
        ]

        if len(available_times) > 0:
            observed = cif.loc[
                available_times[-1]
            ].values[0]
        else:
            observed = 0

        calibration_results.append({
            "decile": int(decile) + 1,
            "n": len(group),
            "predicted_risk": predicted,
            "observed_risk": observed
        })

    calibration_results = pd.DataFrame(
        calibration_results
    )

    if plot_curve:
        # ==================================================
        # 6. Plot calibration curve
        # ==================================================
        plt.figure(figsize=(6, 6))

        max_value = calibration_results[
            ["predicted_risk", "observed_risk"]
        ].max().max()

        max_value = 0.3

        plt.plot(
            [0, max_value],
            [0, max_value],
            "--",
            label="Perfect calibration",
            color="gray",
            alpha=0.5
        )

        plt.scatter(
            calibration_results["predicted_risk"],
            calibration_results["observed_risk"],
            color="#1f77b4",
            s=20,
            label="Calibration curve"
        )

        plt.plot(
            calibration_results["predicted_risk"],
            calibration_results["observed_risk"],
            "-",
            color="#1f77b4"
        )

        plt.xlabel("Predicted 1-year MACE risk")

        plt.ylabel(
            "Observed 1-year MACE risk "
            "(Aalen-Johansen CIF)"
        )

        plt.xlim(2e-6, max_value)
        plt.ylim(2e-6, max_value)

        # plt.xscale("log")
        # plt.yscale("log")

        plt.legend()
        plt.grid(False)
        plt.show()




    # Avoid log(0) and log(1)
    eps = 1e-16


    # Convert predicted probability to log-odds
    eval_df["logit_predicted_risk"] = np.log(
        (eval_df["risk_1yr"] + eps) /
        (1 - eval_df["risk_1yr"] + eps)
    )


    # Define observed 1-year MACE outcome
    # MACE happened AND occurred before/equal to 12 months
    eval_df["MACE_1yr"] = (
        (eval_df["event"] == 1) &
        (eval_df["time"] <= eval_time[0])
    ).astype(int)



    # Logistic calibration model
    # logit(observed MACE) = intercept + slope*logit(predicted risk)

    X = sm.add_constant(
        eval_df["logit_predicted_risk"]
    )

    y = eval_df["MACE_1yr"]


    calibration_model = sm.Logit(
        y,
        X
    ).fit()



    # Extract values

    calibration_intercept = (
        calibration_model.params["const"]
    )

    calibration_slope = (
        calibration_model.params["logit_predicted_risk"]
    )


    print("--------------------------------")
    print("Calibration intercept:",
        calibration_intercept)

    print("Calibration slope:",
        calibration_slope)

    print("--------------------------------")



    return calibration_results, calibration_intercept, calibration_slope


def evaluate_brier_score(
    model,
    data,
    time,
    label,
    eval_time,
    mace_index=0,
    plot_curve=True
):
    """
    Calculate and plot the Brier score for the MACE event.

    Parameters
    ----------
    model :
        Trained DeepHit model.

    data : np.ndarray
        Input feature matrix.

    time : np.ndarray
        Observed event or censoring times.

    label : np.ndarray
        Event labels.

    eval_time : int
        Time bin at which the Brier score is reported.

    mace_index : int, default=0
        Index of the MACE event in the DeepHit output.

    Returns
    -------
    results : dict
        Brier score at the evaluation time, integrated Brier
        score, Brier scores over time, and the time grid.
    """

    # ==================================================
    # 1. Predictions from DeepHit
    # ==================================================

    pred = model.call(
        data,
        training=False
    ).numpy()

    # pred shape:
    # (N patients, K events, T time bins)


    # ==================================================
    # 2. Convert DeepHit PMF to cumulative incidence
    # ==================================================

    # CIF(t) = cumulative sum of event probability over time

    mace_cif = np.cumsum(
        pred[:, mace_index, :],
        axis=1
    )


    # ==================================================
    # 3. Define time grid
    # ==================================================

    time_grid = np.arange(
        mace_cif.shape[1]
    )


    # ==================================================
    # 4. Brier score calculation
    # ==================================================

    brier_scores = []

    for t in time_grid:

        # Predicted probability of MACE by time t
        pred_risk = mace_cif[:, t]

        # Observed MACE outcome at time t
        observed = (
            (label.ravel() == 1) &
            (time.ravel() <= t)
        ).astype(int)

        # Simple Brier score
        bs = np.mean(
            (pred_risk - observed) ** 2
        )

        brier_scores.append(bs)

    brier_scores = np.array(
        brier_scores
    )


    # ==================================================
    # 5. Brier score at evaluation time
    # ==================================================

    brier_at_eval_time = brier_scores[
        eval_time
    ]

    print("--------------------------------")
    print(
        f"Brier score at {eval_time} months:",
        brier_at_eval_time
    )


    # ==================================================
    # 6. Integrated Brier Score
    # ==================================================

    ibs = np.trapezoid(
        brier_scores,
        time_grid
    ) / (
        time_grid[-1] - time_grid[0]
    )

    print(
        "Integrated Brier Score:",
        ibs
    )
    print("--------------------------------")


    # ==================================================
    # 7. Plot Brier score over time
    # ==================================================
    if plot_curve:
        plt.figure(
            figsize=(6, 6)
        )

        plt.plot(
            time_grid,
            brier_scores,
            linewidth=2
        )

        plt.axvline(
            eval_time,
            linestyle="--",
            alpha=0.5,
            label=f"{eval_time} months"
        )

        plt.xlabel("Time")
        plt.ylabel("Brier score")
        plt.legend()
        plt.show()


    # ==================================================
    # 8. Return results
    # ==================================================

    return {
        "brier_at_eval_time": brier_at_eval_time,
        "integrated_brier_score": ibs,
        "brier_scores": brier_scores,
        "time_grid": time_grid
    }




def deep_hit_decision_curve(
    model,
    data,
    time,
    label,
    eval_time
):
    """
    Perform decision-curve analysis for the predicted
    1-year MACE risk.

    Parameters
    ----------
    model
        Trained DeepHit model.

    data : numpy.ndarray
        Model-ready patient feature matrix.

    time : numpy.ndarray
        Observed follow-up times.

    label : numpy.ndarray
        Competing-risk outcome labels:
        0 = censored
        1 = MACE
        2 = competing event

    eval_time : list, tuple, or numpy.ndarray
        Evaluation horizon. For example, [12].

    Returns
    -------
    dca_results : pandas.DataFrame
        Net benefit for the model, treat-all strategy,
        and treat-none strategy at each threshold.
    """

    # Ensure TensorFlow-compatible dtype
    data = data.astype(np.float32)

    # ==================================================
    # 1. Obtain predictions from DeepHit
    # ==================================================
    pred = model.call(
        data,
        training=False
    ).numpy()

    # Expected prediction shape:
    # (patients, events, time bins)

    # DeepHit event index:
    # 0 = MACE
    # 1 = competing event
    mace_index = 0

    # Predicted cumulative MACE risk up to 12 months
    risk_1yr = np.sum(
        pred[:, mace_index, :eval_time[0]],
        axis=1
    )

    # ==================================================
    # 2. Create evaluation dataframe
    # ==================================================
    eval_df = pd.DataFrame({
        "time": time.ravel(),
        "event": label.ravel(),
        "risk_1yr": risk_1yr
    })

    # ==================================================
    # 3. Define observed 1-year MACE outcome
    # ==================================================
    eval_df["MACE_1yr"] = (
        (eval_df["event"] == 1) &
        (eval_df["time"] <= eval_time[0])
    ).astype(int)

    # ==================================================
    # 4. Net benefit function
    # ==================================================
    def calculate_net_benefit(
        y_true,
        predicted_risk,
        threshold
    ):
        # Classify patients as high risk
        predicted_positive = (
            predicted_risk >= threshold
        )

        true_positive = np.sum(
            predicted_positive & (y_true == 1)
        )

        false_positive = np.sum(
            predicted_positive & (y_true == 0)
        )

        number_of_patients = len(y_true)

        net_benefit = (
            true_positive / number_of_patients
            -
            (false_positive / number_of_patients)
            *
            (threshold / (1 - threshold))
        )

        return net_benefit

    # ==================================================
    # 5. Define risk thresholds
    # ==================================================
    thresholds = np.array([
        0,
        0.005,
        0.01,
        0.03,
        0.05,
        0.075,
        0.10
    ])

    y_true = eval_df["MACE_1yr"].values
    predicted_risk = eval_df["risk_1yr"].values

    # ==================================================
    # 6. Calculate model net benefit
    # ==================================================
    model_nb = []

    for threshold in thresholds:

        net_benefit = calculate_net_benefit(
            y_true=y_true,
            predicted_risk=predicted_risk,
            threshold=threshold
        )

        model_nb.append(net_benefit)

    model_nb = np.asarray(model_nb)

    # ==================================================
    # 7. Calculate reference strategies
    # ==================================================
    prevalence = np.mean(y_true)

    # Treat everyone
    all_nb = (
        prevalence
        -
        (1 - prevalence)
        *
        (thresholds / (1 - thresholds))
    )

    # Treat nobody
    none_nb = np.zeros(
        len(thresholds)
    )

    # ==================================================
    # 8. Create results table
    # ==================================================
    dca_results = pd.DataFrame({
        "threshold": thresholds,
        "Predictive_model": model_nb,
        "Treat_all": all_nb,
        "Treat_none": none_nb
    })

    print("\nDecision-curve analysis results:")
    print(dca_results.to_string(index=False))

    # ==================================================
    # 9. Plot decision curves
    # ==================================================
    plt.figure(figsize=(6, 6))

    plt.plot(
        thresholds,
        model_nb,
        marker="o",
        label="Predictive model"
    )

    plt.plot(
        thresholds,
        all_nb,
        linestyle="--",
        label="Treat all"
    )

    plt.plot(
        thresholds,
        none_nb,
        linestyle="--",
        label="Treat none"
    )

    plt.xlabel(
        "Risk threshold (1-year MACE probability)"
    )

    plt.ylabel("Net benefit")

    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    return dca_results



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_feature_risk_comparison(
    model,
    reference_patient,
    varying_feature,
    varying_values,
    comparison_feature,
    comparison_values,
    feature_mean=None,
    feature_std=None,
    comparison_labels=None,
    event_index=0,
    eval_time=12,
):
    """
    Plot predicted risk while varying one feature and comparing
    different values of another feature.
    """

    varying_values = np.asarray(varying_values)

    # ==========================================================
    # 1. Create patients with varying feature values
    # ==========================================================

    patients = pd.DataFrame(
        np.repeat(
            reference_patient.values[None, :],
            len(varying_values),
            axis=0,
        ),
        columns=reference_patient.index,
    )

    # Standardize the varying feature when required
    if feature_mean is not None and feature_std is not None:
        patients[varying_feature] = (
            varying_values - feature_mean
        ) / feature_std
    else:
        patients[varying_feature] = varying_values

    if comparison_labels is None:
        comparison_labels = [
            f"{comparison_feature} = {value}"
            for value in comparison_values
        ]

    results = {
        varying_feature: varying_values
    }

    # ==========================================================
    # 2. Create scenarios and predict risk
    # ==========================================================

    plt.figure(figsize=(6, 6))

    for value, scenario_label in zip(
        comparison_values,
        comparison_labels,
    ):
        scenario_patients = patients.copy()
        scenario_patients[comparison_feature] = value

        pred = model.call(
            scenario_patients,
            training=False,
        ).numpy()

        risk = pred[
            :, event_index, :eval_time
        ].sum(axis=1)

        results[scenario_label] = risk

        plt.plot(
            varying_values,
            risk,
            linewidth=2,
            label=scenario_label,
        )

    # ==========================================================
    # 3. Plot risk curves
    # ==========================================================

    plt.xlabel(varying_feature)
    plt.ylabel(
        f"Predicted {eval_time}-month risk"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()

    return pd.DataFrame(results)



import numpy as np
import pandas as pd
import shap


def compute_time_dependent_shap(
    model,
    data,
    feature_names=None,
    event_index=0,
    horizon=12,
    n_background=50,
    seed=321,
):
    """
    Compute time-dependent SHAP values for a trained DeepHit model.

    This function performs calculations only. It does not create
    plots or save files.
    """

    # ============================================================
    # 1. Prepare data
    # ============================================================

    if isinstance(data, pd.DataFrame):

        if feature_names is None:
            feature_names = data.columns.tolist()

        data_array = data.to_numpy(
            dtype=np.float32
        )

    else:

        data_array = np.asarray(
            data,
            dtype=np.float32,
        )

        if feature_names is None:
            feature_names = [
                f"Feature_{i + 1}"
                for i in range(data_array.shape[1])
            ]

    feature_names = list(feature_names)

    if data_array.ndim != 2:
        raise ValueError(
            "data must have shape: patients × features."
        )

    if len(feature_names) != data_array.shape[1]:
        raise ValueError(
            "The number of feature names must match "
            "the number of data columns."
        )

    # ============================================================
    # 2. DeepHit cumulative-incidence prediction
    # ============================================================

    def predict_event_cif(x):

        x = np.asarray(
            x,
            dtype=np.float32,
        )

        pred = model(
            x,
            training=False,
        ).numpy()

        if pred.ndim != 3:
            raise ValueError(
                "Expected model output with shape "
                "(patients, events, time_bins), "
                f"but received {pred.shape}."
            )

        if event_index >= pred.shape[1]:
            raise ValueError(
                f"event_index={event_index} is outside "
                f"the {pred.shape[1]} model events."
            )

        if horizon >= pred.shape[2]:
            raise ValueError(
                f"horizon={horizon} is outside "
                f"the {pred.shape[2]} time bins."
            )

        event_pmf = pred[
            :,
            event_index,
            :(horizon + 1),
        ]

        return np.cumsum(
            event_pmf,
            axis=1,
        )

    # Check the prediction output
    predict_event_cif(
        data_array[:2]
    )

    # ============================================================
    # 3. Select SHAP background
    # ============================================================

    rng = np.random.default_rng(seed)

    n_background = min(
        n_background,
        data_array.shape[0],
    )

    background_indices = rng.choice(
        data_array.shape[0],
        size=n_background,
        replace=False,
    )

    background_data = data_array[
        background_indices
    ]

    # ============================================================
    # 4. Calculate SHAP values
    # ============================================================

    masker = shap.maskers.Independent(
        background_data,
        max_samples=n_background,
    )

    time_labels = [
        f"Time_{time_index}"
        for time_index in range(horizon + 1)
    ]

    explainer = shap.Explainer(
        predict_event_cif,
        masker=masker,
        algorithm="permutation",
        feature_names=feature_names,
        output_names=time_labels,
        seed=seed,
    )

    max_evaluations = max(
        500,
        2 * data_array.shape[1] + 1,
    )

    explanation = explainer(
        data_array,
        max_evals=max_evaluations,
        batch_size=128,
    )

    time_shap = np.asarray(
        explanation.values
    )

    expected_shape = (
        data_array.shape[0],
        data_array.shape[1],
        horizon + 1,
    )

    if time_shap.shape != expected_shape:
        raise ValueError(
            f"Expected SHAP shape {expected_shape}, "
            f"but received {time_shap.shape}."
        )

    # ============================================================
    # 5. Check SHAP reconstruction
    # ============================================================

    predicted_cif = predict_event_cif(
        data_array
    )

    base_values = np.asarray(
        explanation.base_values
    )

    if base_values.ndim == 1:
        base_values = np.broadcast_to(
            base_values,
            predicted_cif.shape,
        )

    reconstructed_cif = (
        base_values
        + np.sum(time_shap, axis=1)
    )

    reconstruction_error = np.abs(
        predicted_cif - reconstructed_cif
    )

    # ============================================================
    # 6. Calculate feature importance
    # ============================================================

    time_points = np.arange(
        horizon + 1,
        dtype=float,
    )

    if hasattr(np, "trapezoid"):
        integrated_shap = np.trapezoid(
            np.abs(time_shap),
            x=time_points,
            axis=2,
        )
    else:
        integrated_shap = np.trapz(
            np.abs(time_shap),
            x=time_points,
            axis=2,
        )

    global_importance = np.mean(
        integrated_shap,
        axis=0,
    )

    mean_absolute_shap_by_time = np.mean(
        np.abs(time_shap),
        axis=0,
    )

    shap_at_horizon = time_shap[
        :,
        :,
        horizon,
    ]

    importance_table = pd.DataFrame({
        "Feature_index": np.arange(
            len(feature_names)
        ),
        "Feature": feature_names,
        "Mean_integrated_absolute_SHAP":
            global_importance,
    })

    importance_table = importance_table.sort_values(
        "Mean_integrated_absolute_SHAP",
        ascending=False,
    ).reset_index(drop=True)

    # ============================================================
    # 7. Return results
    # ============================================================

    return {
        "explanation": explanation,
        "explain_data": data_array,
        "feature_names": feature_names,
        "time_shap": time_shap,
        "shap_at_horizon": shap_at_horizon,
        "integrated_shap": integrated_shap,
        "global_importance": global_importance,
        "mean_absolute_shap_by_time":
            mean_absolute_shap_by_time,
        "importance_table": importance_table,
        "predicted_cif": predicted_cif,
        "reconstruction_error": reconstruction_error,
        "time_points": time_points,
        "event_index": event_index,
        "horizon": horizon,
    }