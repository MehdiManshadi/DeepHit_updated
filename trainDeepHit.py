import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import os
import random
from utils_eval import c_index
from itertools import product
from ClassDeepHit import DeepHitPlus
import pandas as pd

SEED = 13


def setSeed(seed=SEED):
    """Ensure reproducibility across runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)




def compute_cindex_at_time(model, data, time, label, mask1, eval_horizon):
    """
    Compute event-specific C-index at a given time horizon.
    """
    pred = model.call(data, training=False).numpy()  # (N, K, T)

    if eval_horizon >= pred.shape[2]:
        raise ValueError("eval_horizon exceeds num_Category")

    # Risk up to horizon
    risk = np.sum(pred[:, :, :(eval_horizon + 1)], axis=2)  # (N, K)

    num_Event = mask1.shape[1]
    cidx = np.zeros(num_Event)
    for k in range(num_Event):
        cidx[k] = c_index(
            risk[:, k],
            time,
            (label[:, 0] == k + 1).astype(int),
            eval_horizon
        )
    return cidx

def TrainDeepHit(
    model,
    tr_data, tr_mask1, tr_mask2, tr_time, tr_label,
    eval_time,
    test_size=0.2,
    max_epochs=100,
    batch_size=128,
    learning_rate=1e-3,
    alpha=1.0,
    beta=1.0
):
    """
    Train DeepHit model with validation-based early stopping
    and time-dependent C-index evaluation.
    """

    # ==================================================
    # Optimizer
    # ==================================================
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    num_Category = tr_mask2.shape[1]

    # ==================================================
    # Train step
    # ==================================================
    @tf.function
    def trainStep(x, mask1, mask2, label, timeIndex):
        with tf.GradientTape() as tape:
            pmf = model(x, training=True)

            totalLoss, lossLL, lossRank = model.total_loss(
                pmf,
                mask1,
                mask2,
                label,
                timeIndex,
                alpha=alpha,
                beta=beta
            )

        grads = tape.gradient(totalLoss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return totalLoss, lossLL, lossRank

    # ==================================================
    # Validation step
    # ==================================================
    @tf.function
    def testStep(x, mask1, mask2, label, timeIndex):
        pmf = model(x, training=False)

        totalLoss, lossLL, lossRank = model.total_loss(
            pmf,
            mask1,
            mask2,
            label,
            timeIndex,
            alpha=alpha,
            beta=beta
        )

        return totalLoss, lossLL, lossRank

    # ==================================================
    # Train/Validation split
    # ==================================================
    (tr_data, te_data,
     tr_time, te_time,
     tr_label, te_label,
     tr_mask1, te_mask1,
     tr_mask2, te_mask2) = train_test_split(
        tr_data, tr_time, tr_label, tr_mask1, tr_mask2,
        test_size=test_size,
        random_state=SEED
    )

    num_Event = tr_mask1.shape[1]
    N = tr_data.shape[0]

    print("\n===== START TRAINING =====\n")

    global_iter = 0
    best_test_loss = np.inf
    patience = 3
    best_cindex = -np.inf

    # ==================================================
    # Training loop
    # ==================================================
    batch_loss_history = []
    iteration_history = []
    for epoch in range(max_epochs):
        # reset metrics for this epoch
        epoch_losses = []
        epoch_ll = []
        epoch_rank = []
        idx = np.random.permutation(N)

        for i in range(0, N, batch_size):
            b = idx[i:i + batch_size]

            loss, ll, rank = trainStep(
                tr_data[b],
                tr_mask1[b],
                tr_mask2[b],
                tr_label[b],
                tr_time[b]
            )

            epoch_losses.append(loss.numpy())
            epoch_ll.append(ll.numpy())
            epoch_rank.append(rank.numpy())

            batch_loss_history.append(loss.numpy())
            iteration_history.append(global_iter)
            global_iter += 1
        
        # ==================================================
        # Validation loss (early stopping)
        # ==================================================
        teLoss, _, _ = testStep(
            te_data,
            te_mask1,
            te_mask2,
            te_label,
            te_time
        )

        teLoss= teLoss.numpy()

        if teLoss < best_test_loss:
            best_test_loss = teLoss
            best_weights = model.get_weights()
            patience = 10
        else:
            patience -= 1

        if patience == 0:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # ==================================================
        # C-index evaluation
        # ==================================================
        pred = model.call(te_data, training=False).numpy()
        resultfeat = np.zeros([num_Event, len(eval_time)])

        for t, t_time in enumerate(eval_time):
            eval_horizon = int(t_time)

            if eval_horizon >= num_Category:
                print("ERROR: evaluation horizon is out of range")
                resultfeat[:, t] = -1
                continue

            risk = np.sum(pred[:, :, :(eval_horizon + 1)], axis=2)

            for k in range(num_Event):
                resultfeat[k, t] = c_index(
                    risk[:, k],
                    te_time,
                    (te_label[:, 0] == k + 1).astype(int),
                    eval_horizon
                )

        print(
            f"Epoch {epoch:03d} | "
            f"Train {np.mean(epoch_losses):.4f} "
            f"(LL {np.mean(epoch_ll):.4f}, Rank {np.mean(epoch_rank):.4f}) | "
            f"Test {teLoss:.4f} | "
            f"Test C-index {resultfeat[0, 0]:.4f} "
            f"(competing event: {resultfeat[1, 0]:.4f})"
        )

    model.set_weights(best_weights) # Restore best model weights
    return model


def k_fold_cross_validation(hyperparameters, data, time, label, mask1, mask2, CV_ITERATION=5, eval_time=[12]):
    # ==================================================
    # Train / Validation split
    # ==================================================
    kf = KFold(n_splits=CV_ITERATION, shuffle=True, random_state=SEED)
    splits = list(kf.split(data))
    baseline_cindex = np.zeros((CV_ITERATION, 2))  # Store C-index for each fold

    for i in range(CV_ITERATION):
        tf.keras.backend.clear_session()
        setSeed(SEED)
        train_index, val_index = splits[i]

        tr_data     = data[train_index]
        tr_time     = time[train_index]
        tr_label    = label[train_index]
        tr_mask1    = mask1[train_index]
        tr_mask2    = mask2[train_index]

        va_data     = data[val_index]
        va_time     = time[val_index]
        va_label    = label[val_index]
        va_mask1    = mask1[val_index]
        va_mask2    = mask2[val_index]
        # ==================================================
        num_bins = mask2.shape[1]   # T (time bins)
        num_Event = mask1.shape[1]  # K (number of events)        
        model = DeepHitPlus(
        num_event=num_Event,
        num_bins=num_bins,
        shared_layers=hyperparameters["shared_layers"],
        MR_layers=hyperparameters["MR_layers"],
        CR_layers=hyperparameters["CR_layers"],
        dropout=hyperparameters["dropout"]
                    )
        model = TrainDeepHit(model, tr_data, tr_mask1, tr_mask2, tr_time, tr_label,
                        eval_time = eval_time, 
                        test_size = 0.2, max_epochs = 100, batch_size = 128, learning_rate = 1e-3, alpha=1.0, beta=1.0)

        print("\n===== TRAINING FINISHED =====\n")


        baseline_cindex[i] = compute_cindex_at_time(
            model,
            va_data,
            va_time,
            va_label,
            va_mask1,
            eval_horizon=eval_time[0]
        )

        print(f"Fold {i+1} C-index: {baseline_cindex[i, 0]:.3f} (competing event: {baseline_cindex[i, 1]:.3f})")

    return baseline_cindex


def optimize_hyperparameters_grid_search(data, time, label, mask1, mask2, parameters_grid, CV_ITERATION=5, eval_time=[12]):
    keys = parameters_grid.keys()
    results = []

    for values in product(*parameters_grid.values()):
        params = dict(zip(keys, values))

        baseline_cindex = k_fold_cross_validation(
            data=data,
            time=time,
            label=label,
            mask1=mask1,
            mask2=mask2,
            hyperparameters=params,
            CV_ITERATION=CV_ITERATION,
            eval_time=eval_time
            )

        avg_cindex = np.mean(baseline_cindex, axis=0)

        results.append({
            **params,
            "cindex_event1_per_fold": baseline_cindex[:, 0].tolist(),  # first event across all folds
            "cindex_event2_per_fold": baseline_cindex[:, 1].tolist(),  # second event across all folds
            "cindex_MACE": avg_cindex[0],
            "cindex_competing": avg_cindex[1]
        })

        print(
            f"Params: {params} | "
            f"MACE: {avg_cindex[0]:.3f} | "
            f"Competing: {avg_cindex[1]:.3f}"
        )

        results_df = pd.DataFrame(results)

    return results_df

        

    # Placeholder for hyperparameter optimization logic
    # This function can be implemented to perform grid search or random search
    # over hyperparameters such as learning rate, batch size, alpha, beta, tc.