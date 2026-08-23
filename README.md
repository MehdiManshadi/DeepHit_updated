# DeepHit Updated

TensorFlow/Keras implementation of DeepHit for discrete-time survival analysis with competing risks, updated for current Python workflows and extended with model training, cross-validation, external validation, performance evaluation, and explainability utilities.

This repository is a research implementation based on the original DeepHit method described by Lee et al. (AAAI 2018). It is not the original DeepHit repository and is not distributed as a clinical prediction tool.

## Overview

DeepHit models the joint distribution of event type and event time without imposing a parametric survival distribution. The current implementation is configured for two outcomes:

1. The main event of interest
2. A competing event

The model returns a probability mass function with shape:

```text
patients × events × time bins
```

The repository supports:

- shared and event-specific neural-network layers;
- log-likelihood and ranking losses;
- validation-based early stopping and best-weight restoration;
- event-specific time-dependent C-index calculation;
- K-fold cross-validation and hyperparameter grid search;
- bootstrap confidence intervals;
- external-cohort validation;
- calibration analysis using the Aalen-Johansen estimator;
- Brier-score trajectories and integrated Brier score;
- decision-curve analysis;
- permutation feature importance;
- feature-risk scenario comparisons; and
- time-dependent SHAP analysis.

## Repository structure

| File | Description |
| --- | --- |
| `RunMe.ipynb` | Main notebook demonstrating data loading, preprocessing, hyperparameter search, final training, external validation, evaluation, and explainability. |
| `ClassDeepHit.py` | Defines the `DeepHitPlus` TensorFlow/Keras model and its log-likelihood, ranking, and total-loss functions. |
| `ImportDatabase.py` | Loads the CSV dataset, applies the study-specific time-horizon rules, selects feature groups, and constructs DeepHit masks. |
| `trainDeepHit.py` | Implements training, early stopping, time-dependent C-index evaluation, K-fold cross-validation, and grid search. |
| `model_evaluation.py` | Contains bootstrap C-index, permutation importance, calibration, Brier score, decision-curve, risk-comparison, and time-dependent SHAP utilities. |
| `utils_eval.py` | Implements the cause-specific time-dependent C-index. |
| `runDeepHit.py` | Script-based training and external-validation workflow. |
| `processed_data_analysis.py` | Optional exploratory analyses using UMAP and CatBoost. This file contains study-specific local paths that must be changed before use. |

## Installation

Clone the repository and create an isolated Python environment:

```bash
git clone https://github.com/MehdiManshadi/DeepHit_updated.git
cd DeepHit_updated

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the packages used by the repository:

```bash
pip install \
  tensorflow \
  numpy \
  pandas \
  scikit-learn \
  python-dotenv \
  matplotlib \
  lifelines \
  statsmodels \
  shap \
  seaborn \
  umap-learn \
  catboost \
  jupyter
```

The core DeepHit workflow does not require UMAP or CatBoost; those packages are used only by `processed_data_analysis.py`.

> Dependency versions are not currently pinned. For reproducible research, create and commit a tested `requirements.txt` or environment file.

## Data configuration

The clinical datasets are not included in this repository. Set the `DATA_DIR` environment variable to the directory containing the model-ready CSV files.

Create a `.env` file in the repository root:

```dotenv
DATA_DIR=/absolute/path/to/your/data
```

The notebook currently expects the development dataset to be named:

```text
Region_Sthlm_Gotland_model_data_binary.csv
```

The external-validation example expects:

```text
Region_Uppsala_Örebro_model_data_binary.csv
Region_Väst_model_data_binary.csv
Region_Syd_model_data_binary.csv
Region_Sydöstra_model_data_binary.csv
Region_Norr_model_data_binary.csv
```

Add `.env`, clinical data, trained models, and generated outputs to `.gitignore`. Do not commit sensitive or patient-level data.

## Expected input format

`ImportDatabase.import_dataset()` expects a CSV file in which:

- the first two columns are `time` and `label`;
- `label = 0` represents censoring;
- positive labels represent event types (`1`, `2`, ...); and
- the remaining columns contain model features.

The importer currently selects feature groups using fixed column positions. These positions are specific to the study dataset and must be updated if the input schema changes:

```python
basics = df.columns[2:9]
planned = df.columns[9:13]
atcs = df.columns[13:61]
icds = df.columns[61:77]
icds_specific = df.columns[77:82]
rfs = df.columns[82:87]
icds_summary = df.columns[[87, 88, 89]]
```

The importer also applies a study-specific 12-month horizon by converting time and censoring observations at the horizon. Review this logic before using a dataset with different time units or follow-up rules.

## Running the workflow

The recommended entry point is the notebook:

```bash
jupyter notebook RunMe.ipynb
```

Run the notebook sections in order:

1. Import packages and set random seeds.
2. Load and preprocess the development dataset.
3. Run K-fold cross-validation and hyperparameter grid search.
4. Train the final model using the selected architecture.
5. Evaluate the model internally and across external regions.
6. Calculate calibration, Brier score, and decision-curve results.
7. Run permutation importance and time-dependent SHAP analyses.

The script workflow can be started with:

```bash
python runDeepHit.py
```

Review its dataset paths and settings before execution. The notebook contains the more complete and current end-to-end workflow.

## Minimal Python example

```python
from pathlib import Path
import os

import ImportDatabase as impt
from ClassDeepHit import DeepHitPlus
from trainDeepHit import TrainDeepHit


data_path = (
    Path(os.environ["DATA_DIR"])
    / "Region_Sthlm_Gotland_model_data_binary.csv"
)

x_dim, (data, time, label), (mask1, mask2), feature_names = (
    impt.import_dataset(data_path, time_scale=1.2)
)

model = DeepHitPlus(
    num_event=mask1.shape[1],
    num_bins=mask2.shape[1],
    shared_layers=(32, 16),
    MR_layers=(10, 10),
    CR_layers=(5, 10),
    dropout=0.1,
)

model = TrainDeepHit(
    model=model,
    tr_data=data,
    tr_mask1=mask1,
    tr_mask2=mask2,
    tr_time=time,
    tr_label=label,
    eval_time=[12],
    test_size=0.2,
    max_epochs=100,
    batch_size=128,
    learning_rate=1e-3,
    alpha=1.0,
    beta=1.0,
)

predicted_pmf = model(data, training=False).numpy()
print(predicted_pmf.shape)
```

## Model architecture

`DeepHitPlus` contains:

- a shared multilayer network;
- concatenation of the shared representation with the original features;
- separate event-specific subnetworks;
- dropout before the output layer; and
- a final softmax layer over all event-time combinations.

The total training objective is:

```text
total loss = alpha × log-likelihood loss + beta × ranking loss
```

The current `event_nets` definition contains two event-specific subnetworks. Extend that construction before using more than two event types.

## Model evaluation

### Discrimination

`compute_cindex_at_time()` calculates an event-specific C-index from cumulative predicted risk at a selected horizon. `bootstrap_cindex_at_time()` adds percentile-based bootstrap confidence intervals.

### Calibration

`deep_hit_calibration()` compares predicted risk groups with observed cumulative incidence estimated using Aalen-Johansen and reports a logistic calibration intercept and slope.

### Brier score

`evaluate_brier_score()` returns the Brier score at the selected horizon, the score over time, and an integrated score. The current implementation is an unweighted estimate and does not apply inverse-probability-of-censoring weighting; interpret it accordingly when censoring is present.

### Clinical utility

`deep_hit_decision_curve()` calculates net benefit for the model, treat-all, and treat-none strategies over predefined risk thresholds.

## Explainability

The repository provides two complementary approaches:

- **Permutation importance:** measures the reduction in event-specific C-index after permuting each feature.
- **Time-dependent SHAP:** explains cumulative incidence across time bins and produces integrated global importance, population-level temporal importance, and horizon-specific SHAP values.

Permutation SHAP can be computationally expensive because it evaluates many perturbed feature combinations. `n_background` controls the size of the reference sample; it does not limit the number of patients being explained.

## Reproducibility and external validation

Random seeds are set for Python, NumPy, and TensorFlow. TensorFlow deterministic operations are requested through `TF_DETERMINISTIC_OPS`.

For valid external evaluation:

- estimate preprocessing parameters using the development/training data only;
- apply those same parameters to every validation cohort;
- do not refit preprocessing on an external cohort; and
- do not use external outcomes for training, early stopping, or model selection.

## Important limitations

- The data-import logic and feature positions are study-specific.
- The current neural-network construction is designed for two events.
- Dependency versions are not pinned.
- The repository does not include automated tests.
- Some evaluation utilities use simplified estimators and should be reviewed before formal clinical reporting.
- This research code has not been validated for direct clinical decision-making.

## Reference

If you use this implementation, cite the original DeepHit paper:

> Lee, C., Zame, W. R., Yoon, J., & van der Schaar, M. (2018). DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks. *Proceedings of the AAAI Conference on Artificial Intelligence, 32*(1).

- Paper: https://ojs.aaai.org/index.php/AAAI/article/view/11842
- Original implementation: https://github.com/chl8856/DeepHit

## License

No license file is currently included. Add an appropriate open-source license before redistributing or accepting external contributions.

## Author

Mehdi Dehghan Manshadi  
GitHub: https://github.com/MehdiManshadi
