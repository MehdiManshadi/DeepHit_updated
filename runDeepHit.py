"""
DeepHit Survival Analysis Pipeline
An updated implementation of the DeepHit model according to the new version of python and libraries
for competing risks survival analysis.
The script includes data loading, preprocessing, model training with cross-validation, 
and final evaluation across multiple regions.
----------------------------------
This script trains a DeepHit model for competing risks survival analysis,
evaluates performance using time-dependent C-index, and reports regional results.
"""

import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import KFold
from utils_eval import c_index
import ImportDatabase as impt
from ClassDeepHit import DeepHitPlus
from trainDeepHit import TrainDeepHit


# ==================================================
# CONFIGURATION
# ==================================================
SEED = 13
NUM_EVENTS = 2
EVAL_TIME = 12
CV_FOLDS = 3


def set_seed(seed: int = SEED):
    """Ensure full reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed(SEED)


# ==================================================
# DATA PATHS
# ==================================================
DATA_PATH = (
    "/Users/mehman/Projects/0_Reference_Data/"
    "0_otherRegions/Processed/"
    "Region_Sthlm_Gotland_model_data_binary.csv"
)


# ==================================================
# LOAD + PREPROCESS DATA
# ==================================================
x_dim, (data, time, label), (mask1, mask2) = impt.import_dataset(
    DATA_PATH, time_scale=1.2
)

# Standardization
data[:, 0] = (data[:, 0] - data[:, 0].mean()) / data[:, 0].std()

for j in range(data.shape[1] - 3, data.shape[1]):
    data[:, j] = np.log1p(data[:, j])
    data[:, j] = (data[:, j] - data[:, j].mean()) / data[:, j].std()

# Cast types (TF compatibility)
data = data.astype(np.float32)
mask1 = mask1.astype(np.float32)
mask2 = mask2.astype(np.float32)
time = time.astype(np.float32)
label = label.astype(np.int32)


# ==================================================
# CROSS-VALIDATION SETUP
# ==================================================
kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
splits = list(kf.split(data))


# ==================================================
# MAIN TRAINING LOOP (CROSS-VALIDATION)
# ==================================================
for fold, (train_idx, val_idx) in enumerate(splits):

    print(f"\n===== FOLD {fold + 1}/{CV_FOLDS} =====\n")

    tf.keras.backend.clear_session()
    set_seed(SEED)

    tr_data, va_data = data[train_idx], data[val_idx]
    tr_time, va_time = time[train_idx], time[val_idx]
    tr_label, va_label = label[train_idx], label[val_idx]
    tr_mask1, va_mask1 = mask1[train_idx], mask1[val_idx]
    tr_mask2, va_mask2 = mask2[train_idx], mask2[val_idx]

    # ==================================================
    # MODEL
    # ==================================================
    num_bins = mask2.shape[1]

    model = DeepHitPlus(
        num_event=NUM_EVENTS,
        num_bins=num_bins,
        shared_layers=(30, 15, 5),
        MR_layers=[5, 10],
        CR_layers=(5, 10),
        dropout=0.25
    )

    # sanity forward pass
    _ = model(tr_data[:2], training=False)

    print("Model initialized successfully.")

    # ==================================================
    # TRAINING
    # ==================================================
    model = TrainDeepHit(
        model=model,
        tr_data=tr_data,
        tr_mask1=tr_mask1,
        tr_mask2=tr_mask2,
        tr_time=tr_time,
        tr_label=tr_label,
        va_data=va_data,
        va_mask1=va_mask1,
        va_mask2=va_mask2,
        va_time=va_time,
        va_label=va_label,
        eval_time=[EVAL_TIME],
        test_size=0.2,
        max_epochs=100,
        batch_size=128,
        learning_rate=1e-3,
        alpha=1.0,
        beta=1.0
    )

    print("\n===== TRAINING FINISHED =====\n")


# ==================================================
# FINAL REGIONAL EVALUATION
# ==================================================
REGION_PATHS = np.array([
    "/Users/mehman/Projects/0_Reference_Data/0_otherRegions/Processed/Region_Uppsala_Örebro_model_data_binary.csv",
    "/Users/mehman/Projects/0_Reference_Data/0_otherRegions/Processed/Region_Väst_model_data_binary.csv",
    "/Users/mehman/Projects/0_Reference_Data/0_otherRegions/Processed/Region_Syd_model_data_binary.csv",
    "/Users/mehman/Projects/0_Reference_Data/0_otherRegions/Processed/Region_Sydöstra_model_data_binary.csv",
    "/Users/mehman/Projects/0_Reference_Data/0_otherRegions/Processed/Region_Norr_model_data_binary.csv"
])

REGION_NAMES = {
    0: "Region Uppsala–Örebro",
    1: "Region Väst",
    2: "Region Syd",
    3: "Region Sydöstra",
    4: "Region Norr"
}


print("\n===== REGIONAL PERFORMANCE =====\n")

for i, path in enumerate(REGION_PATHS):

    x_dim, (data, time, label), (mask1, mask2) = impt.import_dataset(
        path, time_scale=1.2
    )

    data[:, 0] = (data[:, 0] - data[:, 0].mean()) / data[:, 0].std()

    for j in range(data.shape[1] - 3, data.shape[1]):
        data[:, j] = np.log1p(data[:, j])
        data[:, j] = (data[:, j] - data[:, j].mean()) / data[:, j].std()

    data = data.astype(np.float32)
    time = time.astype(np.float32)
    label = label.astype(np.int32)

    pred = model.call(data)

    num_Category = mask2.shape[1]
    eval_time = [EVAL_TIME]

    result = np.zeros((NUM_EVENTS, len(eval_time)))

    for t, t_time in enumerate(eval_time):

        eval_horizon = int(t_time)

        if eval_horizon >= num_Category:
            result[:, t] = -1
            continue

        risk = np.sum(pred[:, :, :(eval_horizon + 1)], axis=2)

        for k in range(NUM_EVENTS):
            result[k, t] = c_index(
                risk[:, k],
                time,
                (label[:, 0] == k + 1).astype(int),
                eval_horizon
            )

    print(
        f"{REGION_NAMES[i]:<20} | "
        f"Main risk C-index: {result[0, 0]:.3f} | "
        f"Competing risk C-index: {result[1, 0]:.3f}"
    )