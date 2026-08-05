# This script trains the DeepHit model on the provided dataset and 
# evaluates its performance using the C-index at a specified time horizon.
# It also includes a permutation feature importance analysis to identify
# which features are most influential for the model's predictions at that horizon.
import os
import random
import numpy as np
import tensorflow as tf
from trainDeepHit import *
from sklearn.model_selection import KFold
from utils_eval import c_index
import pandas as pd
import ImportDatabase as impt
from ClassDeepHit import DeepHitPlus

SEED = 13
num_Event = 2
eval_time = 12
CV_ITERATION = 3

def setSeed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"   # deterministic TF ops
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

setSeed(SEED)

# ==================================================
# Load data
# ==================================================
path = (
    "/Users/mehman/Projects/0_Reference_Data/0_otherRegions/Processed/Region_Sthlm_Gotland_model_data_binary.csv"
)

x_dim, (data, time, label), (mask1, mask2) = impt.import_dataset(path,time_scale=1.2)

# Normalize age and log-transform certain features
data[:, 0] = (data[:, 0] - data[:, 0].mean()) / data[:, 0].std()

for j in range(data.shape[1] - 3, data.shape[1]): 
    data[:, j] = np.log1p(data[:, j])
    data[:, j] = (data[:, j] - data[:, j].mean()) / data[:, j].std()

# --------------------------------------------------
# Ensure TensorFlow-compatible dtypes
# --------------------------------------------------
data   = data.astype(np.float32)
mask1  = mask1.astype(np.float32)
mask2  = mask2.astype(np.float32)
time   = time.astype(np.float32)
label  = label.astype(np.int32)   


# ==================================================
# Train / Validation split
# ==================================================
kf = KFold(n_splits=CV_ITERATION, shuffle=True, random_state=SEED)
splits = list(kf.split(data))
for i in [2]:
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
    # Model definition
    # ==================================================
    num_bins = mask2.shape[1]   # T (time bins)

    model = DeepHitPlus(
        num_event=num_Event,
        num_bins=num_bins,
        shared_layers=(30, 15, 5),
        MR_layers=[5, 10],
        CR_layers=(5, 10),
        dropout=0.25   
    )

    # Build model (optional sanity check)
    pmf = model(tr_data[:2], training=False)
    print("PMF shape:", pmf.shape)   # (B, K, T)

    model = TrainDeepHit(model, tr_data, tr_mask1, tr_mask2, tr_time, tr_label,
                    va_data, va_mask1, va_mask2, va_time, va_label, eval_time = [12], 
                    test_size = 0.2, max_epochs = 100, batch_size = 128, learning_rate = 1e-3, alpha=1.0, beta=1.0)

    print("\n===== TRAINING FINISHED =====\n")


    def compute_cindex_at_time(model, data, time, label, eval_horizon):
        """
        Compute event-specific C-index at a given time horizon.
        """
        pred = model(data, training=False).numpy()  # (N, K, T)

        if eval_horizon >= pred.shape[2]:
            raise ValueError("eval_horizon exceeds num_Category")

        # Risk up to horizon
        risk = np.sum(pred[:, :, :(eval_horizon + 1)], axis=2)  # (N, K)

        cidx = np.zeros(num_Event)
        for k in range(num_Event):
            cidx[k] = c_index(
                risk[:, k],
                time,
                (label[:, 0] == k + 1).astype(int),
                eval_horizon
            )
        return cidx

    eval_horizon = 12  # same horizon you report in results

    baseline_cindex = compute_cindex_at_time(
        model,
        va_data,
        va_time,
        va_label,
        eval_horizon
    )

    print("Baseline C-index:", baseline_cindex)

    def permutation_feature_importance(
        model,
        x_val,
        time_val,
        label_val,
        eval_horizon,
        n_repeats=5
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

    pfi = permutation_feature_importance(
        model=model,
        x_val=va_data,
        time_val=va_time,
        label_val=va_label,
        eval_horizon=eval_horizon,
        n_repeats=3
    )

    print("PFI shape:", pfi.shape)  # (num_Event, num_features)

    # Top 10 features per event
    for k in range(num_Event):
        top_idx = np.argsort(-pfi[k])[:10]
        print(f"\nTop features for event {k+1}:")
        print(top_idx)
        print(pfi[k, top_idx])

path = np.array([
    "/Users/mehman/Projects/0_Reference_Data/0_otherRegions/Processed/Region_Uppsala_Örebro_model_data_binary.csv",
    "/Users/mehman/Projects/0_Reference_Data/0_otherRegions/Processed/Region_Väst_model_data_binary.csv",
    "/Users/mehman/Projects/0_Reference_Data/0_otherRegions/Processed/Region_Syd_model_data_binary.csv",
    "/Users/mehman/Projects/0_Reference_Data/0_otherRegions/Processed/Region_Sydöstra_model_data_binary.csv",
    "/Users/mehman/Projects/0_Reference_Data/0_otherRegions/Processed/Region_Norr_model_data_binary.csv"]
    )
Region_names = {
    0: "Region Uppsala_Örebro",
    1: "Region Väst",
    2: "Region Syd",
    3: "Region Sydöstra",
    4: "Region Norr"
}

for i in range(len(path)):

    x_dim, (data, time, label), (mask1, mask2) = impt.import_dataset(path[i],time_scale=1.2)

    num_Category = mask2.shape[1]   # T (time bins)
    eval_time = [12]
    data[:, 0] = (data[:, 0] - data[:, 0].mean()) / data[:, 0].std()

    for j in range(data.shape[1] - 3, data.shape[1]):
        data[:, j] = np.log1p(data[:, j])
        data[:, j] = (data[:, j] - data[:, j].mean()) / data[:, j].std()

    data   = data.astype(np.float32)
    mask1  = mask1.astype(np.float32)
    mask2  = mask2.astype(np.float32)
    time   = time.astype(np.float32)
    label  = label.astype(np.int32) 

    pred = model.call(data)
    resultfeat = np.zeros([num_Event, len(eval_time)])

    for t, t_time in enumerate(eval_time):
        eval_horizon = int(t_time)

        if eval_horizon >= num_Category:
            print('ERROR: evaluation horizon is out of range')
            resultfeat[:, t] = -1
        else:
            # calculate F(t | x, Y, t >= t_M) = \sum_{t_M <= \tau < t} P(\tau | x, Y, \tau > t_M)
            risk = np.sum(pred[:,:,:(eval_horizon+1)], axis=2) #risk score until eval_time

            # Calculate permutation importance for each eval horizon and event
            for k in range(num_Event):
                resultfeat[k, t] = c_index(risk[:,k], time, (label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
                
                #result.loc[k+1, feat, perm_iter, "Delta t = " + "{:03d}".format(t_time)] = result1[k,t] - resultfeat[k,t]
                # since we compare risk_scores, the true label is that event occurs before time horizon
                #result.loc[k+1, feat, perm_iter, "Delta t = " + "{:03d}".format(t_time)] = result1[k,t] - resultfeat[k,t]
                # since we compare risk_scores, the true label is that event occurs before time horizon
    print(
        f"{Region_names[i]:<25}\t"
        f"main risk: {resultfeat[0,0]:.3f}\t"
        f"competing risk: {resultfeat[1,0]:.3f}"
    )
