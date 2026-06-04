import numpy as np
import pandas as pd


# ==================================================
# MASK GENERATION (DeepHit)
# ==================================================

def f_get_fc_mask1(time, label, num_event, num_category):
    """
    Likelihood mask for DeepHit.

    Shape:
        [N, num_event, num_category]
    """
    n_samples = time.shape[0]
    mask = np.zeros((n_samples, num_event, num_category))

    for i in range(n_samples):
        t = int(time[i, 0])

        # event occurred
        if label[i, 0] != 0:
            event_type = int(label[i, 0]) - 1
            mask[i, event_type, t] = 1

        # censored
        else:
            mask[i, :, t + 1:] = 1

    return mask


def f_get_fc_mask2(time, num_category):
    """
    Ranking mask for DeepHit.

    Shape:
        [N, num_category]
    """
    n_samples = time.shape[0]
    mask = np.zeros((n_samples, num_category))

    for i in range(n_samples):
        t = int(time[i, 0])
        mask[i, :t + 1] = 1

    return mask


# ==================================================
# DATA LOADING PIPELINE
# ==================================================

def import_dataset(csv_path, time_scale=1.2):
    """
    Load and preprocess dataset for DeepHit survival analysis.

    Returns:
        x_dim  : number of features
        DATA   : (features, time, label)
        MASK   : (mask1, mask2)
    """

    # ---- Load raw data ----
    df = pd.read_csv(csv_path)

    # ---- Censoring rule (time horizon) ----
    df["time"] = df["time"] * 12
    df["label"] = np.where(df["time"] >= 12, 0, df["label"])
    df["time"] = np.where(df["time"] >= 12, 12, df["time"])

    # ---- Define feature groups ----
    # this part is specific to the dataset structure and should be adapted if the dataset changes
    time_label_cols = df.columns[0:2].tolist()

    basics = df.columns[2:9].tolist()
    planned = df.columns[9:13].tolist()
    atcs = df.columns[13:61].tolist()
    icds = df.columns[61:77].tolist()
    icds_specific = df.columns[77:82].tolist()
    rfs = df.columns[82:87].tolist()
    icds_summary = df.columns[[87, 88, 89]].tolist()

    feature_cols = (
        basics
        + planned
        + atcs
        + rfs
        + icds
        + icds_specific
        + icds_summary
    )
    # Note: the specific column indices for feature groups are based on the original dataset and may need to be updated if the dataset structure changes.
    # ---- Subset dataset ----
    df = df[time_label_cols + feature_cols]

    # ---- Extract arrays ----
    time = df["time"].to_numpy().reshape(-1, 1)
    label = df["label"].to_numpy().astype(int).reshape(-1, 1)
    data = df.iloc[:, 2:].to_numpy()

    # ---- DeepHit configuration ----
    num_category = int(np.round(np.max(time) * time_scale))
    num_event = int(np.max(label))
    x_dim = data.shape[1]

    # ---- Masks ----
    mask1 = f_get_fc_mask1(time, label, num_event, num_category)
    mask2 = f_get_fc_mask2(time, num_category)

    # ---- Output structure ----
    dim = x_dim
    data_tuple = (data, time, label)
    mask_tuple = (mask1, mask2)

    return dim, data_tuple, mask_tuple