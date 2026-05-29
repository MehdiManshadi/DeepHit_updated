import numpy as np
import pandas as pd

def f_get_fc_mask1(time, label, num_Event, num_Category):
    """
    Likelihood mask
    shape: [N, num_Event, num_Category]
    """
    N = int(time.shape[0])
    mask = np.zeros((N, num_Event, num_Category))

    for i in range(N):
        t = int(time[i, 0])
        if label[i, 0] != 0:  # event occurred
            k = int(label[i, 0]) - 1
            mask[i, k, t] = 1
        else:  # censored
            mask[i, :, t + 1:] = 1

    return mask


def f_get_fc_mask2(time, num_Category):
    """
    Ranking mask
    shape: [N, num_Category]
    """
    N = time.shape[0]
    mask = np.zeros((N, num_Category))

    # single-measurement case (DeepHit default)
    for i in range(N):
        t = int(time[i, 0])
        mask[i, :t + 1] = 1

    return mask



def import_dataset(
    csv_path,
    time_scale=1.2
):
    # ---- Load data ----
    df = pd.read_csv(csv_path)
    
    # ---- Enforce censoring beyond horizon (using index) ----
    df['time'] = df['time'] * 12
    df["label"] = np.where(df["time"] >= 12, 0, df["label"])
    df["time"]  = np.where(df["time"] >= 12, 12, df["time"])

    TimeLabel = df.columns[0:2].tolist()

    Basics  = df.columns[2:9].tolist()
    Planned = df.columns[9:13].tolist()
    ATCs    = df.columns[13:34].tolist()
    ICDs    = df.columns[34:53].tolist()
    ICDs_Specific = df.columns[54:58].tolist()
    RFs = df.columns[58:63].tolist()
    ICDs_Summary = df.columns[[63,64,65]].tolist()

    df = df[TimeLabel + Basics + Planned + ATCs + RFs + ICDs_Summary]
    #cols = [0, 1, 2, 41, 8, 20, 40, 24, 25, 7, 17, 18]
    #df = df.iloc[:, cols]
    time = df['time'].to_numpy().reshape(-1, 1)
    label = df['label'].to_numpy().astype(int).reshape(-1, 1)
    # ---- Feature matrix ----
    feature_cols = df.columns[2:]   # keep same convention

    data = df[feature_cols].to_numpy()

    # Age
    #data[:, 0] = (data[:, 0] - data[:, 0].mean()) / data[:, 0].std()
    '''
# Zero-heavy counts
    for j in [37, 38, 39]:
        data[:, j] = np.log1p(data[:, j])
        data[:, j] = (data[:, j] - data[:, j].mean()) / data[:, j].std()
    '''

    # ---- DeepHit parameters (from index, not time) ----
    num_Category = np.round(np.max(time) * time_scale).astype(int)
    num_Event = label.max().astype(int)
    x_dim = data.shape[1]

    # ---- Masks (USE timeIndex, NOT time) ----
    mask1 = f_get_fc_mask1(time, label, num_Event, num_Category=num_Category)
    mask2 = f_get_fc_mask2(time, num_Category=num_Category)

    DIM = x_dim
    DATA = (data, time, label)          # continuous time preserved
    MASK = (mask1, mask2)

    return DIM, DATA, MASK
