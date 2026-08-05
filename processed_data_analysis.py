import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap

files = {
    1: "/Users/mehman/Projects/0_Reference_Data/0_otherRegions/Processed/Region_Sthlm_Gotland_model_data_binary_ALL_Included.csv",
    6: "/Users/mehman/Projects/0_Reference_Data/0_otherRegions/Processed/Region_Norr_model_data_binary.csv"
}

dfs = []
for region, path in files.items():
    tmp = pd.read_csv(path)
    tmp["Region"] = region
    dfs.append(tmp)

df = pd.concat(dfs, axis=0, ignore_index=True)

# ---- Enforce censoring beyond horizon (using index) ----
df['time'] = df['time'] * 12
df["label"] = np.where(df["time"] >= 12, 0, df["label"])
df["time"]  = np.where(df["time"] >= 12, 12, df["time"])

reducer = umap.UMAP(metric='jaccard',
    random_state=42
)

TimeLabel = df.columns[1:2].tolist()
Basics  = df.columns[2:9].tolist()
Planned = df.columns[9:13].tolist()
ATCs    = df.columns[13:61].tolist()
ICDs    = df.columns[61:77].tolist()
ICDs_Specific = df.columns[77:82].tolist()
RFs = df.columns[82:87].tolist()
ICDs_Summary = df.columns[[87,88,89]].tolist()

dff = df[TimeLabel + Basics + Planned + ATCs + RFs + ICDs + ICDs_Specific + ICDs_Summary]

scaled_data = StandardScaler().fit_transform(dff)
embedding = reducer.fit_transform(scaled_data)
plt.figure(figsize=(8,6))

palette = sns.color_palette()

region_names = {
    1: "Region Sthlm_Gotland",
    6: "Region Norr"
}

for region, name in region_names.items():
    idx = df["Region"] == region

    plt.scatter(
        embedding[idx, 0],
        embedding[idx, 1],
        s=5,
        color=palette[region],
        label=region_names[region]
    )

plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the dataset', fontsize=24)

plt.legend(title="Labels")
plt.show()




'''
path = (
    "/Users/mehman/Projects/0_Reference_Data/0_otherRegions/Processed/Region_Sthlm_Gotland_model_data_count.csv"
)
# Example data
df = pd.read_csv(path)

# ---- Enforce censoring beyond horizon (using index) ----
df['time'] = df['time'] * 12
df["label"] = np.where(df["time"] >= 12, 0, df["label"])
df["time"]  = np.where(df["time"] >= 12, 12, df["time"])
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
reducer = umap.UMAP()
dff = df.drop(columns=["time", "label"])
dff = dff.iloc[:, :34]  # Keep only the first 34 features for visualization
scaled_data = StandardScaler().fit_transform(dff)
embedding = reducer.fit_transform(scaled_data)
plt.figure(figsize=(8,6))

palette = sns.color_palette()

label_names = {
    0: "Class 0",
    1: "Class 1",
    2: "Class 2"
}

for cls in [0, 1, 2]:
    idx = df["label"] == cls
    
    plt.scatter(
        embedding[idx, 0],
        embedding[idx, 1],
        s=5,
        color=palette[cls],
        label=label_names[cls]
    )

plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the dataset', fontsize=24)

plt.legend(title="Labels")
plt.show()
'''


from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from ImportDatabase import import_dataset
path = (
    "/Users/mehman/Projects/0_Reference_Data/0_otherRegions/Processed/Region_Sthlm_Gotland_model_data_binary.csv"
)

x_dim, (data, time, label), (mask1, mask2) = import_dataset(path,time_scale=1.2)

X_train, X_test, y_train, y_test = train_test_split(
    data, label, test_size=0.2, random_state=42
)

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function="MultiClass",
    eval_metric='AUC',
    verbose=100,
    class_weights=[1.0, 2.0, 8.0]
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

from sklearn.metrics import f1_score

preds = model.predict(X_test)

f1 = f1_score(y_test, preds, average="macro")
print("Macro F1:", f1)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, preds)

print(cm)