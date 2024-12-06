import pandas as pd
import numpy as np

df = pd.read_csv("annotation.csv")
d = df.iloc[:, -6:].to_numpy()
mask = np.zeros(d.shape[0], dtype=bool)

for i in range(len(d)):
    k = d[i, :]
    sum_nan = sum(pd.isna(i2) for i2 in k)
    div = len(k) - sum_nan

    if sum_nan == 5 or div == 1:
        mask[i] = True

filtered_df = df[mask]
filtered_df.to_csv("filtered_annotation.csv", index=False)
