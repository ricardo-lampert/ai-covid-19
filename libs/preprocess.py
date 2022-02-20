import pandas as pd
import numpy as np


def remove_sparse_columns(df, sparce_threshold=75):
    threshold = int(((100 - sparce_threshold) / 100) * (df.shape[0]) + 1)
    return df.dropna(axis=1, thresh=threshold)


def remove_sparse_rows(df, sparce_threshold=75):
    threshold = int(((100 - sparce_threshold) / 100) * (df.shape[1] - 1) + 1)
    return df.dropna(axis=0, thresh=threshold)


def under_sampling(df, discrete=True):
    count_class_0, count_class_1 = df["LABEL"].value_counts()
    if discrete:
        df_class_0 = df[df["LABEL"] == "NEGATIVO"]
        df_class_1 = df[df["LABEL"] == "POSITIVO"]
    else:
        df_class_0 = df[df["LABEL"] == 0]
        df_class_1 = df[df["LABEL"] == 1]
    if count_class_0 > count_class_1:
        return pd.concat([df_class_0.sample(count_class_1), df_class_1])
    else:
        return pd.concat([df_class_0, df_class_1.sample(count_class_0)], axis=0)


def one_hot_encoded(df):
    one_hot_encoded_X = pd.get_dummies(df.drop(["LABEL"], axis=1))
    one_hot_encoded_y = pd.DataFrame(
        np.where(df["LABEL"].values == "POSITIVO", 1, 0), df.index, columns=["LABEL"]
    )
    return pd.concat([one_hot_encoded_X, one_hot_encoded_y], axis=1)
