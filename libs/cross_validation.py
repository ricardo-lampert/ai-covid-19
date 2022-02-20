# import numpy as np
# import pandas as pd


# def get_folds(df, n_folds=10):
#     folds_list = np.array_split(df.sample(frac=1), n_folds)
#     folds_df = []
#     for index, fold in enumerate(folds_list):
#         train_folds = pd.DataFrame([])
#         test_fold = pd.DataFrame([])
#         for index_b, fold_b in enumerate(folds_list):
#             if index != index_b:
#                 train_folds = pd.concat([train_folds, fold])
#             else:
#                 test_fold = fold
#         folds_df.append(
#             (
#                 train_folds.drop("LABEL", axis=1),
#                 train_folds["LABEL"],
#                 test_fold.drop("LABEL", axis=1),
#                 test_fold["LABEL"],
#             )
#         )
#     return folds_df

from sklearn.model_selection import StratifiedKFold
import pandas as pd


def get_folds(df, n_folds=10, random_state=None, shuffle=False):
    folds_df = []
    X = df.drop("LABEL", axis=1)
    y = df["LABEL"]
    X_values = X.to_numpy()
    y_values = y.to_numpy()
    skf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    for train_index, test_index in skf.split(X_values, y_values):
        X_train, X_test = X_values[train_index], X_values[test_index]
        y_train, y_test = y_values[train_index], y_values[test_index]
        train_df = pd.concat(
            [
                pd.DataFrame(X_train, columns=X.columns),
                pd.DataFrame(y_train, columns=["LABEL"]),
            ],
            axis=1,
        )
        test_df = pd.concat(
            [
                pd.DataFrame(X_test, columns=X.columns),
                pd.DataFrame(y_test, columns=["LABEL"]),
            ],
            axis=1,
        )
        folds_df.append((train_df, test_df))
    return folds_df
