import os
import pandas as pd


def read_dataset(dataset, dataset_path="../datasets", sep=","):
    complete_path = os.path.join(dataset_path, dataset + ".csv")
    # print("Lendo dataset do hospital ", dataset)
    return pd.read_csv(complete_path, sep=sep, header=0)


def split_train_test(train_df,test_df):
    X_train = train_df.drop(["LABEL","grupo"],axis=1)
    y_train = train_df["LABEL"]
    X_test = test_df.drop(["LABEL","grupo"],axis=1)
    y_test = test_df["LABEL"]
    return (X_train, y_train, X_test, y_test)