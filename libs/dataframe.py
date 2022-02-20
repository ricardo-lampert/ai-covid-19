import os
import pandas as pd


def read_dataset(dataset, dataset_path="../datasets", sep=","):
    complete_path = os.path.join(dataset_path, dataset + ".csv")
    # print("Lendo dataset do hospital ", dataset)
    return pd.read_csv(complete_path, sep=sep, header=0)
