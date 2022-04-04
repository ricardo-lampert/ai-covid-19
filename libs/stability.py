from ctypes import Union


def jaccards_index(features_a, features_b):
    intersection = 0
    for feature_a in features_a:
        if feature_a in features_b:
            intersection += 1

    index = intersection / ((len(features_a) + len(features_b)) - intersection)
    return index


def get_jaccard_index(data):
    n = 0
    value = 0
    for i in range(len(data) - 1):
        for j in range(i + 1, len(data)):
            n += 1
            value += jaccards_index(data[i], data[j])
    return value / n
