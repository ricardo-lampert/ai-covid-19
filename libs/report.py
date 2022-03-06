from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
)


def model_report(y_test, y_predicted):

    acc = round(accuracy_score(y_test, y_predicted), 2)
    prec = round(precision_score(y_test, y_predicted), 2)
    rec = round(recall_score(y_test, y_predicted), 2)
    f1_scor = round(f1_score(y_test, y_predicted), 2)
    cm = confusion_matrix(y_test, y_predicted, normalize="true")
    model_dict = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1-score": f1_scor,
        "confusion matrix": {
            "tp": str(round(cm[1][1], 2)),
            "tn": str(round(cm[0][0], 2)),
            "fp": str(round(cm[0][1], 2)),
            "fn": str(round(cm[1][0], 2)),
        },
    }
    return model_dict


def feature_selection_report(features):
    best_features = []
    for feature in features:
        # feature_string = re.sub('_\w*', '', feature)
        # if feature_string not in best_features:
        #     best_features.append(feature_string)
        # best_features.append(feature_string)
        best_features.append(feature)
    return best_features


def dataset_info_report(df):
    dataset_info = {
        "shape": {"instances": df.shape[0], "features": df.shape[1]},
        "features": {},
    }
    features = {}
    for column in df.columns:
        info = df[column].value_counts(dropna=False)
        for info_label in info.items():
            if column not in features:
                features[column] = {}
            features[column][info_label[0]] = info_label[1]
    dataset_info["features"] = features
    return dataset_info
