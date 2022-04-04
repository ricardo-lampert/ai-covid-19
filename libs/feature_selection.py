from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Univariate selection

# f_classif
# ANOVA F-value between label/feature for classification tasks.

# mutual_info_classif
# Mutual information for a discrete target.

# chi2
# Chi-squared stats of non-negative features for classification tasks.

# SelectPercentile
# Select features based on percentile of the highest scores.

# SelectFpr
# Select features based on a false positive rate test.

# SelectFdr
# Select features based on an estimated false discovery rate.

# SelectFwe
# Select features based on family-wise error rate.

# GenericUnivariateSelect
# Univariate feature selector with configurable mode.


def get_k_best(df, X, y, classif, n_features=None):
    if n_features is None:
        n_features = "all"
    if classif == "f_classif":
        k_best_classifier = SelectKBest(score_func=f_classif, k=n_features)
    elif classif == "mutual_info_classif":
        k_best_classifier = SelectKBest(score_func=mutual_info_classif, k=n_features)
    elif classif == "SelectPercentile":
        k_best_classifier = SelectKBest(score_func=mutual_info_classif, k=n_features)
    elif classif == "SelectFpr":
        k_best_classifier = SelectKBest(score_func=mutual_info_classif, k=n_features)
    elif classif == "SelectFdr":
        k_best_classifier = SelectKBest(score_func=mutual_info_classif, k=n_features)
    elif classif == "SelectFwe":
        k_best_classifier = SelectKBest(score_func=mutual_info_classif, k=n_features)
    else:
        k_best_classifier = SelectKBest(score_func=chi2, k=n_features)
    fit = k_best_classifier.fit(X, y)
    scores = []
    for i, score in enumerate(fit.scores_):
        scores.append((df.columns[i], score))
    scores.sort(key=lambda x: x[1], reverse=True)
    features = []
    for i in range(len(scores)):
        features.append(scores[i][0])
    return features


# Feature importance
def get_extra_trees_classifier(X, y, data, n_features=10):
    model = ExtraTreesClassifier(n_estimators=n_features)
    model.fit(X, y)
    scores = []
    features = []
    for i, score in enumerate(model.feature_importances_):
        scores.append((data.columns[i], score))
    scores.sort(key=lambda x: x[1], reverse=True)
    for i in range(n_features):
        features.append(scores[i][0])
    return features


# RFE
def get_rfe(X, y, data, n_features):
    model = LogisticRegression(max_iter=5000)
    selector = RFE(estimator=model, n_features_to_select=n_features)
    features = selector.fit(X, y)
    scores = []
    best_features = []
    for i, score in enumerate(features.ranking_):
        scores.append((data.columns[i], score))
    scores.sort(key=lambda x: x[1])
    for i in range(n_features):
        best_features.append(scores[i][0])
    return best_features


def fs_results_filter(fs_results, n_features):
    fs_filtered_results = []
    ultimo = 0
    for feature, times in fs_results.items():
        if n_features > 0 or times == ultimo:
            fs_filtered_results.append(feature)
            n_features -= 1
            if n_features == 0:
                ultimo = times
        else:
            return fs_filtered_results
    return fs_filtered_results

def get_best_features(features):
    n = len(features[0])
    features_score = {}
    for feature_list in features:
        for index,feature in  enumerate(feature_list):
            if feature not in features_score.keys():
                 features_score[feature]=0
            features_score[feature]+=(n-index)
    features_score_list = [(x,y) for x,y in features_score.items()]
    features_score_list.sort(key=lambda x: x[1], reverse=True)
    best_features=[]
    for item in features_score_list:
        best_features.append(item[0])
    return best_features