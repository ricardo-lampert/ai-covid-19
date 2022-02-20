from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


def get_k_best_features(X, y, n_features=10):

    k_best_classifier = SelectKBest(score_func=f_classif, k=n_features)
    fit = k_best_classifier.fit(X, y)
    cols = fit.get_support(indices=True)
    return X.iloc[:, cols]
