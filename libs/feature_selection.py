from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


def get_k_best_features(X, y, n_features=10):

    k_best_classifier = SelectKBest(score_func=f_classif, k=n_features)
    fit = k_best_classifier.fit(X, y)
    cols = fit.get_support(indices=True)
    return X.iloc[:, cols]

def fs_results_filter(fs_results,n_features):
    fs_filtered_results = []
    ultimo = 0
    for feature,times in fs_results.items():
        if  n_features>0 or times==ultimo:
            fs_filtered_results.append(feature)
            n_features-=1
            if n_features == 0:
                ultimo = times
        else:
            return fs_filtered_results