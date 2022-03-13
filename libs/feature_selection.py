from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2

# f_classif
# ANOVA F-value between label/feature for classification tasks.

# mutual_info_classif
# Mutual information for a discrete target.

# chi2
# Chi-squared stats of non-negative features for classification tasks.
def get_k_best(X, y, classif, n_features=10):
    if classif == "f_classif":
        k_best_classifier = SelectKBest(score_func=f_classif, k=n_features)
    elif classif == "mutual_info_classif":
        k_best_classifier = SelectKBest(score_func=mutual_info_classif, k=n_features)
    else :
        k_best_classifier = SelectKBest(score_func=chi2, k=n_features)
    
    fit = k_best_classifier.fit(X, y)
    cols = fit.get_support(indices=True)
    return X.iloc[:, cols]



# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(max_iter=5000)
# selector = RFE(model, N_FEATURES)
# features = selector.fit(X_raw, y_raw)
# scores=[]
# for i,score in enumerate(features.ranking_):
#   scores.append((data.feature_names[i],score))
# scores.sort(key=lambda x: x[1])
# for i in range(N_FEATURES):
#   print(scores[i][0])
# print("\nObs: Desconsiderar a ordem")
# X = features.transform(X_raw)
# y = y_raw

# from sklearn.ensemble import ExtraTreesClassifier
# model = ExtraTreesClassifier(n_estimators=N_FEATURES)
# model.fit(X_raw, y_raw)
# # print(model.feature_importances_)
# scores=[]
# features=[]
# for i,score in enumerate(model.feature_importances_):
#   scores.append((data.feature_names[i],score))
  
# scores.sort(key=lambda x: x[1],reverse=True)
# for i in range(N_FEATURES):
#   print(scores[i][0])
#   features.append(data.feature_names[i])
# X = df[features]
# y = y_raw


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
    return fs_filtered_results