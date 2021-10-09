from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Logistic Regression
model_classifier(LogisticRegression(C = 0.8, random_state=0, solver='lbfgs'), 'Logistic Regression')

# Support Vector Machine
model_classifier(SVC(kernel = 'rbf', C = 1e6, gamma = 1e-04, probability = True), 'Support Vector Machine') #kernel = 'rbf', C = 1e9, gamma = 1e-07, 

# KNN
model_classifier(knn(), 'K Nearest Neighbors')

# Decision Tree
model_classifier(DecisionTreeClassifier(max_depth=12, min_samples_split=8, random_state=1024), 'Decision Tree')

# Random Forest
model_classifier(RandomForestClassifier(n_estimators = 250, max_depth = 12, min_samples_leaf=16), 'Random Forest')

# LGBM
model_classifier(LGBMClassifier(num_leaves = 32, max_depth=8, learning_rate = 0.03, n_estimators = 250, subsample = 0.8, colsample_bytree = 0.8), 'LGBM')

# XGBoost
model_classifier(XGBClassifier(max_depth = 16, n_estimators = 250, min_child_weight = 8, subsample = 0.8, 
                               learning_rate = 0.02, seed = 42, eval_metric = 'mlogloss', use_label_encoder=False), 'XGBoost')
