# This example has been taken from SciKit documentation and has been
# modifified to suit this assignment. You are free to make changes, but you
# need to perform the task asked in the lab assignment

import pandas as pd
from sklearn.model_selection import train_test_split
#Reading the data into a dataframe
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",header=None)


X = df.values[:, 1:11]
Y = df.values[:, 0]

#Splitting the dataset into test and training data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, test_size =0.2, random_state=42)
X_train.shape
Y_train.shape





#importing relevant models
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier,RandomForestClassifier,AdaBoostClassifier
from xgboost import XGBClassifier



from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10)



#Adding the algorithms to a list
dtc = DecisionTreeClassifier()
mlp = MLPClassifier()
svm = SVC()
gnb = GaussianNB()
logreg = LogisticRegression()
knn = KNeighborsClassifier()
bagging=BaggingClassifier()
RF=RandomForestClassifier()
AdaB=AdaBoostClassifier()
GBoost=GradientBoostingClassifier()
XGBoost=XGBClassifier()
estimators = [dtc,mlp,svm,gnb,logreg,knn,bagging,RF,AdaB,GBoost,XGBoost]



# This is a key step where you define the parameters and their possible values
# that you would like to check.

parameters_dtc = { 'max_features':['auto','sqrt','log2'], 'min_samples_leaf':[1,25,50,100], 'max_depth':[20,10,7], 'min_impurity_decrease':[0.00,0.50,0.75,0.25] } 
parameters_mlp={ 'activation':['relu','tanh','logistic'], 'alpha':[0.0001,0.005,0.5], 'hidden_layer_sizes':[(100,),(200,50,),(50,50,50,)], 'max_iter':[200,500] } 
parameters_svm = { 'C':[1,2,0.5], 'max_iter':[-1,200,400], 'random_state':[100,20,600], 'kernel':['linear','poly'], 'degree':[3,5,7] } 
parameters_gnb = { 'priors':[(0.3,0.4,0.3),(0.5,0.5,0.0)] } 
parameters_logreg = { 'penalty':['l1','l2'], 'C':[0.1,1.0,2.0], 'fit_intercept':['True','False'], 'max_iter': [100,200,50] }
parameters_knn = { 'n_neighbors':[5,2,10], 'weights':['uniform','distance'], 'algorithm': ['auto','ball_tree','kd_tree'], 'p':[1,2,3] } 
parameters_bagging={'n_estimators' : [60,70,80], 'max_samples': [2,3,4], 'max_features': [2,3,4], 'random_state': [100,20,600]}
parameters_RF={'n_estimators' : [60,70,80], 'max_depth': [3,10,7], 'max_features': [2,3,4], 'criterion': ['gini','entropy']}
parameters_AdaB={'n_estimators' : [60,70,80], 'learning_rate': [0.2,0.3,0.4], 'algorithm': ['SAMME','SAMME.R'], 'random_state': [100,20,600]}
parameters_GBoost={'n_estimators' : [60,70,80], 'learning_rate': [0.2,0.3,0.4], 'max_features': [2,3,4], 'max_depth': [3,10,7]}
parameters_XGBoost={'n_estimators' : [60,70,80], 'learning_rate': [0.2,0.3,0.4], 'max_delta_step': [1,2,3,4], 'booster': ['gbtree', 'gblinear','dart']}
parameters = [parameters_dtc,parameters_mlp,parameters_svm,parameters_gnb,parameters_logreg,parameters_knn,parameters_bagging,parameters_RF, parameters_AdaB, parameters_GBoost, parameters_XGBoost]



# We are going to limit ourselves to accuracy score, other options can be
# seen here:
# http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# Some other values used are the predcision_macro, recall_macro


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics

#GridSearch

# scores = ['accuracy']

# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()

#     clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
#                        scoring='%s' % score)
#     clf.fit(X_train, y_train)




for i in range(len(parameters)):
    clf = GridSearchCV(estimators[i], param_grid=parameters[i], cv=skf,
                       scoring='accuracy')
    clf.fit(X_train, Y_train)


    print('Best score: {}'.format(clf.best_score_))
    print("Best parameters set found on development set: \n")
    print(clf.best_params_)
    print("\n")
    #print("Grid scores on development set:")
    #print()
    #means = clf.cv_results_['mean_test_score']
    #stds = clf.cv_results_['std_test_score']
    #for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #    print("%0.3f (+/-%0.03f) for %r"
    #          % (mean, std * 2, params))
    #print()


    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print("\n")
    print("Detailed classification report:")
    print("\n")
    Y_true, Y_pred = Y_test, clf.predict(X_test)
    print(classification_report(Y_true, Y_pred))
    print("\n")
    print("Accuracy Score: \n")
    print(accuracy_score(Y_true, Y_pred))
    print("Detailed confusion matrix: \n") 
    print(confusion_matrix(Y_true, Y_pred))



# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.