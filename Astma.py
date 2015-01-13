# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 13:42:20 2015

@author: olga
"""

import csv
import numpy as np
import pydot, StringIO
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier


from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import train_test_split, KFold, cross_val_score, LeaveOneOut
from sklearn import metrics
from sklearn import preprocessing

from scipy.stats import sem
from class_vis import prettyPicture
######################

def measure_performance(X,y,clf, show_accuracy = True, show_classification_report = True, show_confusion_matrix = True):
    '''
    Helper function to print the statistics:
    takes the set of data and targets, runs a prediction, prints the stats
    '''
    y_pred = clf.predict(X)
    if show_accuracy:
        print "Accuracy:{0:.3f}".format(metrics.accuracy_score(y,y_pred)),"\n"
    if show_classification_report:
        print "Classification report"
        print metrics.classification_report(y,y_pred),"\n"
    if show_confusion_matrix:
        print "Confusion matrix:"
        print metrics.confusion_matrix(y,y_pred),"\n"
#####################
def loo_cv(X_train,y_train, clf):
    loo = LeaveOneOut(X_train[:].shape[0])
    scores = np.zeros(X_train[:].shape[0])
    for train_index, test_index in loo:
        X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
        y_train_cv, y_test_cv = y_train[train_index],y_train[test_index]
        clf = clf.fit(X_train_cv,y_train_cv)
        y_pred = clf.predict(X_test_cv)
        scores[test_index] = metrics.accuracy_score(y_test_cv.astype(int), y_pred.astype(int))
    print "Mean score: ",np.mean(scores), sem(scores)
        
data = np.genfromtxt('atemwege.tsv',delimiter=' ')
data = np.delete(data,0,0)
data = np.delete(data,0,1)


#removing rows with "-1" value (nonexisting answer)
b =[]
for row in range(len(data)):
    if -1 in data[row]:
        b.append(row)

data = np.delete(data,b, 0)

#creating the column which is going to be a classifier and removing that column form numpy array

data_y = data[:,21]
data_X= np.delete(data,[21],1)

#remove some featurres and leaving only 15 of them:
data_X = np.delete(data_X,[8,9,10,13,14,15],1)

scaler = preprocessing.StandardScaler().fit(data_X)

data_X = scaler.transform(data_X)


#choose only 2 feachers to run the algorithm against:
#data_X = np.delete(data_X,[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19],1)
#data_X = data_X[:,11]
feature_names = ['polution zone','allergies', 'astma','mother smokes','faather smokes','parental education','frequent colds','cough','height','sex','weight','lung cap','airflow','airflow 50%','airflow 75%']

X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size = 0.25, random_state = 33)
#scaler = preprocessing.StandardScaler().fit(X_train)
#
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

colors = ["yellow", "red"]
legend = ['healthy', 'sick']

#for i in range(len(colors)):
#    xs = data[:,16][y_train == i]
#    ys = data[:,17][y_train == i]
#    plt.scatter(xs,ys, c = colors[i])
#plt.legend(legend)

##################################

#clf = SVC(C = 100000.0, kernel='rbf')
#clf = SVC(kernel='linear')

#clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, min_samples_leaf = 5)

#clf = KNeighborsClassifier(n_neighbors=5)

#clf = RandomForestClassifier(n_estimators = 10, random_state =33)

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=1, random_state=5).fit(X_train, y_train)

clf = clf.fit(X_train,y_train)
print "score", clf.score(X_test, y_test)
measure_performance(X_test, y_test,clf)
print "Leave One Out"
loo_cv(X_train, y_train, clf)
###############################
#K-fold cross validation


cv = KFold(data_X.shape[0], 5, shuffle = True, random_state = 33)
scores = cross_val_score(clf,data_X, data_y, cv=cv)
print "Cross-validation 5 fold"
print np.mean(scores),"(+/-",sem(scores),")"
###############################
#dot_data = StringIO.StringIO()
#tree.export_graphviz(clf, out_file = dot_data, feature_names = ['polution zone','allergies', 'astma','mother smokes','faather smokes','parental education','frequent colds','cough','height','sex','weight','lung cap','airflow','airflow 50%','airflow 75%'])
#graph = pydot.graph_from_dot_data(dot_data.getvalue())
#graph.write_png('Astma.png')

##############################
if len(X_train[1]) == 2:
    try:
        prettyPicture(clf, X_train, y_train, X_test, y_test)
    except NameError:
        pass



