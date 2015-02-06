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
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.learning_curve import learning_curve
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.cross_validation import train_test_split, KFold, cross_val_score, LeaveOneOut, StratifiedKFold
from sklearn import metrics
from sklearn import preprocessing

from scipy.stats import sem
from class_vis import prettyPicture
from learning_curve import plot_learning_curve
######################
# Mega-comment3

def measure_performance(X,y,clf, show_accuracy = True, show_classification_report = True, show_confusion_matrix = True):
    '''
    Helper function to print the statistics:
    takes the set of data and targets, runs a prediction, prints the stats
    '''
    y_pred = clf.predict(X)
    if show_accuracy:
        print "Accuracy:{0:.3f}".format(metrics.accuracy_score(y,y_pred))#"(+/-",sem(scores),")","\n"
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
    print "Mean score: ",np.mean(scores), "(+/-",sem(scores),")"
######################
    
def size_error(X,y,clf):
    output_error = []
    test_size = 0.1
    while test_size <0.9:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 33)
        clf = clf.fit(X_train,y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        output_error.append([test_size, mean_squared_error(y_train, y_pred_train),mean_squared_error(y_test, y_pred_test)])
        test_size = test_size+ 0.1
        print test_size
    return output_error
######################
        
        
data = np.genfromtxt('atemwege.tsv',delimiter=' ')
data = np.delete(data,0,0)
data = np.delete(data,0,1)


#removing rows with "-1" value (nonexisting answer)
b =[]
for row in range(len(data)):
    if -1 in data[row]:
        b.append(row)

data = np.delete(data,b, 0)

#change sex to 0-femaile, and 1-male
c =[]
for element in data[:,12]:
    if element == 2:
        c.append(0)
    else:
        c.append(element)
data[:,12] = c

d = []
for element in data[:,0]:
    if element == 2:
        d.append(0)
    else:
        d.append(1)
data[:,0] = d
        

#creating the column which is going to be a classifier and removing that column form numpy array

data_y = data[:,21]
data_X= np.delete(data,[21],1)

#remove some featurres and leaving only 15 of them:
data_X = np.delete(data_X,[8,9,10,13,14,15],1)

#scaler = preprocessing.StandardScaler().fit(data_X)
#
#data_X = scaler.transform(data_X)


#choose only 2 feachers to run the algorithm against:
#data_X = np.delete(data_X,[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19],1)
#data_X = data_X[:,11]
feature_names = ['polution zone','allergies', 'astma','mother smokes','father smokes','parental education','frequent colds','cough','height','sex','weight','lung cap','airflow','airflow 50%','airflow 75%']

#t_size = 0.25
#X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size = t_size, random_state = 33)
#scaler = preprocessing.StandardScaler().fit(X_train)
#
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

scaler = preprocessing.StandardScaler().fit(data_X)

data_X = scaler.transform(data_X)


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

#clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, min_samples_leaf = 33)
#clf = clf.fit(X_train,y_train)
#importances = clf.feature_importances_
#indices = np.argsort(importances)[::-1]
#print("Feature ranking:")
#for f in range(10):
#    print " %d. feature %d" % (f + 1, indices[f]), feature_names[f], importances[indices[f]]

#clf = KNeighborsClassifier(n_neighbors=5,  weights = 'distance')

#clf = RandomForestClassifier(n_estimators = 10, random_state =33)

#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=1, random_state=5).fit(X_train, y_train)


#output = np.array(size_error(data_X, data_y, clf))
#figure = plt.figure(1)
#plt.plot(output[:,0]*1322,output[:,1])
#plt.show()

#digits = load_digits()
X, y = data_X, data_y


title = "Learning Curves (Gradient Booster)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
#cv = cross_validation.ShuffleSplit(digits.data.shape[0], n_iter=100,
#                                   test_size=0.2, random_state=0)

cv = cross_validation.ShuffleSplit(data_X.shape[0], n_iter=10,
                                   test_size=0.2, random_state=0)

estimator = GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=1, random_state=5).fit(X, y)
plot_learning_curve(estimator, title, X, y, ylim=(0.5, 1.01), cv=cv, n_jobs=1)

#title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
## SVC is more expensive so we do a lower number of CV iterations:
#cv = cross_validation.ShuffleSplit(digits.data.shape[0], n_iter=10,
#                                   test_size=0.2, random_state=0)
#estimator = SVC(gamma=0.001)
#plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)
#
plt.show()


#clf = clf.fit(X_train,y_train)
#y_pred = clf.predict(X_test)
#print "score", clf.score(X_test, y_test),"=/- {", mean_squared_error(y_test, y_pred),"}"
#measure_performance(X_test, y_test,clf)

#print "Leave One Out"
#loo_cv(X_train, y_train, clf)
###############################
#K-fold cross validation


#cv = KFold(data_X.shape[0], 5, shuffle = True, random_state = 33)

#scores = cross_val_score(clf,data_X, data_y, cv=cv)
#print "Cross-validation 5 fold"
#print np.mean(scores),"(+/-",sem(scores),")"
###############################
#dot_data = StringIO.StringIO()
#tree.export_graphviz(clf, out_file = dot_data, feature_names = ['polution zone','allergies', 'astma','mother smokes','faather smokes','parental education','frequent colds','cough','height','sex','weight','lung cap','airflow','airflow 50%','airflow 75%'])
#graph = pydot.graph_from_dot_data(dot_data.getvalue())
#graph.write_png('Astma.png')

##############################
#if len(X_train[1]) == 2:
#    try:
#        prettyPicture(clf, X_train, y_train, X_test, y_test)
#    except NameError:
#        pass



