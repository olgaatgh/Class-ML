# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 13:42:20 2015

@author: olga
"""

import csv
import numpy as np
import pydot, StringIO

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import preprocessing

import matplotlib.pyplot as plt

def measure_performance(X,y,clf, show_accuracy = True, show_classification_report = True, show_confusion_matrix = True):
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
        
data = np.genfromtxt('miete03.tsv',delimiter='\t')
data = np.delete(data,0,0)
data_y = data[:,0]
data_X= np.delete(data,[0,5],1)

data_X[:,3] = np.abs(data_X[:,3]-2003.0)

y = []
for element in data_y:
    if element < 440:
        element = "$"
    elif element <650 and element > 440:
        element = "$$"
    else:
        element = "$$$"
    y.append(element)
data_y = y

#average_rent = np.mean(data[:,0].astype(np.float))
#average_rent = np.mean(data[:,0])
#min_rent = np.min(data[:,0])
#
#print average_rent, min_rent


#plt.scatter(X_train[:,0], X_train[:,1])
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size = 0.25, random_state = 33)
scaler = preprocessing.StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


#for i in range(len(colors)):
#    xs = X_train[:,0][y_train == i]
#    ys = X_train[:,3][y_train == i]
#    plt.scatter(xs,ys, c = colors[i])
#    print "Yeah"
    
#plt.scatter(X_train[:, 1], X_train[:, 3], c=data_y, cmap=plt.cm.Paired)
plt.scatter(X_train[:, 5], X_train[:, 3])   
#plt.scatter(X_train[:,0],X_train[:,3]) 
  
##################################

#clf = SVC(C = 100000.0, kernel='rbf')
clf = SVC(kernel='linear')

#clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 7, min_samples_leaf = 5)

#clf = KNeighborsClassifier(n_neighbors=10)

#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)

clf = clf.fit(X_train,y_train)
print "score", clf.score(X_test, y_test)
measure_performance(X_test, y_test,clf)
###############################
#dot_data = StringIO.StringIO()
#tree.export_graphviz(clf, out_file = dot_data, feature_names = ['sex','age','first class','second class','third class'])
#graph = pydot.graph_from_dot_data(dot_data.getvalue())
#graph.write_png('titanic.png')

##############################



