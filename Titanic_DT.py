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
from sklearn.cross_validation import train_test_split
from sklearn import metrics

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

with open('titanic.csv',  'rb') as csvfile:
    titanic_reader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
    row = titanic_reader.next()
    feature_names = np.array(row)
    
    titanic_X, titanic_y = [],[]
    for row in titanic_reader:
        titanic_X.append(row)
        titanic_y.append(row[1])
    titanic_X = np.array(titanic_X)
    titanic_y = np.array(titanic_y)
    
##################################
    
titanic_X = titanic_X[:,[0,3,4]]
feature_names = feature_names[[0,3,4]]

#filling missing age data
ages = titanic_X[:,2]


mean_age = np.mean(titanic_X[ages != '',2].astype(np.float))
titanic_X[titanic_X[:,2] == '',2] = mean_age


enc = LabelEncoder()
label_encoder = enc.fit(titanic_X[:,1])
integer_classes = label_encoder.transform(label_encoder.classes_)
t= label_encoder.transform(titanic_X[:,1])
titanic_X[:,1] = t

#one hot encoding
enc = LabelEncoder()
label_encoder = enc.fit(titanic_X[:,0])
integer_classes = label_encoder.transform(label_encoder.classes_).reshape(3,1)
enc = OneHotEncoder()
one_hot_encoder = enc.fit(integer_classes)

num_of_rows = titanic_X.shape[0]
t = label_encoder.transform(titanic_X[:,0]).reshape(num_of_rows,1)
new_features = one_hot_encoder.transform(t)

titanic_X = np.concatenate([titanic_X,new_features.toarray()], axis = 1)
titanic_X = np.delete(titanic_X, [0], 1)
feature_names = ['sex','age','first class','second class','third class']
###############################

X_train, X_test, y_train, y_test = train_test_split(titanic_X, titanic_y, test_size = 0.25, random_state = 33)
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, min_samples_leaf = 5)
clf = clf.fit(X_train,y_train)

measure_performance(X_train, y_train,clf)
###############################
dot_data = StringIO.StringIO()
tree.export_graphviz(clf, out_file = dot_data, feature_names = ['sex','age','first class','second class','third class'])
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_png('titanic.png')

##############################




