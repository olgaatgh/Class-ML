#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

def prettyPicture(clf, X_train, y_train, X_test, y_test ):
    x_min = -3; x_max = 3
    y_min = -3; y_max = 3
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)

    # Plot the train points 
    param1_train_no = [X_train[ii][0] for ii in range(0, len(X_train)) if y_train[ii]==0]
    param2_train_no = [X_train[ii][1] for ii in range(0, len(X_train)) if y_train[ii]==0]
    param1_train_yes = [X_train[ii][0] for ii in range(0, len(X_train)) if y_train[ii]==1]
    param2_train_yes = [X_train[ii][1] for ii in range(0, len(X_train)) if y_train[ii]==1]
    
    plt.scatter(param1_train_no, param2_train_no, color = "y", label="0 - train")
    plt.scatter(param1_train_yes, param2_train_yes, color = "r", label="1 - train")
    
    # Plot the test points 
    param1_test_no = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    param2_test_no = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    param1_test_yes = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    param2_test_yes = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]


    plt.scatter(param1_test_no, param2_test_no, color = "y", marker = "x", label="0 - test")
    plt.scatter(param1_test_yes, param2_test_yes, color = "r", marker = "x", label="1 - test")
    plt.legend(loc=4)
    plt.xlabel("Grain Size")
    plt.ylabel("Max Annealing Temp")

    plt.savefig("test.png")

import base64
import json
import subprocess

def output_image(name, format, bytes):
    image_start = "BEGIN_IMAGE_f9825uweof8jw9fj4r8"
    image_end = "END_IMAGE_0238jfw08fjsiufhw8frs"
    data = {}
    data['name'] = name
    data['format'] = format
    data['bytes'] = base64.encodestring(bytes)
    print image_start+json.dumps(data)+image_end
                                    
