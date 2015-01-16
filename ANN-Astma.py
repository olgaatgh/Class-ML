# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 14:34:54 2015

@author: olgamac
"""
import numpy as np
import neurolab as nl
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer

from time import time

from sklearn import preprocessing

import matplotlib.pyplot as plt

data = np.genfromtxt('atemwege.tsv',delimiter=' ')
#removing first row which has features name and first column which is just consecutve numbers
data = np.delete(data,0,0)
data = np.delete(data,0,1)


#removing rows with "-1" value (nonexisting answer)
b =[]
for row in range(len(data)):
    if -1 in data[row]:
        b.append(row)

data = np.delete(data,b, 0)

#creating the column which is going to be a classifier and removing that column from numpy array

data_y = data[:,21]
data_X= np.delete(data,[21],1)
############################
#net = buildNetwork(11,3,1, hiddenclass=TanhLayer)

# for fitting to the sin(x) optimum parameters are learningrate = 0.01, momentum = 0.01, 10 layers 10000 epochs
#data_X = np.linspace(-7,7,20)
#data_y = np.sin(data_X)
#data_X= data_X.reshape(-1,1)


data_y= data_y.reshape(-1,1)

scaler = preprocessing.StandardScaler().fit(data_X)

data_X = scaler.transform(data_X)


dim = len(data_X[0])

ds = SupervisedDataSet(dim,1)

ds.setField('input',data_X)
ds.setField('target', data_y)


#
#for input, target in ds:
#    print "input ", input, "target ", target
    
net = buildNetwork(dim,3,1)
epoch = 1000
t0 = time()
trainer = BackpropTrainer(net,ds,learningrate = 0.01, momentum = 0.001, verbose = True)
#for i in range(epoch):
#    print "i ",i
#    trainer.train()

x,y = trainer.trainUntilConvergence(verbose = True, validationProportion = 0.15, maxEpochs = epoch)
print "prediction and training time:", round(time()-t0, 3), "s"

with open('ANN-Astma-whole.txt','w') as file_out:
    for i in range(len(x)):
        new_line = str(x[i])+','+str(y[i])+'\n'
        file_out.write(new_line)
        
X = range(0,epoch+2)
x = np.array(x)
y=np.array(y)
#
#fig = plt.figure()
#plt.scatter(x,y)



