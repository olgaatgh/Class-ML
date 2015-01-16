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

from sklearn import preprocessing

import matplotlib.pyplot as plt

data = np.genfromtxt('miete03.tsv',delimiter='\t')
data = np.delete(data,0,0)
data_y = data[:,0]
#data_X= np.delete(data,[0,5],1)
data_X= np.delete(data,[0,4,5],1)

#data_X[:,3] = np.abs(data_X[:,3]-2003.0)

y = []
for element in data_y:
    if element < 550:
        element = 0
    else:
        element = 1
    y.append(element)

data_y = np.array(y)
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
epoch = 10000
trainer = BackpropTrainer(net,ds,learningrate = 0.01, momentum = 0.001, verbose = True)
#for i in range(epoch):
#    print "i ",i
#    trainer.train()

x,y = trainer.trainUntilConvergence(verbose = True, validationProportion = 0.15, maxEpochs = epoch)

with open('onlyZeroAdOne.txt','w') as file_out:
    for i in range(len(x)):
        new_line = str(x[i])+','+str(y[i])+'\n'
        file_out.write(new_line)
        
X = range(0,epoch+2)
x = np.array(x)
y=np.array(y)
#
#fig = plt.figure()
#plt.scatter(x,y)



