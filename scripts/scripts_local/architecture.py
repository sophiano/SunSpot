# -*- coding: utf-8 -*-
"""
Created on Mon May 31 11:41:23 2021

@author: sopmathieu

This script can be used to select the architecture (epochs, number of 
hidden layers and hidden neurons) of the networks. 

"""

import pickle
import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 6.0)
plt.rcParams['font.size'] = 14
import sys
#0 is the script path or working directory
sys.path.insert(1, '/Sunspot/') 
sys.path.insert(2, '../') 
sys.path.insert(3, '../../') 

from SunSpot import NN_training as nn_train

### load data
with open('../../data/Nc_365', 'rb') as fichier:
      my_depickler = pickle.Unpickler(fichier)
      data_stn = my_depickler.load() #standardized bias ('epsilon_{mu2}')
      time = my_depickler.load() #index of time
      station_names = my_depickler.load() #codenames of the series
      dataIC_stn = my_depickler.load() #IC standardized bias without deviations
      pool = my_depickler.load() #index of IC stations
      mu2 = my_depickler.load() #mu2 with levels ('eh')
      mu2_wht_levels = my_depickler.load() #mu2 without levels ('mu2')

(n_obs, n_series) = mu2.shape

data = np.copy(mu2_wht_levels)
dataIC = data[:,pool]

block_length = 100 #100 for 365 days, 50 for 27 days
nmc = 10
neurons = np.arange(10, 100, 10)

######################################################
### regression
######################################################

n_hidden_1 = 1
precision_nmc_1 = np.zeros((nmc, 3))
precision_1 = np.zeros((len(neurons),3))
for i in range(len(neurons)):
    for j in range(nmc):
    
        ### train a regressor
        regressor, scores = nn_train.feed_forward_reg(dataIC, block_length, 
                    n_hidden=n_hidden_1, n_neurons=[neurons[i]], verbose=0)
        precision_nmc_1[j,:] = scores
    
    precision_1[i,:] = np.mean(precision_nmc_1, axis=0)
    

n_hidden_2 = 2
precision_nmc_2 = np.zeros((nmc, 3))
precision_2 = np.zeros((len(neurons),3))
for i in range(len(neurons)):
    for j in range(nmc):
    
        ### train a regressor
        neurons_2 = [neurons[i], int(neurons[i]/2)]
        regressor, scores = nn_train.feed_forward_reg(dataIC, block_length, 
                    n_hidden=n_hidden_2, n_neurons=neurons_2, verbose=0)
        precision_nmc_2[j,:] = scores
    
    precision_2[i,:] = np.mean(precision_nmc_2, axis=0)
    
    
n_hidden_3 = 3
precision_nmc_3 = np.zeros((nmc, 3))
precision_3 = np.zeros((len(neurons),3))
for i in range(len(neurons)):
    for j in range(nmc):
    
        ### train a regressor
        neurons_3 = [neurons[i], int(neurons[i]/2), int(neurons[i]/4)]
        regressor, scores = nn_train.feed_forward_reg(dataIC, block_length, 
                    n_hidden=n_hidden_3, n_neurons=neurons_3, verbose=0)
        precision_nmc_3[j,:] = scores
    
    precision_3[i,:] = np.mean(precision_nmc_3, axis=0)
    
    
#plots
perfs = ['MSE', 'MAE', 'MAPE']
ind = -1
fig = plt.figure()
plt.plot(neurons, precision_1[:,ind], 'o-', label='1 layer')
plt.plot(neurons, precision_2[:,ind], 'o-', label='2 layers')
plt.plot(neurons, precision_3[:,ind], 'o-', label='3 layers')
plt.legend()
plt.xlabel('complexity (neurons)')
plt.ylabel(perfs[ind])
plt.show()
#fig.savefig('../../figures/architecture_MAPE_27.pdf') 


######################################################

epochs = np.arange(10, 110, 10)
precision_nmc = np.zeros((nmc,3))
precision = np.zeros((len(epochs),3))
for i in range(len(epochs)):
    for j in range(nmc):
    
        ### train a regressor
        regressor, scores = nn_train.feed_forward_reg(dataIC, block_length, 
                    n_epochs=epochs[i], verbose=0)
        precision_nmc[j,:] = scores
    
    precision[i,:] = np.mean(precision_nmc, axis=0)
    
#plots
perfs = ['MSE', 'MAE', 'MAPE']
ind = 0
fig = plt.figure()
plt.plot(epochs, precision[:,ind], 'o-')
plt.legend()
plt.xlabel('number of epochs')
plt.ylabel(perfs[ind])
plt.show()  
#fig.savefig('../../figures/epochs_27.pdf')   


######################################################
### classification
######################################################

n_hidden_1 = 1
precision_nmc_1 = np.zeros((nmc, 2))
precision_1 = np.zeros((len(neurons),2))
for i in range(len(neurons)):
    for j in range(nmc):
    
        ### train a classifier
        classifier, scores, matrix = nn_train.feed_forward_class(dataIC, block_length, 
                    n_hidden=n_hidden_1, n_neurons=[neurons[i]], verbose=0)
        precision_nmc_1[j,:] = scores
    
    precision_1[i,:] = np.mean(precision_nmc_1, axis=0)
    

n_hidden_2 = 2
precision_nmc_2 = np.zeros((nmc, 2))
precision_2 = np.zeros((len(neurons),2))
for i in range(len(neurons)):
    for j in range(nmc):
    
        ### train a classifier
        neurons_2 = [neurons[i], int(neurons[i]/2)]
        classifier, scores, matrix = nn_train.feed_forward_class(dataIC, block_length, 
                    n_hidden=n_hidden_2, n_neurons=neurons_2, verbose=0)
        precision_nmc_2[j,:] = scores
    
    precision_2[i,:] = np.mean(precision_nmc_2, axis=0)
    
    
n_hidden_3 = 3
precision_nmc_3 = np.zeros((nmc, 2))
precision_3 = np.zeros((len(neurons),2))
for i in range(len(neurons)):
    for j in range(nmc):
    
        ### train a classifier
        neurons_3 = [neurons[i], int(neurons[i]/2), int(neurons[i]/4)]
        classifier, scores, matrix = nn_train.feed_forward_class(dataIC, block_length, 
                    n_hidden=n_hidden_3, n_neurons=neurons_3, verbose=0)
        precision_nmc_3[j,:] = scores
    
    precision_3[i,:] = np.mean(precision_nmc_3, axis=0)
    
    
#plots
perfs = ['MSE', 'accuracy',]
ind = -1
fig = plt.figure()
plt.plot(neurons, precision_1[:,ind], 'o-', label='1 layer')
plt.plot(neurons, precision_2[:,ind], 'o-', label='2 layers')
plt.plot(neurons, precision_3[:,ind], 'o-', label='3 layers')
plt.legend()
plt.xlabel('complexity (neurons)')
plt.ylabel(perfs[ind])
plt.show()
#fig.savefig('../../figures/architecture_class_365.pdf') 


