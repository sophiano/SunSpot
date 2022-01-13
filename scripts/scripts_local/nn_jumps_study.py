# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 09:29:24 2021

@author: sopmathieu

This script compares the performaces (MAPE) of the recurrent and
feed-forward networks with the support vector machine to identify 
different kinds of deviations. 
The comparison works on the composite (Nc=Ns+10Ng) smoothed on 27 days. 
The methods are also compared with a persisting model, where 
the prediction at time t is simply equal to the value of the series 
at time t-1. 

"""

import pickle
import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 6.0)
plt.rcParams['font.size'] = 14
from keras.models import model_from_json
from sklearn.utils import resample

from SunSpot import bb_methods as bb
from SunSpot import svm_training as svm
from SunSpot import NN_training as nn_train
from statsmodels.tsa.arima_process import ArmaProcess

with open('data/Nc_27', 'rb') as file: #local path
      my_depickler = pickle.Unpickler(file)
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
block_length = 50


### load feed-forward regressor 
json_file = open('nn_models/nn_regression_27.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
reg = model_from_json(loaded_model_json)
# load weights into new model
reg.load_weights("nn_models/nn_regression_27.h5")

### load feed-forward classifier 
json_file = open('nn_models/nn_classification_27.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
clf = model_from_json(loaded_model_json)
# load weights into new model
clf.load_weights("nn_models/nn_classification_27.h5")

#load recurrent regressor
json_file = open('nn_models/rnn_regression_27.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
reg_rnn = model_from_json(loaded_model_json)
# load weights into new model
reg_rnn.load_weights("nn_models/rnn_regression_27.h5")

#load recurrent classifier
json_file = open('nn_models/rnn_classification_27.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
clf_rnn = model_from_json(loaded_model_json)
# load weights into new model
clf_rnn.load_weights("nn_models/rnn_classification_27.h5")

#train svm classifier and regressor
reg_svm, clf_svm = svm.simple_svm(dataIC, wdw_length=block_length,
                    scale=1, C=10)


#========================================================================
### Comparison with persisting model
#=======================================================================

n_test = 50000
wdw_length = 50
verbose = 1
blocks = bb.MBB(data, wdw_length)  
scale = 1

### testing set 
X_test = np.zeros((n_test, wdw_length))
Y_test = np.zeros((n_test))
Y_perst = np.zeros((n_test))
S_test = np.zeros((n_test))
rnd_shifts = np.random.normal(0, scale, n_test)

##### autoregressive models     
ar = np.array([1, -0.9]) #inverse sign 
ma = np.array([1])
AR_object = ArmaProcess(ar, ma)
    
for b in range(0, n_test-2, 3):
    
    shift = rnd_shifts[b]
    series = resample(blocks, replace=True, n_samples=1).flatten()
    #series = np.random.normal(0, 1, size=(block_length))
    #series = AR_object.generate_sample(nsample=block_length) 
    
    for shape in range(3):
    
        boot = np.copy(series)
        
        if shape == 0:
            delay = np.random.randint(0, wdw_length)
            boot[delay:] = boot[delay:] + shift
            S_test[b] = 0
        elif shape == 1:
            power = np.random.uniform(1, 2)
            boot = shift/(500) * (np.arange(wdw_length)**power) + boot
            S_test[b] = 1
        else:
            eta = np.random.uniform(np.pi/(2*wdw_length), 2*np.pi/wdw_length)
            phi = np.random.randint(0, int(wdw_length/4))
            boot = np.sin(eta*np.pi*np.arange(wdw_length) + phi)*shift + boot
            S_test[b] = 2
            
        X_test[b] = boot
        Y_test[b] = shift
        Y_perst[b] = boot[-2] #time t-1
        b += 1

X_test_rnn = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

#predictions
reg.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
pred = reg.predict(X_test).reshape(-1)
reg_rnn.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
pred_rnn = reg_rnn.predict(X_test_rnn).reshape(-1)
pred_svm = reg_svm.predict(X_test)

#MAPE neural networks 
ind = np.where(abs(Y_test) > 0.1)
mape = (1/len(Y_test[ind])) * sum(abs((abs(Y_test[ind]) - abs(pred[ind]))/abs(Y_test[ind])))*100
scores = reg.evaluate(X_test, Y_test, verbose=verbose)
scores.append(mape)
mape = (1/len(Y_test[ind])) * sum(abs((abs(Y_test[ind]) - abs(pred_rnn[ind]))/abs(Y_test[ind])))*100
scores_rnn = reg_rnn.evaluate(X_test_rnn, Y_test, verbose=verbose)
scores_rnn.append(mape)

#MAPE svm
scores_svm = (1/len(Y_test[ind])) * sum(abs((abs(Y_test[ind]) - abs(pred_svm[ind]))/abs(Y_test[ind])))*100

#MAPE persisting model
ind = np.where(abs(Y_test) > 0.1)
score_perst = (1/len(Y_test[ind])) * sum(abs((abs(Y_test[ind]) - abs(Y_perst[ind]))/abs(Y_test[ind])))*100

#MAPE class by class
mape = np.zeros((3)); mape_rnn = np.zeros((3))
mape_svm = np.zeros((3)); mape_perst = np.zeros((3))
for i in range(3):
    ind = np.where((S_test == i) & (abs(Y_test) > 0.1))
    mape[i] = (1/len(Y_test[ind])) * sum(abs((abs(Y_test[ind]) - abs(pred[ind]))/abs(Y_test[ind])))*100
    mape_rnn[i] = (1/len(Y_test[ind])) * sum(abs((abs(Y_test[ind]) - abs(pred_rnn[ind]))/abs(Y_test[ind])))*100
    mape_svm[i] = (1/len(Y_test[ind])) * sum(abs((abs(Y_test[ind]) - abs(pred_svm[ind]))/abs(Y_test[ind])))*100
    mape_perst[i] = (1/len(Y_test[ind])) * sum(abs((abs(Y_test[ind]) - abs(Y_perst[ind]))/abs(Y_test[ind])))*100
    
    # mape[i] = (1/len(Y_test[ind])) * sum(abs((Y_test[ind] - pred[ind])/abs(Y_test[ind])))*100
    # mape_rnn[i] = (1/len(Y_test[ind])) * sum(abs((Y_test[ind] - pred_rnn[ind])/abs(Y_test[ind])))*100
    # mape_svm[i] = (1/len(Y_test[ind])) * sum(abs((Y_test[ind] - pred_svm[ind])/abs(Y_test[ind])))*100
    # mape_perst[i] = (1/len(Y_test[ind])) * sum(abs((Y_test[ind] - Y_perst[ind])/abs(Y_test[ind])))*100








