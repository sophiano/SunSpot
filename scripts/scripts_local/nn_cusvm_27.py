# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 09:29:24 2021

@author: sopmathieu

This script compares the performances of the CUSUM chart, feed-
forward and recurrent networks associated to the adaptive CUSUM chart 
or with simple cut-off values for detecting the deviations 
on unstandardised data smoothed on 27 days. 
A classifier with four classes is also compared to the other methods. 

"""

import pickle
import numpy as np 
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 6.0)
plt.rcParams['font.size'] = 14
import sys
#0 is the script path or working directory
sys.path.insert(1, '/Sunspot/') #inside valusun
sys.path.insert(2, '../') #inside SunSpot
sys.path.insert(3, '../../') #inside scripts

from SunSpot import cusum_design_bb as chart
from SunSpot import NN_limits as nn_limit
from SunSpot import NN_training as nn_train
from SunSpot import svm_training as svm

### load data
with open('../../data/Nc_27', 'rb') as fichier:
      my_depickler = pickle.Unpickler(fichier)
      data_stn = my_depickler.load() #contain all deviations, even from IC stations
      time = my_depickler.load() #time
      station_names = my_depickler.load() #codenames of the stations
      dataIC_stn = my_depickler.load() #IC data without deviations
      pool = my_depickler.load() #pool of IC stations
      mu2 = my_depickler.load() #mu2 with levels ('eh')
      mu2_wht_levels = my_depickler.load() #mu2 without levels ('mu2')

(n_obs, n_series) = mu2.shape

#data to be monitored
data_nn = np.copy(mu2_wht_levels)
dataIC_nn = data_nn[:,pool]
q1_nn = np.quantile(dataIC_nn[~np.isnan(dataIC_nn)], 0.95)
q2_nn = np.quantile(dataIC_nn[~np.isnan(dataIC_nn)], 0.05)
std_nn = np.nanstd(dataIC_nn)

data_cusvm = np.copy(data_stn)
dataIC_cusvm = data_cusvm[:,pool]
q1_cusvm = np.quantile(dataIC_cusvm[~np.isnan(dataIC_cusvm)], 0.95)
q2_cusvm = np.quantile(dataIC_cusvm[~np.isnan(dataIC_cusvm)], 0.05)
std_cusvm = np.nanstd(dataIC_cusvm) #10.5

wdw_length = 50 

####################################################

### train feed-forward networks
#regressor, scores_3 = nn_train.feed_forward_reg(dataIC_nn, wdw_length, scale=3)
#regressor, scores_1 = nn_train.feed_forward_reg(dataIC_nn, wdw_length, scale=1)
reg_adp, scores = nn_train.feed_forward_reg(dataIC_cusvm, wdw_length, scale=3)

### serialize model to JSON
# model_json = reg_adp.to_json()
# with open("../../nn_models/reg_adp_365.json", "w") as json_file:
#     json_file.write(model_json)
# ### serialize weights to HDF5
# reg_adp.save_weights("../../nn_models/reg_adp_365.h5")

### train recurrent networks
#reg_rnn, scores_rnn = nn_train.recurrent_reg(dataIC_nn, wdw_length, scale=1)
reg_rnn_adp, scores_rnn_adp = nn_train.recurrent_reg(dataIC_cusvm, wdw_length, scale=3)

### serialize model to JSON
# model_json = reg_rnn_adp.to_json()
# with open("../../nn_models/reg_rnn_adp_365.json", "w") as json_file:
#     json_file.write(model_json)
# ### serialize weights to HDF5
# reg_rnn_adp.save_weights("../../nn_models/reg_rnn_adp_365.h5")

####################################################

### train networks with four classes 
reg4, scores = nn_train.feed_forward_4reg(dataIC_nn, wdw_length)
clf4, scores, matrix = nn_train.feed_forward_4class(dataIC_nn, wdw_length)

### serialize model to JSON
# model_json = reg4.to_json()
# with open("../../nn_models/nn_reg4_365.json", "w") as json_file:
#     json_file.write(model_json)
# ### serialize weights to HDF5
# reg4.save_weights("../../nn_models/nn_reg4_365.h5")

### serialize model to JSON
# model_json = clf4.to_json()
# with open("../../nn_models/nn_class4_365.json", "w") as json_file:
#     json_file.write(model_json)
# ### serialize weights to HDF5
# clf4.save_weights("../../nn_models/nn_class4_365.h5")


##############################################################

# load feed-forward network (regression)
json_file = open('../../nn_models/nn_regression_27.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
reg = model_from_json(loaded_model_json)
# load weights into new model
reg.load_weights("../../nn_models/nn_regression_27.h5")

#load recurrent network (regression)
json_file = open('../../nn_models/rnn_regression_27.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
reg_rnn = model_from_json(loaded_model_json)
# load weights into new model
reg_rnn.load_weights("../../nn_models/rnn_regression_27.h5")


#========================================================================
### Calibration of the control limits/cut-off values
#=======================================================================

n = 5 

#feed-forward nns with cut-off values
h_loop = np.zeros((n))
for i in range(n): 
    h_loop[i] = nn_limit.cutoff(dataIC_nn, reg, ARL0_threshold=200, L_plus=0.5, 
                  n=2000, nmc=2000, block_length=wdw_length, 
                  BB_method='MBB') 
    
h_nn = np.mean(h_loop) 
h_nn = 0.3532


#feed-forward nns with adaptive CUSUM

### standardised 
h_loop = np.zeros((n))
for i in range(n): 
    h_loop[i] = nn_limit.cutoff(dataIC_cusvm, reg_adp, ARL0_threshold=200, L_plus=40, 
                  n=2000, nmc=2000, block_length=wdw_length, 
                  BB_method='MBB', chart='cusum') 

h_nn_adp = np.mean(h_loop) 
h_nn_adp = 32.2031


### Heuristic decision procedure
#n_L = 10
# h1_nn, h2_nn = nn_limit.cutoff_heuristic(dataIC_nn, regressor, L_plus=4.8, 
#                 l_plus=4, n=2000, nmc= 2000, 
#                   n_L=n_L, block_length=block_length, BB_method='MBB') 
# h1_nn = 2.4 
# h2_nn = 1.8 

##########################################

#recurrent nns with simple cut-off values
h_loop = np.zeros((n))
for i in range(n): 
    h_loop[i] = nn_limit.rnn_cutoff(dataIC_nn, reg_rnn, ARL0_threshold=200, L_plus=0.5, 
                  n=2000, nmc=2000, block_length=wdw_length, 
                  BB_method='MBB') 

h_rnn = np.mean(h_loop) 
h_rnn = 0.3877
    
#recurrent nns with adaptive CUSUM

### standardised 
h_loop = np.zeros((n))
for i in range(n): 
    h_loop[i] = nn_limit.rnn_cutoff(dataIC_cusvm, reg_rnn_adp, L_plus=40, 
                  n=2000, nmc=2000, block_length=wdw_length, 
                  BB_method='MBB', chart='cusum') 

h_rnn_adp = np.mean(h_loop) 
h_rnn_adp = 35.7891


##########################################

#CUSVM method

### standardised 
h_loop = np.zeros((n))
for i in range(n): 
    h_loop[i] = chart.limit_CUSUM(dataIC_cusvm, delta=1, L_plus=40, 
                    missing_values='omit', block_length=wdw_length, 
                    BB_method='MBB') 

h_cusvm = np.mean(h_loop)

h_cusvm =  38.375

#========================================================================
### ARL1 values
#=======================================================================


delt = np.arange(0, 3.2, 0.5) #shift sizes
n_delt = len(delt)
perfs = np.zeros((n_delt, 6))
form = 'oscillation' #shape of the simulated shifts


for i in range(n_delt): 

 
    ### Feed-forward nn with cut-off 
    perfs[i,0] = nn_limit.ARL1_NN(dataIC_nn, reg, h_nn, delta=delt[i]/10.5, 
                      nmc=2000, n=2000, form = form, 
                      block_length=50, wdw_length=wdw_length) 
    
    
    ### Feed-forward nn with adaptive CUSUM
    # perfs[i,1] = nn_limit.ARL1_NN(dataIC_nn, reg, h_nn_adp, delta=delt[i]/10.5, 
    #                   nmc=2000, n=2000, form = form, 
    #                   block_length=50, wdw_length=wdw_length,
    #                   chart='cusum') 
    
    perfs[i,1] = nn_limit.ARL1_NN(dataIC_cusvm, reg_adp, h_nn_adp, delta=delt[i], 
                      nmc=2000, n=2000, form = form, 
                      block_length=50, wdw_length=wdw_length,
                      chart='cusum') 
    
    ### Heuristic
    # perfs[i,1] = nn_limit.ARL1_NN_heuristic(dataIC_nn, reg, h1_nn, h2_nn, 
    #               n_L=n_L, delta=delt[i]/10.5, form=form,
    #               nmc=2000, n=2000, block_length=wdw_length)
        

    ### rnn with cut-off 
    perfs[i,2] = nn_limit.ARL1_RNN(dataIC_nn, reg_rnn, h_rnn,  delta=delt[i]/10.5,
                    form=form, nmc=2000, n=2000, 
                    block_length=50, wdw_length=wdw_length)
    
    
    ### rnn with adaptive CUSUM
    # perfs[i,3] = nn_limit.ARL1_RNN(dataIC_nn, reg_rnn, h_rnn_adp,  delta=delt[i]/10.5,
    #                 form=form, nmc=2000, n=2000, 
    #                 block_length=50, wdw_length=wdw_length ,
    #                 chart='cusum')
    
    perfs[i,3] = nn_limit.ARL1_RNN(dataIC_cusvm, reg_rnn_adp, h_rnn_adp, 
                    delta=delt[i], form=form, nmc=2000, n=2000, 
                    block_length=50, wdw_length=wdw_length ,
                    chart='cusum')
    
    
    ### CUSVM
    perfs[i,4] = chart.ARL1_CUSUM(dataIC_cusvm, h_cusvm, delta=delt[i], k=0.5,
                  block_length=50, form=form, 
                    nmc=2000, n=2000, BB_method='MBB')
    
    
    # perfs[i,4] = chart.ARL1_CUSUM(dataIC_nn, h_cusvm, delta=delt[i]/10.5, 
    #             k=0.025, block_length=50, form=form, 
    #                 nmc=2000, n=2000, BB_method='MBB')

    
    ### 4 classes method  
    perfs[i,5] = nn_limit.ARL1_4class(dataIC_nn, clf4, delta=delt[i]/10.5,
                  form=form, n=2000, nmc=2000, 
                  block_length=50, wdw_length=wdw_length)
        
    
#save into txt files
# f = open('perfs_oscill.txt','w')
# for i in range(n_delt):
#     for j in range(4):
#         if j < 3:
#             f.write("%0.2f, " % perfs[i,j])
#         else:
#             f.write("%0.2f \n" % perfs[i,j])
# f.close()


#plots
start = 0
stop = 16
#delt = delt/21
fig = plt.figure()
plt.plot(delt[start:stop], perfs[start:stop,0], 'o', linestyle=(0, (3, 10, 1, 10)), label='NN-CUT')
plt.plot(delt[start:stop], perfs[start:stop,1], 'X', linestyle=(0, (3, 10, 1, 10)), label='NN-ACUSUM')
plt.plot(delt[start:stop], perfs[start:stop,2], 'v:', label='RNN-CUT')
plt.plot(delt[start:stop], perfs[start:stop,3], '^:', label='RNN-ACUSUM')
plt.plot(delt[start:stop], perfs[start:stop,4], 'p-', label='CUSVM')
plt.plot(delt[start:stop], perfs[start:stop,5], 'd-.',  label='4CLF')
#plt.legend(loc='best', bbox_to_anchor=(0.5, 0.23, 0.5, 0.5), ncol=2, fontsize=11)
plt.legend()
plt.xlabel('shift size')
plt.ylabel('ARL1')
plt.show()
#fig.savefig('ARL1_oscil_27.pdf') 
