# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 10:04:46 2021

@author: sopmathieu

Study of the SVM (kernels, number of input examples, etc) 
and support vectors.

"""

### load packages/files 
import pickle
import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 6.0)
plt.rcParams['font.size'] = 14

from SunSpot import errors as err
from SunSpot import preprocessing as pre
from SunSpot import cusum_design_bb as chart
from SunSpot import alerts as plot
from SunSpot import svm_training as svm
from SunSpot import autocorrelations as bbl

### load data
with open('data/Nc_365', 'rb') as fichier:
      my_depickler = pickle.Unpickler(fichier)
      data = my_depickler.load() #contain all deviations, even from IC stations
      time = my_depickler.load() #time
      station_names = my_depickler.load() #code names of the series
      dataIC = my_depickler.load() #IC data without deviations
      pool = my_depickler.load() #pool
      mu2 = my_depickler.load() #mu2 with levels
      mu2_wht_levels = my_depickler.load() #mu2 with levels

(n_obs, n_series) = data.shape
    

#=========================================================================
### Design of the chart 
#=========================================================================


bb_length = 54
delta_min = 1.5 #fixed value (reproducibility purpose)
ARL0 = 200 #pre-specified ARL0 (controls the false positives rate)

# control_limit = chart.limit_CUSUM(dataIC, delta=delta_min, ARL0_threshold=200,
#                 block_length=bb_length, missing_values='reset') 

control_limit = 19 #fixed value (reproducibility purpose)


#=========================================================================
### Train support vector machine classifier and regressor 
#=========================================================================

### compute the control limit of the chart without missing values
control_limit_mv = chart.limit_CUSUM(dataIC, delta=delta_min, ARL0_threshold=ARL0,
              block_length=bb_length, missing_values='omit')

control_limit_mv = 18.5

### select the length of the input vector 
wdw_length = svm.input_vector_length(dataIC, delta_min, control_limit_mv,
                                block_length=bb_length) 

wdw_length = 80  #fixed value (reproducibility purpose)


n = 21000*3 #number of testing and training instances
scale = 3.5 #scale parameter (~variance) of the halfnormal distribution
n_search = 12000*3 #smaller number of instance (to compute C)

### find an optimal value for C (regularization parameter)
# C_choices = svm.choice_C(dataIC, control_limit_mv, delta_min, wdw_length, 
#                             scale, start = 5, stop = 15, step = 1,
#               delay=True, n=n_search, block_length=bb_length, confusion=False)

# C = C_choices[0] 

### train the classifier and regressor with selected C
C = 14 #for reproduciability purpose

#how much training instances needed to reach similar performances? 
reg, clf = svm.training_svm(dataIC, control_limit_mv, delta_min,
        wdw_length, scale, delay=True, n=n/16, C=C, block_length=bb_length)

# n : 1, 0.99, 0.89 (accuracy), MAPE = 48.83
# n/2 : 1, 0.99, 0.88 (accuracy), MAPE = 48.5
# n/4 = 1, 0.99 0.82 (accuracy), MAPE : 48.46
# n/8 = 1, 0.98 0.78 (accuracy), MAPE : 52.49
# n/16 = 1, 0.97 0.72 (accuracy), MAPE : inf (plus haut)

# other kernels
# reg, clf = svm.training_svm(dataIC, control_limit_mv, delta_min,
#                 wdw_length, scale, delay=True, n=n/2, C=C, kernel='linear',
#                 block_length=bb_length)
# 

reg, clf = svm.training_svm(dataIC, control_limit_mv, delta_min,
                wdw_length, scale, delay=True, n=n/2, C=C, kernel='poly',
                block_length=bb_length)

# reg, clf = svm.training_svm(dataIC, control_limit_mv, delta_min,
#                 wdw_length, scale, delay=True, n=n/2, C=C, kernel='sigmoid',
#                 block_length=bb_length)


#=========================================================================
### Analysis of the support vectors
#=========================================================================

### load the models previously trained 
reg = pickle.load(open('svm_models/svr_Nc_365.sav', 'rb'))
clf = pickle.load(open('svm_models/svc_Nc_365.sav', 'rb'))

# get support vectors
vector_reg = reg.support_vectors_
# get indices of support vectors
ind_reg = reg.support_
# get number of support vectors for each class
num_reg = reg.n_support_

# get support vectors
vectors_clf= clf.support_vectors_
# get indices of support vectors
ind_clf = clf.support_
# get number of support vectors for each class
num_clf = clf.n_support_


reg, clf = svm.training_svm(dataIC, control_limit_mv, delta_min,
        wdw_length, scale, delay=True, n=n/4, C=C, block_length=bb_length)

# get support vectors
vector_reg = reg.support_vectors_
# get indices of support vectors
ind_reg = reg.support_
# get number of support vectors for each class
num_reg = reg.n_support_

# get support vectors
vectors_clf= clf.support_vectors_
# get indices of support vectors
ind_clf = clf.support_
# get number of support vectors for each class
num_clf = clf.n_support_