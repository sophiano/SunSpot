   # -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 11:21:43 2020

@author: sopmathieu

This script applies the monitoring procedure based on the CUSUM chart 
and the support vector machine on the composite (Nc=Ns+10Ng) smoothed on 
27 days. 
This analysis highlights the more sudden jumps or high-frequency
deviations of the stations. 

"""

### load packages/files 
import pickle
import numpy as np 
import pkg_resources as pkg
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 6.0)

from SunSpot import errors as err
from SunSpot import preprocessing as pre
from SunSpot import cusum_design_bb as chart
from SunSpot import alerts as plot
from SunSpot import svm_training as svm
from SunSpot import autocorrelations as bbl

### load data (loaded automatically with package)
data_path = pkg.resource_filename(pkg.Requirement.parse("SunSpot"), 'data')

with open(data_path + '/data_1981', 'rb') as file:
#with open('data/data_1981', 'rb') as file: #local    
    my_depickler = pickle.Unpickler(file)
    Ns = my_depickler.load() #number of spots
    Ng = my_depickler.load() #number of sunspot groups
    Nc = my_depickler.load() #composites: Ns+10Ng
    station_names = my_depickler.load() #codenames of the stations
    time = my_depickler.load() #time

### add new data to station KS (not included in database)
data_ks = np.loadtxt(data_path +'\kisl_wolf.txt', usecols=(0,1,2,3), skiprows=1)
#data_ks = np.loadtxt('data/kisl_wolf.txt', usecols=(0,1,2,3), skiprows=1) #local
Nc_ks = data_ks[9670:23914,3]
Nc[:,24] = Nc_ks

### compute the long-term errors
mu2 = err.long_term_error(Nc, period_rescaling=10, wdw=27)

### discard stations with no values
ind_nan = []
for i in range(mu2.shape[1]):
    if not np.all(np.isnan(mu2[:,i])): 
        ind_nan.append(i)
mu2 = mu2[:,ind_nan]
station_names = [station_names[i] for i in range(len(station_names)) if i in ind_nan]
(n_obs, n_series) = mu2.shape

### apply preprocessing
dataNs = pre.PreProcessing(mu2)  
dataNs.level_removal(wdw=4000)  #remove intrisic levels
dataNs.selection_pools(method='kmeans') #select the IC pool
pool = np.array(dataNs.pool) #pool (number)
pool_ind = [station_names[i] for i in range(n_series) if i in pool] #pool (index)
dataNs.outliers_removal(k=1) #remove outliers

### choose an appropriate value for K
# dataIC = dataNs.dataIC  #IC data without deviations
# data = dataNs.data #data (IC and OC) with deviations
# K = pre.choice_K(data, dataIC, plot=True)

### standardisation
dataNs.standardisation(K=2400) #standardisation of the data
dataIC = dataNs.dataIC  #IC data without deviations
data = dataNs.data #data (IC and OC) with deviations
     
(n_obs, n_series) = mu2.shape
     
#plot the (IC) data
plt.hist(dataIC[~np.isnan(dataIC)], range=[-4,4], bins='auto', density=True, facecolor='b')  
plt.title("Data IC")
plt.text(2, 1, 'mean:' '%4f' %np.nanmean(dataIC))
plt.text(2, 0.8, 'std:' '%4f' %np.nanstd(dataIC))
plt.axis([-4, 4, 0, 1.5])
plt.grid(True)
plt.show()

#plot all data
plt.hist(data[~np.isnan(data)], range=[-4,4], bins='auto', density=True, facecolor='b')  
plt.title("All Data (IC and OC)")
plt.text(2, 1, 'mean:' '%4f' %np.nanmean(data))
plt.text(2, 0.8, 'std:' '%4f' %np.nanstd(data))
plt.axis([-4, 4, 0, 1.5])
plt.grid(True)
plt.show()

n_obs_IC = len(dataIC[~np.isnan(dataIC)])*100 / len(data[~np.isnan(data)])#32%


#=========================================================================
### Design of the chart 
#=========================================================================

### choice of the block length 
#large range
#bb_length = bbl.block_length_choice(data, bbl_min=10, bbl_max=110, bbl_step=10) #50
#smaller range
#bb_length = bbl.block_length_choice(dataIC, 44, 64, 2) #54
bb_length = 54

delta_min = 1 #intial value for the target shift size
ARL0 = 200 #pre-specified ARL0 (controls the false positives rate)

### adjust the control limits
delta_min = chart.shift_size(data, pool, dataIC=dataIC,
                delta=delta_min, ARL0_threshold=ARL0, block_length=bb_length, 
                qt=0.5, missing_values ='reset')[1]


delta_min = 1.4 #fixed value (reproducibility purpose)

control_limit = chart.limit_CUSUM(dataIC, delta=delta_min, ARL0_threshold=200,
                block_length=bb_length, missing_values='reset')

control_limit = 13 #fixed value (reproducibility purpose)


### compute the performance of the chart
# ARL1 = chart.ARL1_CUSUM(dataIC, control_limit, delta=delta_min, 
#                             missing_values='reset', block_length=bb_length) 


#=========================================================================
### train support vector machine classifier and regressor 
#=========================================================================

### compute the control limit of the chart without missing values
control_limit_mv = chart.limit_CUSUM(dataIC, delta=delta_min, ARL0_threshold=ARL0,
              block_length=bb_length, missing_values='omit') 

control_limit_mv = 15.5 #fixed value (reproducibility purpose)

### select the length of the input vector 
wdw_length = svm.input_vector_length(dataIC, delta_min, control_limit_mv,
                                block_length=bb_length) 

wdw_length = 70 #fixed value (reproducibility purpose)
   

scale = 3.5 #scale parameter (~variance) of the halfnormal distribution
n = 21000*3 #number of testing and training instances
n_search = 12000*3 #smaller number of instance (to compute C)

### find an optimal value for C (regularization parameter)
C_choices = svm.choice_C(dataIC, control_limit_mv, delta_min, wdw_length, 
                            scale, start = 5, stop = 15, step = 1,
              delay=True, n=n_search, block_length=bb_length, confusion=False)

C = C_choices[2] #13 for reproducibility purpse

### find an optimal kernel function
# reg, clf = svm.training_svm(dataIC, control_limit_mv, delta_min,
#                 wdw_length, scale, delay=True, n=n_search, C=C, kernel='linear',
#                 block_length=bb_length)

# reg, clf = svm.training_svm(dataIC, control_limit_mv, delta_min,
#                 wdw_length, scale, delay=True, n=n_search, C=C, kernel='poly',
#                 block_length=bb_length)

# reg, clf = svm.training_svm(dataIC, control_limit_mv, delta_min,
#                 wdw_length, scale, delay=True, n=n_search, C=C, kernel='sigmoid',
#                 block_length=bb_length)


### train the classifier and regressor with selected C and kernel
C = 13 #for reproduciability purpose
reg, clf = svm.training_svm(dataIC, control_limit_mv, delta_min,
                wdw_length, scale, delay=True, n=n, C=C, block_length=bb_length)

### save models 
# filename = 'svr_Nc_27.sav'
# pickle.dump(reg, open(filename, 'wb'))
# filename = 'svc_Nc_27.sav'
# pickle.dump(clf, open(filename, 'wb'))

### or load the models previously trained 
### Those are available on github (svm_models). The following commands
### may be used to open them (with appropriate paths).

#reg = pickle.load(open('svm_models/svr_Nc_27.sav', 'rb'))
#clf = pickle.load(open('svm_models/svc_Nc_27.sav', 'rb'))

#=========================================================================
### Run the control chart and plot results (with svm predictions)
#=========================================================================

#for all stations
for i in range(n_series): 
    data_indv = data[:,i]
    level_indv = mu2[:,i]
    
    [form_plus, form_minus, size_plus, size_minus,
    C_plus, C_minus] = plot.alerts_info(data_indv, control_limit, 
            delta_min, wdw_length, clf, reg)
    
    fig = plot.plot_4panels(data_indv, level_indv, control_limit, time, 
                     form_plus, form_minus, size_plus, size_minus, 
                     C_plus, C_minus, station_names[i], time_start=1981)
        
    
    
#for a particular station
stat = [i for i in range(len(station_names)) if station_names[i] == 'SG'][0]

for i in range(stat, stat+1): 
    data_indv = data[:,i]
    level_indv = mu2[:,i]
    
    [form_plus, form_minus, size_plus, size_minus,
    C_plus, C_minus] = plot.alerts_info(data_indv, control_limit, 
            delta_min, wdw_length, clf, reg)
                           
    fig = plot.plot_4panels(data_indv, level_indv, control_limit, time, 
                     form_plus, form_minus, size_plus, size_minus, 
                     C_plus, C_minus, station_names[i], time_start=1981)
    
    fig = plot.plot_3panels(data_indv, control_limit, time, 
                     form_plus, form_minus, size_plus, size_minus, 
                     C_plus, C_minus, station_names[i], time_start=2004, 
                     time_stop=2008)
    
    fig = plot.plot_1panel(data_indv, time, form_plus, form_minus, size_plus, 
                      size_minus, station_names[i],
                      time_start=2006.8, time_stop=2008.1)
    
  
    #fig.savefig('figures/jumps_%s.pdf' %station_names[i]) #save figures

