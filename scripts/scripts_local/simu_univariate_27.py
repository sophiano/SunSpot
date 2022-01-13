# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 10:47:33 2021

@author: sopmathieu

Comparison between the proposed control scheme based on a pool of IC
stations to a purely univariate method design on each series 
separately. This analysis is done for data smoothed on 27 days. 

"""

### load packages/files 
import pickle
import numpy as np 
import sys
#0 is the script path or working directory
sys.path.insert(1, '/Sunspot/') 
sys.path.insert(2, '../') 
sys.path.insert(3, '../../') 
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 6.0)
plt.rcParams['font.size'] = 14

from SunSpot import bb_methods as bb
from SunSpot import errors as err
from SunSpot import preprocessing as pre
from SunSpot import cusum_design_bb as chart
from sklearn.utils import resample

#=========================================================================
### Design of the chart with the selected pool
#=========================================================================

# pool = 100 stations,  34.5% 

### load data
with open('../../data/Nc_27', 'rb') as fichier:
      my_depickler = pickle.Unpickler(fichier)
      data = my_depickler.load() #contain all deviations, even from IC stations
      time = my_depickler.load() #time
      station_names = my_depickler.load() #code names of the series
      dataIC = my_depickler.load() #IC data without deviations
      pool = my_depickler.load() #pool
      mu2 = my_depickler.load() #mu2 with levels
      mu2_wht_levels = my_depickler.load() #mu2 with levels

(n_obs, n_series) = data.shape

n_data = len(data[~np.isnan(data)])
n_perc = n_data*100/(n_series*n_obs)

### choice of the block length 
bb_length = 54
ARL0 = 200 #pre-specified ARL0 (controls the false positives rate)
delta_min = 1.4 #fixed value (reproducibility purpose)

n = 5
h_loop = np.zeros((n))
for i in range(n):  
    h_loop[i] = chart.limit_CUSUM(dataIC, delta=delta_min, ARL0_threshold=200,
                 block_length=bb_length, missing_values='reset')

h = np.mean(h_loop) #12.7
h = 13

#=========================================================================
### Design of the chart with other pools
#=========================================================================


with open('../../data/data_1981', 'rb') as file: #local    
    my_depickler = pickle.Unpickler(file)
    Ns = my_depickler.load() #number of spots
    Ng = my_depickler.load() #number of sunspot groups
    Nc = my_depickler.load() #composites: Ns+10Ng
    station_names = my_depickler.load() #codenames of the stations
    time = my_depickler.load() #time

### add new data to station KS (not included in database)
data_ks = np.loadtxt('../../data/kisl_wolf.txt', usecols=(0,1,2,3), skiprows=1) #local
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


### small pool of 50 stations,  8.9% of obs

# dataNs = pre.PreProcessing(mu2)  
# dataNs.level_removal(wdw=4000)  #remove intrisic levels

# #pool
# dataNs.selection_pools(method='fixed', nIC=50) #select the IC pool
# pool = np.array(dataNs.pool) #pool (number) = 50
# pool_ind = [station_names[i] for i in range(n_series) if i in pool] #pool (index)

# dataNs.outliers_removal(k=1) #remove outliers
# dataNs.standardisation(K=2400) #standardisation of the data
# dataIC = dataNs.dataIC  #IC data without deviations
# data = dataNs.data #data (IC and OC) with deviations

# n_obs_IC = len(dataIC[~np.isnan(dataIC)])*100 / len(data[~np.isnan(data)]) #8.9%

# n = 5
# h_loop = np.zeros((n))
# for i in range(n):  
#     h_loop[i]= chart.limit_CUSUM(dataIC, delta=delta_min, ARL0_threshold=200,
#                  block_length=bb_length, missing_values='reset')

# h_small = np.mean(h_loop) #13.496


### all stations 
dataNs = pre.PreProcessing(mu2)  
dataNs.level_removal(wdw=4000)  #remove intrisic levels

#pool
dataNs.selection_pools(method='fixed', nIC=n_series) #select the IC pool
pool = np.array(dataNs.pool) #pool (number) 
pool_ind = [station_names[i] for i in range(n_series) if i in pool] #pool (index)

dataNs.outliers_removal(k=1) #remove outliers
dataNs.standardisation(K=2400) #standardisation of the data
dataIC = dataNs.dataIC  #IC data without deviations
data = dataNs.data #data (IC and OC) with deviations

n_obs_IC = len(dataIC[~np.isnan(dataIC)])*100 / len(data[~np.isnan(data)]) #64.61%

n = 5
h_loop = np.zeros((n))
for i in range(n):  
    h_loop[i]= chart.limit_CUSUM(dataIC, delta=delta_min, ARL0_threshold=200,
                 L_plus=20, block_length=bb_length, missing_values='reset')

h_large = np.mean(h_loop) #14.11
h_large = 14.11

#=========================================================================
### Design of the chart without pool (univariate)
#=========================================================================

stat = [i for i in range(len(station_names)) if station_names[i] == 'SM'][0]

data_ind = data[:,stat]
n_values = len(data_ind[~np.isnan(data_ind)])

qinf = np.quantile(data_ind[~np.isnan(data_ind)], 0.25)
qsup = np.quantile(data_ind[~np.isnan(data_ind)], 0.75)


#non-overlapping blocks
blocks = bb.NBB(data_ind.reshape(-1,1), bb_length, NaN=True) 

blocks_IC = []
for i in range(len(blocks)):
    count = 0 
    for j in range(bb_length):
        if blocks[i,j] < qsup and blocks[i,j] > qinf:
            count += 1
    if count >= int(bb_length/2):
        blocks_IC.append(blocks[i,:])
        
#IC data
indv_IC = np.array(blocks_IC).reshape(-1,1)

        
n = 5
h_loop = np.zeros((n))
for i in range(n):  
    h_loop[i] = chart.limit_CUSUM(indv_IC, delta=delta_min, ARL0_threshold=200,
        L_plus=40, block_length=bb_length, missing_values='reset')

h_ind = np.mean(h_loop) 

h_ind = 5.27 #34.8125 SM out of control and 5.27 for FU 

#=========================================================================
### Performances
#=========================================================================

delt = np.arange(0, 3, 0.2) #shift sizes
n_delt = len(delt)
perfs = np.zeros((n_delt,3))
form = 'jump' #shape of the simulated shifts


for i in range(n_delt):
 
    perfs[i, 0] = chart.ARL1_CUSUM(indv_IC, h, delta=delt[i], k=0.7,
                  block_length=bb_length, form=form, 
                    nmc=2000, n=2000, BB_method='MBB', missing_values='reset')
    
    perfs[i, 1] = chart.ARL1_CUSUM(indv_IC, h_large, delta=delt[i], k=0.7,
                  block_length=bb_length, form=form, 
                    nmc=2000, n=2000, BB_method='MBB', missing_values='reset')
    
    perfs[i, 2] = chart.ARL1_CUSUM(indv_IC, h_ind, delta=delt[i], k=0.7,
                  block_length=bb_length, form=form, 
                    nmc=2000, n=2000, BB_method='MBB', missing_values='reset')
    
fig = plt.figure()
plt.plot(delt, perfs[:,0], 'o-', label='actual pool')
plt.plot(delt, perfs[:,2], 'D--', label='one station')
plt.plot(delt, perfs[:,1], 'v:', label='all stations')
plt.legend(fontsize=16)
plt.xlabel('shift size')
plt.ylabel('ARL1')
plt.show()
#fig.savefig('ic_oscill_27_2.pdf') 

