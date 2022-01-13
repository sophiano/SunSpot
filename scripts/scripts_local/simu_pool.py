# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 12:01:55 2021

@author: sopmathieu

Comparison between the proposed control scheme and others 
designed on different pools.
This analysis is done for data smoothed on 365 days. 

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

#=========================================================================
### Design of the chart with the selected pool
#=========================================================================

# pool = 119 stations

### load data
with open('../../data/Nc_365', 'rb') as fichier:
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
delta_min = 1.4 #fixed value (reproducibility purpose)

n = 5
h_loop = np.zeros((n))
for i in range(n):  
    h_loop[i] = chart.limit_CUSUM(dataIC, delta=delta_min, L_plus=30,
                 block_length=bb_length, missing_values='reset')

h = np.mean(h_loop) 
h = 21.21


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
mu2 = err.long_term_error(Nc, period_rescaling=10, wdw=365)

### discard stations with no values
ind_nan = []
for i in range(mu2.shape[1]):
    if not np.all(np.isnan(mu2[:,i])): 
        ind_nan.append(i)
mu2 = mu2[:,ind_nan]
station_names = [station_names[i] for i in range(len(station_names)) if i in ind_nan]
(n_obs, n_series) = mu2.shape


### small pool of 60 stations
dataNs = pre.PreProcessing(mu2)  
dataNs.level_removal(wdw=4000)  #remove intrisic levels

#pool
dataNs.selection_pools(method='fixed', nIC=60) #select the IC pool
pool = np.array(dataNs.pool) #pool (number) = 60
pool_ind = [station_names[i] for i in range(n_series) if i in pool] #pool (index)

dataNs.outliers_removal(k=1) #remove outliers
dataNs.standardisation(K=2400) #standardisation of the data
dataIC_small = dataNs.dataIC  #IC data without deviations
data = dataNs.data #data (IC and OC) with deviations

n_obs_IC = len(dataIC_small[~np.isnan(dataIC_small)])*100 /  \
                    len(data[~np.isnan(data)]) #10.8%

n = 5
h_loop = np.zeros((n))
for i in range(n):  
    h_loop[i]= chart.limit_CUSUM(dataIC_small, delta=delta_min, L_plus=30,
                  block_length=bb_length, missing_values='reset')

h_small = np.mean(h_loop) 
h_small = 21.41

#####################################

### tiny pool of 40 stations
dataNs = pre.PreProcessing(mu2)  
dataNs.level_removal(wdw=4000)  #remove intrisic levels

#pool
dataNs.selection_pools(method='fixed', nIC=40) #select the IC pool
#dataNs.selection_pools(method='kmeans', nIC_inf=0.90) #select the IC pool
pool = np.array(dataNs.pool) #pool (number) = 30
pool_ind = [station_names[i] for i in range(n_series) if i in pool] #pool (index)

dataNs.outliers_removal(k=1) #remove outliers
dataNs.standardisation(K=2400) #standardisation of the data
dataIC_tiny = dataNs.dataIC  #IC data without deviations
data = dataNs.data #data (IC and OC) with deviations

n_obs_IC = len(dataIC_tiny[~np.isnan(dataIC_tiny)])*100 /  \
                    len(data[~np.isnan(data)]) #5.34%

n = 5
h_loop = np.zeros((n))
for i in range(n):  
    h_loop[i]= chart.limit_CUSUM(dataIC_tiny, delta=delta_min, L_plus=30,
                  block_length=bb_length, missing_values='reset')

h_tiny = np.mean(h_loop) 
h_tiny = 18.81

###########################################################

### all stations (243)
dataNs = pre.PreProcessing(mu2)  
dataNs.level_removal(wdw=4000)  #remove intrisic levels

#pool
dataNs.selection_pools(method='fixed', nIC=n_series) #select the IC pool
pool = np.array(dataNs.pool) #pool (number) 
pool_ind = [station_names[i] for i in range(n_series) if i in pool] #pool (index)

dataNs.outliers_removal(k=1) #remove outliers
dataNs.standardisation(K=4600) #standardisation of the data
dataIC_large = dataNs.dataIC  #IC data without deviations
data_ = dataNs.data #data (IC and OC) with deviations

n_obs_IC = len(dataIC_large[~np.isnan(dataIC_large)])*100 / \
    len(data[~np.isnan(data)]) #100%
    
n = 5
h_loop = np.zeros((n))
for i in range(n):  
    h_loop[i]= chart.limit_CUSUM(dataIC_large, delta=delta_min, ARL0_threshold=200,
                 L_plus=20, block_length=bb_length, missing_values='reset')

h_large = np.mean(h_loop) 

h_large = 14.94 #15.25-14.94


#=========================================================================
### Performances
#=========================================================================

delt = np.arange(0, 3.2, 0.5) #shift sizes
n_delt = len(delt)
perfs = np.zeros((n_delt,4))
form = 'oscillation' #shape of the simulated shifts


for i in range(n_delt):
 
    perfs[i, 0] = chart.ARL1_CUSUM(dataIC, h, delta=delt[i], k=0.7,
                  block_length=bb_length, form=form, 
                    nmc=2000, n=2000, BB_method='MBB', missing_values='reset')
        
    perfs[i, 1] = chart.ARL1_CUSUM(dataIC, h_tiny, delta=delt[i], k=0.7,
                  block_length=bb_length, form=form, 
                    nmc=2000, n=2000, BB_method='MBB', missing_values='reset')
    
    
    perfs[i, 2] = chart.ARL1_CUSUM(dataIC, h_small, delta=delt[i], k=0.7,
                  block_length=bb_length, form=form, 
                    nmc=2000, n=2000, BB_method='MBB', missing_values='reset')
    
    perfs[i, 3] = chart.ARL1_CUSUM(dataIC, h_large, delta=delt[i], k=0.7,
                  block_length=bb_length, form=form, 
                    nmc=2000, n=2000, BB_method='MBB', missing_values='reset')
    
fig = plt.figure()
plt.plot(delt, perfs[:,1], 'v:', label='40') 
plt.plot(delt, perfs[:,2], 'd-.', label='60') 
plt.plot(delt, perfs[:,0], 'o-', label='119 (proposed)')
plt.plot(delt, perfs[:,3], 'D--', label='243 (all)')#243 stations
plt.legend(fontsize=16)
plt.xlabel('shift size')
plt.ylabel('ARL1')
plt.show()
#fig.savefig('pool_oscill_same.pdf') 