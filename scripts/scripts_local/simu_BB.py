# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 08:20:08 2021

@author: sopmathieu

Comparison between the proposed control scheme and others 
using the NBB bootstrap or MBB bootstrap, with different block lenghts. 

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
from SunSpot import autocorrelations as bbl
from SunSpot import errors as err
from SunSpot import preprocessing as pre
from SunSpot import cusum_design_bb as chart


#=========================================================================
### Design of the chart with BBL=54
#=========================================================================

# pool = 100 stations,  34.5% 

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

#n_data = len(data[~np.isnan(data)])
#n_perc = n_data*100/(n_series*n_obs)

### choice of the block length 
bb_length = 54
ARL0 = 200 #pre-specified ARL0 (controls the false positives rate)
delta_min = 1.5 #fixed value (reproducibility purpose)

n = 5
h_loop = np.zeros((n))
for i in range(n):  
    h_loop[i] = chart.limit_CUSUM(dataIC, delta=delta_min, ARL0_threshold=200,
                 block_length=bb_length, missing_values='reset')

h = np.mean(h_loop) 
h = 18.67

#=========================================================================
### Design of the chart with smaller blocks
#=========================================================================

### choice of the block length 
bb_length = 5
ARL0 = 200 #pre-specified ARL0 (controls the false positives rate)
delta_min = 1.5 #fixed value (reproducibility purpose)

n = 20
h_loop = np.zeros((n))
for i in range(n):  
    h_loop[i] = chart.limit_CUSUM(dataIC, delta=delta_min, ARL0_threshold=200,
                 block_length=bb_length, missing_values='reset')

h_5 = np.mean(h_loop) 
h_5 = 9.043

#=========================================================================

### choice of the block length 
bb_length = 10
ARL0 = 200 #pre-specified ARL0 (controls the false positives rate)
delta_min = 1.5 #fixed value (reproducibility purpose)

n = 10
h_loop = np.zeros((n))
for i in range(n):  
    h_loop[i] = chart.limit_CUSUM(dataIC, delta=delta_min, ARL0_threshold=200,
                 block_length=bb_length, missing_values='reset')

h_10 = np.mean(h_loop) 
h_10 = 13.496

#=========================================================================

### choice of the block length 
bb_length = 27
ARL0 = 200 #pre-specified ARL0 (controls the false positives rate)
delta_min = 1.5 #fixed value (reproducibility purpose)

n = 10
h_loop = np.zeros((n))
for i in range(n):  
    h_loop[i] = chart.limit_CUSUM(dataIC, delta=delta_min, ARL0_threshold=200,
                 block_length=bb_length, missing_values='reset')

h_27 = np.mean(h_loop) 
h_27 = 19.7244


#=========================================================================
### Design of the chart with larger blocks
#=========================================================================

### choice of the block length 
bb_length = 100
ARL0 = 200 #pre-specified ARL0 (controls the false positives rate)
delta_min = 1.5 #fixed value (reproducibility purpose)

n = 5
h_loop = np.zeros((n))
for i in range(n):  
    h_loop[i] = chart.limit_CUSUM(dataIC, delta=delta_min, ARL0_threshold=200,
                 block_length=bb_length, missing_values='reset')

h_100 = np.mean(h_loop) 
h_100 = 9.8

#=========================================================================

### choice of the block length 
bb_length = 200
ARL0 = 200 #pre-specified ARL0 (controls the false positives rate)
delta_min = 1.5 #fixed value (reproducibility purpose)

n = 10
h_loop = np.zeros((n))
for i in range(n):  
    h_loop[i] = chart.limit_CUSUM(dataIC, delta=delta_min, ARL0_threshold=200,
                 block_length=bb_length, missing_values='reset')

h_200 = np.mean(h_loop) 
h_200 = 0.923

#=========================================================================
### Design of the chart with other BB methods
#=========================================================================

# nbb_length = bbl.block_length_choice(data, bbl_min=10, bbl_max=110,
#                             bbl_step=10, BB_method='NBB') #50
# nbb_length = bbl.block_length_choice(data, bbl_min=40, bbl_max=62,
#                             bbl_step=2, BB_method='NBB') #50

nbb_length = 50

n = 10
h_loop = np.zeros((n))
for i in range(n):  
    h_loop[i]= chart.limit_CUSUM(dataIC, delta=delta_min, ARL0_threshold=200,
                  block_length=nbb_length, BB_method='NBB', 
                                        missing_values='reset')

h_nbb = np.mean(h_loop) 
h_nbb = 19.67


#============================================================

# cbb_length = bbl.block_length_choice(data, bbl_min=10, bbl_max=110,
#                            bbl_step=10, BB_method='CBB') #50
# cbb_length = bbl.block_length_choice(data, bbl_min=40, bbl_max=62,
#                             bbl_step=2, BB_method='CBB') #50

cbb_length = 52

n = 10
h_loop = np.zeros((n))
for i in range(n):  
    h_loop[i]= chart.limit_CUSUM(dataIC, delta=delta_min, ARL0_threshold=200,
                  block_length=nbb_length, BB_method='CBB', 
                                        missing_values='reset')

h_cbb = np.mean(h_loop) 
h_cbb = 19.615

#=========================================================================
### Performances
#=========================================================================

delt = np.arange(0, 3, 0.2) #shift sizes
n_delt = len(delt)
perfs = np.zeros((n_delt,5))
form = 'oscillation' #shape of the simulated shifts


for i in range(n_delt):
    
    perfs[i, 0] = chart.ARL1_CUSUM(dataIC, h_5, delta=delt[i], k=0.75,
                  block_length=500, form=form, 
                    nmc=2000, n=2000, BB_method='NBB', missing_values='reset')
    

    perfs[i, 1] = chart.ARL1_CUSUM(dataIC, h_10, delta=delt[i], k=0.75,
                  block_length=500, form=form, 
                    nmc=2000, n=2000, BB_method='MBB', missing_values='reset')
    
    #nbb 
    perfs[i, 2] = chart.ARL1_CUSUM(dataIC, h_nbb, delta=delt[i], k=0.75,
                  block_length=500, form=form, 
                    nmc=2000, n=2000, BB_method='MBB', missing_values='reset')
    
    
    # perfs[i, 2] = chart.ARL1_CUSUM(dataIC, h_27, delta=delt[i], k=0.75,
    #               block_length=500, form=form, 
    #                 nmc=2000, n=2000, BB_method='MBB', missing_values='reset')
    
    
    #cusvm
    perfs[i, 3] = chart.ARL1_CUSUM(dataIC, h, delta=delt[i], k=0.75,
                  block_length=500, form=form, 
                    nmc=2000, n=2000, BB_method='MBB', missing_values='reset')
    
    #large blocks
    perfs[i, 4] = chart.ARL1_CUSUM(dataIC, h_100, delta=delt[i], k=0.75,
                  block_length=500, form=form, 
                    nmc=2000, n=2000, BB_method='MBB', missing_values='reset')
        

    
fig = plt.figure()
plt.plot(delt, perfs[:,0], 'o-', label='BBL=5')
plt.plot(delt, perfs[:,1], 'D--', label='BBL=10')
#plt.plot(delt, perfs[:,2], 'v:', label='BBL=27')
plt.plot(delt, perfs[:,2], 'v:', label='NBB')
plt.plot(delt, perfs[:,3], '^-.', label='BBL=54')
plt.plot(delt, perfs[:,4], 'p', linestyle=(0, (3, 10, 1, 10)), label='BBL=100')
plt.legend(fontsize=16)
plt.xlabel('shift size')
plt.ylabel('ARL1')
plt.show()
#fig.savefig('bbl_simu_500_oscill.pdf') 





