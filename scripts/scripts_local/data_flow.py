# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 14:19:37 2021

@author: sopmathieu

This script creates graphs (data workflow) for the article and the thesis. 
It illustrates the different steps of the pre-processing of the long-term bias. 

"""

### load packages/files 
import pickle
import numpy as np 
from scipy.stats import iqr
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8.0, 5.0)
plt.rcParams['font.size'] = 18
import sys
sys.path.insert(1, '/Sunspot/')
sys.path.insert(2, '../Sunspot')
sys.path.insert(3, '../')

from SunSpot import errors as err
from SunSpot import preprocessing as pre

### load data
with open('data/data_1981', 'rb') as file:
    my_depickler = pickle.Unpickler(file)
    Ns = my_depickler.load() #number of spots
    Ng = my_depickler.load() #number of sunspot groups
    Nc = my_depickler.load() #composite: Ns+10Ng
    station_names = my_depickler.load() #codenames of the stations
    time = my_depickler.load() #time
    
### add new data to station KS
data_ks = np.loadtxt('data/kisl_wolf.txt', usecols=(0,1,2,3), skiprows=1)
Nc_ks = data_ks[9670:23914,3]
Nc[:,24] = Nc_ks

#for a particular station
stat = [i for i in range(len(station_names)) if station_names[i] == 'UC'][0]


## observations (Yit)

Y_UC = Nc[:,stat]

plt.hist(Y_UC[~np.isnan(Y_UC )], bins='auto', density=True)  
plt.text(200, 0.015, 'mean: ' '%.3f' %np.nanmean(Y_UC ))
plt.text(200, 0.0125, 'std: ' '%.3f' %np.nanstd(Y_UC ))
#plt.axis([0,150, 0, 0.08])
plt.yticks(np.arange(0, 0.02, 0.005))
plt.xlabel('$Y_i(t)$ (for %s)' %station_names[stat])
plt.ylabel('Density')
plt.grid(True)
#plt.savefig('hist_1.pdf') #save figures
plt.show()


plt.plot(time, Y_UC)
plt.ylabel('$Y_i(t)$ (for %s)' %station_names[stat])
plt.xlabel('time (year)')
#plt.savefig('flow_1.pdf') #save figures


### remove the solar signal
ratio = err.long_term_error(Nc, period_rescaling=10, wdw=1)
ratio_UC = ratio[:,stat]

plt.hist(ratio_UC[~np.isnan(ratio_UC)], range=[-1,4],bins='auto', density=True, facecolor='b')  
plt.text(2, 2, 'mean: ' '%.3f' %np.nanmean(ratio_UC))
plt.text(2, 1.5, 'std: ' '%.3f' %np.nanstd(ratio_UC))
plt.xlabel('$Y_i(t)/M_t$ (for %s)' %station_names[stat])
plt.ylabel('Density')
plt.grid(True)
#plt.savefig('hist_2.pdf') 
plt.show()

plt.plot(time,ratio_UC)
plt.ylabel('$Y_i(t)/M_t$ (for %s)' %station_names[stat])
plt.xlabel('time (year)')
#plt.savefig('flow_2.pdf') 


### remove the short-term error
eh = err.long_term_error(Nc, period_rescaling=10, wdw=27)
eh_UC = eh[:,stat]

plt.hist(eh_UC[~np.isnan(eh_UC)], range=[-0.6,3.4],bins='auto', density=True, facecolor='b')  
plt.text(2, 2, 'mean: ' '%.3f' %np.nanmean(eh_UC))
plt.text(2, 1.5, 'std: ' '%.3f' %np.nanstd(eh_UC))
plt.xlabel('$\widehat{eh}(i,t)$ (for %s)' %station_names[stat])
plt.ylabel('Density')
plt.grid(True)
#plt.savefig('hist_3.pdf') 
plt.show()

plt.plot(time, eh_UC)
plt.ylabel('$\widehat{eh}(i,t)$ (for %s)' %station_names[stat])
plt.xlabel('time (year)')
#plt.savefig('flow_3.pdf') 

### remove the levels

### discard stations with no values
ind_nan = []
for i in range(eh.shape[1]):
    if not np.all(np.isnan(eh[:,i])): 
        ind_nan.append(i)
eh = eh[:,ind_nan]
station_names = [station_names[i] for i in range(len(station_names)) if i in ind_nan]
(n_obs, n_series) = eh.shape

### apply preprocessing
dataNs = pre.PreProcessing(eh)  
dataNs.level_removal(wdw=4000)  #remove intrisic levels
mu2 = dataNs.data 
mu2_UC = mu2[:,stat]

plt.hist(mu2_UC[~np.isnan(mu2_UC)], range=[-1,1.5],bins='auto', density=True, facecolor='b')  
plt.text(0.75, 2, 'mean: ' '%.3f' %np.nanmean(mu2_UC))
plt.text(0.75, 1.5, 'std: ' '%.3f' %np.nanstd(mu2_UC))
plt.xlabel('$\hat \mu_2(i,t)$ (for %s)' %station_names[stat])
plt.ylabel('Density')
plt.grid(True)
#plt.savefig('hist_4.pdf')
plt.show()

plt.plot(time, mu2_UC)
plt.ylabel('$\hat \mu_2(i,t)$ (for %s)' %station_names[stat])
plt.xlabel('time (year)')
#plt.savefig('flow_4.pdf') 


#################################################################"
###############################################################"

#compute the mse
mse = np.zeros(n_series)
for i in range(n_series):
    mse[i] = (np.nanmedian(mu2[:,i]))**2 + iqr(mu2[:,i], nan_policy='omit') 
ordered_mse = np.argsort(mse)

plt.hist(mse, bins='auto', density=True, facecolor='b')  
plt.xlabel('$\hat \mu_2(i,t)$ (for %s)' %station_names[stat])
plt.ylabel('Density')
plt.grid(True)
plt.show()

kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(ordered_mse.reshape(-1,1))
s = np.linspace(0, n_series)
e = kde.score_samples(s.reshape(-1,1))
plt.plot(s, e)


#################################################################"
###############################################################"

dataNs.selection_pools(method='kmeans') #select the IC pool
pool = np.array(dataNs.pool) #pool (number)
pool_ind = [station_names[i] for i in range(n_series) if i in pool] #pool (index)

mu2_IC = mu2[:,pool]
len_before = len(mu2_IC[~np.isnan(mu2_IC)])
dataNs.outliers_removal(k=1) #remove outliers
mu2_IC_out = dataNs.dataIC 
len_after = len(mu2_IC_out[~np.isnan(mu2_IC_out)])

Perc_removed = len_after*100/len_before #91.188 with k=1 ; 99.1026 with k=2

