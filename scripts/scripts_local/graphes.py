# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 10:31:38 2020

@author: sopmathieu

This script reproduces the first figure of the second article which shows the 
long-term errors and the median of the network as a function of time. 

"""

import pickle
import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 6.0)
import sys
sys.path.insert(0, '../Sunspot/')
sys.path.insert(1, '../')

from SunSpot import errors as err

with open('data/data_1981', 'rb') as file: # load all stations
     my_depickler = pickle.Unpickler(file)
     Ns = my_depickler.load() #number of spots
     Ng = my_depickler.load() #number of sunspot groups
     Nc = my_depickler.load() #Ns+10Ng
     station_names = my_depickler.load() #index of the stations
     time = my_depickler.load() #time


mu2_wht_level_27 = err.long_term_error(Ns, period_rescaling=8, wdw=27, level=True)
mu2_wht_level_365 = err.long_term_error(Ns, period_rescaling=8, wdw=365, level=True)
Mt = err.median(Ns, period_rescaling=8, with_rescaling=True)#med of observations

#### plot the long-term errors without level 
start = int(np.where(time == 1981)[0])
stat = [i for i in range(len(station_names)) if station_names[i] == 'MT'][0]

plt.rcParams['font.size'] = 13
plt.rcParams['figure.figsize'] = (10.0, 6.0)
plt.figure(1)
plt.subplot(2, 1, 1)
#plt.title("Long-term error in %s" %station_names[stat])
plt.ylabel('$\hat \mu_2(i,t)$')
plt.plot(time[start:], mu2_wht_level_27[start:,stat], ':',c='tab:purple', label='27 days')
plt.plot(time[start:], mu2_wht_level_365[start:,stat], c='tab:green',  lw=2, label= '365 days')
plt.legend(loc='upper right')
plt.plot([time[start], time[-1]], [0, 0], 'k-', lw=2)
axes = plt.gca()
axes.set_ylim([-1.3,2.7])

plt.subplot(2, 1, 2)
#plt.title("Solar cycle")
plt.ylabel('$M_t$')
plt.xlabel('year')
plt.plot(time[start:], Mt[start:])
plt.tight_layout()
plt.savefig('cycle.pdf')  
#plt.show()
    