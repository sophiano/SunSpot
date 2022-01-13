# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 11:09:09 2021

@author: sopmathieu

This script studies in particular the level (h) of the stations. 
"""

### load packages/files 
import pickle
import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 6.0)
plt.rcParams['font.size'] = 18
import sys
sys.path.insert(1, '/Sunspot/')
sys.path.insert(2, '../Sunspot')
sys.path.insert(3, '../')

from SunSpot import errors as err

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

########################################

### levels and long term errors 
eh = err.long_term_error(Ns, period_rescaling=8, wdw=27)
eh_1 = err.long_term_error(Ns, period_rescaling=8, wdw=365)

plt.hist(eh[~np.isnan(eh)],bins=40, density=True, facecolor='b')  
plt.text(3, 1.4, 'mean: ' '%.3f' %np.nanmean(eh))
plt.text(3, 1, 'std: ' '%.3f' %np.nanstd(eh))
plt.xlabel('$\widehat{eh}$')
plt.ylabel('Density')
plt.grid(True)
#plt.savefig('hist_eh.pdf')
plt.show()

plt.subplot(2,1,1)
stat = [i for i in range(len(station_names)) if station_names[i] == 'SM'][0]
plt.plot(time, eh[:,stat], ':', c='tab:blue', label='$\widehat{eh}$ (27)')
plt.plot(time, eh_1[:,stat], '-', c='tab:orange', label='$\widehat{eh}$ (1)')
plt.plot([time[0], time[-1]], [1, 1], 'k-', lw=2)
plt.ylabel('$\widehat{eh}$ (for %s)' %station_names[stat])
plt.xticks([])
#plt.axis([time[0], time[-1], 0.5, 2.2])
plt.legend(loc='upper right')
plt.subplot(2,1,2)
stat = [i for i in range(len(station_names)) if station_names[i] == 'FU'][0]
plt.plot(time, eh[:,stat], ':', c='tab:blue', label='$\widehat{eh}$ (27)')
plt.plot(time, eh_1[:,stat], '-', c='tab:orange', label='$\widehat{eh}$ (1)')
plt.plot([time[0], time[-1]], [1, 1], 'k-', lw=2)
plt.ylabel('$\widehat{eh}$ (for %s)' %station_names[stat])
plt.xlabel('time (year)')
#plt.axis([time[0], time[-1], 0.5, 2.2])
plt.legend(loc='upper right')
#plt.savefig('eh_time.pdf')
plt.show()

##################################################
### levels

def level(x, wdw, center=True):
    """
    Removes the intrinsic levels of an initial array of series.
    
    This function applies a smoothing process in time by a moving-average
    (MA) filter on each individual series. Then, the smoothed series are
    substracted from the initial series to remove the levels of the 
    processes.
    
    Parameters
    ---------
    x : 2D-array 
        A panel of series (rows: time, columns: series).
    wdw : int
        Length of the MA window length, expressed in days. 
    center : str, optional
        Flag to indicate that the moving windows should be centered with respect 
        to the data. Otherwise, the windows are left-shifted to include only 
        the current and past observations. The default is True.
        
    Returns
    -------
   x_wht_levels : 2D-array
        The array without intrinsic levels.
        
    """
    assert np.ndim(x) == 2, "Input data must be a 2D array"
    (n_obs, n_stations) = x.shape
    levels = np.zeros((n_obs, n_stations))
    levels[:] = np.nan
    
    assert wdw > 0, "Window length must be a postive integer"
    wdw = np.round(wdw)
        
    if center:
        if wdw % 2 == 1:
            halfwdw = int((wdw - 1)/2)
        else:
            wdw += 1
            halfwdw = int((wdw - 1)/2)
        for i in range(n_stations):
            for j in range(n_obs):
                if j < halfwdw: #beginning
                    levels[j,i] = np.nanmean(x[0:wdw,i])
                elif j >= halfwdw and j < n_obs - halfwdw: #middle
                    levels[j,i] = np.nanmean(x[j-halfwdw :j+halfwdw+1,i])
                else: #end
                    levels[j,i] = np.nanmean(x[n_obs-wdw :n_obs,i])

            
    if not center:
        for i in range(n_stations):
            for j in range(n_obs):
                if j < wdw: #beginning
                    levels[j,i] = np.nanmean(x[0:wdw,i]) 
                else: #remaining part
                    levels[j,i] = np.nanmean(x[j - wdw+1:j+1,i]) 
            
    return levels

h = level(eh, 4000)
h_1 = level(eh_1, 4000)

plt.hist(h[~np.isnan(h)],bins=40, density=True, facecolor='b')  
plt.text(1.5, 1.5, 'mean: ' '%.3f' %np.nanmean(h))
plt.text(1.5, 1, 'std: ' '%.3f' %np.nanstd(h))
plt.xlabel('$\hat h$')
plt.ylabel('Density')
plt.grid(True)
#plt.savefig('hist_h.pdf')
plt.show()

plt.subplot(2,1,1)
stat = [i for i in range(len(station_names)) if station_names[i] == 'SM'][0]
plt.plot(time, h[:,stat], '-', c='tab:blue', label='$\widehat{eh}$ (27)')
#plt.plot(time, h_1[:,stat], '-', c='tab:orange', label='$\widehat{eh}$ (1)')
plt.plot([time[0], time[-1]], [1, 1], 'k-', lw=2)
plt.ylabel('$\hat h$ (for %s)' %station_names[stat])
plt.xticks([])
plt.axis([time[0], time[-1], 0.6, 1.7])
#plt.legend(loc='upper right')
plt.subplot(2,1,2)
stat = [i for i in range(len(station_names)) if station_names[i] == 'FU'][0]
plt.plot(time, h[:,stat], '-', c='tab:blue', label='$\widehat{eh}$ (27)')
#plt.plot(time, h_1[:,stat], '-', c='tab:orange', label='$\widehat{eh}$ (1)')
plt.plot([time[0], time[-1]], [1, 1], 'k-', lw=2)
plt.ylabel('$\hat h$ (for %s)' %station_names[stat])
plt.xlabel('time (year)')
plt.axis([time[0], time[-1], 0.6, 1.7])
#plt.legend(loc='upper right')
#plt.savefig('h_time.pdf')
plt.show()

#############################################"
### long-term error

mu2 = err.long_term_error(Nc, period_rescaling=10, wdw=27, level=True, 
                          wdw_level=4000)
mu2_1 = err.long_term_error(Nc, period_rescaling=10, wdw=365, level=True, 
                          wdw_level=4000)

plt.hist(mu2[~np.isnan(mu2)],bins='auto', density=True, facecolor='b')  
plt.text(1.5, 2.5, 'mean: ' '%.3f' %np.nanmean(mu2))
plt.text(1.5, 2, 'std: ' '%.3f' %np.nanstd(mu2))
plt.xlabel('$\hat \mu_2$')
plt.ylabel('Density')
plt.grid(True)
#plt.savefig('hist_mu2.pdf')
plt.show()

plt.subplot(2,1,1)
stat = [i for i in range(len(station_names)) if station_names[i] == 'SM'][0]
plt.plot(time, mu2[:,stat], ':', c='tab:blue', label='$\hat \mu_2$ (27)')
plt.plot(time, mu2_1[:,stat], '-', c='tab:orange', label='$\hat \mu_2$ (1)')
plt.plot([time[0], time[-1]], [0, 0], 'k-', lw=2)
plt.ylabel('$\hat \mu_2$ (for %s)' %station_names[stat])
plt.xticks([])
#plt.axis([time[0], time[-1], 0.6, 1.7])
plt.legend(loc='upper right', fontsize=14)
plt.subplot(2,1,2)
stat = [i for i in range(len(station_names)) if station_names[i] == 'FU'][0]
plt.plot(time, mu2[:,stat], ':', c='tab:blue', label='$\hat \mu_2$ (27)')
plt.plot(time, mu2_1[:,stat], '-', c='tab:orange', label='$\hat \mu_2$ (1)')
plt.plot([time[0], time[-1]], [0, 0], 'k-', lw=2)
plt.ylabel('$\hat \mu_2$ (for %s)' %station_names[stat])
plt.xlabel('time (year)')
#plt.axis([time[0], time[-1], 0.6, 1.7])
plt.legend(loc='upper right', fontsize=14)
plt.savefig('mu2_time.pdf')
plt.show()