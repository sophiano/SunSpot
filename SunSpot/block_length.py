# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 08:23:31 2020

@author: sopmathieu

"""

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.utils import resample
from kneed import KneeLocator
# plot parameters
plt.rcParams['figure.figsize'] = (10.0, 6.0)


def autocorr(x, t=1):
    """ 
    Computes the autocorrelation of a time-series with missing values 
    at a specified lag. 
    
    Parameters
    ----------
    x : 1D-array
       The time-series.
    t : int, optional
       The lag at which the autocorrelation should be computed. 
       Default is one.
       
    Returns
    -------
    corr : float
        The autocorrelation of the series at a specified lag.
    """
    if t >= 1:
        ts1 = x[:-t]; ts2 = x[t:]
    else: 
        ts1 = x; ts2 = x
    a = np.ma.masked_invalid(ts1)
    b = np.ma.masked_invalid(ts2)
    msk = (~a.mask & ~b.mask)
    corr = (np.corrcoef([a[msk], b[msk]]))[0,1]
    return corr



def block_length_choice(data, n_corr=50, wdw_min=10, wdw_max=110, 
                    wdw_step=10, nmc=200, BB_method='MBB', plot=True):
    """
    Computes an appropriate value for the block length.
    
    The algorithm works as follows.
    For each block length tested over the specified range, this function resamples
    several series of observations using a block bootstrap procedure.
    Then, it computes the mean squared error (mse) of the mean,
    the standard deviation and the autocorrelation at different lags 
    of the resampled series (with respect to the original data). 
    Small block lengths represent the variance and the mean of the data properly 
    (mse of the mean and the variance increases when block length augments).
    While large block lengths better account for the autocorrelation of the data
    (mse of the autocorrelation diminishes when block length increases). 
    The appropriate value for the block length is finally selected as the first
    value such that the mse of the autocorrelation starts to stabilize
    ("knee" of the curve).
    This value intuitively corresponds to the smallest length which is 
    able to represent the main part of the autocorrelation of the series.

    Parameters
    ----------
    data : 2D-array
        A matrix representing a panel of time-series to be resampled by
        a block boostrap procedure. (rows: time, columns: series)
        To save computational time, only the IC series may be used. 
    n_corr : int, optional
        Maximal lag up to which the autocorrelation is evaluated.
        The default is 50.
    wdw_min : int, optional
        Lower value for the block length. The block lengths are tested in
        the range [wdw_min, wdw_max]. Default is 10.
    wdw_max : int, optional
        Upper value for the block length. The block lengths are tested in
        the range [wdw_min, wdw_max]. Default is 110.
    wdw_step : int, optional
        Step value for the block length. The block lengths are tested in 
        the range [wdw_min, wdw_max], with step equal to wdw_step.
        Default is 10.
    nmc  : int > 0, optional
        Number of resampled series used to compute the mses.
    BB_method : str, optional
       String that designates the block boostrap method chosen for sampling data. 
       Values for the string should be selected among: 
       'MBB': moving block bootstrap
       'NBB': non-overlapping block bootstrap
       If the matched block bootstrap is intended to be use, 'NBB' may be 
       selected (the matched block bootstrap is based on the 'NBB'). 
       'CBB': circular block bootstrap
       Default is 'MBB'.
    plot : bool, optional
       Flag to show the figures (and print some results). The default is True. 
      
    Returns
    -------
    block-length : int
        The selected block length. 

    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    (n_obs, n_series) = data.shape
    assert BB_method in ['MBB', 'NBB', 'CBB'], "Undefined block bootstrap procedure"
    
    #compute the autocorrelation of the initial data
    corr_data = np.zeros((n_series, n_corr))  
    for j in range(n_series):
        for i in range(n_corr):
            corr_data[j,i] = autocorr(data[:,j],i+1)
      
    ### parameters 
    n_wdw = int(np.ceil((wdw_max - wdw_min)/wdw_step)) #number of block length sizes tested
    mse_mean_series = np.zeros((n_wdw, n_series)); mse_std_series = np.zeros((n_wdw, n_series))
    mse_corr_lag = np.zeros((n_corr, n_series)) ; mse_corr_series = np.zeros((n_wdw, n_series))
    wdw = np.zeros(n_wdw)
    c = 0
    for block_length in range(wdw_min, wdw_max, wdw_step): 
        
        ### Create blocks (moving block bootstrap)
        wdw[c] = block_length
        n_blocks = int(np.ceil(n_obs/block_length))
        
        if BB_method == 'MBB':
            N = n_obs - block_length + 1
            blocks = np.zeros((N, block_length, n_series)) 
            for j in range(n_series):
                for i in range(N):
                    blocks[i,:,j] = data[i:i+block_length, j] #series by series
        elif BB_method == 'NBB':
            N = int(np.floor(n_obs / block_length))
            blocks = np.zeros((N, block_length, n_series))
            for j in range(n_series):
                cc = 0
                it = 0
                for i in range(0, N):
                    blocks[cc,:,j] = data[it:it+block_length,j] #non-overlapping
                    it += block_length
                    cc += 1
        elif BB_method == 'CBB':
                N = n_obs
                blocks = np.zeros((N, block_length, n_series))
                for j in range(n_series): 
                    cc = 0
                    data_dup = np.concatenate((data[:,j], data[:,j]))
                    for i in range(0, N):
                        blocks[cc,:,j] = data_dup[i:i+block_length] #overlapping
                        cc += 1
       
        for j in range(n_series):
            corr_boot = np.zeros((n_corr, nmc)); mean_boot = np.zeros((nmc)); std_boot = np.zeros((nmc))
            corr_boot[:] = np.nan; mean_boot[:] = np.nan; std_boot[:] = np.nan
            
            for b in range(nmc):   
                boot = resample(blocks[:,:,j], replace=True, n_samples=n_blocks).flatten()
                #boot = boot[~np.isnan(boot)]
                for i in range(n_corr):
                    corr_boot[i,b] = autocorr(boot, i+1)  
                mean_boot[b] = np.nanmean(boot)
                std_boot[b] = np.nanstd(boot)
        
            ### results per station
            mse_mean_series[c,j] = (np.nanmean(mean_boot) - np.nanmean(data[:,j]))**2 + np.nanvar(mean_boot)
            mse_std_series[c,j] = (np.nanmean(std_boot) - np.nanstd(data[:,j]))**2 + np.nanvar(std_boot)
            for i in range(n_corr):
                mse_corr_lag[i,j] = (np.nanmean(corr_boot[i,:]) - corr_data[j,i])**2 + np.nanvar(corr_boot[i,:])
            mse_corr_series[c,j] = np.nanmean(mse_corr_lag, axis=0)[j]
        c += 1
    
    #for all stations
    mse_mean = np.nanmean(mse_mean_series, axis=1)
    mse_std = np.nanmean(mse_std_series, axis=1)  
    mse_corr = np.nanmean(mse_corr_series, axis=1) 
        
    x = wdw
    y = mse_corr
    
    if plot: 
        plt.plot(x, y); plt.xlabel('block length')
        plt.ylabel('mse of autocorrelation')
        plt.title('MSE of the autocorrelation as a function of the block length')
        plt.show()
        print('Block length which minimizes the mse of the mean:', x[np.argmin(mse_mean)])#0
        print('Block length which minimizes the mse of the std:', x[np.argmin(mse_std)])#0
        print('Block length which minimizes the mse of the autocorrelation:', x[np.argmin(mse_corr)]) #100
    
    #select the knee of the curve
    coef = np.polyfit(x, y, deg=1)
    coef_curve = np.polyfit(x, y, deg=2)
    if coef_curve[0] < 0: 
        curve = 'concave'        
    else: 
        curve = 'convex'
    if coef[0] < 0: #slope is positive
        direction = 'decreasing'
    else: #slope is negative
        direction = 'increasing'
    kn = KneeLocator(x, y, curve=curve, direction=direction)
    block_length = kn.knee 
    
    return block_length
    

# if __name__ == "__main__":
    
    # import pickle
    
    # ### load pre-processed data 
    # with open('data/Ns_27', 'rb') as fichier:
    #       my_depickler = pickle.Unpickler(fichier)
    #       data = my_depickler.load() #contain all deviations, even from IC stations
    #       time = my_depickler.load()
    #       station_names = my_depickler.load()
    #       dataIC = my_depickler.load()
    #       pool = my_depickler.load()
             
    # block_length = block_length_choice(dataIC, n_corr=50, wdw_step= 60, 
    #                                    BB_method='CBB')