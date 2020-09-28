# -*- coding: utf-8 -*-
"""
Created on Thu May  7 17:30:50 2020

@author: sopmathieu
"""

import numpy as np 
from sklearn.utils import resample

import sys
sys.path.insert(0, '../Sunspot/')
sys.path.insert(1, '../')

import BB_methods as BB

def ARL0_CUSUM_MV(data, L_plus, L_minus=None, delta=1.5, k=None, nmc=4000,
                  n=4000, two_sided=True, missing_values='omit', 
                  gap=0, block_length=None, BB_method='MBB'):
    """ 
    Compute the ARL0 of the CUSUM chart for a control limit value
    and a particular shift size in presence of missing values.
    
    Arguments:
    data:           IC dataset (not the blocks)
    delta:          shift size that we aim to detect
    k:              allowance parameter
    L_plus:         value of the upper control limit
    L_minus:        value of the lower control limit
    nmc:            number of Monte-Carlo runs
    n:              length of the resampled series 
    two_sided:      boolean (True=two-sided chart, False=one-sided chart (upper))
    missing_values: how to treat the missing values. 
                    'omit' removes the blocks containing missing values 
                    'filled' fills-up the MV by the mean of each series 
                    'reset' reset the chart after a gap of # days
    gap:            length of the gap (period where the chart statistic is propagated)
    block_length:   length of the block
    BB_method:      block boostrap method among: 
                    'MBB': moving block bootstrap
                    'NBB': non-overlapping block bootstrap
                    'CBB': circular block bootstrap
                    'MABB': matched block bootstrap
    
    Return:
    ARL:            ARL_0 value   
    """  
    (n_obs, n_series) = data.shape    
    if missing_values == 'filled':
        for i in range(n_series):
            data[np.isnan(data[:,i]),i] = np.nanmean(data[:,i]) #fill obs by the mean of the series
            
    ##Block bootstrap
    if missing_values == 'filled' or missing_values == 'omit':
        if BB_method == 'MBB': 
            blocks = BB.MBB(data, block_length)    
        elif BB_method == 'NBB':
            blocks = BB.NBB(data, block_length) 
        elif BB_method == 'CBB':
            blocks = BB.CBB(data, block_length) 
        
    else:
        if BB_method == 'MBB': 
            blocks = BB.MBB(data, block_length, NaN=True)    
        elif BB_method == 'NBB':
            blocks = BB.NBB(data, block_length, NaN=True) 
        elif BB_method == 'CBB':
            blocks = BB.CBB(data, block_length, NaN=True)
   
    if 'blocks' in locals():
        n_blocks = int(np.ceil(n/blocks.shape[1]))
      
    #chart parameters
    if k is None:
        k = delta/2
    if L_minus is None:
        L_minus = -L_plus

    RL_minus = np.zeros((nmc,1)); RL_plus = np.zeros((nmc,1)) 
    RL_minus[:] = np.nan; RL_plus[:] = np.nan
    for j in range(nmc):
        
        if BB_method == 'MABB': 
            boot = BB.resample_MatchedBB(data, block_length, n=n)
        else:
            boot = resample(blocks, replace=True, n_samples=n_blocks).flatten()[:n]
    
        ### Monitoring ###
        C_plus = np.zeros((n,1)) 
        cp = 0; nan_p = 0
        for i in range(1, n):
            if not np.isnan(boot[i]):
                C_plus[i] = max(0, C_plus[i-1] + boot[i] - k)
                cp = 0
            elif (np.isnan(boot[i]) and cp < gap):
                C_plus[i] = C_plus[i-1]
                cp += 1; nan_p += 1
            else: 
                C_plus[i] = 0 
                nan_p += 1
            if C_plus[i] > L_plus:
                RL_plus[j] = i #-nan_p
                break 
            
        C_minus = np.zeros((n,1))
        cm = 0; nan_m = 0
        for i in range(1, n):
            if not np.isnan(boot[i]):
                 C_minus[i] = min(0, C_minus[i-1] + boot[i] + k)
                 cm = 0
            elif (np.isnan(boot[i]) and cm < gap):
                C_minus[i] = C_minus[i-1]
                cm += 1; nan_m += 1
            else: 
                C_minus[i] = 0 
                nan_m += 1
            if C_minus[i] < L_minus:
                RL_minus[j] = i #-nan_m
                break 
    
        if np.isnan(RL_plus[j]):
            RL_plus[j] = n #-nan_p
        if np.isnan(RL_minus[j]):
            RL_minus[j] = n  #-nan_m
           
    if two_sided:
        ARL = (1/(np.mean(RL_minus)) + 1/(np.mean(RL_plus)))**(-1) 
    else: 
        ARL = np.mean(RL_plus)
    return ARL


def search_CUSUM_MV(data, delta=1.5, k=None, ARL0_threshold=200, rho=2, L_plus=20, 
                    L_minus=0, nmc=4000, n=4000, two_sided=True, verbose=True,
                    missing_values='omit', gap=0, block_length=None, BB_method='MBB'):
    """ 
    Compute the control limit of the CUSUM chart for a pre-specified value of ARL0
    and a particular shift size in presence of missing values.
    
    Arguments:
    data:            IC dataset 
    delta:           shift size that we aim to detect
    k:               allowance parameter
    ARL0_threshold:  pre-specified value of ARL0 
    rho:             accuracy: |ARL0-ARL0_threshold|>rho
    L_plus:          upper value of the control limit
    L_minus:         lower value of the control limit
    nmc:             number of Monte-Carlo runs
    n:               length of the resampled series 
    two_sided:       boolean (True=two sided chart, False=one_sided)
    verbose:         boolean (when True: print intermediate results)
    missing_values:  how to treat the missing values. 
                     'omit': removes the blocks containing missing values 
                     'filled': fills-up the MV by the mean of each series 
                     'reset': resets the chart after a gap of # days
    gap:             length of the gap (period where the chart statistic is propagated)
    block_length:    length of the block 
    BB_method:       block boostrap method among: 
                     'MBB': moving block bootstrap
                     'NBB': non-overlapping block bootstrap
                     'CBB': circular block bootstrap
                     'MABB': matched block bootstrap
    
    Return:
    L:               control limit of the chart
    """
    
    (n_obs, n_series) = data.shape    
    if missing_values == 'filled':
        for i in range(n_series):
            data[np.isnan(data[:,i]),i] = np.nanmean(data[:,i]) #fill obs by the mean of the series
           
    ##Block bootstrap
    if missing_values == 'filled' or missing_values == 'omit':
        if BB_method == 'MBB': 
            blocks = BB.MBB(data, block_length)    
        elif BB_method == 'NBB':
            blocks = BB.NBB(data, block_length) 
        elif BB_method == 'CBB':
            blocks = BB.CBB(data, block_length) 
        
    else:
        if BB_method == 'MBB': 
            blocks = BB.MBB(data, block_length, NaN=True)    
        elif BB_method == 'NBB':
            blocks = BB.NBB(data, block_length, NaN=True) 
        elif BB_method == 'CBB':
            blocks = BB.CBB(data, block_length, NaN=True)
   
    if 'blocks' in locals():
        n_blocks = int(np.ceil(n/blocks.shape[1]))
        
        
    #chart parameters
    if k is None:
        k = delta/2
    L = (L_plus+L_minus)/2

    ARL = 0
    while (np.abs(ARL - ARL0_threshold) > rho):
        RL_minus = np.zeros((nmc,1)); RL_plus = np.zeros((nmc,1)) 
        RL_minus[:] = np.nan; RL_plus[:] = np.nan
        for j in range(nmc):
            
            if BB_method == 'MABB': 
                boot = BB.resample_MatchedBB(data, block_length, n=n)
            else:
                boot = resample(blocks, replace=True, n_samples=n_blocks).flatten()[:n]
    
            
            ### Monitoring ###
            C_plus = np.zeros((n,1)) 
            cp = 0; nan_p = 0
            for i in range(1, n):
                if not np.isnan(boot[i]):
                    C_plus[i] = max(0, C_plus[i-1] + boot[i] - k)
                    cp = 0
                elif (np.isnan(boot[i]) and cp < gap):
                    C_plus[i] = C_plus[i-1]
                    cp += 1; nan_p += 1
                else: 
                    C_plus[i] = 0 
                    nan_p += 1
                if C_plus[i] > L:
                    RL_plus[j] = i #-nan_p
                    break 
                
            C_minus = np.zeros((n,1))
            cm = 0; nan_m = 0
            for i in range(1, n):
                if not np.isnan(boot[i]):
                     C_minus[i] = min(0, C_minus[i-1] + boot[i] + k)
                     cm = 0
                elif (np.isnan(boot[i]) and cm < gap):
                    C_minus[i] = C_minus[i-1]
                    cm += 1; nan_m += 1
                else: 
                    C_minus[i] = 0 
                    nan_m += 1
                if C_minus[i] < -L:
                    RL_minus[j] = i #- nan_m
                    break 
        
            if np.isnan(RL_plus[j]):
                RL_plus[j] = n #- nan_p
            if np.isnan(RL_minus[j]):
                RL_minus[j] = n #- nan_m
                  
        if two_sided:
            ARL = (1/(np.mean(RL_minus)) + 1/(np.mean(RL_plus)))**(-1) 
        else: 
            ARL = np.mean(RL_plus)
        if ARL < ARL0_threshold:
            L_minus = (L_minus + L_plus)/2
        elif ARL > ARL0_threshold:
            L_plus = (L_minus + L_plus)/2
        L = (L_plus + L_minus)/2
            
        if verbose:
            print(ARL)
            print(L)
            
    return L

def ARL1_CUSUM_MV(data, L_plus, L_minus=None, form='jump', delta=1.5, k=None, 
                  nmc=4000, n=4000, two_sided=True, eta=0.25, 
                  missing_values='omit', gap=0, block_length=None, BB_method='MBB'):
    """ 
    Compute the ARL1 of the CUSUM chart for a simulated shift of size delta
    and form between 'jump', 'oscill' and 'trend'.
    
    Arguments:
    data:            IC dataset
    L_plus:          value of the upper control limit
    L_minus:         value of the lower control limit
    form:            form of the shift: 'jump', 'oscill' or 'trend'
    delta:           shift size 
    k:               allowance parameter
    nmc:             number of Monte-Carlo runs
    n:               length of the resampled series
    two_sided:       boolean (True=two-sided chart, False=one-sided chart (upper))
    eta:             parameter of the oscillating shift
    missing_values:  how to treat the missing values. 
                     'omit': removes the blocks containing missing values 
                     'filled': fills-up the MV by the mean of each series 
                     'reset': resets the chart after a gap of # days
    gap:             length of the gap (period where the chart statistic is propagated)
    block_length:    length of the block
    BB_method:       block boostrap method among: 
                     'MBB': moving block bootstrap
                     'NBB': non-overlapping block bootstrap
                     'CBB': circular block bootstrap
                     'MABB': matched block bootstrap
    
    Return:
    ARL1:            ARL1 value of the chart
    """
    (n_obs, n_series) = data.shape    
    if missing_values == 'filled':
        for i in range(n_series):
            data[np.isnan(data[:,i]),i] = np.nanmean(data[:,i]) #fill obs by the mean of the series
        
    ##Block bootstrap
    if missing_values == 'filled' or missing_values == 'omit':
        if BB_method == 'MBB': 
            blocks = BB.MBB(data, block_length)    
        elif BB_method == 'NBB':
            blocks = BB.NBB(data, block_length) 
        elif BB_method == 'CBB':
            blocks = BB.CBB(data, block_length) 
        
    else:
        if BB_method == 'MBB': 
            blocks = BB.MBB(data, block_length, NaN=True)    
        elif BB_method == 'NBB':
            blocks = BB.NBB(data, block_length, NaN=True) 
        elif BB_method == 'CBB':
            blocks = BB.CBB(data, block_length, NaN=True)
   
    if 'blocks' in locals():
        n_blocks = int(np.ceil(n/blocks.shape[1]))
         
    #parameters
    shift = delta
    if k is None:
        k = delta/2
    if L_minus is None:
        L_minus = -L_plus

    RL1_plus = np.zeros((nmc,1)); RL1_minus = np.zeros((nmc,1))
    RL1_plus[:] = np.nan; RL1_minus[:] = np.nan
    for b in range(nmc):
        
        if BB_method == 'MABB': 
            boot = BB.resample_MatchedBB(data, block_length, n=n)
        else:
            boot = resample(blocks, replace=True, n_samples=n_blocks).flatten()[:n]
    
        if form == 'oscill':
            boot = np.sin(eta*np.pi*np.arange(n))*shift + boot
            pass
        elif form == 'trend':
            boot = shift/(150)*(np.arange(n)**1.5) + boot
            pass
        else: 
            boot = boot + shift
            pass

        C_plus = np.zeros((n,1)) 
        cp = 0; nan_p = np.zeros((n,1))
        for i in range(1, n):
            if not np.isnan(boot[i]):
                C_plus[i] = max(0, C_plus[i-1] + boot[i] - k)
                cp = 0
            elif (np.isnan(boot[i]) and cp < gap):
                C_plus[i] = C_plus[i-1]
                cp += 1; nan_p[i] = 1
            else: 
                C_plus[i] = 0
                nan_p[i] = 1
            if C_plus[i] > L_plus: 
                ind = nan_p[:i]
                RL1_plus[b] = i #-sum(ind)
                break 
            
        C_minus = np.zeros((n,1)) 
        cm = 0; nan_m = np.zeros((n,1))      
        for j in range(1, n):
            if not np.isnan(boot[j]):
                C_minus[j] = min(0, C_minus[j-1] + boot[j] + k)
                cm = 0
            elif (np.isnan(boot[j]) and cm < gap):
                C_minus[j] = C_minus[j-1]
                cm += 1; nan_m[j] = 1
            else: 
                C_minus[j] = 0
                nan_m[j] = 1
            if C_minus[j] < L_minus:
                ind = nan_m[:j]
                RL1_minus[b] = j #-sum(ind)
                break
            
        if np.isnan(RL1_plus[b]): 
            RL1_plus[b] = n
        if np.isnan(RL1_minus[b]):
            RL1_minus[b] = n
            
           
    if two_sided:
        ARL1 = (1/(np.nanmean(RL1_minus)) + 1/(np.nanmean(RL1_plus)))**(-1) 
    else: 
        ARL1 = np.mean(RL1_plus)
    return ARL1


def ARL_values_MV(data, L_plus, L_minus=None, form='jump', delta=1.5, k=None,
                  nmc=4000, n=8000, two_sided=True, eta=0.25,
                  missing_values='omit', gap=0, block_length=None, BB_method='MBB'):
    """ 
    Compute the ARL1 and the ARL0 of the CUSUM chart for a simulated shift of size 
    delta and form between 'jump', 'oscill' and 'trend'.
    
    Arguments:
    data:             IC dataset
    L_plus:           value of the upper control limit
    L_minus:          value of the lower control limit
    form:             form of the shift: 'jump', 'oscill' or 'trend'
    delta:            shift size 
    k:                allowance parameter
    nmc:              number of Monte-Carlo runs
    n:                length of the resampled series 
    two_sided:        boolean (True=two-sided chart, False=one-sided chart (upper))
    eta:              parameter of the oscillating shifts
    missing_values:   how to treat the missing values. 
                      'omit': removes the blocks containing missing values 
                      'filled': fills-up the MV by the mean of each series 
                      'reset': resets the chart after a gap of # days
    gap:              length of the gap (period where the chart statistic is propagated)
    block_length:     length of the block
    BB_method:        block boostrap method among: 
                      'MBB': moving block bootstrap
                      'NBB': non-overlapping block bootstrap
                      'CBB': circular block bootstrap
                      'MABB': matched block bootstrap
    
    Return:
    ARL1, ARL0:     ARL1 and ARL0 values
    """
    (n_obs, n_series) = data.shape    
    if missing_values == 'filled':
        for i in range(n_series):
            data[np.isnan(data[:,i]),i] = np.nanmean(data[:,i]) #fill obs by the mean of the series
        
    ##Block bootstrap
    if missing_values == 'filled' or missing_values == 'omit':
        if BB_method == 'MBB': 
            blocks = BB.MBB(data, block_length)    
        elif BB_method == 'NBB':
            blocks = BB.NBB(data, block_length) 
        elif BB_method == 'CBB':
            blocks = BB.CBB(data, block_length) 
        
    else:
        if BB_method == 'MBB': 
            blocks = BB.MBB(data, block_length, NaN=True)    
        elif BB_method == 'NBB':
            blocks = BB.NBB(data, block_length, NaN=True) 
        elif BB_method == 'CBB':
            blocks = BB.CBB(data, block_length, NaN=True)
   
    if 'blocks' in locals():
        n_blocks = int(np.ceil(n/blocks.shape[1]))
    
    #chart parameters
    shift = delta
    if k is None:
        k = delta/2
    if L_minus is None:
        L_minus = -L_plus
    n_shift = int(n/2)

    FP_minus = np.zeros((nmc,1)); FP_plus = np.zeros((nmc,1))
    RL1_plus = np.zeros((nmc,1)); RL1_minus = np.zeros((nmc,1))
    RL1_plus[:] = np.nan; RL1_minus[:] = np.nan
    for b in range(nmc):
        
        if BB_method == 'MABB': 
            boot = BB.resample_MatchedBB(data, block_length, n=n)
        else:
            boot = resample(blocks, replace=True, n_samples=n_blocks).flatten()[:n]
    
        if form == 'oscill':
            boot[n_shift:] = np.sin(eta*np.pi*np.arange(n_shift))*shift + boot[n_shift:]
            pass
        elif form == 'trend':
            boot[n_shift:] = shift/(150)*(np.arange(n_shift)**1.5) + boot[n_shift:]
            pass
        else: 
            boot[n_shift:] = boot[n_shift:] + shift
            pass
        
        cnt_plus = 0; cp = 0 
        C_plus = np.zeros((n,1)); nan_p = np.zeros((n,1))
        for i in range(1, n):
            if not np.isnan(boot[i]):
                C_plus[i] = max(0, C_plus[i-1] + boot[i] - k)
                C_plus[n_shift] = 0
                cp = 0
            elif (np.isnan(boot[i]) and cp < gap):
                C_plus[n_shift] = 0
                C_plus[i] = C_plus[i-1]
                cp += 1; nan_p[i] = 1
            else: 
                C_plus[i] = 0 
                nan_p[i] = 1
            if C_plus[i] > L_plus and i < n_shift + 1 and cnt_plus == 0:
                ind = nan_p[0:i]
                FP_plus[b] = i #-sum(ind)
                cnt_plus += 1
            elif C_plus[i] > L_plus and i > n_shift: 
                ind = nan_p[n_shift:i]
                RL1_plus[b] = i - n_shift #-sum(ind)
                break 
            
        cnt_minus = 0; cm = 0
        C_minus = np.zeros((n,1)); nan_m = np.zeros((n,1))      
        for j in range(1, n):
            if not np.isnan(boot[j]):
                C_minus[j] = min(0, C_minus[j-1] + boot[j] + k)
                C_minus[n_shift] = 0
                cm = 0
            elif (np.isnan(boot[j]) and cm < gap):
                C_minus[n_shift] = 0
                C_minus[j] = C_minus[j-1]
                cm += 1; nan_m[j] = 1
            else: 
                C_minus[j] = 0 
                nan_m[j] = 1
            if C_minus[j] < L_minus and j <n_shift + 1 and cnt_minus == 0: # first false positive 
                ind = nan_p[0:j]
                FP_minus[b] = j #-sum(ind) 
                cnt_minus += 1
            elif C_minus[j] < L_minus and j > n_shift: 
                ind = nan_m[n_shift:j]
                RL1_minus[b] = j - n_shift #-sum(ind)
                break
            
        if np.isnan(RL1_plus[b]): 
            RL1_plus[b] = n - n_shift
        if np.isnan(RL1_minus[b]): 
            RL1_minus[b] = n - n_shift
            
        if FP_minus[b] == 0: 
            FP_minus[b] = n_shift
        if FP_plus[b] == 0: 
            FP_plus[b] = n_shift
            
           
    if two_sided:
        ARL1 = (1/(np.nanmean(RL1_minus)) + 1/(np.nanmean(RL1_plus)))**(-1) 
        ARL0 = (1/(np.nanmean(FP_minus)) + 1/(np.nanmean(FP_plus)))**(-1)
    else: 
        ARL1 = np.mean(RL1_plus)
        ARL0 = np.nanmean(FP_plus)
    return (ARL1, ARL0)

#############################
    
def shiftSize(data, L_plus, L_minus=None, qt = 0.5, delta=1.5, k=None, nmc=4000, 
              n=2000, two_sided=True, block_length=None, missing_values='omit', 
                 gap=0, BB_method='MBB'):
    """ 
    Estimate an appropriate shift size (parameter delta). 
    The sizes of the shifts are estimated after each alert using Montgomery's formula
    on OC series. Then, the appropriate shift size is defined 
    as a particular quantile of the shift size distribution. 
    
    Arguments:
    data:             OC dataset 
    L_plus:           value of the upper control limit
    L_minus:          value of the lower control limit
    qt:               particular quantile of the distribution
    delta:            initial value for the shift size
    k:                allowance parameter
    nmc:              number of Monte-Carlo runs
    n:                length of the resampled series 
    two_sided:        boolean (True=two-sided chart, False=one-sided chart (upper)) 
    block_length:     length of the blocks 
    missing_values:   how to treat the missing values. 
                      'omit' removes the blocks containing missing values 
                      'filled' fills-up the MV by the mean of each series 
                      'reset' reset the chart after a gap of # days
    gap:              length of the gap (period where the chart statistic is propagated)
    BB_method:        block boostrap method among: 
                      'MBB': moving block bootstrap
                      'NBB': non-overlapping block bootstrap
                      'CBB': circular block bootstrap
                      'MABB': matched block bootstrap
    Return:
    estimated_shift:  estimated shift size     
    """
    
    if k is None:
        k = delta/2
    if L_minus is None:
        L_minus = -L_plus
    (n_obs, n_series) = data.shape 
    
    if missing_values == 'filled':
        for i in range(n_series):
            data[np.isnan(data[:,i]),i] = np.nanmean(data[:,i]) #fill obs by the mean of the series
            
    ##Block bootstrap
    if missing_values == 'filled' or missing_values == 'omit':
        if BB_method == 'MBB': 
            blocks = BB.MBB(data, block_length)    
        elif BB_method == 'NBB':
            blocks = BB.NBB(data, block_length) 
        elif BB_method == 'CBB':
            blocks = BB.CBB(data, block_length) 
        
    else:
        if BB_method == 'MBB': 
            blocks = BB.MBB(data, block_length, NaN=True)    
        elif BB_method == 'NBB':
            blocks = BB.NBB(data, block_length, NaN=True) 
        elif BB_method == 'CBB':
            blocks = BB.CBB(data, block_length, NaN=True)
   
    if 'blocks' in locals():
        n_blocks = int(np.ceil(n/blocks.shape[1]))


    shift_hat_plus = np.zeros((nmc, 1)); shift_hat_minus = np.zeros((nmc, 1))
    shift_hat_plus[:] = np.nan; shift_hat_minus[:] = np.nan
    for b in range(nmc):

        if BB_method == 'MABB': 
            boot = BB.resample_MatchedBB(data, block_length, n=n)
        else:
            boot = resample(blocks, replace=True, n_samples=n_blocks).flatten()[:n]
    
        C_plus = np.zeros((n,1)); cp = 0
        for i in range(1, n):
            if not np.isnan(boot[i]):
                C_plus[i] = max(0, C_plus[i-1] + boot[i] - k)
                cp = 0
            elif (np.isnan(boot[i]) and cp < gap):
                C_plus[i] = C_plus[i-1]
                cp += 1
            else: 
                C_plus[i] = 0 
            if C_plus[i] > L_plus:
                last_zero = np.where(C_plus[:i] == 0)[0][-1]
                shift_hat_plus[b] = k + C_plus[i]/(i - last_zero)
                break 
            
        C_minus = np.zeros((n,1)); cm = 0     
        for j in range(1, n):
            if not np.isnan(boot[j]):
                C_minus[j] = min(0, C_minus[j-1] + boot[j] + k)
                cm = 0
            elif (np.isnan(boot[j]) and cm < gap):
                C_minus[j] = C_minus[j-1]
                cm += 1
            else: 
                C_minus[j] = 0 
            if C_minus[j] < L_minus:
                last_zero = np.where(C_minus[:j] == 0)[0][-1]
                shift_hat_minus[b] = -k - C_minus[j]/(j - last_zero)
                break
                        
    if two_sided:
        shifts = np.concatenate((shift_hat_plus[np.where(~np.isnan(shift_hat_plus))],
                                shift_hat_minus[np.where(~np.isnan(shift_hat_minus))]))
    else:
        shifts = shift_hat_plus[np.where(~np.isnan(shift_hat_plus))]
    
    estimated_shift = np.quantile(np.abs(shifts), qt) 
        
    return estimated_shift


def reccurentDesign_CUSUM(data, indexIC, dataIC=None, delta=1.5, k=None, 
                          ARL0_threshold=200, block_length=None, qt=0.5, accuracy=0.1, 
                          L_up=30, missing_values ='omit', gap=0, verbose=True, 
                          BB_method='MBB'):
    """ 
    Recurrent algorithm to compute the control limits of the chart. 
    
    First, the program computes the control limit for an initial value of delta. 
    Then, it estimates the actual shifts on the OC processes using Montgomery's 
    formula. After that, the control limits are readjusted for the new delta. 
    The procedure is iterated until delta converges. 
    
    Arguments:
    data:              dataset containing the OC and IC processes
    indexIC:           array with the index of the IC processes
                       among all processes
    dataIC:            IC dataset 
    delta:             initial size of the shift
    k:                 allowance parameter
    ARL0_threshold:    pre-specified value of ARL0 
    block_length:      length of the blocks 
    qt:                particular quantile 
    accuracy:          accuracy of the convergence between the deltas 
    L_up:              upper value of the control limit
    missing_values:    how to treat the missing values. 
                       'omit' removes the blocks containing missing values 
                       'filled' fills-up the MV by the mean of each series 
                       'reset' reset the chart after a gap of # days
    gap:               length of the gap (period where the chart statistic is propagated)
    verbose:           boolean. If true, print intermediate results
    BB_method:         block boostrap method among: 
                       'MBB': moving block bootstrap
                       'NBB': non-overlapping block bootstrap
                       'CBB': circular block bootstrap
                       'MABB': matched block bootstrap
                       
    Return: 
    L:                 value of the control limit
    delta:             estimated shift size
    """
    (n_obs, n_series) = data.shape 
    if dataIC is None:
        dataIC = data[:,indexIC]
    indexOC = np.in1d(np.arange(n_series), indexIC)
    dataOC = data[:,~indexOC]
    delta_prev = 0
    
    while (np.abs(delta - delta_prev) > accuracy):
        L = search_CUSUM_MV(dataIC, delta=delta, k=k, L_plus=L_up, ARL0_threshold=ARL0_threshold, 
                            block_length=block_length, missing_values=missing_values,
                            gap=gap, verbose=False, BB_method=BB_method)
        delta_prev = delta
        delta = shiftSize(dataOC, L_plus=L, delta=delta, k=k, qt=qt, 
                          missing_values=missing_values, gap=gap, 
                          block_length=block_length, BB_method=BB_method) 
        if verbose:
            print(L)
            print(delta)
        
    return (L, delta)

##########################################################################"
""" Tests """

if __name__ == "__main__":            

    
    #### iid normal data tests ####
    data = np.random.normal(0 ,1, size=(10000,1))
    #paper 'basics about the CUSUM'
    t1 = ARL0_CUSUM_MV(data, delta=1, L_plus=4, block_length=1) #168
    t1a = ARL0_CUSUM_MV(data, delta=1, L_plus=4, block_length=1, BB_method='CBB') #168
    t1b = ARL0_CUSUM_MV(data, delta=1, L_plus=4, block_length=1, BB_method='NBB') #168
    t1c = ARL0_CUSUM_MV(data, delta=1, L_plus=4, block_length=1, BB_method='MABB') #168
    t2 = ARL0_CUSUM_MV(data, delta=1, L_plus=5, block_length=1) #465
    
    t3 = search_CUSUM_MV(data, delta=1, ARL0_threshold=465, verbose=True, block_length=1)#5
    t3a = search_CUSUM_MV(data, delta=1, ARL0_threshold=465, block_length=1, BB_method='CBB')#5
    t3b = search_CUSUM_MV(data, delta=1, ARL0_threshold=465, block_length=1, BB_method='NBB')#5
    t3c = search_CUSUM_MV(data, delta=1, ARL0_threshold=465, block_length=1, BB_method='MABB')#5
    
    t4 = ARL1_CUSUM_MV(data, delta=2, L_plus=4, k=1/2, block_length=1) #3.34
    t4a = ARL1_CUSUM_MV(data, delta=2, L_plus=4, k=1/2, block_length=1, BB_method='CBB') #3.34
    t4b = ARL1_CUSUM_MV(data, delta=2, L_plus=4, k=1/2, block_length=1, BB_method='NBB') #3.34
    t4c = ARL1_CUSUM_MV(data, delta=2, L_plus=4, k=1/2, block_length=1, BB_method='MABB') #3.34
    t5 = ARL1_CUSUM_MV(data, delta=2, L_plus=5, k=1/2, block_length=1) #4.01
    #Qiu example 4.3
    t6 = ARL1_CUSUM_MV(data, delta=0.25, L_plus=3.502, k=1/2, block_length=1) #52.836
    
    t7 = ARL_values_MV(data, delta=2, L_plus=4, k=1/2, block_length=1) #3.34-168
    t7a = ARL_values_MV(data, delta=2, L_plus=4, k=1/2, block_length=1, BB_method='CBB') #3.34-168
    t7c = ARL_values_MV(data, delta=2, L_plus=4, k=1/2, block_length=1, BB_method='MABB') #3.34-168
    t8 = ARL_values_MV(data, delta=2, L_plus=5, k=1/2, block_length=1) #4.01-465
    
    #####################"
    
    n = 1000
    delta = 2
    dataIC = np.random.normal(0, 1, size=(n,4))
    dataOC = np.random.normal(delta, 1, size=(n,4))
    data = np.hstack((dataIC, dataOC))
    indexIC = np.arange(0, 4)
    
    L1 = search_CUSUM_MV(dataIC, L_plus=5, delta=1.5, block_length=1) #2.9-3
    d1 = shiftSize(data, L1, block_length=1) #1.8-2
    L2, d2 = reccurentDesign_CUSUM(data=data, indexIC=indexIC, delta=delta, 
                          block_length=1, L_up=5) #delta=2.4-2.3 and L=1.8-1.9
