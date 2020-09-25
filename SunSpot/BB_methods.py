# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:58:08 2020
@author: sopmathieu
"""

import numpy as np    
import scipy.stats as ss
from statsmodels.tsa.arima_process import ArmaProcess
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample


def MBB(x, block_length=None, NaN=False):
    """
    Moving block bootstrap for a panel of time-series 
    with potentially missing observations.
    
    Arguments:
    x:               dataset (row= obs, col= series)
    block_length:    length of the block 
    NaN:             boolean. True= blocks with NaNs, False= remove NaNs from blocks
    
    Return:
    blocks:          matrix where each row corresponds to a block of consecutive obs 
    """
    assert np.ndim(x) == 2, "Input data must be a 2D array"
    (row_x, column_x) = x.shape   
    
    if block_length is None:
        block_length = int(np.floor(row_x**(1/3))) 
    
    try:     
        assert block_length > 0
    except AssertionError:
        print("The block length should be superior to 0.")
    else:
        block_length = int(block_length)
    

    N = row_x - block_length + 1
    blocks = np.zeros((N*column_x, block_length))
    c = 0
    for j in range(column_x): 
        for i in range(N):
            blocks[c,:] = x[i:i+block_length,j]
            c += 1
    if not NaN:
        blocks = blocks[~np.isnan(blocks).any(axis=1)] #remove NaNs
    return blocks


def resample_MBB(x, block_length=None, n=None, NaN=False):
    """
    Sample of length n randomly selected (with repetitions) from the original 
    series x using the MBB method 
    
    Arguments:
    x:               dataset (row= obs, col= series)
    block_length:    length of the block 
    n:               length of the desired series
    NaN:             boolean. True= sample with NaNs. False= sample without NaNs
    
    Return:
    sample:          series of obs resampled by MBB
    """
    (row_x, column_x) = x.shape
    if n is None:
        n = row_x #return a series with same length as the original by default
    blocks = MBB(x, block_length)    
    if not NaN:
        blocks = blocks[~np.isnan(blocks).any(axis=1)] #remove NaN
    n_blocks = int(np.ceil(n / block_length))
    sample = resample(blocks, replace=True, n_samples=n_blocks)
    return sample.flatten()


def NBB(x, block_length=None, NaN=False):
    """
    Non-overlapping block bootstrap for a panel of time-series
    with potentially missing obs.
    
    Arguments:
    x:               dataset (row= obs, col= series)
    block_length:    length of the block 
    NaN:             boolean. True= blocks with NaNs, False= remove NaNs from blocks
    
    Return:
    blocks:          matrix where each row corresponds to a block of consecutive obs 
    """
    (row_x, column_x) = x.shape
    
    if block_length is None:
        block_length = int(np.floor(row_x**(1/3)))  

    N = int(np.floor(row_x / block_length))
    blocks = np.zeros((N*column_x, block_length))
    c = 0
    for j in range(column_x):
        it = 0
        for i in range(0,N):
            blocks[c,:] = x[it:it+block_length,j] #non-overlapping
            it += block_length
            c += 1
    if not NaN:
        blocks = blocks[~np.isnan(blocks).any(axis=1)] #remove NaN
    return blocks

def resample_NBB(x, block_length=None, n=None, NaN=False):
    """ 
    Sample of length n randomly selected (with repetitions) from the original 
    series x using the NBB method 
        
    Arguments:
    x:               dataset (row= obs, col= series)
    block_length:    length of the block 
    n:               length of the desired series
    NaN:             boolean. True= sample with NaNs. False= sample without NaNs
    
    Return:
    sample:          series of obs resampled by NBB
    """
    (row_x, column_x) = x.shape
    if n is None:
        n = row_x #return a series with same length as the original by default
    blocks = NBB(x, block_length)    
    if not NaN:
        blocks = blocks[~np.isnan(blocks).any(axis=1)] #remove NaN
    n_blocks = int(np.ceil(n / block_length))
    sample = resample(blocks, replace=True, n_samples=n_blocks)
    return sample.flatten()

def CBB(x, block_length=None, NaN=False):
    """
    Circular block bootstrap for a panel of time-series
    with potentially missing obs.
    
    Arguments:
    x:               dataset (row= obs, col= series)
    block_length:    length of the block 
    NaN:             boolean. True= blocks with NaNs, False= remove NaNs from blocks
    
    Return:
    blocks:          matrix where each row corresponds to a block of consecutive obs 
    """
    (row_x, column_x) = x.shape
    
    if block_length is None:
        block_length = int(np.floor(row_x**(1/3)))  

    N = row_x
    blocks = np.zeros((N*column_x, block_length))
    c = 0
    for j in range(column_x): 
        x_dup = np.concatenate((x[:,j],x[:,j]))
        for i in range(0,N):
            blocks[c,:] = x_dup[i:i+block_length] #overlapping
            c += 1
    if not NaN:
        blocks = blocks[~np.isnan(blocks).any(axis=1)] #remove NaN
    return blocks
   

def resample_CBB(x, block_length=None, n=None, NaN=False):
    """
    Sample of length n randomly selected (with repetitions) from the original 
    series x using the CBB method 
            
    Arguments:
    x:               dataset (row= obs, col= series)
    block_length:    length of the block 
    n:               length of the desired series
    NaN:             boolean. True= sample with NaNs. False= sample without NaNs
    
    Return:
    sample:          series of obs resampled by CBB
    """
    (row_x, column_x) = x.shape
    if n is None:
        n = row_x #return a series with same length as the original by default
    blocks = CBB(x, block_length)    
    if not NaN:
        blocks = blocks[~np.isnan(blocks).any(axis=1)] #remove NaN
    n_blocks = int(np.ceil(n / block_length))
    sample = resample(blocks, replace=True, n_samples=n_blocks)
    return sample.flatten()


def resample_MatchedBB(x, block_length=None, k_block=None, n=None):
    """
    Matched block bootstrap for a panel of time-series
    with potentially missing obs.
    
    Arguments:
    x:               dataset (row= obs, col= series)
    block_length:    length of the block 
    k_block:         parameter (Ri-k, Ri+k)
    n:               length of the desired series
    
    Return:
    boot_matched:    series of obs resampled by matched BB, of length n
    """
    (row_x, column_x) = x.shape
    
    if block_length is None:
        block_length = int(np.floor(row_x**(1/3))) 
        
    if k_block is None:
        k_block = 0.84*row_x**(1/5)
        
    if n is None:
        n = row_x #return a series with same length as the original by default
        
    #create non-overlapping blocks
    blocks = NBB(x, block_length) 

    last_values = np.zeros((len(blocks)))
    for i in range(len(blocks)):
        last = blocks[i,-7:]
        if not np.isnan(np.min(last)):
            first_not_nan = np.where(~np.isnan(last))[0][-1] #first non nan value in the last seven values
            last_values[i] = last[first_not_nan]
        else:
            last_values[i] = blocks[i, block_length-1] #nan

    blocks = blocks[~np.isnan(last_values)]#remove blocks whose last values is nan
    last_values = last_values[~np.isnan(last_values)]
    ranks = ss.rankdata(last_values, method='ordinal') - 1 #start at 0 
    sort_ind = np.argsort(ranks)
    n_blocks = int(np.ceil(n / block_length))
    b = len(blocks)
    if k_block > b: #really small number of blocks
        k_block = 1
    
    seed = np.random.randint(b)
    boot_matched = blocks[seed,:] #first block is randomly chosen from all blocks
    rk_boot = ranks[seed]
    for i in range(n_blocks):
        u = int(np.random.uniform(rk_boot-k_block, rk_boot+k_block))
        if u < 0: #beginning
            u = -u
        elif u > b - 1: #end series
            u = 2*b - 1 - u
        loc = sort_ind[u] #location of the block (in original series)
        if loc == b - 1:
            loc = 2*b - 3 - loc
        next_block = blocks[loc+1,:]
        rk_boot = ranks[loc+1]
        boot_matched = np.concatenate((boot_matched, next_block))
        
    return boot_matched

##########################################################################"
    
""" Tests """

if __name__ == "__main__":
    test = np.arange(198)
    test_matrix = np.transpose(np.tile(test,(4,1)))
    test_MBB = MBB(test_matrix)
    test_NBB = NBB(test_matrix)
    test_CBB = CBB(test_matrix)

"""Tests matched block bootstrap """

#### simulate an AR(0.9)
if __name__ == "__main__":
    
    n = 5080
    phi1 = -0.9
    ar1 = np.array([1, phi1])
    ma1 = np.array([1])
    AR_object1 = ArmaProcess(ar1, ma1)
    data = AR_object1.generate_sample(nsample=(n,2))
    (n,N_series) = data.shape
    #plt.figure(1)
    #plt.plot(data[4700:,0])
    #plt.scatter(data[0:n-1,0], data[1:,0]) #correspond to ar(-0.9)
    
    ### apply an MA(81) to create long-memory series
    wdw = 81
    for i in range(N_series):
        ts = pd.Series(data[:,i], index=range(n))
        data[:,i] = ts.rolling(wdw, center=True).mean()
    data = data[40:n-40,:]    
    (n, N_series) = data.shape
    plt.figure(1)
    plt.plot(data[4700:,0])
    
    block_length = 30
    boot_matched = resample_MatchedBB(data ,block_length, 5, n)
    boot_sampled = resample_NBB(data, block_length, n) #simple BB
    
    """Plot the result"""
    stop = min(len(boot_sampled), len(boot_matched))
    start = stop - 10*block_length
    x = np.arange(start, stop)
    
    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.title("Matched block bootstrap")
    plt.plot(x, boot_matched[start:stop])
    for i in range(start, stop, block_length):
        plt.axvline(x = i)
    axes = plt.gca()
    #axes.set_ylim([-100,100])
    
    plt.subplot(2, 1, 2)
    plt.title("Simple Block bootstrap")
    plt.plot(x, boot_sampled[start:stop])
    for i in range(start, stop, block_length):
        plt.axvline(x=i) #vertical lines
    axes = plt.gca()
    #axes.set_ylim([-10,20])
    plt.tight_layout()
    plt.show()