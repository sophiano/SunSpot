# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:41:10 2021

@author: sopmathieu

This file contains several functions to calibrate the cut-off values for 
the predictions of the neural networks.
Those values are similar to the limits of the traditional control charts. 

The file also contains functions to compute the in-control and out-of-control 
average run lengths of the networks. 

"""

import numpy as np 
from sklearn.utils import resample

from SunSpot import bb_methods as bb


#===========================================================================
### Feed-forward neural networks
#==========================================================================

def cutoff(data, reg, ARL0_threshold=200, rho=2, L_plus=2, L_minus=0, 
                 nmc=4000, n=4000, verbose=True, block_length=None,
                 wdw_length=None, BB_method='MBB', chart='shewhart'):
    """ 
   Adjusts the cut-off values of a neural network using a bisection 
   searching algorithm.

    
   The cut-off values are similar to the limits of the traditional 
   control charts. Two different methods are implemented in this function. 
   (1) When chart = 'shewhart', the cut-off values are computed directly 
   on the predicted shift sizes. Hence the sizes that are outside of the cut-off 
   values trigger an alert. 
   (2) When chart = 'cusum', an apdative cusum chart is used to trigger 
   an alert. This chart is similar to the cusum but adapts its allowance 
   parameter (k) at each time, as a function of the predicted sizes. 
   The cut-off values correspond thus here to the control limits of the adaptive 
   cusum and are adjusted by the algorithm.
 
   In both cases, the algorithm works as follows.
   For each monte-carlo run, a new series of observations is sampled from the 
   IC data using a block boostrap procedure. 
   The series is then split in moving windows and fed to the neural network
   which predicts the size of the deviations. 
   The IC average run length (ARL0) is calculated over the runs with the 
   selected approach ('shewhart' or 'cusum'). 
   With the former, the run lengths are recorded until the predicted 
   shift sizes exceed the cut-off values. 
   With the latter, the run lengths are computed until the adaptive cusum 
   triggers an alert (i.e. when the cumulative sum of its deviations are 
   outside of the cut-off values). 
   If the actual ARL0 is inferior (resp. superior) to the
   pre-specified ARL0, the cut-off is increased (resp. decreased).
   This algorithm is iterated until the actual ARL0 reaches the pre-specified ARL0
   at the desired accuracy.
    
    Parameters
    ---------
    data : 2D-array
        IC dataset (rows: time, columns: IC series).
    reg : a neural network model (for regression)
        The trained neural network.
    ARL0_threshold : float > 0, optional
        Pre-specified value for the IC average run length (ARL0). 
        This value is inversely proportional to the rate of false positives.
        Typical values are 100, 200 or 500. Default is 200.
    rho : float > 0, optional
        Accuracy to reach the pre-specified value for ARL0: 
        the algorithm stops when |ARL0-ARL0_threshold| < rho.
        The default is 2.
    L_plus : float, optional
        Upper value for the positive cut-off value. Default is 2.
    L_minus : float, optional
        Lower value for the positive cut-off. Default is 0. 
    nmc : int > 0, optional
        Number of Monte-Carlo runs. This parameter has typically a large value.
        Default is 4000. 
    n : int > 0, optional
        Length of the resampled series (by the block bootstrap procedure).
        Default is 4000. 
    Verbose : bool, optional
        Flag to print intermediate results. Default is True.
    block_length :  int > 0, optional
        The length of the blocks. It is equal to the length of the input 
        vector of the network here. Default is None. 
        When None, the length is computed using an optimal formula. 
    BB_method : str, optional
       String that designates the block boostrap method chosen for sampling data. 
       Values for the string should be selected among: 
       'MBB': moving block bootstrap
       'NBB': non-overlapping block bootstrap
       'CBB': circular block bootstrap
       'MABB': matched block bootstrap
       Default is 'MBB'.
    chart : str, optional
        String that designated the type of control chart that should 
        be used. Value for 'chart' should be selected among:
        'shewhart' : Shewhart chart on the predicted shift sizes ;
        'cusum' : adaptive CUSUM chart.
    
    Returns
    ------
    L : float
       The positive cut-off value of the network (with this algorithm,
       it has the same value as the negative cut-off, with opposite sign). 
       
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    (n_obs, n_series) = data.shape  
  
    ##Block bootstrap
    assert BB_method in ['MBB', 'NBB', 'CBB', 'MABB'], "Undefined block bootstrap procedure"
    if BB_method == 'MBB': 
        blocks = bb.MBB(data, block_length)  
    elif BB_method == 'NBB':
        blocks = bb.NBB(data, block_length) 
    elif BB_method == 'CBB':
        blocks = bb.CBB(data, block_length) 
        
    if 'blocks' in locals():
        block_length = blocks.shape[1]
        n_blocks = int(np.ceil((n + block_length)/block_length))                       
    if wdw_length is None:
        wdw_length = block_length
    wdw_length = int(wdw_length)
                               
    #other parameters
    assert L_plus > L_minus, "L_plus should be superior than L_minus"
    L = (L_plus + L_minus)/2
    n = int(n)
    assert n > 0, "n must be strictly positive"
    nmc = int(nmc)
    assert nmc > 0, "nmc must be strictly positive"
    assert rho > 0, "rho must be strictly positive"
    assert ARL0_threshold > 0, "ARL0_threshold must be strictly positive"

    
    #####################################
    if chart == 'shewhart':
        
        ARL = 0
        while (np.abs(ARL - ARL0_threshold) > rho):
            RL_minus = np.zeros((nmc,1)); RL_plus = np.zeros((nmc,1)) 
            RL_minus[:] = np.nan; RL_plus[:] = np.nan
            for j in range(nmc):
                
                ## simulate new series with block bootstrap
                if BB_method == 'MABB': 
                    boot = bb.resample_MatchedBB(data, block_length, n=(int(n*1.1)-1))
                else:
                    boot = resample(blocks, replace=True, n_samples=n_blocks).flatten()
                    
                #prepare the series into moving windows for the network
                boot = bb.MBB(boot.reshape(-1,1), wdw_length)[:n,:]
                        
                ## feed the input vectors to the network
                size_pred = reg.predict(boot)
    
                ## Monitoring 
                for i in range(1, n):
                    if size_pred[i] > L:
                        RL_plus[j] = i 
                        break 
                    
                for i in range(1, n):
                    if size_pred[i] < -L:
                        RL_minus[j] = i 
                        break 
            
                if np.isnan(RL_plus[j]):
                    RL_plus[j] = n 
                if np.isnan(RL_minus[j]):
                    RL_minus[j] = n 
                      
            ## adjust control limits
            ARL = (1/(np.mean(RL_minus)) + 1/(np.mean(RL_plus)))**(-1) 
            if ARL < ARL0_threshold:
                L_minus = (L_minus + L_plus)/2
            elif ARL > ARL0_threshold:
                L_plus = (L_minus + L_plus)/2
            L = (L_plus + L_minus)/2
                
            if verbose:
                print(ARL)
                print(L)
                
    #########################################
    if chart == 'cusum':
        
        ARL = 0
        while (np.abs(ARL - ARL0_threshold) > rho):
            RL_minus = np.zeros((nmc,1)); RL_plus = np.zeros((nmc,1)) 
            RL_minus[:] = np.nan; RL_plus[:] = np.nan
            for j in range(nmc):
                
                ## simulate new series with block bootstrap
                if BB_method == 'MABB': 
                    boot = bb.resample_MatchedBB(data, block_length, n=(int(n*1.1)-1))
                else:
                    boot = resample(blocks, replace=True, n_samples=n_blocks).flatten()
                     
                #prepare the series into moving windows for the network
                boot = bb.MBB(boot.reshape(-1,1), wdw_length)[:n,:]
        
                ## feed the input vectors to the network
                size_pred = reg.predict(boot)
                #k = np.maximum(abs(size_pred/2), 0.25)
                k = abs(size_pred/2)
                
                
                ## Monitoring 
                C_plus = np.zeros((n,1)) 
                for i in range(1, n):
                    C_plus[i] = max(0, C_plus[i-1] + boot[i,-1] - k[i])
                    if C_plus[i] > L/k[i]:    
                        RL_plus[j] = i 
                        break 
            
                C_minus = np.zeros((n,1))
                for i in range(1, n):
                    C_minus[i] = min(0, C_minus[i-1] + boot[i,-1] + k[i])
                    if C_minus[i] < -L/k[i]:
                        RL_minus[j] = i 
                        break 
            
                if np.isnan(RL_plus[j]):
                    RL_plus[j] = n 
                if np.isnan(RL_minus[j]):
                    RL_minus[j] = n 
                      
            ## adjust control limits
            ARL = (1/(np.mean(RL_minus)) + 1/(np.mean(RL_plus)))**(-1) 
            if ARL < ARL0_threshold:
                L_minus = (L_minus + L_plus)/2
            elif ARL > ARL0_threshold:
                L_plus = (L_minus + L_plus)/2
            L = (L_plus + L_minus)/2
                
            if verbose:
                print(ARL)
                print(L)
            
    return L

#===========================================================================
#==========================================================================


def cutoff_heuristic(data, reg, L_plus, L_minus=0, ARL0_threshold=200, 
                 n_L=5,  rho=2, l_plus=None, l_minus=0, 
                 nmc=4000, n=4000, verbose=True, block_length=None, 
                 BB_method='MBB', alternance=False):
    """ 
   Adjusts the cut-off values of a neural network for an heuristic decision 
   procedure with a bisection searching algorithm.
    
   The cut-off values are similar to the limits of the traditional 
   control charts. With the heuristic decision procedure, a series
   is out-of-control when: (a) the predicted shift sizes are outside of 
   [-L, L] or (b) when n_L observations are consecutively outside of [-l, l]. 
   Hence, the series is in alert when a large shift (>L) is detected or 
   when n_L small shifts (>l) are consecutively recorded.
   
   The values of L and l are adjusted by a searching algorithm as follows.
   From initial values for L and l, the actual IC average run 
   length (ARL0) is computed on 'nmc' processes that are sampled with repetition 
   from the IC data by the block bootstrap procedure.
   If the actual ARL0 is inferior (resp. superior) to the pre-specified ARL0, 
   the cut-off values are alternatively increased (resp. decreased).
   This algorithm is iterated until the actual ARL0 reaches the pre-specified ARL0
   at the desired accuracy.
   If 'alternance' is set to false, then the procedure will only calibrate 'l', 
   the lower cut-off. The value of L = (L_plus+L_minus)/2 will not change.
    
    Parameters
    ---------
    data : 2D-array
        IC dataset (rows: time, columns: IC series).
    reg : a neural network model (for regression)
        The trained neural network.
    L_plus : float
        Upper value for the positive (upper) cut-off. 
    L_minus : float, optional
        Lower value for the positive (upper) cut-off. Default is 0. 
    ARL0_threshold : int > 0, optional
        Pre-specified value for the IC average run length (ARL0). 
        This value is inversely proportional to the rate of false positives.
        Typical values are 100, 200 or 500. Default is 200.
    n_L : int >= 0, optional 
        Number of consecutive values that should be outside [l_minus, l_plus] 
        to trigger an alert. The default is 5. 
    rho : float > 0, optional
        Accuracy to reach the pre-specified value for ARL0: 
        the algorithm stops when |ARL0-ARL0_threshold| < rho.
        The default is 2.
    l_plus : float, optional
        Upper value for the positive (lower) cut-off. Default is 2/3 of 
        L_plus. 
    L_minus : float, optional
        Lower value for the positive (lower) cut-off. Default is 0. 
    nmc : int > 0, optional
        Number of Monte-Carlo runs. This parameter has typically a large value.
        Default is 4000. 
    n : int > 0, optional
        Length of the resampled series (by the block bootstrap procedure).
        Default is 4000. 
    Verbose : bool, optional
        Flag to print intermediate results. Default is True.
    block_length :  int > 0, optional
        The length of the blocks. It is equal to the length of the input 
        vector of the network here. Default is None. 
        When None, the length is computed using an optimal formula. 
    BB_method : str, optional
       String that designates the block boostrap method chosen for sampling data. 
       Values for the string should be selected among: 
       'MBB': moving block bootstrap
       'NBB': non-overlapping block bootstrap
       'CBB': circular block bootstrap
       'MABB': matched block bootstrap
       Default is 'MBB'.
    alternance : bool, optional
        Boolean that allows the calibrations of both the upper and lower 
        cut-off values, alternatively. The default is False. 
    
    Returns
    ------
    L : float
       The positive upper cut-off values of the network (with this algorithm,
       it has the same value as the negative upper cut-off, 
       with opposite sign).
    l : float
       The positive lower cut-off of the network (with this algorithm,
       it has the same value as the negative lower cut-off, 
       with opposite sign).
    
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    (n_obs, n_series) = data.shape  
  
    ##Block bootstrap
    assert BB_method in ['MBB', 'NBB', 'CBB', 'MABB'], "Undefined block bootstrap procedure"
    if BB_method == 'MBB': 
        blocks = bb.MBB(data, block_length)    
    elif BB_method == 'NBB':
        blocks = bb.NBB(data, block_length) 
    elif BB_method == 'CBB':
        blocks = bb.CBB(data, block_length) 
        
    if 'blocks' in locals():
        block_length = blocks.shape[1]
        n_blocks = int(np.ceil((n + block_length)/block_length))
        
    #limits
    assert L_plus > L_minus, "L_plus should be superior to L_minus"
    if l_plus is None:
        l_plus = 2*L_plus/3
    assert l_plus > l_minus, "l_plus should be superior to l_minus"
    assert l_plus <= L_plus, "l_plus should be inferior or equal to L_plus"
    L = (L_plus + L_minus)/2
    l = (l_plus + l_minus)/2
    
    #other parameters
    n = int(n)
    assert n > 0, "n must be strictly positive"
    n_L = int(n_L)
    assert n_L >= 0, "n_L must be superior or equal to zero"
    nmc = int(nmc)
    assert nmc > 0, "nmc must be strictly positive"
    assert rho > 0, "rho must be strictly positive"
    assert ARL0_threshold > 0, "ARL0_threshold must be strictly positive"

        
    ARL = 0 ; count = 1
    while (np.abs(ARL - ARL0_threshold) > rho):
        RL_minus = np.zeros((nmc,1)); RL_plus = np.zeros((nmc,1)) 
        RL_minus[:] = np.nan; RL_plus[:] = np.nan
        for j in range(nmc):
            
            if BB_method == 'MABB': 
                boot = bb.resample_MatchedBB(data, block_length, n=(int(n*1.1)-1))
            else:
                boot = resample(blocks, replace=True, n_samples=n_blocks).flatten()

            #prepare the series into moving windows for the network
            boot = bb.MBB(boot.reshape(-1,1), block_length)[:n,:]
            
            ## feed the input vectors to the network
            size_pred = reg.predict(boot)
            

            ## Monitoring 
            for i in range(1, n):
                if size_pred[i] > L:
                    RL_plus[j] = i 
                    break 
                if i > n_L and np.all(size_pred[i-n_L+1:i+1] > l):
                    RL_plus[j] = i 
                    break 
                
            for i in range(1, n):
                if  size_pred[i] < -L:
                    RL_minus[j] = i 
                    break 
                if i > n_L and np.all(size_pred[i-n_L+1:i+1] < -l):
                    RL_minus[j] = i 
                    break 
        
            if np.isnan(RL_plus[j]):
                RL_plus[j] = n 
            if np.isnan(RL_minus[j]):
                RL_minus[j] = n 
                  
        if alternance :
            count =  -count
        
        ARL = (1/(np.mean(RL_minus)) + 1/(np.mean(RL_plus)))**(-1) 
        
        if ARL < ARL0_threshold:
            if alternance: 
                if count < 0 and (l_plus + (l_minus + l_plus)/2)/2 < L:
                    l_minus = (l_minus + l_plus)/2
                else:
                    L_minus = (L_minus + L_plus)/2
            else: 
                  l_minus = (l_minus + l_plus)/2
                
        elif ARL > ARL0_threshold:
            if alternance: 
                if count < 0 and (l_minus + (l_minus + l_plus)/2)/2 < L:
                    l_plus = (l_minus + l_plus)/2
                else:
                    L_plus = (L_minus + L_plus)/2
            else:
                l_plus = (l_minus + l_plus)/2
                
        L = (L_plus + L_minus)/2
        l = (l_plus + l_minus)/2
            
        if verbose:
            print(ARL)
            print("L: ", L)
            print("l: ", l)
                 
    return L, l


#===========================================================================
#==========================================================================


def ARL1_NN(data, reg, L_plus, L_minus=None, form='jump', delta=0.5, 
                  nmc=4000, n=4000, block_length=None, wdw_length=None,
                  BB_method='MBB', chart='shewhart'):
    """ 
    Computes the out-of-control (OC) average run length (ARL1)
    of a neural network.
    
    The algorithm works as follows.
    For each monte-carlo run, a new series of observations is sampled from the 
    IC data using a block boostrap procedure. Then, a shift of specified form 
    and size is simulated on top of the sample. 
    The series is then slit in moving windows and fed to the neural network
    which predicts the size of the deviations. 
    The run length is then evaluated for the selected approach ('shewhart' or 
    'cusum'). With the former, the run lengths are recorded until the predicted 
    shift sizes exceed the cut-off values. 
    With the latter, the run lengths are computed until the adaptive cusum 
    triggers an alert (i.e. when the cumulative sum of its deviations are 
    outside of the cut-off values). 
    Finally, the average run length is calculated over the runs.
    
    Parameters
    ----------
    data : 2D-array
        IC dataset (rows: time, columns: IC series).
    reg : a neural network model (for regression)
        The trained neural network.
    L_plus : float 
        Value for the positive cut-off.
    L_minus : float, optional
        Value for the negative cut-off. Default is None. 
        When None, L_minus = - L_plus. 
    form :  str, optional
         String that represents the forms of the shifts that are simulated. 
         The value of the string should be chosen among: 'jump', 'oscillation' 
         or 'drift'.
         Default is 'jump'.
    delta : float, optional
        The shift size. Default is 0.5.
    nmc : int > 0, optional
        Number of Monte-Carlo runs. This parameter has typically a large value.
        Default is 4000. 
    n : int > 0, optional
        Length of the resampled series (by the block bootstrap procedure).
        Default is 4000. 
    block_length :  int > 0, optional
        The length of the blocks. It is equal to the length of the input 
        vector of the network here. Default is None. 
        When None, the length is computed using an optimal formula. 
    BB_method : str, optional
       String that designates the block boostrap method chosen for sampling data. 
       Values for the string should be selected among: 
       'MBB': moving block bootstrap
       'NBB': non-overlapping block bootstrap
       'CBB': circular block bootstrap
       'MABB': matched block bootstrap
       Default is 'MBB'.
    
    Returns
    ------
    ARL1 : float            
         The OC average run length (ARL1) of the network.
         
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    (n_obs, n_series) = data.shape 
        
    ##Block bootstrap
    assert BB_method in ['MBB', 'NBB', 'CBB', 'MABB'], "Undefined block bootstrap procedure"
    if BB_method == 'MBB': 
        blocks = bb.MBB(data, block_length)    
    elif BB_method == 'NBB':
        blocks = bb.NBB(data, block_length) 
    elif BB_method == 'CBB':
        blocks = bb.CBB(data, block_length) 
    
    if 'blocks' in locals():
        block_length = blocks.shape[1]
        n_blocks = int(np.ceil((n + block_length)/block_length))
    if wdw_length is None:
        wdw_length = block_length
    wdw_length = int(wdw_length)
        
    #parameters
    assert form in ['jump','drift','oscillation'], "Undefined shift form"
    shift = delta
    if L_minus is None:
        L_minus = -L_plus
    n = int(n)
    assert n > 0, "n must be strictly positive"
    nmc = int(nmc)
    assert nmc > 0, "nmc must be strictly positive"


    if chart == 'shewhart':
        
        RL1_plus = np.zeros((nmc,1)); RL1_minus = np.zeros((nmc,1))
        RL1_plus[:] = np.nan; RL1_minus[:] = np.nan
        for b in range(nmc):
            
            if BB_method == 'MABB': 
                boot = bb.resample_MatchedBB(data, block_length, n=(int(n*1.1)-1))
            else:
                boot = resample(blocks, replace=True, n_samples=n_blocks).flatten()
            
            ## create artificial deviation on top of the IC series
            if form == 'oscillation':
                eta = np.random.uniform(0.02, 0.2)
                boot[wdw_length-2:] = np.sin(eta*np.pi*np.arange(len(boot)-wdw_length+2))*shift + boot[wdw_length-2:]
                #boot = np.sin(eta*np.pi*np.arange(len(boot)))*shift + boot
                pass
            elif form == 'drift':
                power = np.random.uniform(1.5, 2)
                boot[wdw_length-2:] = shift/(500)*(np.arange(len(boot)-wdw_length+2)**power) + boot[wdw_length-2:]
                #boot = shift/(500)*(np.arange(len(boot))**power) + boot
                pass
            else: 
                boot[wdw_length-1:] = boot[wdw_length-1:] + shift
                #boot = boot + shift
                pass
            
            #prepare the series into moving windows for the network
            boot = bb.MBB(boot.reshape(-1,1), wdw_length)[:n,:]
            
            ## feed the input vectors to the network
            size_pred = reg.predict(boot)
            
            ## Monitoring 
            for i in range(1, n):
                if size_pred[i] > L_plus: 
                    RL1_plus[b] = i 
                    break 
                
            for j in range(1, n):
                if size_pred[j] < L_minus:
                    RL1_minus[b] = j 
                    break
                
            if np.isnan(RL1_plus[b]): 
                RL1_plus[b] = n
            if np.isnan(RL1_minus[b]):
                RL1_minus[b] = n
                
               
        ARL1 = (1/(np.nanmean(RL1_minus)) + 1/(np.nanmean(RL1_plus)))**(-1) 

    ##############################
    if chart == 'cusum':
        
        RL1_plus = np.zeros((nmc,1)); RL1_minus = np.zeros((nmc,1))
        RL1_plus[:] = np.nan; RL1_minus[:] = np.nan
        for b in range(nmc):
            
            if BB_method == 'MABB': 
                boot = bb.resample_MatchedBB(data, block_length, n=(int(n*1.1)-1))
            else:
                boot = resample(blocks, replace=True, n_samples=n_blocks).flatten()
                
                
            ## create artificial deviation on top of the IC series
            if form == 'oscillation':
                eta = np.random.uniform(0.02, 0.2)
                boot[wdw_length-2:] = np.sin(eta*np.pi*np.arange(len(boot)-wdw_length+2))*shift + boot[wdw_length-2:]
                #boot = np.sin(eta*np.pi*np.arange(len(boot)))*shift + boot
                pass
            elif form == 'drift':
                power = np.random.uniform(1.5, 2)
                boot[wdw_length-2:] = shift/(500)*(np.arange(len(boot)-wdw_length+2)**power) + boot[wdw_length-2:]
                #boot = shift/(500)*(np.arange(len(boot))**power) + boot
                pass
            else: 
                boot[wdw_length-1:] = boot[wdw_length-1:] + shift
                #boot = boot + shift
                pass
            
            #prepare the series into moving windows for the network
            boot = bb.MBB(boot.reshape(-1,1), wdw_length)[:n,:]
            
            ## feed the input vectors to the network
            size_pred = reg.predict(boot)
            #size_pred = gaussian_filter1d(size_pred, 3)
            #k = np.maximum(abs(size_pred/2), 0.25)
            k = abs(size_pred/2)
            
            ## Monitoring 
            C_plus = np.zeros((n,1)) 
            for i in range(1, n):
                C_plus[i] = max(0, C_plus[i-1] + boot[i,-1] - k[i])
                if C_plus[i] > L_plus/k[i]:
                    RL1_plus[b] = i 
                    break 
            
            C_minus = np.zeros((n,1))
            for j in range(1, n):
                C_minus[j] = min(0, C_minus[j-1] + boot[j,-1] + k[j])
                if C_minus[j] < L_minus/k[j]:
                    RL1_minus[b] = j
                    break 
                
            if np.isnan(RL1_plus[b]): 
                RL1_plus[b] = n
            if np.isnan(RL1_minus[b]):
                RL1_minus[b] = n
                
               
        ARL1 = (1/(np.nanmean(RL1_minus)) + 1/(np.nanmean(RL1_plus)))**(-1) 

    return ARL1


def ARL1_NN_heuristic(data, reg, L_plus, l_plus, L_minus=None, l_minus=None,
                      n_L=5, form='jump',  delta=0.5,  nmc=4000, 
                      n=4000, block_length=None, 
                      BB_method='MBB'):
    """ 
    Computes the out-of-control (OC) average run length (ARL1)
    of a neural network for the heuristic decision procedure.
    
    The algorithm works as follows.
    For each monte-carlo run, a new series of observations is sampled from the 
    IC data using a block boostrap procedure. Then, a shift of specified form 
    and size is simulated on top of the sample.  
    The series is then slit in moving windows and fed to the neural network
    which predicts the size of the deviations. 
    The run length is then evaluated as the number of observations that are
    below the cut-off values (i.e. which does not trigger an alert). 
    Finally, the average run length is calculated over the runs.
    
    Parameters
    ----------
    data : 2D-array
        IC dataset (rows: time, columns: IC series).
    reg : a neural network model (for regression)
        The trained neural network.
    L_plus : float 
        Value for the positive (upper) cut-off value.
    l_plus : float
        Value for the positive (lower) cut-off value.         
    L_minus : float, optional
        Value for the negative (upper) cut-off. Default is None. 
        When None, L_minus = - L_plus. 
    l_minus : float, optional
        Value for the negative (lower) cut-off. Default is None.
        When None, l_minus = - l_plus. 
    n_L : int >= 0, optional 
        Number of consecutive values that should be outside [l_minus, l_plus] 
        to trigger an alert. The default is 5.
    form :  str, optional
         String that represents the forms of the shifts that are simulated. 
         The value of the string should be chosen among: 'jump', 'oscillation' 
         or 'drift'.
         Default is 'jump'.
    delta : float, optional
        The shift size. Default is 0.5.
    nmc : int > 0, optional
        Number of Monte-Carlo runs. This parameter has typically a large value.
        Default is 4000. 
    n : int > 0, optional
        Length of the resampled series (by the block bootstrap procedure).
        Default is 4000. 
    block_length :  int > 0, optional
        The length of the blocks. It is equal to the length of the input 
        vector of the network here. Default is None. 
        When None, the length is computed using an optimal formula. 
    BB_method : str, optional
       String that designates the block boostrap method chosen for sampling data. 
       Values for the string should be selected among: 
       'MBB': moving block bootstrap
       'NBB': non-overlapping block bootstrap
       'CBB': circular block bootstrap
       'MABB': matched block bootstrap
       Default is 'MBB'.
    
    Returns
    ------
    ARL1 : float            
         The OC average run length (ARL1) of the network.

    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    (n_obs, n_series) = data.shape    
    ##Block bootstrap
    assert BB_method in ['MBB', 'NBB', 'CBB', 'MABB'], "Undefined block bootstrap procedure"
    if BB_method == 'MBB': 
        blocks = bb.MBB(data, block_length)    
    elif BB_method == 'NBB':
        blocks = bb.NBB(data, block_length) 
    elif BB_method == 'CBB':
        blocks = bb.CBB(data, block_length) 
    
    if 'blocks' in locals():
        block_length = blocks.shape[1]
        n_blocks = int(np.ceil((n + block_length)/block_length))
          
    #other parameters
    assert form in ['jump','drift','oscillation'], "Undefined shift form"
    shift = delta
    if L_minus is None:
        L_minus = -L_plus
    if l_minus is None:
        l_minus = -l_plus
    n_L = int(n_L)
    assert n_L >= 0, "n_L must be superior or equal to zero"
    n = int(n)
    assert n > 0, "n must be strictly positive"
    nmc = int(nmc)
    assert nmc > 0, "nmc must be strictly positive"

    RL1_plus = np.zeros((nmc,1)); RL1_minus = np.zeros((nmc,1))
    RL1_plus[:] = np.nan; RL1_minus[:] = np.nan
    for b in range(nmc):
        
        if BB_method == 'MABB': 
            boot = bb.resample_MatchedBB(data, block_length, n=(int(n*1.1)-1))
        else:
            boot = resample(blocks, replace=True, n_samples=n_blocks).flatten()
    
        ## create artificial deviation on top of the IC series
        if form == 'oscillation':
            eta = np.random.uniform(0.02, 0.2)
            boot[block_length-2:] = np.sin(eta*np.pi*np.arange(len(boot)-block_length+2))*shift + boot[block_length-2:]
            #boot = np.sin(eta*np.pi*np.arange(len(boot)))*shift + boot
            pass
        elif form == 'drift':
            power = np.random.uniform(1.5, 2)
            boot[block_length-2:] = shift/(500)*(np.arange(len(boot)-block_length+2)**power) + boot[block_length-2:]
            #boot = shift/(500)*(np.arange(len(boot))**power) + boot
            pass
        else: 
            boot[block_length-1:] = boot[block_length-1:] + shift
            #boot = boot + shift
            pass
        
        #prepare the series into moving windows for the network
        boot = bb.MBB(boot.reshape(-1,1), block_length)[:n,:]

        ## feed the input vectors to the network
        size_pred = reg.predict(boot)
        
        ## Monitoring
        for i in range(n): 
            if size_pred[i] > L_plus: 
                RL1_plus[b] = i + 1
                break 
            if i > n_L and np.all(size_pred[i-n_L+1:i+1] > l_plus):
                RL1_plus[b] = i + 1
                break
            
          
        for j in range(n):
            if size_pred[j] < L_minus:
                RL1_minus[b] = j + 1
                break
            if j > n_L and np.all(size_pred[j-n_L+1:j+1] < l_minus):
                RL1_minus[b] = j + 1
                break 
            
            
        if np.isnan(RL1_plus[b]): 
            RL1_plus[b] = n
        if np.isnan(RL1_minus[b]):
            RL1_minus[b] = n
            
           
    ARL1 = (1/(np.nanmean(RL1_minus)) + 1/(np.nanmean(RL1_plus)))**(-1) 
 
    return ARL1


#===========================================================================
### Recurrent neural networks
#==========================================================================

def rnn_cutoff(data, reg, ARL0_threshold=200, rho=2, L_plus=2, L_minus=0, 
                 nmc=4000, n=4000, verbose=True, block_length=None, 
                 wdw_length=None, BB_method='MBB', chart='shewhart'):
    """ 
   Adjusts the cut-off values of a recurrent neural network using a bisection
   searching algorithm. 
    
   The cut-off values are similar to the limits of the traditional 
   control charts. Two different methods are implemented in this function. 
   (1) When chart = 'shewhart', the cut-off values are computed directly 
   on the predicted shift sizes. Hence the sizes that are outside of the cut-off 
   values trigger an alert. 
   (2) When chart = 'cusum', an apdative cusum chart is used to trigger 
   an alert. This chart is similar to the cusum but adapts its allowance 
   parameter (k) at each time, as a function of the predicted sizes. 
   The cut-off values correspond thus here to the control limits of the adaptive 
   cusum and are adjusted by the algorithm.
 
   In both cases, the algorithm works as follows.
   For each monte-carlo run, a new series of observations is sampled from the 
   IC data using a block boostrap procedure. 
   The series is then split in moving windows and fed to the neural network
   which predicts the size of the deviations. 
   The IC average run length (ARL0) is calculated over the runs with the 
   selected approach ('shewhart' or 'cusum'). 
   With the former, the run lengths are recorded until the predicted 
   shift sizes exceed the cut-off values. 
   With the latter, the run lengths are computed until the adaptive cusum 
   triggers an alert (i.e. when the cumulative sum of its deviations are 
   outside of the cut-off values). 
   If the actual ARL0 is inferior (resp. superior) to the
   pre-specified ARL0, the cut-off is increased (resp. decreased).
   This algorithm is iterated until the actual ARL0 reaches the pre-specified ARL0
   at the desired accuracy.
    
    Parameters
    ---------
    data : 2D-array
        IC dataset (rows: time, columns: IC series).
    reg : a recurrent neural network model (for regression)
        The trained recurrent neural network.
    ARL0_threshold : int > 0, optional
        Pre-specified value for the IC average run length (ARL0). 
        This value is inversely proportional to the rate of false positives.
        Typical values are 100, 200 or 500. Default is 200.
    rho : float > 0, optional
        Accuracy to reach the pre-specified value for ARL0: 
        the algorithm stops when |ARL0-ARL0_threshold| < rho.
        The default is 2.
    L_plus : float, optional
        Upper value for the positive cut-off value. Default is 2.
    L_minus : float, optional
        Lower value for the positive cut-off value. Default is 0. 
    nmc : int > 0, optional
        Number of Monte-Carlo runs. This parameter has typically a large value.
        Default is 4000. 
    n : int > 0, optional
        Length of the resampled series (by the block bootstrap procedure).
        Default is 4000. 
    Verbose : bool, optional
        Flag to print intermediate results. Default is True.
    block_length :  int > 0, optional
        The length of the blocks. It is equal to the length of the input 
        vector of the network here. Default is None. 
        When None, the length is computed using an optimal formula. 
    BB_method : str, optional
       String that designates the block boostrap method chosen for sampling data. 
       Values for the string should be selected among: 
       'MBB': moving block bootstrap
       'NBB': non-overlapping block bootstrap
       'CBB': circular block bootstrap
       'MABB': matched block bootstrap
       Default is 'MBB'.
    
    Returns
    ------
    L : float
       The positive cut-off value of the network (with this algorithm,
       it has the same value as the negative cut-off, with opposite sign). 
       
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    (n_obs, n_series) = data.shape  
  
    ##Block bootstrap
    assert BB_method in ['MBB', 'NBB', 'CBB', 'MABB'], "Undefined block bootstrap procedure"
    if BB_method == 'MBB': 
        blocks = bb.MBB(data, block_length)  
    elif BB_method == 'NBB':
        blocks = bb.NBB(data, block_length) 
    elif BB_method == 'CBB':
        blocks = bb.CBB(data, block_length) 
        
    if 'blocks' in locals():
        block_length = blocks.shape[1]
        n_blocks = int(np.ceil((n + block_length)/block_length))
    if wdw_length is None:
        wdw_length = block_length
    wdw_length = int(wdw_length)
                
    #other parameters
    assert L_plus > L_minus, "L_plus should be superior than L_minus"
    L = (L_plus + L_minus)/2
    n = int(n)
    assert n > 0, "n must be strictly positive"
    nmc = int(nmc)
    assert nmc > 0, "nmc must be strictly positive"
    assert rho > 0, "rho must be strictly positive"
    assert ARL0_threshold > 0, "ARL0_threshold must be strictly positive"

    if chart == 'shewhart':
        
        ARL = 0
        while (np.abs(ARL - ARL0_threshold) > rho):
            RL_minus = np.zeros((nmc,1)); RL_plus = np.zeros((nmc,1)) 
            RL_minus[:] = np.nan; RL_plus[:] = np.nan
            for j in range(nmc):
                
                if BB_method == 'MABB': 
                    boot = bb.resample_MatchedBB(data, block_length, n=(int(n*1.1)-1))
                else:
                    boot = resample(blocks, replace=True, n_samples=n_blocks).flatten()
                    
                #prepare the series into moving windows for the network
                boot = bb.MBB(boot.reshape(-1,1), wdw_length)[:n,:]
                    
                ## feed the input vectors to the network
                boot = np.reshape(boot, (boot.shape[0], 1, boot.shape[1]))
                size_pred = reg.predict(boot)
    
                ## Monitoring 
                for i in range(1, n):
                    if size_pred[i] > L:
                        RL_plus[j] = i 
                        break 
                    
                for i in range(1, n):
                    if size_pred[i] < -L:
                        RL_minus[j] = i 
                        break 
            
                if np.isnan(RL_plus[j]):
                    RL_plus[j] = n 
                if np.isnan(RL_minus[j]):
                    RL_minus[j] = n 
                      
    
            ARL = (1/(np.mean(RL_minus)) + 1/(np.mean(RL_plus)))**(-1) 
            if ARL < ARL0_threshold:
                L_minus = (L_minus + L_plus)/2
            elif ARL > ARL0_threshold:
                L_plus = (L_minus + L_plus)/2
            L = (L_plus + L_minus)/2
                
            if verbose:
                print(ARL)
                print(L)
                
    ###########################
    if chart =='cusum':
        
        ARL = 0
        while (np.abs(ARL - ARL0_threshold) > rho):
            RL_minus = np.zeros((nmc,1)); RL_plus = np.zeros((nmc,1)) 
            RL_minus[:] = np.nan; RL_plus[:] = np.nan
            for j in range(nmc):
                
                if BB_method == 'MABB': 
                    boot = bb.resample_MatchedBB(data, block_length, n=(int(n*1.1)-1))
                else:
                    boot = resample(blocks, replace=True, n_samples=n_blocks).flatten()
                    
                #prepare the series into moving windows for the network
                boot = bb.MBB(boot.reshape(-1,1), wdw_length)[:n,:]   
                    
                ## feed the input vectors to the network
                boot_pred = np.reshape(boot, (boot.shape[0], 1, boot.shape[1]))
                size_pred = reg.predict(boot_pred)
                k = abs(size_pred/2)
    
                ## Monitoring 
                C_plus = np.zeros((n,1)) 
                for i in range(1, n):
                    C_plus[i] = max(0, C_plus[i-1] + boot[i,-1] - k[i])
                    if C_plus[i] > L/k[i]:
                        RL_plus[j] = i 
                        break 
            
                C_minus = np.zeros((n,1))
                for i in range(1, n):
                    C_minus[i] = min(0, C_minus[i-1] + boot[i,-1] + k[i])
                    if C_minus[i] < -L/k[i]:
                        RL_minus[j] = i 
                        break 
            
                if np.isnan(RL_plus[j]):
                    RL_plus[j] = n 
                if np.isnan(RL_minus[j]):
                    RL_minus[j] = n 
                      
    
            ARL = (1/(np.mean(RL_minus)) + 1/(np.mean(RL_plus)))**(-1) 
            if ARL < ARL0_threshold:
                L_minus = (L_minus + L_plus)/2
            elif ARL > ARL0_threshold:
                L_plus = (L_minus + L_plus)/2
            L = (L_plus + L_minus)/2
                
            if verbose:
                print(ARL)
                print(L)        
    return L


def ARL1_RNN(data, reg, L_plus, L_minus=None, form='jump', delta=0.5, 
                  nmc=4000, n=4000, block_length=None, wdw_length=None,
                  BB_method='MBB', chart='shewhart'):
    """ 
    Computes the out-of-control (OC) average run length (ARL1)
    of a recurrent neural network.
    
    The algorithm works as follows.
    For each monte-carlo run, a new series of observations is sampled from the 
    IC data using a block boostrap procedure. Then, a shift of specified form 
    and size is simulated on top of the sample. 
    The series is then slit in moving windows and fed to the neural network
    which predicts the size of the deviations. 
    The run length is then evaluated for the selected approach ('shewhart' or 
    'cusum'). With the former, the run lengths are recorded until the predicted 
    shift sizes exceed the cut-off values.  
    With the latter, the run lengths are computed until the adaptive cusum 
    triggers an alert (i.e. when the cumulative sum of its deviations are 
    outside of the cut-off values). 
    Finally, the average run length is calculated over the runs.
    
    Parameters
    ----------
    data : 2D-array
        IC dataset (rows: time, columns: IC series).
    reg : a recurrent neural network model (for regression)
        The trained recurrent neural network.
    L_plus : float 
        Value for the positive cut-off value.
    L_minus : float, optional
        Value for the negative cut-off. Default is None. 
        When None, L_minus = - L_plus. 
    form :  str, optional
         String that represents the forms of the shifts that are simulated. 
         The value of the string should be chosen among: 'jump', 'oscillation' 
         or 'drift'.
         Default is 'jump'.
    delta : float, optional
        The shift size. Default is 0.5.
    nmc : int > 0, optional
        Number of Monte-Carlo runs. This parameter has typically a large value.
        Default is 4000. 
    n : int > 0, optional
        Length of the resampled series (by the block bootstrap procedure).
        Default is 4000. 
    block_length :  int > 0, optional
        The length of the blocks. It is equal to the length of the input 
        vector of the network here. Default is None. 
        When None, the length is computed using an optimal formula. 
    BB_method : str, optional
       String that designates the block boostrap method chosen for sampling data. 
       Values for the string should be selected among: 
       'MBB': moving block bootstrap
       'NBB': non-overlapping block bootstrap
       'CBB': circular block bootstrap
       'MABB': matched block bootstrap
       Default is 'MBB'.
    
    Returns
    ------
    ARL1 : float            
         The OC average run length (ARL1) of the network.
         
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    (n_obs, n_series) = data.shape 
        
    ##Block bootstrap
    assert BB_method in ['MBB', 'NBB', 'CBB', 'MABB'], "Undefined block bootstrap procedure"
    if BB_method == 'MBB': 
        blocks = bb.MBB(data, block_length)    
    elif BB_method == 'NBB':
        blocks = bb.NBB(data, block_length) 
    elif BB_method == 'CBB':
        blocks = bb.CBB(data, block_length) 
    
    if 'blocks' in locals():
        block_length = blocks.shape[1]
        n_blocks = int(np.ceil((n + block_length)/block_length))
    if wdw_length is None:
        wdw_length = block_length
    wdw_length = int(wdw_length)
        
    #other parameters
    assert form in ['jump','drift','oscillation'], "Undefined shift form"
    shift = delta
    if L_minus is None:
        L_minus = -L_plus
    n = int(n)
    assert n > 0, "n must be strictly positive"
    nmc = int(nmc)
    assert nmc > 0, "nmc must be strictly positive"

    if chart =='shewhart':
        RL1_plus = np.zeros((nmc,1)); RL1_minus = np.zeros((nmc,1))
        RL1_plus[:] = np.nan; RL1_minus[:] = np.nan
        for b in range(nmc):
            
            if BB_method == 'MABB': 
                boot = bb.resample_MatchedBB(data, block_length, n=(int(n*1.1)-1))
            else:
                boot = resample(blocks, replace=True, n_samples=n_blocks).flatten()
    
            ## create artificial deviation on top of the IC series
            if form == 'oscillation':
                eta = np.random.uniform(0.02, 0.2)
                boot[wdw_length-2:] = np.sin(eta*np.pi*np.arange(len(boot)-wdw_length+2))*shift + boot[wdw_length-2:]
                #boot = np.sin(eta*np.pi*np.arange(len(boot)))*shift + boot
                pass
            elif form == 'drift':
                power = np.random.uniform(1.5, 2)
                boot[wdw_length-2:] = shift/(500)*(np.arange(len(boot)-wdw_length+2)**power) + boot[wdw_length-2:]
                #boot = shift/(500)*(np.arange(len(boot))**power) + boot
                pass
            else: 
                boot[wdw_length-1:] = boot[wdw_length-1:] + shift
                #boot = boot + shift
                pass
            
            #prepare the series into moving windows for the network
            boot = bb.MBB(boot.reshape(-1,1), wdw_length)[:n,:]
          
            ## feed the input vectors to the network
            boot = np.reshape(boot, (boot.shape[0], 1, boot.shape[1]))
            size_pred = reg.predict(boot)
            
            ## Monitoring 
            for i in range(n):
                if size_pred[i] > L_plus: 
                    RL1_plus[b] = i + 1
                    break 
                
            for j in range(n):
                if size_pred[j] < L_minus:
                    RL1_minus[b] = j + 1
                    break
                
            if np.isnan(RL1_plus[b]): 
                RL1_plus[b] = n
            if np.isnan(RL1_minus[b]):
                RL1_minus[b] = n
                
               
        ARL1 = (1/(np.nanmean(RL1_minus)) + 1/(np.nanmean(RL1_plus)))**(-1) 

    #######################################
    if chart =='cusum': 
        
        RL1_plus = np.zeros((nmc,1)); RL1_minus = np.zeros((nmc,1))
        RL1_plus[:] = np.nan; RL1_minus[:] = np.nan
        for b in range(nmc):
            
            if BB_method == 'MABB': 
                boot = bb.resample_MatchedBB(data, block_length, n=(int(n*1.1)-1))
            else:
                boot = resample(blocks, replace=True, n_samples=n_blocks).flatten()

             ## create artificial deviation on top of the IC series
            if form == 'oscillation':
                eta = np.random.uniform(0.02, 0.2)
                boot[wdw_length-2:] = np.sin(eta*np.pi*np.arange(len(boot)-wdw_length+2))*shift + boot[wdw_length-2:]
                #boot = np.sin(eta*np.pi*np.arange(len(boot)))*shift + boot
                pass
            elif form == 'drift':
                power = np.random.uniform(1.5, 2)
                boot[wdw_length-2:] = shift/(500)*(np.arange(len(boot)-wdw_length+2)**power) + boot[wdw_length-2:]
                #boot = shift/(500)*(np.arange(len(boot))**power) + boot
                pass
            else: 
                boot[wdw_length-1:] = boot[wdw_length-1:] + shift
                #boot = boot + shift
                pass
            
            #prepare the series into moving windows for the network
            boot = bb.MBB(boot.reshape(-1,1), wdw_length)[:n,:]
          
            ## feed the input vectors to the network
            boot_pred = np.reshape(boot, (boot.shape[0], 1, boot.shape[1]))
            size_pred = reg.predict(boot_pred)
            k = abs(size_pred/2)
            
            ## Monitoring 
            C_plus = np.zeros((n,1)) 
            for i in range(1, n):
                C_plus[i] = max(0, C_plus[i-1] + boot[i,-1] - k[i])
                if C_plus[i] > L_plus/k[i]:
                    RL1_plus[b] = i 
                    break 
            
            C_minus = np.zeros((n,1))
            for j in range(1, n):
                C_minus[j] = min(0, C_minus[j-1] + boot[j,-1] + k[j])
                if C_minus[j] < L_minus/k[j]:
                    RL1_minus[b] = j
                    break 
                
            if np.isnan(RL1_plus[b]): 
                RL1_plus[b] = n
            if np.isnan(RL1_minus[b]):
                RL1_minus[b] = n
                
               
        ARL1 = (1/(np.nanmean(RL1_minus)) + 1/(np.nanmean(RL1_plus)))**(-1) 

    return ARL1

######################################################################"
###################################################################


def ARL1_4class(data, clf, form='jump', delta=0.5, nmc=4000, n=4000,
            block_length=None, wdw_length=None, BB_method='MBB'):
    """ 
    Computes the out-of-control (OC) average run length (ARL1)
    of a neural network classifier.
    
    The algorithm works as follows.
    For each monte-carlo run, a new series of observations is sampled from the 
    IC data using a block boostrap procedure. Then, a shift of specified form 
    and size is simulated on top of the sample. 
    The series is then slit in moving windows and fed to the neural network
    which predicts the size of the deviations. 
    The run length is then evaluated for the selected approach ('shewhart' or 
    'cusum'). With the former, the run lengths are recorded until the predicted 
    shift sizes exceed the cut-off values. 
    With the latter, the run lengths are computed until the adaptive cusum 
    triggers an alert (i.e. when the cumulative sum of its deviations are 
    outside of the cut-off values). 
    Finally, the average run length is calculated over the runs.
    
    Parameters
    ----------
    data : 2D-array
        IC dataset (rows: time, columns: IC series).
    clf : a neural network model (for classification)
        The trained neural network.
    form :  str, optional
         String that represents the forms of the shifts that are simulated. 
         The value of the string should be chosen among: 'jump', 'oscillation' 
         or 'drift'.
         Default is 'jump'.
    delta : float, optional
        The shift size. Default is 0.5.
    nmc : int > 0, optional
        Number of Monte-Carlo runs. This parameter has typically a large value.
        Default is 4000. 
    n : int > 0, optional
        Length of the resampled series (by the block bootstrap procedure).
        Default is 4000. 
    block_length :  int > 0, optional
        The length of the blocks. It is equal to the length of the input 
        vector of the network here. Default is None. 
        When None, the length is computed using an optimal formula. 
    BB_method : str, optional
       String that designates the block boostrap method chosen for sampling data. 
       Values for the string should be selected among: 
       'MBB': moving block bootstrap
       'NBB': non-overlapping block bootstrap
       'CBB': circular block bootstrap
       'MABB': matched block bootstrap
       Default is 'MBB'.
    
    Returns
    ------
    ARL1 : float            
         The OC average run length (ARL1) of the network.
         
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    (n_obs, n_series) = data.shape 
        
    ##Block bootstrap
    assert BB_method in ['MBB', 'NBB', 'CBB', 'MABB'], "Undefined block bootstrap procedure"
    if BB_method == 'MBB': 
        blocks = bb.MBB(data, block_length)    
    elif BB_method == 'NBB':
        blocks = bb.NBB(data, block_length) 
    elif BB_method == 'CBB':
        blocks = bb.CBB(data, block_length) 
    
    if 'blocks' in locals():
        block_length = blocks.shape[1]
        n_blocks = int(np.ceil((n + block_length)/block_length))
    if wdw_length is None:
        wdw_length = block_length
    wdw_length = int(wdw_length)
        
    #parameters
    assert form in ['jump','drift','oscillation'], "Undefined shift form"
    shift = delta
    n = int(n)
    assert n > 0, "n must be strictly positive"
    nmc = int(nmc)
    assert nmc > 0, "nmc must be strictly positive"

    
    RL1 = np.zeros((nmc,1)); RL1[:] = np.nan
    for b in range(nmc):
        
        if BB_method == 'MABB': 
            boot = bb.resample_MatchedBB(data, block_length, n=(int(n*1.1)-1))
        else:
            boot = resample(blocks, replace=True, n_samples=n_blocks).flatten()
        
        ## create artificial deviation on top of the IC series
        if form == 'oscillation':
            eta = np.random.uniform(0.02, 0.2)
            boot[wdw_length-2:] = np.sin(eta*np.pi*np.arange(len(boot)-wdw_length+2))*shift + boot[wdw_length-2:]
            pass
        elif form == 'drift':
            power = np.random.uniform(1.5, 2)
            boot[wdw_length-2:] = shift/(500)*(np.arange(len(boot)-wdw_length+2)**power) + boot[wdw_length-2:]
            pass
        else: 
            boot[wdw_length-1:] = boot[wdw_length-1:] + shift
            pass
        
        #prepare the series into moving windows for the network
        boot = bb.MBB(boot.reshape(-1,1), wdw_length)[:n,:]
        
        ## feed the input vectors to the network
        form_pred = np.argmax(clf.predict(boot), axis=-1)
        
        ## Monitoring 
        for i in range(n):
            if form_pred[i] != 0: #first class correspond to 'no shift'
                RL1[b] = i + 1
                break 
                           
        if np.isnan(RL1[b]): 
            RL1[b] = n

    ARL1 = np.nanmean(RL1)
    
    return ARL1
