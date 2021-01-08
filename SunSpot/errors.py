# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:27:18 2020

@author: sopmathieu

This code proposes a set of functions to estimate the solar signal and the 
different errors (at long-term, short-term and solar minima) of the stations. 
For more information, we refer to the journal paper: "Uncertainty quantification
in sunspot counts", from S. Mathieu, R. von Sachs, V Delouillle, L. Lefèvre 
and C. Ritter (2019). 
"""

import pickle
import numpy as np 
from scipy.stats import iqr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
import warnings
warnings.filterwarnings("ignore")


def rescaling(data, period_rescaling):
    """
    Rescales the data wrt the median of the panel.
    
    This function rescales the observations on the median of the panel using 
    piece-wise constant scaling-factors. These factors are computed using a 
    simple linear regression of the observations on the median without 
    intercept.
    
    Parameters
    ----------
    data : 2D-array 
        A matrix of observations to be rescaled: either Ns, Ng or Nc
        (rows: time, columns: stations).
    period_rescaling : float>0
        Length of the period where the scaling-factors are assumed to be 
        constant, expressed in months.
        Using a statistical procedure based on Kruskal-Wallis tests, 
        the minimum value for this parameter is found to be equal to 
        8 months for Ns, 14 months for Ng and 10 months for Nc.
        
    Returns: 
    -------
    obs_rescaled : 2D-array
        A matrix with the rescaled observations
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    assert period_rescaling > 0, "Period must be strictly positive"
    (n_obs, n_stations) = data.shape
    month = int(365/12)
    step = int(period_rescaling*month)
    n = int(np.ceil(n_obs/step))
    
    med = np.nanmedian(data, axis=1)#create warnings: sometimes all values are Nans

    #### linear regression
    slopeOLSY = np.ones((n, n_stations)) 
    for j in range(n_stations):
        c = 0
        for i in range(0, n):
            X = data[c:c+step,j]
            Y = med[c:c+step]
            ind = np.where(~np.isnan(X))
            Y = Y[ind].reshape(-1,1) 
            X = X[ind].reshape(-1,1)
            c += step
            if len(Y > 0) > 2 and len(X > 0) > 2:
                reg = LinearRegression(fit_intercept=False).fit(X, Y)
                slopeOLSY[i,j] = reg.coef_ #slope
        
    
    ### effective rescaling 
    obs_rescaled = np.zeros((n_obs, n_stations)) 
    for j in range(n_stations):
        c = 0
        for i in range(0, n):
            if slopeOLSY[i,j] > 0 and not np.isnan(slopeOLSY[i,j]):
                obs_rescaled[c:c+step,j] = data[c:c+step,j] * slopeOLSY[i,j]
            c += step
    
    return obs_rescaled

###############################
    
def median(data, period_rescaling, with_rescaling=True):
    """ 
    Computes the median of a panel of obs. (rescaled or not) along the time.
    
    Parameters
    ----------
    data : 2D-array 
        A matrix of observations: either Ns, Ng or Nc
        (rows: time, columns: stations).
    period_rescaling : float>0
        Length of the period where the scaling-factors are assumed to be 
        constant, expressed in months.
        Using a statistical procedure based on Kruskal-Wallis tests, 
        the minimum value for this parameter is found to be equal to 
        8 months for Ns, 14 months for Ng and 10 months for Nc.
    with_rescaling :  bool, optional 
        A flag to compute the median on rescaled obs. Default is True.
        
    Returns
    ------
    Mt : 1D-array
        Daily median (reference) of the panel.
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    (nobs, n_stations) = data.shape
    if rescaling: 
        data_rescaled = rescaling(data, period_rescaling)
        Mt = np.round(np.nanmedian(data_rescaled, axis=1))
    else: 
        Mt = np.round(np.nanmedian(data, axis=1))
    return Mt
        
###############################

# def level_removal_old(x, wdw):
#     """
#     Removes the intrinsic levels of an initial array of series.
    
#     This function applies a smoothing process in time by a moving-average
#     (MA) filter on each individual series. Then, the smoothed series are
#     substracted from the initial series to remove the levels of the 
#     processes.
    
#     Parameters
#     ---------
#     x : 2D-array 
#         A panel of series (rows: time, columns: series).
#     wdw : int
#         Length of the MA window length, expressed in days. 
        
#     Returns
#     -------
#     x_wht_levels : 2D-array
#         The array without intrinsic levels.
#     """
#     assert np.ndim(x) == 2, "Input data must be a 2D array"
#     (n_obs, n_stations) = x.shape
#     x_wht_levels = np.copy(x)
    
#     assert wdw > 0, "Window length must be a postive integer"
#     wdw = np.round(wdw)
#     if wdw % 2 == 1:
#         halfwdw = int((wdw - 1)/2)
#     else:
#         wdw += 1
#         halfwdw = int((wdw - 1)/2)
        
#     for i in range(n_stations):
#         m = np.nanmean(x[:,i])
#         ma = np.ones((n_obs))*m
#         for j in range(halfwdw, n_obs - halfwdw):
#             ma[j] = np.nanmean(x[j-halfwdw :j+halfwdw+1,i])
#         ma[np.isnan(ma)] = m
#         x_wht_levels[:,i] = x_wht_levels[:,i] - ma
        
#     return x_wht_levels


def level_removal(x, wdw, center=True):
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
    x_wht_levels = np.copy(x)
    
    assert wdw > 0, "Window length must be a postive integer"
    wdw = np.round(wdw)
        
    if center:
        if wdw % 2 == 1:
            halfwdw = int((wdw - 1)/2)
        else:
            wdw += 1
            halfwdw = int((wdw - 1)/2)
        for i in range(n_stations):
            m = np.nanmean(x[:,i])
            ma = np.ones((n_obs))*m
            for j in range(n_obs):
                if j < halfwdw: #beginning
                    ma[j] = np.nanmean(x[0:wdw,i])
                elif j >= halfwdw and j < n_obs - halfwdw: #middle
                    ma[j] = np.nanmean(x[j-halfwdw :j+halfwdw+1,i])
                else: #end
                    ma[j] = np.nanmean(x[n_obs-wdw :n_obs,i])
            ma[np.isnan(ma)] = m
            x_wht_levels[:,i] = x_wht_levels[:,i] - ma
            
    if not center:
        for i in range(n_stations):
            m = np.nanmean(x[:,i])
            ma = np.ones((n_obs))*m
            for j in range(n_obs):
                if j < wdw: #beginning
                    ma[j] = np.nanmean(x[0:wdw,i]) 
                else: #remaining part
                    ma[j] = np.nanmean(x[j - wdw+1:j+1,i]) 
            ma[np.isnan(ma)] = m
            x_wht_levels[:,i] = x_wht_levels[:,i] - ma
            
        
    return x_wht_levels
    
# def long_term_error_old(data, period_rescaling, wdw=81, min_perc_per_wdw=10,
#                     level=False, wdw_level=4000):
#     """ 
#     This function computes the long-term error (mu2(i,t)), i.e. errors 
#     superior or equal to 27 days, of the stations.
    
#     Parameters
#     ---------
#     data : 2D-array 
#         A matrix of observations: either Ns, Ng or Nc
#         (rows: time, columns: stations).
#     period_rescaling : float>0
#         Length of the period where the scaling-factors are assumed to be 
#         constant, expressed in months.
#         Using a statistical procedure based on Kruskal-Wallis tests, 
#         the minimum value for this parameter is found to be equal to 
#         8 months for Ns, 14 months for Ng and 10 months for Nc.
#     wdw : int>0, optional
#         Length of the MA window, expressed in days. 
#         This parameter should be superior than or equal to 27 days. 
#         Common scales are 27, 81, 365 or 900 days. The default is 81.
#     min_perc_per_wdw : int, optional
#         Minimum percentage of obs required per window to compute a value for
#         that day (otherwise NaN). Default is ten percents. 
#     level: bool, optional 
#         Flag to remove the level of the stations (similar to 'k' factors).
#         When True, wdw_level should be set to integer. The default is False. 
#     wdw_level : int, optional
#         Length of the MA window length to remove the levels, expressed in days. 
#         The default is 4000 (which correspond to roughly eleven years or 
#         one solar cycle).
        
#     Returns
#     ------
#     mu2 : 2D-array
#         The long-term errors of the stations.
#     """
#     assert np.ndim(data) == 2, "Input data must be a 2D array"
#     (n_obs, n_stations) = data.shape
#     data_rescaled = rescaling(data, period_rescaling)
#     Mt = np.round(np.nanmedian(data_rescaled, axis=1))
    
#     ratio = np.zeros((n_obs, n_stations))
#     ratio[:] = np.nan
#     for i in range(n_obs):
#         if not np.isnan(Mt[i]) and Mt[i] > 0:
#             ratio[i,:] = data[i,:]/Mt[i] #Yi(t)/Mt

#     assert wdw > 0, "Window length must be strictly positive"
#     if wdw % 2 == 1:
#         halfwdw = int((wdw - 1)/2)
#     else:
#         wdw += 1
#         halfwdw = int((wdw - 1)/2)
        
#     ### smoothing procedure
#     mu2 = np.ones((n_obs, n_stations)); mu2[:] = np.nan
#     for i in range(n_stations):
#         for j in range(n_obs):
#             if j > -1 and j < halfwdw: #beginning
#                 m = ratio[0:j + halfwdw + 1,i]
#             elif j >= halfwdw and j < n_obs - halfwdw: #middle
#                 m = ratio[j - halfwdw:j + halfwdw + 1,i]
#             else: #end
#                 m = ratio[j - halfwdw:n_obs,i]
#             if len(m[~np.isnan(m)]) > np.round(wdw/min_perc_per_wdw):
#                 mu2[j,i] = np.nanmean(m) #(Yi(t)/Mt)*
                
#     if level: 
#         mu2 = level_removal(mu2, wdw_level)
        
#     return mu2


def long_term_error(data, period_rescaling, wdw=81, min_perc_per_wdw=10,
                    level=False, wdw_level=4000, center=True):
    """ 
    This function computes the long-term error (mu2(i,t)), i.e. errors 
    superior or equal to 27 days, of the stations using a smoothing process.
    
    Parameters
    ---------
    data : 2D-array 
        A matrix of observations: either Ns, Ng or Nc
        (rows: time, columns: stations).
    period_rescaling : float>0
        Length of the period where the scaling-factors are assumed to be 
        constant, expressed in months.
        Using a statistical procedure based on Kruskal-Wallis tests, 
        the minimum value for this parameter is found to be equal to 
        8 months for Ns, 14 months for Ng and 10 months for Nc.
    wdw : int>0, optional
        Length of the MA window, expressed in days. 
        This parameter should be superior than or equal to 27 days. 
        Common scales are 27, 81, 365 or 900 days. The default is 81.
    min_perc_per_wdw : int, optional
        Minimum percentage of obs required per window to compute a value for
        that day (otherwise NaN). Default is ten percents. 
    level: bool, optional 
        Flag to remove the level of the stations (similar to 'k' factors).
        When True, wdw_level should be set to integer. The default is False. 
    wdw_level : int, optional
        Length of the MA window length to remove the levels, expressed in days. 
        The default is 4000 (which correspond to roughly eleven years or 
        one solar cycle).
    center : str, optional
        Flag to indicate that the moving windows should be centered with respect 
        to the data. Otherwise, the windows are left-shifted to include only 
        the current and past observations. The default is True. 
        
    Returns
    ------
    mu2 : 2D-array
        The long-term errors of the stations.
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    (n_obs, n_stations) = data.shape
    data_rescaled = rescaling(data, period_rescaling)
    Mt = np.round(np.nanmedian(data_rescaled, axis=1))
    
    ratio = np.zeros((n_obs, n_stations))
    ratio[:] = np.nan
    for i in range(n_obs):
        if not np.isnan(Mt[i]) and Mt[i] > 0:
            ratio[i,:] = data[i,:]/Mt[i] #Yi(t)/Mt

    assert wdw > 0, "Window length must be strictly positive"
        
    ### smoothing procedure
    mu2 = np.ones((n_obs, n_stations)); mu2[:] = np.nan
    
    if not center:
        for i in range(n_stations):
            for j in range(n_obs):
                if j < wdw: #beginning
                    m = ratio[0:wdw,i]
                else: #remaining part
                    m = ratio[j - wdw+1:j+1,i]
                if len(m[~np.isnan(m)]) > np.round(wdw/min_perc_per_wdw):
                    mu2[j,i] = np.nanmean(m) #(Yi(t)/Mt)*
                    
                
    elif center:           
        if wdw % 2 == 1:
            halfwdw = int((wdw - 1)/2)
        else:
            wdw += 1
            halfwdw = int((wdw - 1)/2)
        
        for i in range(n_stations):
            for j in range(n_obs):
                if j < halfwdw: #beginning
                    m = ratio[0:wdw,i]
                elif j >= halfwdw and j < n_obs - halfwdw: #middle
                    m = ratio[j - halfwdw:j + halfwdw + 1,i]
                else: #end
                    m = ratio[n_obs - wdw:n_obs,i]
                if len(m[~np.isnan(m)]) > np.round(wdw/min_perc_per_wdw):
                    mu2[j,i] = np.nanmean(m) #(Yi(t)/Mt)*
                
    if level: 
        mu2 = level_removal(mu2, wdw_level, center=center)
        
    return mu2

#################################
    
def anscombe(x, inverse=False, alpha=4.2):
    """
    Applies an Anscombe transform on an initial array. 
    
    This transformation stabilizes the variance of the array. 
    
    Parameters
    ---------
    x : 1D-array     
        Initial array to be transformed.
    inverse : bool, optional. 
        Flag to apply the inverse Anscombe transform. Default is False.
    alpha : float, optional
        Parameter of the Anscombe transform. The value alpha=4.2 is optimal 
        for Nc as demonstrated by T. Dudok de wit (2016) and may also
        be used for Ns and Ng. The default is 4.2.
        
    Returns
    ------
    y : 1D-array
      The transformed array.
    """
    assert np.ndim(x) == 1, "Input data must be a 1D array"
    if inverse:
        y = ((x/2*alpha)**2 - 3/8*alpha**2) / alpha
    
    else:
        y = 2/alpha * np.sqrt(alpha*x + 3/8*alpha**2)
        
    return y


def median_transformed(data, period_rescaling):
    """
    This function computes the transformed version of the median along the time.
    
    The transformation is composed of an Anscombe transform (to stabilize
    the variance) and a Wiener filter (to remove high frequencies).
    
    Parameters
    ---------
    data : 2D-array 
        A matrix of observations: either Ns, Ng or Nc
        (rows: time, columns: stations).
    period_rescaling : float>0
        Length of the period where the scaling-factors are assumed to be 
        constant, expressed in months.
        Using a statistical procedure based on Kruskal-Wallis tests, 
        the minimum value for this parameter is found to be equal to 
        8 months for Ns, 14 months for Ng and 10 months for Nc.
    
    Returns
    -------
    mu_s : 1D-array             
        The transformed median of the panel (along the time).  
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    (n_obs, n_stations) = data.shape
    data_rescaled = rescaling(data, period_rescaling)
    med = np.nanmedian(data_rescaled, axis=1) #not rounded
    ind_nan = np.isnan(med) #index of NaNs
    med = med[~ind_nan] 
    
    med = anscombe(med) #Anscombe
    med_fft = np.fft.fft(med) #fast fourier transform 
    
    #threshold at 7 days 
    thr = 7  
    L = len(med)
    if L % 2 == 1: 
        P2 = abs(med_fft/L) #two-sided spectre 
        P1 = P2[0:int(np.round(L/2+1))]
        P1[1:-1] = 2*P1[1:-1] #single sided amplitude spectrum
        Fs = 1 
        f = Fs*np.arange(0,L/2)/L #frequency domain
        f = 1./f  #time domain
        
        inf_7 = np.where(f < thr) #remove periods before 7 days (high frequencies)
        P1[inf_7] = 0
        P1[1:-1] = P1[1:-1]/2
        P2[0:int(np.round(L/2+1))] = P1
        P1_rev = np.transpose(P1)[::-1]
        P2[int(np.round(L/2+1)):L] = P1_rev[0:len(P1_rev)-1]
        inf_thr = np.where(P2 == 0)
        med_fft[inf_thr] = 0
    else:
        P2 = abs(med_fft/L) #two-sided spectre 
        P1 = P2[0:int(np.round(L/2+1))]
        P1[1:-1] = 2*P1[1:-1] #single sided amplitude spectrum
        Fs = 1 
        f = Fs*np.arange(0,L/2)/L #frequency domain
        f = 1./f  #time domain
        
        inf_7 = np.where(f < thr) #remove periods before 7 days (high frequencies)
        P1[inf_7] = 0
        P1[1:-1] = P1[1:-1]/2
        P2[0:int(np.round(L/2+1))] = P1
        P1_rev = np.transpose(P1)[::-1]
        P2[int(np.round(L/2+1))-1:L] = P1_rev[0:len(P1_rev)-1]
        inf_thr = np.where(P2 == 0)
        med_fft[inf_thr] = 0

    med_ifft = abs(np.fft.ifft(med_fft)) #inverse fourier transform 
    med_inv = anscombe(med_ifft, inverse=True) #inverse Anscombe
    med_inv[med_inv<0] = 0 #remove negative values if any
    mu_s = np.zeros((n_obs)); mu_s[:] = np.nan #reconstruct array with MV
    mu_s[~ind_nan] = med_inv
    return np.round(mu_s)

def error_at_minima(data, period_rescaling):
    """ 
    This function computes the error at solar minima (epsilon3(i,t))
    of the stations.
    
    Parameters
    ------
    data : 2D-array 
        A matrix of observations: either Ns, Ng or Nc
        (rows: time, columns: stations).
    period_rescaling : float>0
        Length of the period where the scaling-factors are assumed to be 
        constant, expressed in months.
        Using a statistical procedure based on Kruskal-Wallis tests, 
        the minimum value for this parameter is found to be equal to 
        8 months for Ns, 14 months for Ng and 10 months for Nc.
        
    Returns
    -------
    e3 : 2D-array               
        The errors of the stations at solar minima.
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    (n_obs, n_stations) = data.shape
    data_rescaled = rescaling(data, period_rescaling)
    mu_s = median_transformed(data, period_rescaling)
    
    e3 = np.zeros((n_obs, n_stations)); e3[:] = np.nan
    for i in range(n_obs):
        if mu_s[i] == 0: #when mu_s(t)=0
            e3[i,:] = data_rescaled[i,:] #Zi(t)

    return e3

def short_term_error(data, period_rescaling):
    """ 
    This function computes the short-term error (epsilon_tilde(i,t) := epsilon1(i,t)
    + epsilon2(i,t)), i.e. errors inferior to 27 days, of the stations.
    
    Parameters
    ----------
    data : 2D-array 
        A matrix of observations: either Ns, Ng or Nc
        (rows: time, columns: stations).
    period_rescaling : float>0
        Length of the period where the scaling-factors are assumed to be 
        constant, expressed in months.
        Using a statistical procedure based on Kruskal-Wallis tests, 
        the minimum value for this parameter is found to be equal to 
        8 months for Ns, 14 months for Ng and 10 months for Nc.
        
    Returns
    ------
    e1 : 2D-array   
        The short-term errors of the stations.
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    (n_obs, n_stations) = data.shape
    data_rescaled = rescaling(data, period_rescaling)
    mu_s = median_transformed(data, period_rescaling)
    
    e1 = np.zeros((n_obs, n_stations)); e1[:] = np.nan
    for i in range(n_obs):
        if mu_s[i] > 0 and mu_s[i] != np.nan:
            e1[i,:] = data_rescaled[i,:]/mu_s[i] #Z(i,t)/mu_s(t)
    return e1 

#=========================================================================
#========================================================================

def stats(x, station=True, period=None, robust=False):
    """
    Computes the mean and the standard deviation of a quantity 'x'. 
    This function may be applied in general on any quantity (errors, 
    observations, median, etc.).

    Parameters
    ----------
    x : 2D-array
        An initial array (rows: time, columns: stations).
        1D-arrays may be transformed into 2D-arrays using 'data.reshape(-1,1)'.
    station : bool, optional
        Flag to compute the mean and the standard deviation station by station.
        Otherwise, all stations are agglomerated in the computations.
        The default is True.
    period : integer > 0, optional
        Period on which the mean and the standard deviation are computed, 
        expressed in days (365 = one year). 
        When None, the entire period is used. The default is None.
    robust : bool, optional
        Flag to use robust estimator of the mean and the standard deviation 
        (i.e.: the median and the interquartile range). The default is False.
    Returns
    -------
    x_mean : 2D-array
         The mean of the quantity 'x'.
    x_std : 2D-array
         The standard deviation of the quantity 'x'.
    """
    assert np.ndim(x) == 2, "Input data must be a 2D array."
    (n_obs, n_stations) = x.shape
    
    if period is None:
        period = n_obs #entire period used
    
    n_rows = int(np.ceil(n_obs / period))
    if not station: #all station together
        x_mean = np.zeros((n_rows,1)); x_mean[:] = np.nan
        x_std = np.zeros((n_rows,1)); x_std[:] = np.nan
        c = 0
        for i in range(0, n_obs, period):
             if i < n_obs-period:
                if not robust: 
                    x_mean[c] = np.nanmean(x[i:i+period])
                    x_std[c] = np.nanstd(x[i:i+period])
                else: #robust estimators
                    x_mean[c] = np.nanmedian(x[i:i+period])
                    x_std[c] = iqr(x[i:i+period], nan_policy='omit')
             else:
                if not robust: 
                    x_mean[c] = np.nanmean(x[i:i+n_obs])
                    x_std[c] = np.nanstd(x[i:i+n_obs])
                else: #robust estimators
                    x_mean[c] = np.nanmedian(x[i:i+n_obs])
                    x_std[c] = iqr(x[i:i+n_obs], nan_policy='omit')
             c += 1
            
        
    if station: ### station by station
        x_mean = np.zeros((n_rows, n_stations)); x_mean[:] = np.nan
        x_std = np.zeros((n_rows, n_stations)); x_std[:] = np.nan
        c = 0
        for i in range(0, n_obs, period):
            if i < n_obs-period:
                for j in range(n_stations):
                    if not robust:
                        x_mean[c,j] = np.nanmean(x[i:i+period,j])
                        x_std[c,j] = np.nanstd(x[i:i+period,j])
                    else: #robust estimators
                        x_mean[c,j] = np.nanmedian(x[i:i+period,j])
                        x_std[c,j] = iqr(x[i:i+period,j], nan_policy='omit')
            else: #last observations 
                for j in range(n_stations):
                    if not robust:
                        x_mean[c,j] = np.nanmean(x[i:i+n_obs,j])
                        x_std[c,j] = np.nanstd(x[i:i+n_obs,j])
                    else: #robust estimators
                        x_mean[c,j] = np.nanmedian(x[i:i+n_obs,j])
                        x_std[c,j] = iqr(x[i:i+n_obs,j], nan_policy='omit')
            c += 1
                
    return x_mean, x_std

def mse_criterion(x, names, ref=None, robust=False):
    """
    Computes the mean-squarred error (mse) of a quantity 'x' in each station.
    
    This criterion combines the variance of the series with their bias
    with respect to a reference. It may be used as a stability criterion, 
    for clustering or ranking the stations. 

    Parameters
    ----------
    x : 2D-array
        An initial array (rows: time, columns: stations).
        This function may be applied in general on any quantity that is station- 
        dependent such as the errors or the observations (Ns, Ng, Nc).
    names : list
        A list containing the code names of the stations.
    ref : float, optional
        The reference value for the quantity 'x' (used to compute the bias).
        When None, the reference is taken as the mean value of 
        x. The default is None.
    robust : bool, optional
        Flag to compute the mse on robust estimators (i.e. the median 
        and the interquartile range). The default is False.

    Returns
    -------
    mse : 1D-array
        The mean-squarred error in each station.
    order_names : list
        The index of the stations sorted from the most stable series (minimum
        mse value) to the most unstable one (max mse value).

    """
    assert np.ndim(x) == 2, "Input data must be a 2D array"
    (n_obs, n_stations) = x.shape
    
    if ref is None:
        if robust:
            ref = np.nanmedian(x)
        else:
            ref = np.nanmean(x)
    
    mse = np.zeros(n_stations)
    for i in range(n_stations):
        if robust:
            mse[i] = (np.nanmedian(x[:,i]-ref))**2 + iqr(x[:,i], nan_policy='omit') 
        else:
            mse[i] = (np.nanmean(x[:,i]-ref))**2 + np.nanvar(x[:,i]) 
        
    ind_order = np.argsort(mse)
    order_names = [names[i] for i in ind_order]
            
    return mse, order_names

def error_bars(data, period_rescaling):
    """
    Computes an error bar for the stations at each time.
    
    This function computes an additive error for the stations 
    at each time (Yi(t) = s(t) + s(t)(epsilon_tilde -1) ).

    Parameters
    ----------
    data : 2D-array
        A matrix of observations: either Ns, Ng or Nc
        (rows: time, columns: stations).
    period_rescaling : float>0
        Length of the period where the scaling-factors are assumed to be 
        constant, expressed in months.
        Using a statistical procedure based on Kruskal-Wallis tests, 
        the minimum value for this parameter is found to be equal to 
        8 months for Ns, 14 months for Ng and 10 months for Nc.

    Returns
    -------
    bars : 2D-array
        The error bars of the data.
    ref : 1D-array
        The transformed median (reference) of the network.
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    (n_obs, n_stations) = data.shape
    
    ref = median_transformed(data, period_rescaling)
    e1 = short_term_error(data, period_rescaling) 
    
    bars = np.zeros((n_obs, n_stations)); bars[:] = np.nan
    for i in range(n_obs):
        for j in range(n_stations):
            if ref[i] > 0: 
                bars[i,j] = ref[i]*(e1[i,j]-1)
             
    return bars, ref

if __name__ == "__main__":
    
    ####################################################################
    ### load data

    with open('../data/data_35_1947', 'rb') as file: # load subset of 21 stations
    #with open('data/data_1981', 'rb') as file: # load all stations
         my_depickler = pickle.Unpickler(file)
         Ns = my_depickler.load() #number of spots
         Ng = my_depickler.load() #number of sunspot groups
         Nc = my_depickler.load() #Ns+10Ng
         station_names = my_depickler.load() #index of the stations
         time = my_depickler.load() #time (fraction of years)
         
    Ns_rescaled = rescaling(Ns, 8)    

    ####################################    
    ### Solar signal
    mus_Ns = median_transformed(Ns, period_rescaling=8)
    mus_Ng = median_transformed(Ng, period_rescaling=14)
    mus_Nc = median_transformed(Nc, period_rescaling=10)
    
    ### histograms
    plt.rcParams['figure.figsize'] = (10.0, 6.0)
    plt.figure(1)  
    plt.hist(mus_Ns[~np.isnan(mus_Ns)], range=[0,150], bins='auto', density=True, facecolor='b')  
    plt.title("Solar signal (Ns)")
    plt.text(60, 0.04, 'mean: ' '%4f' %np.nanmean(mus_Ns))
    plt.text(60, 0.03, 'std: ' '%4f' %np.nanstd(mus_Ns))
    plt.axis([0,150, 0, 0.08])
    plt.grid(True)
    plt.show()
    
    plt.figure(2)  
    plt.hist(mus_Ng[~np.isnan(mus_Ng)], range=[0,20], bins=20, density=True, facecolor='b')  
    plt.title("Solar signal (Ng)")
    plt.text(15, 0.2, 'mean: ' '%4f' %np.nanmean(mus_Ng))
    plt.text(15, 0.15, 'std: ' '%4f' %np.nanstd(mus_Ng))
    plt.axis([0, 20, 0, 0.25])
    plt.grid(True)
    plt.show()
    
    plt.figure(3)  
    plt.hist(mus_Nc[~np.isnan(mus_Nc)], range=[0,300], bins='auto', density=True, facecolor='b')  
    plt.title("Solat signal (Nc)")
    plt.text(150, 0.015, 'mean: ' '%4f' %np.nanmean(mus_Nc))
    plt.text(150, 0.0125, 'std: ' '%4f' %np.nanstd(mus_Nc))
    plt.axis([0, 300, 0, 0.03])
    plt.grid(True)
    plt.show()
    

    ####################################    
    ### Long-term error 
    mu2_81 = long_term_error(Ns, period_rescaling=8, wdw=81)
    mu2_1 = long_term_error(Ns, period_rescaling=8, wdw=365)
    mu2_2 = long_term_error(Ns, period_rescaling=8, wdw=912)
    
    #stability criterion
    #mse_mu2, names = mse_criterion(mu2_1, station_names, ref=None)
    mse_mu2, names = mse_criterion(mu2_1, station_names, ref=1)
    
    start = np.where(time == 1960)[0][0]
    stop = np.where(time == 2010)[0][0]
    stat = 19
    plt.plot(time[start:stop], mu2_81[start:stop, stat], ':', c='tab:green', label='mu2_81')
    plt.plot(time[start:stop], mu2_1[start:stop, stat], '--', c='tab:red', label='mu2_1')
    plt.plot(time[start:stop], mu2_2[start:stop, stat], lw=3, c='tab:blue', label='mu2_2')
    plt.plot([time[start], time[stop]], [1, 1], 'k-', lw=2)
    plt.legend(loc='upper right')
    #f4.set_ylim([-10,20]); f4.set_xlim([time[start-20], time[stop+20]])
    if stop-start < 4000:
        x_ticks = np.arange(np.round(time[start]), np.round(time[stop])+1, 1)
    else :
        x_ticks = np.arange(np.round(time[start]), np.round(time[stop])+1, 5)
    plt.xticks(x_ticks)
    plt.title("Long-term error for Ns in %s" %station_names[stat])
    plt.ylabel('mu2')
    plt.xlabel('year')
    plt.tick_params(axis='x', which='major')
    plt.show()
    
    ####################################
    ### Error at minima
    e3_Ns = error_at_minima(Ns, period_rescaling=8)
    e3_Ng = error_at_minima(Ng, period_rescaling=14)
    e3_Nc = error_at_minima(Nc, period_rescaling=10)
    
    ### histograms
    binning = int(6/(3.5*np.nanstd(e3_Ns)*len(e3_Ns)**(-1/3))) #Scott's rule
    plt.rcParams['figure.figsize'] = (10.0, 6.0)
    plt.figure(1)  
    plt.hist(e3_Ns[~np.isnan(e3_Ns)], range=[0,5], bins=binning, density=True, facecolor='b')  
    plt.title("Error at minima (Ns)")
    plt.text(2, 0.8, 'mean:' '%4f' %np.nanmean(e3_Ns))
    plt.text(2, 0.6, 'std:' '%4f' %np.nanstd(e3_Ns))
    plt.axis([0, 5, 0, 1])
    plt.grid(True)
    plt.show()
    
    binning = int(6/(3.5*np.nanstd(e3_Ng)*len(e3_Ng)**(-1/3)))
    plt.figure(2)  
    plt.hist(e3_Ng[~np.isnan(e3_Ng)], range=[0,5], bins=binning, density=True, facecolor='b')  
    plt.title("Error at minima (Ng)")
    plt.text(2, 0.8, 'mean:' '%4f' %np.nanmean(e3_Ng))
    plt.text(2, 0.6, 'std:' '%4f' %np.nanstd(e3_Ng))
    plt.axis([0, 5, 0, 1])
    plt.grid(True)
    plt.show()
    
    binning = int(6/(3.5*np.nanstd(e3_Nc)*len(e3_Nc)**(-1/3)))
    plt.figure(3)  
    plt.hist(e3_Nc[~np.isnan(e3_Nc)], range=[0,30], bins=binning, density=True, facecolor='b')  
    plt.title("Error at minima (Nc)")
    plt.text(10, 0.03, 'mean:' '%4f' %np.nanmean(e3_Nc))
    plt.text(10, 0.02, 'std:' '%4f' %np.nanstd(e3_Nc))
    plt.axis([0, 30, 0, 0.04])
    plt.grid(True)
    plt.show()
    
    ##################
    ### Short-term error
    e1_Ns = short_term_error(Ns, period_rescaling=8)
    e1_Ng = short_term_error(Ng, period_rescaling=14)
    e1_Nc = short_term_error(Nc, period_rescaling=10)
    
    #stability criterion
    #mse_e1, names = mse_criterion(e1_Ns, station_names, ref=None)
    mse_e1, names = mse_criterion(e1_Ns, station_names, ref=1)
    
    ###histograms
    binning = int(6/0.0328) #Scott's rule for the binning
    plt.rcParams['figure.figsize'] = (10.0, 6.0)
    plt.figure(1)  
    plt.hist(e1_Ns[~np.isnan(e1_Ns)], range=[0,5], bins=binning, density=True, facecolor='b')  
    plt.title("Short-term error (Ns)")
    plt.text(2, 1, 'mean:' '%4f' %np.nanmean(e1_Ns))
    plt.text(2, 0.6, 'std:' '%4f' %np.nanstd(e1_Ns))
    plt.axis([0, 5, 0, 3.5])
    plt.grid(True)
    plt.show()
    
    binning = int(6/0.0328)
    plt.figure(2)  
    plt.hist(e1_Ng[~np.isnan(e1_Ng)], range=[0,5], bins=binning, density=True, facecolor='b')  
    plt.title("Short-term error (Ng)")
    plt.text(2, 1, 'mean:' '%4f' %np.nanmean(e1_Ng))
    plt.text(2, 0.6, 'std:' '%4f' %np.nanstd(e1_Ng))
    plt.axis([0, 5, 0, 3.5])
    plt.grid(True)
    plt.show()
    
    binning = int(6/0.0433)
    plt.figure(3)  
    plt.hist(e1_Nc[~np.isnan(e1_Nc)], range=[0,5], bins=binning, density=True, facecolor='b')  
    plt.title("Short-term error (Nc)")
    plt.text(2, 1, 'mean:' '%4f' %np.nanmean(e1_Nc))
    plt.text(2, 0.6, 'std:' '%4f' %np.nanstd(e1_Nc))
    plt.axis([0, 5, 0, 3.5])
    plt.grid(True)
    plt.show()

    #====================================================================
    #====================================================================
    #stability criterion combining short and long-term
    
    ### Long-term error without levels
    mu2_81 = long_term_error(Ns, period_rescaling=8, wdw=81, level=True, wdw_level=4000)
    mu2_1 = long_term_error(Ns, period_rescaling=8, wdw=365, level=True, wdw_level=4000)
    mu2_2 = long_term_error(Ns, period_rescaling=8, wdw=912, level=True, wdw_level=4000)
        
    
    # mse_combined = mse_criterion(e1_Ns, station_names, ref=None)[0] + \
    #                     mse_criterion(mu2_1, station_names, ref=None)[0]
    # ind_order = np.argsort(mse_combined)
    # names = [station_names[i] for i in ind_order]
    
    mse_add = mse_criterion(e1_Ns, station_names, ref=1)[0] + \
                        mse_criterion(mu2_1, station_names, ref=0)[0]
    ind_order = np.argsort(mse_add)
    names_add = [station_names[i] for i in ind_order]
    
    #should be the same if the errors were perfectly independent
    mse_comb, names_comb = mse_criterion(e1_Ns+mu2_1, station_names, ref=1)
    
    #====================================================================
    #error bars
    
    bars, ref = error_bars(Ns, period_rescaling=8)
    
    start = np.where(time == 2005)[0][0]
    stop = np.where(np.round(time,1) == 2005.5)[0][0]
    stat = 20
    plt.stem(time[start:stop], ref[start:stop]+bars[start:stop, stat], label='errors', markerfmt='C0.', basefmt='C0-')
    plt.plot(time[start:stop], ref[start:stop], c='tab:purple', label='reference', lw=3)
    plt.legend(loc='upper right')
    if stop-start < 4000:
        x_ticks = np.arange(np.round(time[start]), np.round(time[stop])+1, 1)
    else :
        x_ticks = np.arange(np.round(time[start]), np.round(time[stop])+1, 5)
    plt.xticks(x_ticks)
    plt.title("Additive errors in %s" %station_names[stat])
    plt.ylabel('Yit')
    plt.xlabel('year')
    plt.tick_params(axis='x', which='major')
    plt.show()
