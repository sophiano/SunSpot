# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 08:55:28 2020

@author: sopmathieu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.svm import SVR
from sklearn import svm
from scipy.stats import halfnorm 
from sklearn.metrics import plot_confusion_matrix
from scipy import interpolate

from SunSpot import BB_methods as BB

def selection_input_vector(data, delta_min, L, block_length, k=None, nmc=2000,
                           n=2000, qt=0.7, plot=True):
    """ 
    Compute the length of the input vector for the support vector machine 
    procedures. It is defined as a particular quantile 'qt' of the run length
    distribution for a shift size equal to 'delta_min'.
    
    Arguments:
    data:           IC dataset (not the blocks)
    delta_min:      mininum shift size to detect
    L:              value of the upper control limit
    block_length:   length of the blocks
    k:              allowance parameter
    nmc:            number of Monte-Carlo runs
    n:              length of the resampled series 
    qt:             quantile of the run length distribution
    plot:           boolean. Show the histogram of the run length distribution
                    if true. 
         
    Return:
    m:              length of the input vector   
    """
    if k is None:
        k = delta_min/2
    blocks = BB.MBB(data, block_length)    
    n_blocks = int(np.ceil(n/blocks.shape[1]))
        
    RL1_plus = np.zeros((nmc,1)); RL1_minus = np.zeros((nmc,1))
    RL1_plus[:] = np.nan; RL1_minus[:] =np.nan
    for b in range(nmc):
        
        #sample data with BB and shift them by delta_min 
        boot = resample(blocks, replace=True, n_samples=n_blocks).flatten()[:n]
        boot = boot + delta_min 
        
        C_plus = np.zeros((n, 1))
        for i in range(1, n):
            C_plus[i] = max(0, C_plus[i-1] + boot[i] - k)
            if C_plus[i] > L:
                RL1_plus[b] = i
                break 
            
        C_minus = np.zeros((n, 1))       
        for j in range(1, n):
            C_minus[j] = min(0, C_minus[j-1] + boot[j] + k)
            if C_minus[j] < -L:
                RL1_minus[b] = j
                break
        
        if np.isnan(RL1_plus[b]): 
            RL1_plus[b] = n
        if np.isnan(RL1_minus[b]):
            RL1_minus[b] = n
            

    RL = (1/(RL1_minus) + 1/(RL1_plus))**(-1)
    
    if plot: 
        plt.figure(1)
        plt.hist(RL[~np.isnan(RL)], range=[-4,100], bins='auto', density=True, facecolor='b')  
        plt.title("Run length distribution")
        plt.axis([-4, 100, 0, 0.2])
        plt.grid(True)
        plt.show()
        
    m = int(np.quantile(RL[~np.isnan(RL)], qt)) 
    return m 


def training_svr_svm(data, L, delta_min, n, wdw_length, scale, block_length, k=None, 
                     n_series=500, power=1.5, N=150, C=10.0, epsilon=0.001,
                     precision=True, confusion=True): 
    """
    Train the support vector machine classifier (svc) and regressor (svr)
    to predict the size of the shifts in a continuous range starting from
    'delta_min' and the form of the shifts among "jumps", "trends" and "oscillating
    shifts".
    
    Arguments:
    data:           IC dataset (not the blocks)
    L:              control limit
    delta_min:      min shift size to detect
    n:              number of training and validating instances (multiple of 3)
    wdw_length:     length of the input vector
    scale:          scale parameter of the halfnormal distribution
    block_length:   length of the blocks
    k:              allowance parameter
    n_series:       length of the resampled series
    power:          exponent of the trend (power-law function)
    N:              parameter of the trend
    C:              parameter of the svr and svc (tradeoff between regularization
                    and mis-classifications)
    epsilon:        parameter of the svr (approximation accuracy)  
    precision:      boolean. If true, print accuracy measures
    confusion:      boolean. If true, show confusion matrix 
          
    Return:
    clf:            trained classifier
    regressor       trained regressor
    """    
    blocks = BB.MBB(data, block_length)    
    n_blocks = int(np.ceil(n_series/blocks.shape[1]))
    
    if k is None:
        k = delta_min/2
    eta = 1/ wdw_length
    sign = 1
    n_test = int(n/5) #n testing instances
    n_train = n - n_test #n training instances
    
    ### training
    input_train = np.zeros((n_train, wdw_length))
    size_train = np.zeros((n_train))
    form_train = np.zeros((n_train))
    rnd = halfnorm(scale=scale).rvs(size=n_train) + delta_min #size of shifts
    for b in range(0, n_train-2, 3):
        
        shift = rnd[b]*sign
        series = resample(blocks, replace=True, n_samples=n_blocks).flatten()[:n_series]
        delay = np.random.randint(wdw_length) #simulate a random delay 
        
        for rnd_form in range(3):
    
            boot = np.copy(series)
            
            if rnd_form == 0:         
                boot[wdw_length:] = boot[wdw_length:] + shift
                form_train[b] = 0
            elif rnd_form == 1:
                boot = shift/(N) * (np.arange(0,n_series)**power) + boot
                form_train[b] = 1
            else:
                boot[wdw_length:] = np.sin(eta*np.pi*np.arange(n_series-wdw_length))*shift + boot[wdw_length:]
                form_train[b] = 2
            size_train[b] = shift
            
            C_plus = np.zeros((n_series, 1))
            for i in range(wdw_length + delay, n_series): #start the monitoring after random delay 
                C_plus[i] = max(0, C_plus[i-1] + boot[i] - k)
                if C_plus[i] > L:
                    input_plus = boot[i+1-wdw_length:i+1] 
                    break 
                
            C_minus = np.zeros((n_series, 1))       
            for j in range(wdw_length + delay, n_series):
                C_minus[j] = min(0, C_minus[j-1] + boot[j] + k)
                if C_minus[j] < -L:
                    input_minus = boot[j+1-wdw_length:j+1] 
                    break
                
            if i > j: #save first alert recorded
                input_train[b,:] = input_minus
            else:
                input_train[b,:] = input_plus
                
            b += 1
        sign = -sign
        
    ### train the models
    regressor = SVR(C=C, epsilon=epsilon)
    regressor.fit(input_train, size_train)
    clf = svm.SVC(C=C)
    clf.fit(input_train, form_train)
    
    ###testing 
    input_test = np.zeros((n_test, wdw_length))
    label_test = np.zeros((n_test))
    form_test = np.zeros((n_test))
    rnd = halfnorm(scale=scale).rvs(size=n_test) + delta_min
    for b in range(0, n_test-2, 3):
         
        shift = rnd[b]*sign
        series = resample(blocks, replace=True, n_samples=n_blocks).flatten()[:n_series]
        delay = np.random.randint(wdw_length)
        
        for rnd_form in range(3):
            
            boot = np.copy(series)
            
            if rnd_form == 0:
                boot[wdw_length:] = boot[wdw_length:] + shift
                form_test[b] = 0
            elif rnd_form == 1:
                boot = shift/(N) * (np.arange(0,n_series)**power) + boot
                form_test[b] = 1
            else:
                boot[wdw_length:] = np.sin(eta*np.pi*np.arange(n_series-wdw_length))*shift + boot[wdw_length:]
                form_test[b] = 2
            label_test[b] = shift
            
            C_plus = np.zeros((n_series, 1))
            for i in range(wdw_length + delay, n_series):
                C_plus[i] = max(0, C_plus[i-1] + boot[i] - k)
                if C_plus[i] > L:
                    input_plus = boot[i+1-wdw_length:i+1] 
                    break 
                
            C_minus = np.zeros((n_series, 1))       
            for j in range(wdw_length + delay, n_series):
                C_minus[j] = min(0, C_minus[j-1] + boot[j] + k)
                if C_minus[j] < -L:
                    input_minus = boot[j+1-wdw_length:j+1]
                    break
                
            if i > j: #first alert recorded
                input_test[b,:] = input_minus
            else:
                input_test[b,:] = input_plus
                
            b += 1     
        sign = -sign
    
    ### compute accuracy and other precision measures 
    label_pred = regressor.predict(input_test)
    label_pred_clf = clf.predict(input_test)
    
    if precision :
        #regressor
        MAPE = (1/len(label_pred)) * sum(np.abs((label_test - label_pred)/label_test))*100
        NRMSE = np.sqrt(sum((label_test - label_pred)**2) / sum(label_test**2))
        print('MAPE=', MAPE) 
        print('NRMSE=', NRMSE) 
        #classifier
        accuracy = sum(label_pred_clf == form_test)*100 / len(label_pred_clf)
        MAE = (1/len(label_pred_clf)) * sum(np.abs(form_test - label_pred_clf))
        MSE = (1/len(label_pred_clf)) * sum((form_test - label_pred_clf)**2)
        print('Accuracy=', accuracy) 
        print('MAE=', MAE) 
        print('MSE=', MSE) 
        
    ### compute the confusion matrix 
    if confusion : 
        class_names = ['jump', 'trend', 'oscill.']
        titles_options = [("Confusion matrix, without normalization", None),
                          ("Normalized confusion matrix", 'true')]
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(clf, input_test, form_test,
                                         display_labels=class_names,
                                         cmap=plt.cm.Blues,
                                         normalize=normalize)
            disp.ax_.set_title(title)
            print(title)
            print(disp.confusion_matrix)
        plt.show()
    
    return (regressor, clf)


def interpolate1d(X):
    """ 
    Interpolate intermediate NaN values in a 1D array.
    
    Arguments: 
    X:       1D array
    Return: 
    X_int:   new array with interpolated values
    """
    ind = np.arange(X.shape[0])
    ind_not_nans = np.where(~np.isnan(X))
    f = interpolate.interp1d(ind[ind_not_nans], X[ind_not_nans], bounds_error=False)
    X_int = np.where(np.isfinite(X), X, f(ind))
    return X_int

def fill_nan(x):
    """
    Fill first NaN values by the first value that is not NaN and interpolate 
    intermediate NaNs in a 2D array. 
    Arguments: 
    x:      2D array. Each row will be interpolated. 
    
    Return: 
    new_x:  new array with interpolated values
    ind:    index of non-NaN rows
    """
    (nRows, wdw) = x.shape
    new_x = np.zeros((nRows,wdw)); new_x[:] = np.nan
    for i in range(nRows):
        indMissing = np.where(np.isnan(x[i,:]))[0]
        l = len(x[i,indMissing]) #number of MVs
        if l < 4*wdw/5: #20% available values otherwise discarded
            new_x[i,:] = x[i,:]
            if l > 0 and indMissing[0] == 0: #missing value at index 0 
                c = 0
                while c + 1 < len(indMissing) and indMissing[c+1] == indMissing[c] + 1:
                    c += 1
                new_x[i,:c+1] = x[i,c+1] #first nans replaced by first non nan value
                indMissing = np.where(np.isnan(new_x[i,:]))[0]
                l = len(new_x[i,indMissing])
            if l > 0 and indMissing[0] > 0:
                new_x[i,:] = interpolate1d(new_x[i,:]) #interpolate intermediate nans
    ind = np.where(~np.isnan(new_x).all(axis=1))[0]
    new_x = new_x[ind] #remove NaNs 
    return new_x, ind
