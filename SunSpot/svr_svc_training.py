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
from kneed import KneeLocator
from scipy.stats import halfnorm 
from sklearn.metrics import plot_confusion_matrix
from scipy import interpolate

from SunSpot import bb_methods as bb

def input_vector_length(data, delta_min, L_plus, L_minus=None, k=None,
                nmc=4000, n=2000, qt=None, block_length=None, 
                BB_method='MBB', plot=True):
    """ 
    Computes the length of the input vector for the support vector machine 
    procedures (svms). 
    
    The length of the input vector represents the number of past observations 
    that are fed to the svms after each alert. The regressor and classifier 
    then predict the form and the size of the shift that causes the alert
    based on the input vector.
    Intuitively, the length should be sufficiently large to ensure that most 
    of the shifts are contained within the input vector while maintaining 
    the computing efficiency of the method. This is usually 
    not a problem for the large shifts that are quickly detected by the chart. 
    However the smallest shifts may be detected only after a certain amount 
    of time and therefore require larger vectors. 
    Hence, the length is selected as an upper quantile of the run length 
    distribution, computed on data shifted by the smallest shift size 
    that we aim to detect.
    
    It is implemented as follows.
    For each monte-carlo run, a new series of observations is sampled from the 
    IC data using a block boostrap procedure. Then, a jump of size 
    "delta_min' is simulated on top of the sample. 
    The run length of the chart is then evaluated. The length of the input 
    vector is finally selected as a specified quantile of the run length
    distribution. If the quantile is unspecified, an optimal quantile 
    is selected by locating the 'knee' of the quantiles curve.
        
    
    Parameters: 
    ---------
    data : 2D-array
        IC dataset (rows: time, columns: IC series).
    delta_min : float >= 0
        The target minimum shift size. 
    L_plus : float 
        Value for the positive control limit.
    L_minus : float, optional
        Value for the negative control limit. Default is None. 
        When None, L_minus = - L_plus. 
    k : float, optional
        The allowance parameter. The default is None. 
        When None, k = delta/2 (optimal formula for normal data).
    nmc : int > 0, optional
        Number of Monte-Carlo runs. This parameter has typically a large value.
        Default is 2000. 
    n : int >= 0, optional
        Length of the resampled series (by the block bootstrap procedure).
        Default is 2000. 
    qt :  float in [0,1], optional
        Quantile of the run length distribution (used to select an appropriate
        input vector length). Default is None. 
        When None, the appropriate quantile is selected with a knee locator. 
    block_length :  int > 0, optional
        The length of the blocks. Default is None. 
        When None, the length is computed using an optimal formula. 
    BB_method : str, optional
       String that designates the block boostrap method chosen for sampling data. 
       Values for the string should be selected among: 
       'MBB': moving block bootstrap
       'NBB': non-overlapping block bootstrap
       'CBB': circular block bootstrap
       'MABB': matched block bootstrap
       Default is 'MBB'.
    plot :  bool, optional 
        Flag to show the histogram of the run length distribution. 
        Default is True.
         
    Returns
    -------
    m : int > 0
        The length of the input vector.
    """
    assert BB_method in ['MBB', 'NBB', 'CBB', 'MABB'], "Undefined block bootstrap procedure"
    if BB_method == 'MBB': 
        blocks = bb.MBB(data, block_length)    
    elif BB_method == 'NBB':
        blocks = bb.NBB(data, block_length) 
    elif BB_method == 'CBB':
        blocks = bb.CBB(data, block_length) 
                    
    if 'blocks' in locals():
        n_blocks = int(np.ceil(n/blocks.shape[1]))
        
    if k is None:
        k = delta_min/2
    if L_minus is None:
        L_minus = -L_plus
    n = int(n)
    assert n >= 0, "n must be superior or equal to zero"
    nmc = int(nmc)
    assert nmc > 0, "nmc must be strictly positive"
        
    RL1_plus = np.zeros((nmc,1)); RL1_minus = np.zeros((nmc,1))
    RL1_plus[:] = np.nan; RL1_minus[:] =np.nan
    for b in range(nmc):
        
        #sample data with BB and shift them by delta_min 
        if BB_method == 'MABB': 
            boot = bb.resample_MatchedBB(data, block_length, n=n)
        else:
            boot = resample(blocks, replace=True, n_samples=n_blocks).flatten()[:n]
        boot = boot + delta_min 
        
        C_plus = np.zeros((n, 1))
        for i in range(1, n):
            C_plus[i] = max(0, C_plus[i-1] + boot[i] - k)
            if C_plus[i] > L_plus:
                RL1_plus[b] = i
                break 
            
        C_minus = np.zeros((n, 1))       
        for j in range(1, n):
            C_minus[j] = min(0, C_minus[j-1] + boot[j] + k)
            if C_minus[j] < L_minus:
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
        
    if qt is not None:
        ### select m with a specified quantile 
        m = int(np.quantile(RL[~np.isnan(RL)], qt)) 
    else:
        ### select m with knee locator
        y = np.zeros((100))
        c=0 ; x = np.arange(1, 0.3, -0.05)
        for q in np.arange(1, 0.3, -0.05):
            y[c] = np.quantile(RL[~np.isnan(RL)], q)
            c += 1
                
        y = y[:len(x)]
        if plot: 
            plt.plot(x, y); plt.xlabel('quantile')
            plt.ylabel('run length')
            plt.title('Run length at different quantiles')
            plt.show()
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
        knee = kn.knee 
        m = int(np.quantile(RL[~np.isnan(RL)], knee))
    
    return m


def training_svr_svm(data, L_plus, delta_min, wdw_length, scale, 
              delay=True, L_minus=None,  k=None, n=63000, n_series=500,
            C=1.0, epsilon=0.001, kernel='rbf', degree=3, 
            block_length=None, BB_method='MBB', 
            precision=True, confusion=True): 
    """
    Trains the support vector machine classifier (svc) and regressor (svr).
    
    The training (and validating) procedure works as follows.
    For each monte-carlo run, a new series of observations is sampled from the 
    data using a block boostrap procedure.
    A shift size is then sampled from a halfnormal distribution (supported by 
    [delta_min, +inf]) with a specified scale parameter.
    A jump, an oscillating shift (with random frequency in the interval 
    [pi/(2*wdw_length), 2*pi/wdw_length]) and a drift (with random power-law
    functions in the range [1.5,2]) of previous size 
    are then added on top of the sample to create artificial deviations. 
    The classifer is then trained to recognize the form of deviations among the 
    three general classes: 'jump', 'drift' or 'oscillation'.
    Whereas the regressor learns to predict the size of the shifts in the
    continuous ranges [-inf, -delta_min] and [delta_min, +inf]. 
    Once the learning is finished, a validation step is also applied on 
    unseen deviations to evaluate the performances of the svr and svc.
    Three different criteria are computed: the absolute mean squared error (MAPE)
    together with the normalized root mean square error (NRMSE) for the regressor
    and the accuracy for the classifier.
    
    Parameters
    ---------
    data : 2D-array
        IC dataset (rows: time, columns: IC series).
    L_plus : float 
        Value for the positive control limit.
    delta_min : float > 0
        The target minimum shift size. 
    wdw_length : int > 0
        The length of the input vector.
    scale : float > 0
         The scale parameter of the halfnormal distribution 
         (similar to the variance of a normal distribution). 
         A typical range of values for scale is [1,4], depending on the size
         of the actual deviations
    delay : bool, optional
        Flag to start the chart after a delay randomly selected from the
        interval [0, wdw_lengt]. Default is True. 
        This flag should be activated only for noisy data that are quickly 
        detected by the chart. 
    L_minus :  float, optional
        Value for the negative control limit. Default is None. 
        When None, L_minus = - L_plus. 
    k : float, optional
        The allowance parameter. The default is None. 
        When None, k = delta/2 (optimal formula for normal data).
    n : int > 0, optional      
        Number of training and validating instances. This value is 
        typically large. Default is 63000.
    n_series : int > 0, optional
        Length of the resampled series (by the block bootstrap procedure).
        Default is 500. 
    C : float > 0, optional
        Regularization parameter of the svr and svc (the strength of the 
        regularization is inversely proportional to C).
        Default is 1. Typical range is [1, 10].
    epsilon : float, optional
        Parameter of the svr, which represents the approximation accuracy. 
        Default is 0.001.
    kernel : str, optional
        The kernel function to be used in the suppor vector machine procedures. 
        Values should be selected among: 'rbf', 'linear', 'sigmoid' and 'poly'. 
        Default is 'rbf'.
    degree : int > 0, optional
        The degree of the polynomial kernel. Only used when kernel='poly'.
        Default is 3.
    block_length :  int > 0, optional
        The length of the blocks. Default is None. 
        When None, the length is computed using an optimal formula. 
    BB_method : str, optional
       String that designates the block boostrap method chosen for sampling data. 
       Values for the string should be selected among: 
       'MBB': moving block bootstrap
       'NBB': non-overlapping block bootstrap
       'CBB': circular block bootstrap
       'MABB': matched block bootstrap
       Default is 'MBB'.
    precision : bool, optional    
        Flag to print accuracy measures. Default is True.
    confusion : bool, optional 
        Flag to show the confusion matrix (measure of the classification accuracy, 
        class by class). Default is True.        
          
    Returns 
    ------
    clf : support vector classification model
        The trained classifier.
    regressor : support vector regression model
        The trained regressor.
    """     
    assert BB_method in ['MBB', 'NBB', 'CBB', 'MABB'], "Undefined block bootstrap procedure"
    if BB_method == 'MBB': 
        blocks = bb.MBB(data, block_length)    
    elif BB_method == 'NBB':
        blocks = bb.NBB(data, block_length) 
    elif BB_method == 'CBB':
        blocks = bb.CBB(data, block_length) 
                    
    if 'blocks' in locals():
        n_blocks = int(np.ceil(n_series/blocks.shape[1]))
    
    wdw_length = int(np.ceil(wdw_length)) #should be integer
    
    n = int(n)
    assert n > 0, "n must be strictly positive"
    if n % 3 == 2:  #n should be multiple of 3
        n += 1
    if n % 3 == 1: 
        n += 2
        
    if L_minus is None:
        L_minus = -L_plus
    if k is None:
        k = delta_min/2
        
    assert degree > 0, "degree must be strictly positive"
    degree = int(degree)
        
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
        if BB_method == 'MABB': 
            series = bb.resample_MatchedBB(data, block_length, n=n_series)
        else:
            series = resample(blocks, replace=True, n_samples=n_blocks).flatten()[:n_series]
        if delay:
            delay = np.random.randint(wdw_length/2) #simulate a random delay
        else: 
            delay = 0
        
        for rnd_form in range(3):
            boot = np.copy(series)
            
            if rnd_form == 0:         
                boot[wdw_length:] = boot[wdw_length:] + shift
                form_train[b] = 0
                #plt.plot(boot); plt.show()
            elif rnd_form == 1:
                power = np.random.uniform(1.5,2)
                boot = shift/(n_series) * (np.arange(n_series)**power) + boot
                form_train[b] = 1
            elif rnd_form == 2:
                #eta = np.random.uniform(np.pi/(2*wdw_length), 2*np.pi/wdw_length)
                eta = np.random.uniform(np.pi/(wdw_length), 3*np.pi/wdw_length)
                boot = np.sin(eta*np.pi*np.arange(n_series))*shift*boot
                form_train[b] = 2
            
            size_train[b] = shift
            
            input_plus = boot[wdw_length:wdw_length*2] #default is not alert
            C_plus = np.zeros((n_series, 1))
            for i in range(wdw_length + delay, n_series): #start the monitoring after random delay 
                C_plus[i] = max(0, C_plus[i-1] + boot[i] - k)
                if C_plus[i] > L_plus:
                    input_plus = boot[i+1-wdw_length:i+1] 
                    break 
                
            input_minus = boot[wdw_length:wdw_length*2] #default is not alert
            C_minus = np.zeros((n_series, 1))       
            for j in range(wdw_length + delay, n_series):
                C_minus[j] = min(0, C_minus[j-1] + boot[j] + k)
                if C_minus[j] < L_minus:
                    input_minus = boot[j+1-wdw_length:j+1] 
                    break
                
            if i > j: #save first alert recorded
                input_train[b,:] = input_minus
            else:
                input_train[b,:] = input_plus
            #plt.plot(input_train[0,:]); plt.show()
            #plt.plot(input_train[1,:]); plt.show()
            #plt.plot(input_train[2,:]); plt.show() 
            
            b += 1
        sign = -sign
        
    ### train the models
    regressor = SVR(C=C, epsilon=epsilon, kernel=kernel, degree=degree)
    regressor.fit(input_train, size_train)
    clf = svm.SVC(C=C, kernel=kernel, degree=degree)
    clf.fit(input_train, form_train)
    
    ###testing 
    input_test = np.zeros((n_test, wdw_length))
    label_test = np.zeros((n_test))
    form_test = np.zeros((n_test))
    rnd = halfnorm(scale=scale).rvs(size=n_test) + delta_min
    for b in range(0, n_test-2, 3):
         
        shift = rnd[b]*sign
        if BB_method == 'MABB': 
            series = bb.resample_MatchedBB(data, block_length, n=n_series)
        else:
            series = resample(blocks, replace=True, n_samples=n_blocks).flatten()[:n_series]
        if delay:
            delay = np.random.randint(wdw_length/2) #simulate a random delay
        else: 
            delay = 0
        
        for rnd_form in range(3):
            
            boot = np.copy(series)
            
            if rnd_form == 0:
                boot[wdw_length:] = boot[wdw_length:] + shift
                form_test[b] = 0
            elif rnd_form == 1:
                power = np.random.uniform(1.5,2)
                boot = shift/(n_series) * (np.arange(n_series)**power) + boot
                form_test[b] = 1
            else:
                #eta = np.random.uniform(np.pi/(2*wdw_length), 2*np.pi/wdw_length)
                eta = np.random.uniform(np.pi/(wdw_length), 3*np.pi/wdw_length)
                #boot[wdw_length:] = np.sin(eta*np.pi*np.arange(n_series-wdw_length))*shift + boot[wdw_length:]
                boot = np.sin(eta*np.pi*np.arange(n_series))*shift*boot
                form_test[b] = 2
                
            label_test[b] = shift
            
            input_plus = boot[wdw_length:wdw_length*2] #default is not alert
            C_plus = np.zeros((n_series, 1))
            for i in range(wdw_length + delay, n_series):
                C_plus[i] = max(0, C_plus[i-1] + boot[i] - k)
                if C_plus[i] > L_plus:
                    input_plus = boot[i+1-wdw_length:i+1] 
                    break 
                
            input_minus = boot[wdw_length:wdw_length*2] #default is not alert
            C_minus = np.zeros((n_series, 1))       
            for j in range(wdw_length + delay, n_series):
                C_minus[j] = min(0, C_minus[j-1] + boot[j] + k)
                if C_minus[j] < L_minus:
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
        print('MAPE =', MAPE) 
        print('NRMSE =', NRMSE) 
        
        label_pred = abs(label_pred)
        label_test = abs(label_test)
        MAPE = (1/len(label_pred)) * sum(np.abs((label_test - label_pred)/label_test))*100
        NRMSE = np.sqrt(sum((label_test - label_pred)**2) / sum(label_test**2))
        print('MAPE without signs =', MAPE) 
        print('NRMSE without signs =', NRMSE) 
        
        #classifier
        accuracy = sum(label_pred_clf == form_test)*100 / len(label_pred_clf)
        MAE = (1/len(label_pred_clf)) * sum(np.abs(form_test - label_pred_clf))
        MSE = (1/len(label_pred_clf)) * sum((form_test - label_pred_clf)**2)
        print('Accuracy =', accuracy) 
        #print('MAE=', MAE) 
        #print('MSE=', MSE) 
        
    ### compute the confusion matrix 
    if confusion : 
        class_names = ['jump', 'drift', 'oscill.']
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
            print(disp.confusion_matrix[2,1]/n_test)
        plt.show()
    
    return (regressor, clf)


def choice_of_C(data, L_plus, delta_min, wdw_length, scale, start=1, stop=10,
                step=1, delay=True, L_minus=None,  k=None, n=36000, n_series=500, 
                epsilon=0.001, block_length=None, BB_method='MBB', confusion=True, 
                verbose=True):
    """
    Selects an appropriate value for the regularization parameter (C) of the 
    support vector machine procedures. 
    
    The procedure is implemented as follows.
    For each value of C, the regressor and classifier are trained and validated.
    Then, the values of C that maximize/minimize different performance 
    measures are returned. 
    The training (and validating) procedure works as explained below.
    For each monte-carlo run, a new series of observations is sampled from the 
    data using a block boostrap procedure.
    A shift size is then sampled from a halfnormal distribution (supported by 
    [delta_min, +inf]) with a specified scale parameter.
    A jump, an oscillating shift (with random frequency in the interval 
    [pi/(2*wdw_length), 2*pi/wdw_length]) and a drift (with random power-law
    functions in the range [1.5,2]) of previous size 
    are then added on top of the sample to create artificial deviations. 
    The classifer is then trained to recognize the form of deviations among the 
    three general classes: 'jump', 'drift' or 'oscillation'.
    Whereas the regressor learns to predict the size of the shifts in the
    continuous ranges [-inf, -delta_min] and [delta_min, +inf]. 
    Once the learning is finished, a validation step is also applied on 
    unseen deviations to evaluate the performances of the svr and svc. 
    Three different criteria are computed: the absolute mean squared error (MAPE)
    together with the normalized root mean square error (NRMSE) for the regressor
    and the accuracy for the classifier.
    
    Parameters
    ---------
    data : 2D-array
        IC dataset (rows: time, columns: IC series).
    L_plus : float 
        Value for the positive control limit.
    delta_min : float > 0
        The target minimum shift size. 
    wdw_length : int > 0
        The length of the input vector.
    scale : float > 0
         The scale parameter of the halfnormal distribution 
         (similar to the variance of a normal distribution). 
         A typical range of values for scale is [1,4], depending on the size
         of the actual deviations
    start : float > 0, optional
        Starting value for C. Default is 1.
    stop : float > 0, optional
        Stopping value for C. Default is 10.
    step : float > 0, optional
        Step value for C. The function tests different values of C in the 
        range [start, stop] with step value equal to 'step'. Default is 1.
    delay : bool, optional
        Flag to start the chart after a delay randomly selected from the
        interval [0, wdw_lengt]. Default is True. 
        This flag should be activated only for noisy data that are quickly 
        detected by the chart. 
    L_minus :  float, optional
        Value for the negative control limit. Default is None. 
        When None, L_minus = - L_plus. 
    k : float, optional
        The allowance parameter. The default is None. 
        When None, k = delta/2 (optimal formula for normal data).
    n : int > 0, optional      
        Number of training and validating instances. This value is 
        typically large. Default is 36000.
    n_series : int > 0, optional
        Length of the resampled series (by the block bootstrap procedure).
        Default is 500. 
    epsilon : float, optional
        Parameter of the svr, which represents the approximation accuracy. 
        Default is 0.001.
    block_length :  int > 0, optional
        The length of the blocks. Default is None. 
        When None, the length is computed using an optimal formula. 
    BB_method : str, optional
       String that designates the block boostrap method chosen for sampling data. 
       Values for the string should be selected among: 
       'MBB': moving block bootstrap
       'NBB': non-overlapping block bootstrap
       'CBB': circular block bootstrap
       'MABB': matched block bootstrap
       Default is 'MBB'.
    confusion : bool, optional 
        Flag to show the confusion matrix (measure of the classification accuracy, 
        class by class). Default is True.  
    verbose : bool, optional    
        Flag to print infos about C. Default is True.
          
    Returns 
    ------
    min_MAPE : float > 0
        The value of C that minimizes the MAPE (mean absolute percentage error).
    min_NRMSE : float > 0
        The value of C that minimizes the NRMSE (normalized root mean squarred 
        error).
    max_accuracy : float > 0
        The value of C that maximizes the accuracy.
    """   
    assert BB_method in ['MBB', 'NBB', 'CBB', 'MABB'], "Undefined block bootstrap procedure"
    if BB_method == 'MBB': 
        blocks = bb.MBB(data, block_length)    
    elif BB_method == 'NBB':
        blocks = bb.NBB(data, block_length) 
    elif BB_method == 'CBB':
        blocks = bb.CBB(data, block_length) 
                    
    if 'blocks' in locals():
        n_blocks = int(np.ceil(n_series/blocks.shape[1]))
    
    wdw_length = int(np.ceil(wdw_length)) #should be integer
    
    n = int(n)
    assert n > 0, "n must be strictly positive"
    if n % 3 == 2: #n should be multiple of 3
        n += 1
    if n % 3 == 1: 
        n += 2
        
    if L_minus is None:
        L_minus = -L_plus
    if k is None:
        k = delta_min/2
        
    sign = 1
    n_test = int(n/5) #n testing instances
    n_train = n - n_test #n training instances
    
    n_C = int(np.ceil((stop-start)/step))
    MAPE = np.zeros((n_C)); NRMSE = np.zeros((n_C)); accuracy = np.zeros((n_C))
    count = 0
    C_values = np.arange(start, stop, step)
    for C in np.arange(start, stop, step):
        ### training
        input_train = np.zeros((n_train, wdw_length))
        size_train = np.zeros((n_train))
        form_train = np.zeros((n_train))
        rnd = halfnorm(scale=scale).rvs(size=n_train) + delta_min #size of shifts
        for b in range(0, n_train-2, 3):
            
            shift = rnd[b]*sign
            if BB_method == 'MABB': 
                series = bb.resample_MatchedBB(data, block_length, n=n_series)
            else:
                series = resample(blocks, replace=True, n_samples=n_blocks).flatten()[:n_series]
            if delay:
                delay = np.random.randint(wdw_length/2) #simulate a random delay
            else: 
                delay = 0
            
            for rnd_form in range(3):
        
                boot = np.copy(series)
                
                if rnd_form == 0:         
                    boot[wdw_length:] = boot[wdw_length:] + shift
                    form_train[b] = 0
                elif rnd_form == 1:
                    power = np.random.uniform(1.5,2)
                    boot = shift/(n_series) * (np.arange(0,n_series)**power) + boot
                    form_train[b] = 1
                else:
                    eta = np.random.uniform(np.pi/(wdw_length), 3*np.pi/wdw_length)
                    boot = np.sin(eta*np.pi*np.arange(n_series))*shift*boot
                    form_train[b] = 2
                
                size_train[b] = shift
                
                input_plus = boot[wdw_length:wdw_length*2]
                C_plus = np.zeros((n_series, 1))
                for i in range(wdw_length + delay, n_series): #start the monitoring after random delay 
                    C_plus[i] = max(0, C_plus[i-1] + boot[i] - k)
                    if C_plus[i] > L_plus:
                        input_plus = boot[i+1-wdw_length:i+1] 
                        break 
                    
                input_minus = boot[wdw_length:wdw_length*2]
                C_minus = np.zeros((n_series, 1))       
                for j in range(wdw_length + delay, n_series):
                    C_minus[j] = min(0, C_minus[j-1] + boot[j] + k)
                    if C_minus[j] < L_minus:
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
            if BB_method == 'MABB': 
                series = bb.resample_MatchedBB(data, block_length, n=n_series)
            else:
                series = resample(blocks, replace=True, n_samples=n_blocks).flatten()[:n_series]
            if delay:
                delay = np.random.randint(wdw_length/2) #simulate a random delay
            else: 
                delay = 0
            
            for rnd_form in range(3):
                
                boot = np.copy(series)
                
                if rnd_form == 0:
                    boot[wdw_length:] = boot[wdw_length:] + shift
                    form_test[b] = 0
                elif rnd_form == 1:
                    power = np.random.uniform(1.5,2)
                    boot = shift/(n_series) * (np.arange(0,n_series)**power) + boot
                    form_test[b] = 1
                else:
                    eta = np.random.uniform(np.pi/(wdw_length), 3*np.pi/wdw_length)
                    boot = np.sin(eta*np.pi*np.arange(n_series))*shift*boot
                    form_test[b] = 2
                label_test[b] = shift
                
                input_plus = boot[wdw_length:wdw_length*2]
                C_plus = np.zeros((n_series, 1))
                for i in range(wdw_length + delay, n_series):
                    C_plus[i] = max(0, C_plus[i-1] + boot[i] - k)
                    if C_plus[i] > L_plus:
                        input_plus = boot[i+1-wdw_length:i+1] 
                        break 
                
                input_minus = boot[wdw_length:wdw_length*2]
                C_minus = np.zeros((n_series, 1))       
                for j in range(wdw_length + delay, n_series):
                    C_minus[j] = min(0, C_minus[j-1] + boot[j] + k)
                    if C_minus[j] < L_minus:
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
        
        #regressor
        MAPE[count] = (1/len(label_pred)) * sum(np.abs((label_test - label_pred)/label_test))*100
        NRMSE[count] = np.sqrt(sum((label_test - label_pred)**2) / sum(label_test**2))
        #classifier
        accuracy[count] = sum(label_pred_clf == form_test)*100 / len(label_pred_clf)
         
        ### compute the confusion matrix 
        if confusion : 
            class_names = ['jump', 'drift', 'oscill.']
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
        
        count += 1
           
    min_MAPE = C_values[np.argmin(MAPE)]
    min_NRMSE = C_values[np.argmin(NRMSE)] 
    max_accuracy = C_values[np.argmax(accuracy)]
    
    if verbose:
        print('C value that minimizes the MAPE:', min_MAPE)
        print('C value that minimizes the NRMSE:', min_NRMSE)
        print('C value that maximizes the accuracy:', max_accuracy)
    
    return min_MAPE, min_NRMSE, max_accuracy


#========================================================================
#========================================================================

def interpolate1d(X):
    """ 
    Interpolates intermediate NaN values in an array.
    
    Parameters
    ---------
    X : 1D-array
        Array to be interpolated.
    
    Returns
    -------
    X_int : 1D-array
        New array with interpolated values.
    """
    ind = np.arange(X.shape[0])
    ind_not_nans = np.where(~np.isnan(X))
    f = interpolate.interp1d(ind[ind_not_nans], X[ind_not_nans], bounds_error=False)
    X_int = np.where(np.isfinite(X), X, f(ind))
    return X_int

def fill_nan(x):
    """
    Fills first NaN values by the first value that is not NaN and interpolates 
    intermediate NaNs in an initial array. 
    
    Parameters
    ----------
    x : 2D-array
        Array where each row will be interpolated. 
    
    Returns
    -------
    new_x : 2D-array
        New array with interpolated values.
    ind : 1D-array  
        Array which contains the indexes of non-NaN rows (i.e. rows that contain
        values other than NaNs).
    """
    (n_rows, wdw) = x.shape
    new_x = np.zeros((n_rows,wdw)); new_x[:] = np.nan
    for i in range(n_rows):
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
