# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:35:31 2020

@author: sopmathieu
"""

#load useful packages
import numpy as np
import matplotlib.pyplot as plt
import SVR_SVC_training as SVM


def firstNonNan(x):
  """ Find first non NaN value in a 1D array. 
  Arguments: 
  x:    1D numpy array
  Return: 
  ind:   index of first non NaN value
  """
  for ind in range(len(x)):
    if not np.isnan(x[ind]):
      return ind

def CUSUM_monitoring(data, L, delta, wdw_length, clf, reg, k=None):
    """
    Apply the two-sided CUSUM chart on 'data' and compute the input vectors 
    after each alert.
    
    Parameters:
    data:       1D dataset containing only one series
    L:          control limit of the chart
    delta:      shift size
    wdw_length: length of the input vector
    clf:        trained classifier
    reg:        trained regressor
    k:          allowance parameter
    
    Return 
    form_plus:   predicted shift forms after positive alerts
    form_minus:  predicted shift forms after negative alerts
    size_plus:   predicted shift sizes after positive alerts
    size_minus:  predicted shift sizes after negative alerts
    C_plus:      values for the positivie chart statistic
    C_minus:     values for the negative chart statistic
    """
    n_obs = len(data)
    if k is None:
        k = delta/2
    
    length = len(data[~np.isnan(data)])
    
    input_minus = np.zeros((n_obs, wdw_length)); input_plus = np.zeros((n_obs, wdw_length))
    input_minus[:] = np.nan; input_plus[:] = np.nan
    flag_p = np.zeros((n_obs)); flag_m = np.zeros((n_obs))
    C_plus = np.zeros((n_obs)); C_minus = np.zeros((n_obs))
    
    for i in range(wdw_length, n_obs):
        
        ## CUSUM monitoring
        C_plus[i] = max(0, C_plus[i-1] + data[i] - k) #max(0,np.nan)=0
        C_minus[i] = min(0, C_minus[i-1] + data[i] + k)
        if C_plus[i] > L:
            flag_p[i] = 1
            input_plus[i,:] = data[i+1-wdw_length:i+1] 
        if C_minus[i] < -L:
            flag_m[i] = 1
            input_minus[i,:] = data[i+1-wdw_length:i+1]
    
   
    ## compute percentage of alerts
    oc_p = np.nonzero(flag_p)
    oc_m = np.nonzero(flag_m)
    #if alert both for pos and neg limits, count for only one alert
    oc_both = len(set(np.concatenate((oc_p[0], oc_m[0]))))
    #OC_perc = oc_both*100/N_obs #total period
    OC_perc = oc_both*100/length #observing period
    print(OC_perc) #print percentages of alerts 
        
    ## interpolate NaNs in input vectors
    input_minus_valid, ind_minus = SVM.fill_nan(input_minus)
    input_plus_valid, ind_plus = SVM.fill_nan(input_plus)
    
    ##apply classifier and regressor on (filled-up) input vectors
    size_minus = np.zeros((n_obs)); size_plus = np.zeros((n_obs))
    size_minus[:] = np.nan; size_plus[:] = np.nan
    form_minus = np.zeros((n_obs)); form_plus = np.zeros((n_obs))
    form_minus[:] = np.nan; form_plus[:] = np.nan
    if len(ind_minus)>0: #at least one value
        size_minus[ind_minus] = reg.predict(input_minus_valid)
        form_minus[ind_minus] = clf.predict(input_minus_valid)
    if len(ind_plus)>0:
        size_plus[ind_plus] = reg.predict(input_plus_valid)
        form_plus[ind_plus] = clf.predict(input_plus_valid)
    
    return (form_plus, form_minus, size_plus, size_minus, C_plus, C_minus)
    
    
    
def plot_performances(data, level, L, time, form_plus, form_minus, size_plus, 
                      size_minus, C_plus, C_minus, name, start=None, length=None):
    """
    Plot the performances of the chart. 
    The first panel represents the data with levels, the second shows the standardized 
    residuals (without levels), the third displays the CUSUM statistics
    and the last panel shows the predicted shift sizes and forms.
    
    Parameters:
    data:        1D dataset containing only one series
    level:       data with levels
    L:           control limit of the chart
    time:        array with the time
    form_plus:   predicted shift forms after positive alerts
    form_minus:  predicted shift forms after negative alerts
    size_plus:   predicted shift sizes after positive alerts
    size_minus:  predicted shift sizes after negative alerts
    C_plus:      values for the positivie chart statistic
    C_minus:     values for the negative chart statistic
    name:        names of the series
    start:       start of the period to show
    length:      length of the period to show
    """
    n_obs = len(data)
    if start is None:
        start = firstNonNan(data)
    if length is None:
        length = int(len(data)/4)
    stop = start + length  
    
    #colors
    colorInd_plus = np.where(~np.isnan(form_plus))[0][:]
    colorInd_minus = np.where(~np.isnan(form_minus))[0][:]
    color_graph_p = np.ones((n_obs))*3; color_graph_m = np.ones((n_obs))*3 
    color_graph_m[colorInd_minus] = form_minus[colorInd_minus]
    color_graph_p[colorInd_plus] = form_plus[colorInd_plus]
    
    #jumps
    jump_p = np.zeros((n_obs)); jump_p[:]=np.nan
    jump_m = np.zeros((n_obs)); jump_m[:]=np.nan
    jump_m[np.where(color_graph_m == 0)[0]] = size_minus[np.where(color_graph_m == 0)[0]]
    jump_p[np.where(color_graph_p == 0)[0]] = size_plus[np.where(color_graph_p == 0)[0]]
    
    #trends
    trend_p = np.zeros((n_obs)); trend_p[:] = np.nan
    trend_m = np.zeros((n_obs)); trend_m[:] = np.nan
    trend_m[np.where(color_graph_m == 1)[0]] = size_minus[np.where(color_graph_m == 1)[0]]
    trend_p[np.where(color_graph_p == 1)[0]] = size_plus[np.where(color_graph_p == 1)[0]]
    
    #oscillating shifts
    oscill_p = np.zeros((n_obs)); oscill_p[:] = np.nan
    oscill_m = np.zeros((n_obs)); oscill_m[:] = np.nan
    oscill_m[np.where(color_graph_m == 2)[0]] = size_minus[np.where(color_graph_m == 2)[0]]
    oscill_p[np.where(color_graph_p == 2)[0]] = size_plus[np.where(color_graph_p == 2)[0]]


    plt.rcParams['figure.figsize'] = (7.0, 10.0)
    plt.rcParams['font.size'] = 12
    
    fig = plt.figure()
    f1 = fig.add_subplot(4, 1, 1)
    plt.title("Monitoring in %s" %name)
    plt.ylabel('$\hat \widetilde{\eta}(i,t)$')
    plt.plot(time[start:stop], level[start:stop])
    plt.plot([time[start], time[stop]], [1, 1], 'k-', lw=2)
    f1.axes.get_xaxis().set_ticklabels([]) 
    f1.set_ylim([0,5])

    f2 = fig.add_subplot(4, 1, 2)
    plt.ylabel('$\hat \epsilon_{\hat \mu_2(i,t)}$')
    plt.plot(time[start:stop], data[start:stop])
    plt.plot([time[start], time[stop]], [0, 0], 'k-', lw=2)
    f2.set_ylim([-10,20]); 
    f2.axes.get_xaxis().set_ticklabels([]) 
    
    f3 = fig.add_subplot(4, 1, 3)
    plt.ylabel('CUSUM statistics')
    plt.plot(time[start:stop], np.sqrt(C_plus[start:stop]), c='tab:red', label='C+')
    plt.plot(time[start:stop], -np.sqrt(-C_minus[start:stop]), '--', c='tab:brown', label='C-')
    #draw horizontal line
    plt.plot([time[start], time[stop]], [np.sqrt(L), np.sqrt(L)], 'k-', lw=2.2)
    plt.plot([time[start], time[stop]], [-np.sqrt(L), -np.sqrt(L)], 'k-', lw=2.2)
    plt.legend(loc='upper right')
    f3.set_ylim([-25,25]); 
    f3.axes.get_xaxis().set_ticklabels([]) 
    
    f4 = fig.add_subplot(4, 1, 4)
    plt.ylabel('Shifts')
    plt.plot(time[start:stop], jump_m[start:stop], '--', c='tab:purple', label='jumps')
    plt.plot(time[start:stop], jump_p[start:stop], '--', c='tab:purple')
    plt.plot(time[start:stop], trend_m[start:stop],  c='tab:green', label='trends')
    plt.plot(time[start:stop], trend_p[start:stop],  c='tab:green')
    plt.plot(time[start:stop], oscill_m[start:stop], ':', c='tab:orange', label='oscill')
    plt.plot(time[start:stop], oscill_p[start:stop], ':', c='tab:orange')
    plt.plot([time[start], time[stop]], [0, 0], 'k-', lw=2)
    plt.legend(loc='upper right')
    f4.set_ylim([-10,20]); f4.set_xlim([time[start-20], time[stop+20]])
    if stop-start <4000:
        x_ticks = np.arange(np.round(time[start]), np.round(time[stop])+1, 1)
    else :
        x_ticks = np.arange(np.round(time[start]), np.round(time[stop])+1, 5)
    plt.xticks(x_ticks)
    plt.xlabel('year')
    plt.tick_params(axis='x', which='major')
    #plt.tight_layout() 
    plt.show()
    return fig 
