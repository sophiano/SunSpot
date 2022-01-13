# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:27:18 2020

@author: sopmathieu

This file contains a class that can be used to preprocess the data. 

The class contains functions to remove a general 'level' of the data, to 
automatically select a set of in-control series and to standardise the 
data.

"""

import pickle
from zipfile import ZipFile
import numpy as np 
from scipy.stats import iqr, skew, kurtosis
from sklearn.cluster import MeanShift, estimate_bandwidth,KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter1d
from kneed import KneeLocator
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 6.0)

from SunSpot import errors as err


class PreProcessing():
    """
    A class used to encapsulate the preprocessing of the series. 

    Attributes
    ----------
    obs : 2D-array
        A matrix representing a panel of time-series to be monitored.
        (rows: time, columns: series)

    Methods
    -------
    level_removal(level=True, wdw=None)
        Removes the intrinsic levels of the processes.
    clustering(x, method, ref, nIC_inf, nIC):
        Clusters the data into two groups using different clustering methods.
    selection_pools(method='kmeans', ref=0, nIC_inf=0.25, nIC1=None, nIC2=None):
        Selects the IC pool.
    outliers_removal(k=1):
        Remove the outliers from the IC series at each time. 
    standardisation(K):
        Standardizes the observations with time-dependent parameters. 
        
    """
    
    def __init__(self, obs):
        """
        Parameters
        ----------
        obs : 2D-array
            A matrix representing a panel of time-series to be monitored.
            (rows: time, columns: series)
        """
        assert np.ndim(obs) == 2, "Input data must be a 2D array"
        self.obs = obs        
          
    #================================================================================
    #================================================================================
    def level_removal(self, level=True, wdw=None):
        """
        Removes the intrinsic levels of the processes.
        
        This function applies a smoothing process in time by a moving-average
        (MA) filter on each individual series. Then, the smoothed series are
        substracted from the initial series to remove the levels of the 
        processes.
        
        Parameters
        ---------
        level: bool, optional 
            Flag to remove the level of the stations.
            When True, wdw should be set to integer. The default is True. 
        wdw : int, optional
            Length of the MA window length, expressed in days. 
            The default is None.
        """
        self.data = np.copy(self.obs)
        if level: 
            (n_obs, n_stations) = self.data.shape
            
            assert wdw is not None, "Window length must be a postive integer"
            assert wdw > 0, "Window length must be a postive integer"
            wdw = np.round(wdw)
            if wdw % 2 == 1:
                halfwdw = int((wdw - 1)/2)
            else:
                wdw += 1
                halfwdw = int((wdw - 1)/2)
                
            # for i in range(n_stations):
            #     m = np.nanmean(self.data[:,i])
            #     ma = np.ones((n_obs))*m
            #     for j in range(halfwdw, n_obs - halfwdw):
            #         ma[j] = np.nanmean(self.data[j-halfwdw :j+halfwdw+1,i])
            #     ma[np.isnan(ma)] = m
            #     self.data[:,i] = self.data[:,i] - ma
                
            for i in range(n_stations):
                m = np.nanmean(self.data[:,i])
                ma = np.ones((n_obs))*m
                for j in range(n_obs):
                    if j < halfwdw: #beginning
                        ma[j] = np.nanmean(self.data[0:wdw,i])
                    elif j >= halfwdw and j < n_obs - halfwdw: #middle
                        ma[j] = np.nanmean(self.data[j-halfwdw :j+halfwdw+1,i])
                    else: #end
                        ma[j] = np.nanmean(self.data[n_obs-wdw :n_obs,i])
                ma[np.isnan(ma)] = m
                self.data[:,i] = self.data[:,i] - ma
            
    #================================================================================
    #================================================================================
    def clustering(self, x, method, ref, nIC_inf, nIC):
        """
        Clusters the data into two groups using different clustering methods.
        
        A robust version of the mean-squarred error (mse) is calculated for each series
        of the panel. Then, a clustering algorithm groups the mse into two groups: 
        the in-control (IC) or stable group and out-of-control (OC) or unstable group.
        
    
        Parameters
        ----------
        x : 2D-array
           A matrix representing a panel of time-series to be clustered.
           (rows: time, columns: series) 
        ref : float 
           The 'true' reference of the panel. If None, the median of the network is 
           used as a reference. 
        method : str 
             String that designates the clustering method that will be used.
             Values for 'method' should be selected from:
            'leftmedian': all series whose mse is inferior to the median of the mse 
            are selected ;
            'kmeans': K-means clustering ;
            'agg': agglomerative hierarchical clustering ;
            'ms': mean-shift clustering ;
            'dbscan': DBSCAN clustering ;
            'gmm': gaussian mixture models used for clustering ;
            'fixed': selects the first 'nIC' most stable processes
        nIC_inf : float in [0,1]
            Lower bound, expressed in percentage, for the number of IC processes.
            Typical value are 0.25 or 0.3. 
            The number of IC series are computed to be in range [nIC_inf*n_series ; 
            (1-nIC_inf)*n_series], where n_series designates the number of 
            diffrent series. 
        nIC : int 
          Number of IC stations selected (if method='fixed').
        """
        if ref is None: 
            ref = np.nanmedian(x)
            
        (n_obs, n_stations) = x.shape
        #compute the mse
        mse = np.zeros(n_stations)
        for i in range(n_stations):
            mse[i] = (np.nanmedian(x[:,i]-ref))**2 + iqr(x[:,i], nan_policy='omit') 
            #mse[i] = (np.nanmean(x[:,i]-ref))**2 + np.nanvar(x[:,i]) 
        ordered_stations = np.argsort(mse)
            
        ### Different methods:
        ### All series that are more stable than the median 
        if method == 'leftmedian':
            pool = list(np.where(mse < np.median(mse))[0])
            
        ###K-means
        if method == 'kmeans':
            mse_init = np.copy(mse)
            kmeans = KMeans(n_clusters=2, init='k-means++', random_state=100).fit(mse.reshape(-1,1))
            labels_km = kmeans.labels_
            best = labels_km[ordered_stations[0]] #label of the most stable series
            n_best = len(labels_km[labels_km == best])
            worst = labels_km[ordered_stations[-1]] #label of the most stable series
            n_worst = len(labels_km[labels_km == worst])
            pool = [i for i in range(n_stations) if labels_km[i] == best]
            
            #too much series into OC group
            while n_worst > (1-nIC_inf)*n_stations:
                ind_bad = np.where(labels_km==worst)[0]
                mse = mse[ind_bad]
                ordered_array = np.argsort(mse)
                kmeans = KMeans(n_clusters=2, init='k-means++', random_state=100).fit(mse.reshape(-1,1))
                labels_km = kmeans.labels_
                worst = labels_km[ordered_array[-1]] #label of the worst series
                n_worst = len(labels_km[labels_km == worst])
                pool = [i for i in range(n_stations) if mse_init[i] not in mse[labels_km == worst]]
                
            #too much series into IC group
            while n_best > (1-nIC_inf)*n_stations: 
                ind_good = np.where(labels_km==best)[0]
                mse = mse[ind_good]
                ordered_array = np.argsort(mse)
                kmeans = KMeans(n_clusters=2, init='k-means++', random_state=100).fit(mse.reshape(-1,1))
                labels_km = kmeans.labels_
                best = labels_km[ordered_array[0]] #label of the most stable series
                n_best = len(labels_km[labels_km == best])
                pool = [i for i in range(n_stations) if mse_init[i] in mse[labels_km == best]]
            
        ####################################
        ### Agglomerative clustering
        if method == 'agg':
            mse_init = np.copy(mse)
            agg = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward').fit(mse.reshape(-1, 1))
            labels_agg = agg.labels_
            best = labels_agg[ordered_stations[0]]
            n_best = len(labels_agg[labels_agg == best])
            worst = labels_agg[ordered_stations[-1]] 
            n_worst = len(labels_agg[labels_agg == worst])
            pool = [i for i in range(n_stations) if labels_agg[i] == best]
            
            #too much series into OC group
            while n_worst > (1-nIC_inf)*n_stations:
                ind_bad = np.where(labels_agg==worst)[0]
                mse = mse[ind_bad]
                ordered_array = np.argsort(mse)
                agg = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward').fit(mse.reshape(-1, 1))
                labels_agg = agg.labels_
                worst = labels_agg[ordered_array[-1]] #label of the worst series
                n_worst = len(labels_agg[labels_agg == worst])
                pool = [i for i in range(n_stations) if mse_init[i] not in mse[labels_agg == worst]]
                
            #too much series into IC group
            while n_best > (1-nIC_inf)*n_stations: 
                ind_good = np.where(labels_agg==best)[0]
                mse = mse[ind_good]
                ordered_array = np.argsort(mse)
                agg = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward').fit(mse.reshape(-1, 1))
                labels_agg = agg.labels_
                best = labels_agg[ordered_array[0]] #label of the most stable series
                n_best = len(labels_agg[labels_agg == best])
                pool = [i for i in range(n_stations) if mse_init[i] in mse[labels_agg == best]]
                
            
        ###########################################"
        ### Gaussian mixture models clustering 
        if method == 'gmm': 
            mse_init = np.copy(mse)
            gmm = GaussianMixture(n_components=2).fit(mse.reshape(-1, 1))
            labels_gmm = gmm.predict(mse.reshape(-1, 1))
            best = labels_gmm[ordered_stations[0]]
            n_best = len(labels_gmm[labels_gmm == best])
            worst = labels_gmm[ordered_stations[-1]] 
            n_worst = len(labels_gmm[labels_gmm == worst])
            pool = [i for i in range(n_stations) if labels_gmm[i] == best]
            
            #too much series into OC group
            while n_worst > (1-nIC_inf)*n_stations:
                ind_bad = np.where(labels_gmm==worst)[0]
                mse = mse[ind_bad]
                ordered_array = np.argsort(mse)
                gmm = GaussianMixture(n_components=2).fit(mse.reshape(-1, 1))
                labels_gmm = gmm.predict(mse.reshape(-1, 1))
                worst = labels_gmm[ordered_array[-1]] #label of the worst series
                n_worst = len(labels_gmm[labels_gmm == worst])
                pool = [i for i in range(n_stations) if mse_init[i] not in mse[labels_gmm == worst]]
                
            #too much series into IC group
            while n_best > (1-nIC_inf)*n_stations: 
                ind_good = np.where(labels_gmm==best)[0]
                mse = mse[ind_good]
                ordered_array = np.argsort(mse)
                gmm = GaussianMixture(n_components=2).fit(mse.reshape(-1, 1))
                labels_gmm = gmm.predict(mse.reshape(-1, 1))
                best = labels_gmm[ordered_array[0]] #label of the most stable series
                n_best = len(labels_gmm[labels_gmm == best])
                pool = [i for i in range(n_stations) if mse_init[i] in mse[labels_gmm == best]]
                
        ###################################
        ### Mean-shifts
        if method == 'ms':
            bandwidth = estimate_bandwidth(mse.reshape(-1, 1), quantile=0.2, n_samples=len(mse))
            ms = MeanShift(bandwidth).fit(mse.reshape(-1, 1))
            labels_ms = ms.labels_
            best = [] 
            best.append(labels_ms[ordered_stations[0]]) #find label of most stable station
            for i in range(1, n_stations):
                n = len(labels_ms[np.isin(labels_ms, best)]) #number of stations with label included into 'best'
                new_label = labels_ms[ordered_stations[i]] #label of next stable station
                if n < nIC_inf*n_stations and new_label not in best:
                    best.append(new_label) 
            pool = [i for i in range(n_stations) if labels_ms[i] in best]
    
        ####################################
        ### Density-based clustering
        if method == 'dbscan':
            dbscan = DBSCAN(eps=0.01, min_samples=2).fit(mse.reshape(-1, 1))
            labels_db = dbscan.labels_
            best = [] 
            best.append(labels_db[ordered_stations[0]]) 
            for i in range(1, n_stations):
                n = len(labels_db[np.isin(labels_db, best)])
                new_label = labels_db[ordered_stations[i]] 
                if n < nIC_inf*n_stations and new_label not in best:
                    best.append(new_label)
            pool = [i for i in range(n_stations) if labels_db[i] in best]
     
        if method == 'fixed': 
            nIC = int(nIC)
            assert nIC > 0, "nIC should be stricly positive!"
            pool = ordered_stations[:nIC]
                
        return pool
        
    #================================================================================
    #================================================================================
    def selection_pools(self, method='kmeans', ref=0, nIC_inf=0.25, nIC=None):
        """
        Selects the IC pool.
        
        This function automatically selects the IC processes 
        of the pool using different clustering methods. 
        
        Parameters
        ----------
        method : str, optional
             String that designates the clustering method that will be used.
             Values for 'method' should be selected from:
            'leftmedian': all series whose mse is inferior to the median of the mse 
            are selected ;
            'kmeans': K-means clustering ;
            'agg': agglomerative hierarchical clustering ;
            'ms': mean-shift clustering ;
            'dbscan': DBSCAN clustering ;
            'gmm': gaussian mixture models used for clustering ;
            'fixed': selects the first 'nIC' most stable processes
            The default is 'kmeans'.
        ref : float, optional
           The 'true' reference of the panel. If None, the median of the network is 
           used as a reference. The default is zero.
        nIC_inf : float in [0,1], optional
            Lower bound, expressed in percentage, for the range of IC processes number.
            The number of IC series are computed to be in range [nIC_inf*n_series ; 
            (1-nIC_inf)*n_series], where n_series designates the number of 
            diffrent series. Default is 0.25.
        nIC : int, optional
             Number of IC stations of the pool (if method='fixed').
             Default is None.  
        """
        (n_obs, n_stations) = self.data.shape
        if nIC is None: 
            nIC = np.round(n_stations/2)
        self.pool = self.clustering(self.data, method, ref, nIC_inf, nIC)
                        
    #================================================================================
    #================================================================================
    def outliers_removal(self, k=1):
        """
        Removes the outliers from the IC series at each time. 
        
        This function removes at each time the observations of the IC series
        that do not fall into a multiple of the standard deviation 
        around the mean.
        
        Parameters
        ---------
        k : float, optional
           Multiple of the standard deviation that defines an outlier. 
           Typical values are 1, 1.5 or 2 (depending of the data noise) but should 
           be larger than or equal to 1 (otherwise too many IC data are suppressed).
           Default is 1. 
        """
        (n_obs, n_stations) = self.data.shape
        self.dataIC = self.data[:,self.pool]
        for i in range(n_obs):
            avail = np.where(~np.isnan(self.data[i,:]))
            avail_IC = np.where(~np.isnan(self.dataIC[i,:]))
            #index = np.where((self.dataIC[i,:] > np.nanmedian(self.data[i,:]) + k*iqr(self.data[i,:], nan_policy='omit')) | (self.dataIC[i,:]<np.nanmedian(self.data[i,:]) - k*iqr(self.data[i,:], nan_policy='omit')))
            index = np.where((self.dataIC[i,:] > np.nanmean(self.data[i,:]) + k*np.nanstd(self.data[i,:])) | (self.dataIC[i,:] < np.nanmean(self.data[i,:]) - k*np.std(self.data[i,:])))
            if ((index is not None) and len(avail[0]) > len(avail_IC[0])*1.1):
                self.dataIC[i,index] = np.nan
        
    #================================================================================
    #================================================================================
    def standardisation(self, K):
        """ 
        Standardises the observations with time-dependent parameters. 
        
        The time-varying mean and the variance are computed across the IC series 
        and along a small window of time using K nearest neighbors
        (KNN) estimators.
        
        Parameters
        ----------
        K :  int > 0
            Number of nearest neighbors. 
        """
        assert K > 0, "K must be a postive integer"
        K = np.round(K)
        (n_obs, n_stations) = self.data.shape
        mu_0 = np.nanmean(self.dataIC); sigma_0 = np.nanstd(self.dataIC)
        mu_t = np.ones((n_obs))*mu_0; sigma_t = np.ones((n_obs))*sigma_0
        for t in range(n_obs):
            c = 0
            mylist = [] 
            for i in range(n_obs):
                if c < K:
                    if t+i < n_obs:
                        data_tb = self.dataIC[t+i,:]
                        nb = data_tb[~np.isnan(data_tb)]
                        mylist.extend(nb)
                        c += len(nb) 
                    if t-i >= 0 and i > 0:
                        data_ta = self.dataIC[t-i,:]
                        na = data_ta[~np.isnan(data_ta)]
                        mylist.extend(na)
                        c += len(na) 
                if ((c >= K and len(mylist) > 0) or (i == n_obs - 1 and len(mylist) > 0)):  
                    mu_t[t] = np.mean(mylist)
                    sigma_t[t] = np.std(mylist)
                    break
            
        sigma_t[sigma_t == 0] = sigma_0 #don't divide by zero         
        for i in range(n_stations):
            self.data[:,i] = (self.data[:,i] - mu_t)/sigma_t
            
        for i in range(len(self.pool)):
            self.dataIC[:,i] = (self.dataIC[:,i] - mu_t)/sigma_t




def standardisation(data, dataIC, K):
    """ 
    Standardises the input data with time-dependent parameters. 
    
    The time-varying mean and the variance are computed across the IC series 
    and along a small window of time using K nearest neighbors (KNN) 
    regression method.
    
    Parameters
    ----------
    data : 2D-array
        A matrix representing a panel of time-series to be standardised.
        (rows: time, columns: series)
    dataIC : 2D-array
        A subset of the panel containing stable time-series to be standardised.
        The time-varying mean and variance are estimated on this stable subset.
    K :  int > 0
        Number of nearest neighbors. 
    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    assert np.ndim(dataIC) == 2, "Input data must be a 2D array"
    assert K > 0, "K must be a postive integer"
    K = np.round(K)
    
    (n_obs, n_stations) = data.shape
    mu_0 = np.nanmean(dataIC); sigma_0 = np.nanstd(dataIC)
    mu_t = np.ones((n_obs))*mu_0; sigma_t = np.ones((n_obs))*sigma_0
    for t in range(n_obs):
        c = 0
        mylist = [] 
        for i in range(n_obs):
            if c < K:
                if t+i < n_obs:
                    data_tb = dataIC[t+i,:]
                    nb = data_tb[~np.isnan(data_tb)]
                    mylist.extend(nb)
                    c += len(nb) 
                if t-i >= 0 and i > 0:
                    data_ta = dataIC[t-i,:]
                    na = data_ta[~np.isnan(data_ta)]
                    mylist.extend(na)
                    c += len(na) 
            if ((c >= K and len(mylist) > 0) or (i == n_obs - 1 and len(mylist) > 0)):  
                mu_t[t] = np.mean(mylist)
                sigma_t[t] = np.std(mylist)
                break
        
    sigma_t[sigma_t == 0] = sigma_0 #don't divide by zero 
        
    data_stn = np.copy(data)
    dataIC_stn = np.copy(dataIC)
    for i in range(n_stations):
        data_stn[:,i] = (data_stn[:,i] - mu_t)/sigma_t
        
    for i in range(dataIC.shape[1]):
        dataIC_stn[:,i] = (dataIC_stn[:,i] - mu_t)/sigma_t  
    
    return data_stn, dataIC_stn


def choice_K(data, dataIC, plot=True, start=200, stop=10000, step=200):
    """ 
    Computes an appropriate value for the number of nearest neighbors (K).
    
    This function evaluates the standard deviation of the standardised data 
    for different values of K. Then, the value where the standard deviation 
    stabilises (the 'knee' of the curve) is selected using a 'knee' 
    locator. 
    
    Parameters
    ----------
    data : 2D-array
        A matrix representing a panel of time-series to be standardised
        by K-NN regression method. (rows: time, columns: series)
    dataIC : 2D-array
        A subset of the panel containing stable time-series to be standardised.
        The time-varying mean and variance are estimated on this stable subset. 
    plot : bool, optional 
        Flag to plot the mean and the standard deviation of the standardised 
        data as a function of K. Default is True.
    start : int, optional
        Lower value for K. The numbers of nearest neighbours are tested in
        the range [start, stop]. Default is 200.
    stop : int, optional
        Upper value for K. The numbers of nearest neighbours are tested in
        the range [start, stop]. Default is 10000.
    step : int, optional
        Step value for K. The numbers of nearest neighbours are tested in 
        the range [start, stop], with step equal to 'step'.
        Default is 200.
        
    Returns
    ------
    K : int>0
       The selected number of nearest neighbors.
    """
    start = int(start); stop = int(stop); step = int(step)
    n = int(np.ceil((stop - start)/step))
    mean_data = np.zeros((n)); std_data = np.zeros((n))
    c=0 ; x = np.arange(start,stop,step)
    for K in range(start, stop, step):
        #standardise the data with K-NN estimators
        data_stn = standardisation(data, dataIC, K)[0]
        mean_data[c] = np.nanmean(data_stn)
        std_data[c] = np.nanstd(data_stn)
        c += 1
        
        
    y = std_data[:len(x)]
    y_smooth = gaussian_filter1d(y, 1)
    coef = np.polyfit(x, y_smooth, deg=1)
    coef_curve = np.polyfit(x, y_smooth, deg=2)
    if coef_curve[0] < 0: 
        curve = 'concave'        
    else: 
        curve = 'convex'
    if coef[0] < 0: #slope is positive
        direction = 'decreasing'
    else: #slope is negative
        direction = 'increasing'
    kn = KneeLocator(x, y_smooth, curve=curve, direction=direction)
    K = kn.knee 
    
    if plot: 
        plt.rcParams['figure.figsize'] = (8.0, 8.0)
        plt.rcParams['font.size'] = 14
        plt.plot(x, mean_data[:len(x)], marker='o');  plt.xlabel('K'); plt.ylabel('mean')
        plt.title('Mean of the data as function of K')
        if K is not None:
            plt.axvline(x=K, color='orange', linestyle='--', label='K selected \n (knee)')
            plt.legend()
        plt.show()
        plt.plot(x, std_data[:len(x)], marker='o'); plt.xlabel('K'); plt.ylabel('std')
        plt.title('Std of the data as function of K')
        if K is not None:
            plt.axvline(x=K, color='orange', linestyle='--', label='K selected \n (knee)')
            plt.legend()
        plt.show()
    
    return K

if __name__ == "__main__":

       
    ### load data
    with open('data/data_1981', 'rb') as file: # load all stations
         my_depickler = pickle.Unpickler(file)
         Ns = my_depickler.load() #number of spots
         Ng = my_depickler.load() #number of sunspot groups
         Nc = my_depickler.load() #Ns+10Ng
         station_names = my_depickler.load() #index of the stations
         time = my_depickler.load() #time
         
    ### add new data to station KS
    data_ks = np.loadtxt('data/kisl_wolf.txt', usecols=(0,1,2,3), skiprows=1)
    Nc_ks = data_ks[9670:23914,3]
    Nc[:,24] = Nc_ks
         
    ### Compute the long-term errors
    mu2 = err.long_term_error(Nc, period_rescaling=10, wdw=27)
        
    #discard stations with no values
    ind_nan = []
    for i in range(mu2.shape[1]):
        if not np.all(np.isnan(mu2[:,i])): 
            ind_nan.append(i)
    mu2 = mu2[:,ind_nan]
    station_names = [station_names[i] for i in range(len(station_names)) if i in ind_nan]
    (n_obs, n_series) = mu2.shape
    
    #percentage of missing values (total)
    perc_missing = len(mu2[np.isnan(mu2)])/(n_obs*n_series) #76%
    
    ### Apply preprocessing
    dataNs = PreProcessing(mu2)  
    dataNs.level_removal(wdw=4000)  
    mu2_wht_level = dataNs.data
    data_nn = np.copy(mu2_wht_level)
    dataNs.selection_pools(method='kmeans')
    pool = np.array(dataNs.pool)
    pool_ind = [station_names[i] for i in range(n_series) if i in pool]
    dataNs.outliers_removal(k=1) 
    
    ### Choose K 
    #dataIC = dataNs.dataIC
    #data = dataNs.data
    #K_knee = choice_K(data, dataIC) #2400
        
    #scale = K_knee/len(pool) #7
    dataNs.standardisation(K=2400)
    dataIC = dataNs.dataIC
    data = dataNs.data
    
    #plot all data
    plt.hist(data[~np.isnan(data)], range=[-4,4], bins='auto', density=True, facecolor='b')  
    plt.title("Data")
    plt.text(2, 1, 'mean:' '%4f' %np.nanmean(data))
    plt.text(2, 0.8, 'std:' '%4f' %np.nanstd(data))
    plt.text(2, 0.6, 'skewness:' '%4f' %skew(data[~np.isnan(data)]))
    plt.text(2, 0.4, 'kurtosis:' '%4f' %kurtosis(data[~np.isnan(data)]))
    plt.axis([-4, 4, 0, 1.5])
    plt.grid(True)
    plt.show()
        
    #plot the IC data
    plt.hist(dataIC[~np.isnan(dataIC)], range=[-4,4], bins='auto', density=True, facecolor='b')  
    plt.title("Data IC")
    plt.text(2, 1, 'mean:' '%4f' %np.nanmean(dataIC))
    plt.text(2, 0.8, 'std:' '%4f' %np.nanstd(dataIC))
    plt.text(2, 0.6, 'skewness:' '%4f' %skew(dataIC[~np.isnan(dataIC)]))
    plt.text(2, 0.4, 'kurtosis:' '%4f' %kurtosis(dataIC[~np.isnan(dataIC)]))
    plt.axis([-4, 4, 0, 1.5])
    plt.grid(True)
    plt.show()
    
    #=====================================================
    ## export data 
    #=====================================================
    
    ## serialized format 
    # with open('Nc_27', 'wb') as file: 
    #     my_pickler = pickle.Pickler(file)
    #     my_pickler.dump(data) #data (IC and OC) with deviations
    #     my_pickler.dump(time) #time
    #     my_pickler.dump(station_names) #index of the stations
    #     my_pickler.dump(dataIC) #IC series without deviations (for chart design)
    #     my_pickler.dump(pool) #index of series in the pool 
    #     my_pickler.dump(mu2) #long-term error with level (eh)
    #     my_pickler.dump(data_nn) #long-term error without levels (mu2)
    
    # ### save files into txt format
    # np.savetxt('data.txt', data, delimiter=',')
    # np.savetxt('time.txt', time, delimiter=',')
    # f = open('station_names.txt','w')
    # for ele in station_names:
    #     f.write(ele+'\n')
    # f.close()
    # np.savetxt('dataIC.txt', dataIC, delimiter=',')
    # np.savetxt('pool.txt', pool, delimiter=',')
    # np.savetxt('mu2.txt', mu2, delimiter=',')
    # np.savetxt('data_nn.txt', data_nn, delimiter=',')
    
    # ## create a ZipFile object
    # zipObj = ZipFile('Nc_27.zip', 'w')
    # zipObj.write('data.txt')
    # zipObj.write('time.txt')
    # zipObj.write('station_names.txt')
    # zipObj.write('dataIC.txt')
    # zipObj.write('pool.txt')
    # zipObj.write('mu2.txt')
    # zipObj.write('data_nn.txt')
    # zipObj.close()
    
    
    #======================================================================
    #=====================================================================
    
    ### Compute the long-term errors
    mu2 = err.long_term_error(Nc, period_rescaling=10, wdw=365)
        
    #discard stations with no values
    ind_nan = []
    for i in range(mu2.shape[1]):
        if not np.all(np.isnan(mu2[:,i])): 
            ind_nan.append(i)
    mu2 = mu2[:,ind_nan]
    station_names = [station_names[i] for i in range(len(station_names)) if i in ind_nan]
    (n_obs, n_series) = mu2.shape
    
    #percentage of missing values (total)
    perc_missing = len(mu2[np.isnan(mu2)])/(n_obs*n_series)#68, less MVs
    
    ### Apply preprocessing
    dataNs = PreProcessing(mu2)  
    dataNs.level_removal(wdw=4000) #11 ans
    mu2_wht_level = dataNs.data
    data_nn = np.copy(mu2_wht_level)
    dataNs.selection_pools(method='kmeans')
    pool = np.array(dataNs.pool)
    pool_ind = [station_names[i] for i in range(n_series) if i in pool]
    dataNs.outliers_removal(k=1) 
        
    ### Choose K 
    #dataIC = dataNs.dataIC
    #data = dataNs.data 
    #K_knee = choice_K(data, dataIC) #4600 
    
    #standardisation
    #scale = K_knee/len(pool) #25~26
    dataNs.standardisation(K=4600)
    dataIC = dataNs.dataIC
    data = dataNs.data
        
    #plot all data
    plt.hist(data[~np.isnan(data)], range=[-4,4], bins='auto', density=True, facecolor='b')  
    plt.title("Data")
    plt.text(2, 1, 'mean:' '%4f' %np.nanmean(data))
    plt.text(2, 0.8, 'std:' '%4f' %np.nanstd(data))
    plt.text(2, 0.6, 'skewness:' '%4f' %skew(data[~np.isnan(data)]))
    plt.text(2, 0.4, 'kurtosis:' '%4f' %kurtosis(data[~np.isnan(data)]))
    plt.axis([-4, 4, 0, 1.5])
    plt.grid(True)
    plt.show()
        
    #plot the IC data
    plt.hist(dataIC[~np.isnan(dataIC)], range=[-4,4], bins='auto', density=True, facecolor='b')  
    plt.title("Data IC")
    plt.text(2, 1, 'mean:' '%4f' %np.nanmean(dataIC))
    plt.text(2, 0.8, 'std:' '%4f' %np.nanstd(dataIC))
    plt.text(2, 0.6, 'skewness:' '%4f' %skew(dataIC[~np.isnan(dataIC)]))
    plt.text(2, 0.4, 'kurtosis:' '%4f' %kurtosis(dataIC[~np.isnan(dataIC)]))
    plt.axis([-4, 4, 0, 1.5])
    plt.grid(True)
    plt.show()
       
    ## export data 
    # with open('Nc_365', 'wb') as file: 
    #     my_pickler = pickle.Pickler(file)
    #     my_pickler.dump(data) #data (IC and OC) with deviations
    #     my_pickler.dump(time) #time
    #     my_pickler.dump(station_names) #index of the stations
    #     my_pickler.dump(dataIC) #IC series without deviations (for chart design)
    #     my_pickler.dump(pool) #index of series in the pool 
    #     my_pickler.dump(mu2) #long-term error
    #     my_pickler.dump(data_nn) #long-term error without levels

    # ### save files into txt format
    # np.savetxt('data.txt', data, delimiter=',')
    # np.savetxt('time.txt', time, delimiter=',')
    # f = open('station_names.txt','w')
    # for ele in station_names:
    #     f.write(ele+'\n')
    # f.close()
    # np.savetxt('dataIC.txt', dataIC, delimiter=',')
    # np.savetxt('pool.txt', pool, delimiter=',')
    # np.savetxt('mu2.txt', mu2, delimiter=',')
    # np.savetxt('data_nn.txt', data_nn, delimiter=',')
    
    # ## create a ZipFile object
    # zipObj = ZipFile('Nc_365.zip', 'w')
    # zipObj.write('data.txt')
    # zipObj.write('time.txt')
    # zipObj.write('station_names.txt')
    # zipObj.write('dataIC.txt')
    # zipObj.write('pool.txt')
    # zipObj.write('mu2.txt')
    # zipObj.write('data_nn.txt')
    # zipObj.close()
