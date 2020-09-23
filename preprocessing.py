# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:27:18 2020

@author: sopmathieu
"""

import pickle
import numpy as np 
from sklearn.linear_model import LinearRegression
from scipy.stats import iqr, skew, kurtosis
from sklearn.cluster import MeanShift, estimate_bandwidth,KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from collections import Counter
import matplotlib.pyplot as plt
from operator import itemgetter


class PreProcessing():
    
    def __init__(self, obs, time, stationNames):
        self.obs = obs
        self.time = time
        self.stationNames = stationNames
        
    
    def rescaling(self, period_rescaling=8):
        """ 
        This function rescales the observations on the median of the panel using 
        piece-wise constant scaling-factors. These factors are computed using a 
        simple linear regression of the observations on the median without 
        intercept.
        
        Parameters:
        periodRescaling:   length of the period where the scaling-factors
                           are assumed to be constant, expressed in months
        """
        (nObs, nStations) = self.obs.shape
        month = int(365/12)
        step = int(period_rescaling*month)
        n = int(np.ceil(nObs/step))
        
        med = np.nanmedian(self.obs, axis=1)

        #### linear regression
        slopeOLSY = np.ones((n, nStations)) 
        for j in range(nStations):
            c = 0
            for i in range(0, n):
                X = self.obs[c:c+step,j]
                Y = med[c:c+step]
                ind = np.where(~np.isnan(X))
                Y = Y[ind].reshape(-1,1) 
                X = X[ind].reshape(-1,1)
                c += step
                if len(Y > 0) > 2 and len(X > 0) > 2:
                    reg = LinearRegression(fit_intercept=False).fit(X, Y)
                    slopeOLSY[i,j] = reg.coef_ #slope
            
        
        ### effective rescaling 
        obs_rescaled = np.zeros((nObs, nStations)) 
        for j in range(nStations):
            c = 0
            for i in range(0, n):
                if slopeOLSY[i,j] > 0 and not np.isnan(slopeOLSY[i,j]):
                    obs_rescaled[c:c+step,j] = self.obs[c:c+step,j] * slopeOLSY[i,j]
                c += step
        
        self.obs_rescaled = obs_rescaled
        print('Data Rescaled!')
        
    #########################################################
    def commonSignalRemoval(self, rescaling=True):
        """ 
        This function removes the common component of the processes. 
        
        Parameters:
        rescaling:   boolean, if true, the reference is computed on rescaled obs
        """
        (nObs, nStations) = self.obs.shape
        if rescaling: 
            Mt = np.round(np.nanmedian(self.obs_rescaled, axis=1))
            print('Long-term error with rescaling!')
        else: 
            Mt = np.round(np.nanmedian(self.obs, axis=1))
            print('Long-term error without rescaling!')
        
        ratio = np.zeros((nObs, nStations))
        ratio[:] = np.nan
        for i in range(nObs):
            if not np.isnan(Mt[i]) and Mt[i] > 0:
                ratio[i,:] = self.obs[i,:]/Mt[i]
        self.raw_ratio = ratio
        
        
    ###########################################################
    def movingAverage(self, wdw=27, min_obs_per_wdw=10):
        """ 
        This function applies a moving average (MA) of length 'wdw' on the data.
        
        Parameters:
        wdw:              length of the MA window
        min_obs_per_wdw:  min number of obs required per window to compute a value
                          for that day (otherwise NaN)
        """
        (nObs, nStations) = self.raw_ratio.shape
        if wdw % 2 == 1:
            halfwdw = int((wdw - 1)/2)
        else:
            wdw += 1
            halfwdw = int((wdw - 1)/2)
            
        ma = np.ones((nObs, nStations)); ma[:] = np.nan
        for i in range(nStations):
            for j in range(nObs):
                if j > -1 and j < halfwdw: #beginning
                    m = self.raw_ratio[0:j + halfwdw + 1,i]
                elif j >= halfwdw and j < nObs - halfwdw: #middle
                    m = self.raw_ratio[j - halfwdw:j + halfwdw + 1,i]
                else: #end
                    m = self.raw_ratio[j - halfwdw:nObs,i]
                if len(m[~np.isnan(m)]) > int(wdw/min_obs_per_wdw):
                    ma[j,i] = np.nanmean(m)
        self.ratio = ma
        print('Moving Average on length:', wdw)
          
    ###################################################################
    def levelRemoval(self, wdw=240):
        """
        This function removes the individual level of the processes by substracting 
        a MA of window length "wdw" in each series. 
        
        Parameters:
        wdw:    MA window length, expressed in days
        """
        self.data = np.copy(self.ratio)
        (nObs, nStations) = self.data.shape
        if wdw % 2 == 1:
            halfwdw = int((wdw - 1)/2)
        else:
            wdw += 1
            halfwdw = int((wdw - 1)/2)
        for i in range(nStations):
            m = np.nanmean(self.data[:,i])
            ma = np.ones((nObs))*m
            for j in range(halfwdw, nObs - halfwdw):
                ma[j] = np.nanmean(self.data[j-halfwdw :j+halfwdw+1,i])
            ma[np.isnan(ma)] = m
            self.data[:,i] = (self.data[:,i] - ma)
        print('Level removed by MA of length: ', wdw)
       
    ########################################################################     
    def selectionIC(self, method='kmeans', reference=0, nIC1=None, nIC2=None):
        """
        This function automatically selects the number of IC processes 
        in P1 (moderately stable pool) and P2 (very stable pool) using different
        clustering methods. 
        
        Parameters:
        method:         clustering methods:
                        'leftmean' chooses all stations more stable 
                         than the mean of the stations ;
                        'leftmedian' chooses all stations more stable
                         than the median of the stations ;
                        'kmeans': k-means clustering ;
                        'agg': agglomerative hierarchical clustering ;
                        'ms': mean shift clustering ;
                        'dbscan': DBscan clustering ;
                        'gmm': gaussian mixture model ;
                        'fixed': selects the first 'nIC' most stable stations ;
                        'all': all methods (selects the most common pool)
        reference:      reference of the panel 
        nIC1:           number of IC stations in P1 (if method='fixed')
        nIC2:           number of IC stations in P2 (if method='fixed')
        """
        
        def pool_selection(x, station_names, ref, method, nIC):
            """
            Parameters:
            x:              dataset
            station_names:  code names of the stations
            ref:            reference of the panel 
            method:         clustering methods:
                            'leftmean' chooses all stations more stable 
                             than the mean of the stations ;
                            'leftmedian' chooses all stations more stable
                             than the median of the stations ;
                            'kmeans': k-means clustering ;
                            'agg': agglomerative hierarchical clustering ;
                            'ms': mean shift clustering ;
                            'dbscan': DBscan clustering ;
                            'gmm': gaussian mixture model ;
                            'fixed': selects the first 'nIC' most stable stations ;
                            'all': all methods (selects the most common pool) 
            nIC:            number of IC stations (if method='fixed')
            """
            if ref is None: 
                ref = np.nanmedian(x)
                
            (nObs, nStations) = x.shape
            #compute the mse
            mse = np.zeros(nStations)
            for i in range(nStations):
                mse[i] = (np.nanmedian(x[:,i]-ref))**2 + iqr(x[:,i], nan_policy='omit') 
            ordered_stations = np.argsort(mse)
        
            ind_stable = np.where(mse < np.mean(mse) - 1*np.std(mse))[0]
            too_stable =  [station_names[i] for i in ind_stable]
            ind_rob = np.where((mse < np.mean(mse) + 1.5*np.std(mse)) & (mse > np.mean(mse) - 1.5*np.std(mse)))[0]
            mse_rob = mse[ind_rob] #remove outliers 
            ordered_stations_rob = np.argsort(mse_rob) 
            station_names_rob = itemgetter(*ind_rob)(station_names)
        
            
            list_pool = [] #nested list (each element is a list)
            if method == 'leftmean' or method == 'all':
                list_pool.append(list(np.where(mse < np.mean(mse_rob))[0]))
                
            if method == 'leftmedian' or method == 'all':
                list_pool.append(list(np.where(mse < np.median(mse_rob))[0]))
                
            if method == 'kmeans' or method == 'all':
                kmeans = KMeans(n_clusters=2, init='k-means++', random_state=100).fit(mse_rob.reshape(-1,1))
                labels_km = kmeans.labels_
                best = labels_km[ordered_stations_rob[0]] #label of the most stable series
                pool_kmeans = np.where(labels_km == best)[0]
                names_kmeans = list(itemgetter(*pool_kmeans)(station_names_rob))
                names_kmeans = list(set(names_kmeans + too_stable)) # remove double 
                list_pool.append([i for i in range(len(station_names)) if station_names[i] in names_kmeans])
        
                
            if method == 'agg' or method == 'all':
                agg = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward').fit(mse_rob.reshape(-1, 1))
                labels_agg = agg.labels_
                best = labels_agg[ordered_stations_rob[0]]
                pool_agg = np.where(labels_agg == best)[0]
                names_agg = list(itemgetter(*pool_agg)(station_names_rob))
                names_agg = list(set(names_agg + too_stable))
                list_pool.append([i for i in range(len(station_names)) if station_names[i] in names_agg])
                
            if method == 'ms' or method == 'all':
                bandwidth = estimate_bandwidth(mse_rob.reshape(-1, 1), quantile=0.2, n_samples=len(mse_rob))
                ms = MeanShift(bandwidth).fit(mse_rob.reshape(-1, 1))
                labels_ms = ms.labels_
                best = [] 
                best.append(labels_ms[ordered_stations_rob[0]]) #find label of most stable station
                for i in range(1, len(mse_rob)):
                    n_stations = len(labels_ms[np.isin(labels_ms, best)]) #number of stations with label included into 'best'
                    new_label = labels_ms[ordered_stations_rob[i]] #label of next stable station
                    if n_stations < len(mse_rob)/3 and new_label not in best:
                        best.append(new_label)
                pool_ms = np.arange(len(mse_rob))[np.isin(labels_ms, best)] 
                names_ms = list(itemgetter(*pool_ms)(station_names_rob))
                names_ms = list(set(names_ms + too_stable))
                list_pool.append([i for i in range(len(station_names)) if station_names[i] in names_ms])
        
                
            if method == 'dbscan'or method == 'all':
                dbscan = DBSCAN(eps=0.01, min_samples=2).fit(mse_rob.reshape(-1, 1))
                labels_db = dbscan.labels_
                best = [] 
                best.append(labels_db[ordered_stations_rob[0]]) 
                for i in range(1, len(mse_rob)):
                    n_stations = len(labels_db[np.isin(labels_db, best)])
                    new_label = labels_db[ordered_stations_rob[i]] 
                    if n_stations < len(mse_rob)/3 and new_label not in best:
                        best.append(new_label)
                pool_db = np.arange(len(mse_rob))[np.isin(labels_db, best)]   
                names_db = list(itemgetter(*pool_db)(station_names_rob))
                names_db = list(set(names_db + too_stable))
                list_pool.append([i for i in range(len(station_names)) if station_names[i] in names_db])
         
                
            if method == 'gmm' or method == 'all': 
                gmm = GaussianMixture(n_components=2).fit(mse_rob.reshape(-1, 1))
                labels_gmm = gmm.predict(mse_rob.reshape(-1, 1))
                best = labels_gmm[ordered_stations_rob[0]]
                pool_gmm = np.where(labels_gmm == best)[0]
                names_gmm = list(itemgetter(*pool_gmm)(station_names_rob))
                names_gmm = list(set(names_gmm + too_stable))
                list_pool.append([i for i in range(len(station_names)) if station_names[i] in names_gmm])
         
            if method == 'fixed': 
                list_pool.append(ordered_stations[:nIC])
                
        
            length = []
            for i in range(len(list_pool)):
                length.append(len(list_pool[i]))
            most_common, num_most_common = Counter(length).most_common(1)[0]
            pool = list_pool[[i for i in range(len(list_pool)) if len(list_pool[i]) == most_common][0]]
            #ordered_stations[:most_common]
        
            if len(pool) < 2: #only one station ! 
                pool = ordered_stations[:nIC]
        
            return pool
        
        ##############
        (nObs, nStations) = self.data.shape
        if nIC1 is None: 
            nIC1 = int(np.round(nStations/2))
        self.pool = pool_selection(self.data, self.stationNames, reference, method, nIC1)
        print('P1 containing  %s stations' %len(self.pool))
                
        if nIC2 is None: 
            nIC2 = int(np.floor(len(self.pool)/2))
        self.p2 = pool_selection(self.data[:,self.pool], self.stationNames, reference, method, nIC2)
        print('P2 containing  %s stations' %len(self.p2))
            
    def outliersRemoval(self, k=1):
        """
        Remove the extreme outliers (at each time) from the IC stations. 
        This rocedure is similar to an adaptive shewhart chart. 
        
        Parameters
        k:      constant value, multiple of IQR that defines an outlier  
        """
        (nObs, nStations) = self.data.shape
        self.dataIC = self.data[:,self.pool]
        for i in range(nObs):
            avail = np.where(~np.isnan(self.data[i,:]))
            avail_IC = np.where(~np.isnan(self.dataIC[i,:]))
            index = np.where((self.dataIC[i,:] > np.nanmedian(self.data[i,:]) + k*iqr(self.data[i,:], nan_policy='omit')) | (self.dataIC[i,:]<np.nanmedian(self.data[i,:]) - k*iqr(self.data[i,:], nan_policy='omit')))
            if ((index is not None) and len(avail[0]) > len(avail_IC[0]) + 1):
                self.dataIC[i,index] = np.nan
        print('Outliers removed!')
        
    def standardisation(self, K=200):
        """ 
        Standardisation using the mean and the variance across the IC panel 
        and along a small window of time using K nearest neigbors estimators.
        
        Parameters
        K:           number of nearest neighbors
        """
        nICParameters = int(len(self.p2))
        print('nICParameters: ', nICParameters)
        self.dataICParameters = self.dataIC[:,self.p2]
        
        (nObs, nStations) = self.data.shape
        mu_0 = np.nanmean(self.dataICParameters); sigma_0 = np.nanstd(self.dataICParameters)
        mu_t = np.ones((nObs))*mu_0; sigma_t = np.ones((nObs))*sigma_0
        for t in range(K*2, nObs-K*2):
            c = 0
            list1 = np.array([]); 
            for i in range(K*2): #K nearest neigbors constrained with time window=2K
                if c < K:
                    data_tb = self.dataICParameters[t+i,:]
                    nb = data_tb[~np.isnan(data_tb)]
                    data_ta = self.dataICParameters[t-i,:]
                    na = data_ta[~np.isnan(data_ta)]
                    if i > 0:
                        list1 = np.append(list1, nb); list1 = np.append(list1, na)
                        c = len(nb) + len(na) + c
                    else: 
                        list1 = np.array(nb)
                        c = len(nb) + c
                if ((c >= K and len(list1) > 0) or (i == K*2 - 1 and len(list1) > 0)):  
                    mu_t[t] = np.mean(list1)
                    sigma_t[t] = np.std(list1)
                    break
            
        sigma_t[sigma_t == 0] = sigma_0 #don't divide by zero         
        for i in range(nStations):
            self.data[:,i] = (self.data[:,i] - mu_t)/sigma_t
            
        for i in range(len(self.pool)):
            self.dataIC[:,i] = (self.dataIC[:,i] - mu_t)/sigma_t
        print('Data standardized!')
        
    def runAll(self, nIC1=None, nIC2=None, method='kmeans'):
        """ 
        Apply the entire pre-processing.
        
        Parameters
        nIC1:           number of IC stations in P1
        nIC2:           number of IC series in P2
        method:         clustering methods:
                        'leftmean' chooses all stations more stable 
                         than the mean of the stations ;
                        'leftmedian' chooses all stations more stable
                         than the median of the stations ;
                        'kmeans': k-means clustering ;
                        'agg': agglomerative hierarchical clustering ;
                        'ms': mean shift clustering ;
                        'dbscan': DBscan clustering ;
                        'gmm': gaussian mixture model ;
                        'fixed': selects the first 'nIC' most stable stations ;
                        'all': all methods (selects the most common pool) 
        """
        self.rescaling()
        self.commonSignalRemoval() 
        self.movingAverage() 
        self.levelRemoval()
        self.selectionIC(method=method, nIC1=nIC1, nIC2=nIC2) 
        self.outliersRemoval() 
        self.standardisation() 
        print('Applied all functions!')
    

if __name__ == "__main__":
    
    ####################################################################
    ### Number of spots 

    with open('datasets/data_35_47', 'rb') as file: # load subset of 21 stations
#    with open('datasets/data_all_47', 'rb') as file: # load all stations
    
         my_depickler = pickle.Unpickler(file)
         Ns = my_depickler.load() #number of spots
         Ng = my_depickler.load() #number of sunspot groups
         Nc = my_depickler.load() #Ns+10Ng
         station_names = my_depickler.load()
         time = my_depickler.load()
         
    ### Apply preprocessing
    dataNs = PreProcessing(Ns, time, station_names)  
    
    dataNs.rescaling()
    #Ns = dataNs.obs_rescaled 
    dataNs.commonSignalRemoval() 
    #ratio= dataNs.raw_ratio
    dataNs.movingAverage() 
    level = dataNs.ratio #mu2
    
    dataNs.levelRemoval()
    dataNs.selectionIC(method='kmeans')
    pool = dataNs.pool
    #p2 = dataNs.p2
    
    dataNs.outliersRemoval()    
    dataNs.standardisation()
    data = np.copy(dataNs.data)
    dataIC = dataNs.dataIC
        
    plt.rcParams['figure.figsize'] = (10.0, 6.0)
    plt.figure(1)  
    plt.hist(dataIC[~np.isnan(dataIC)], range=[-4,4], bins='auto', density=True, facecolor='b')  
    plt.title("Data IC")
    plt.text(2, 1, 'mean:' '%4f' %np.nanmean(dataIC))
    plt.text(2, 0.8, 'std:' '%4f' %np.nanstd(dataIC))
    plt.text(2, 0.6, 'skewness:' '%4f' %skew(dataIC[~np.isnan(dataIC)]))
    plt.text(2, 0.4, 'kurtosis:' '%4f' %kurtosis(dataIC[~np.isnan(dataIC)]))
    plt.axis([-4, 4, 0, 1.5])
    plt.grid(True)
    plt.show()
    
    #### export data 
    with open('Ns', 'wb') as file: # subset of 21 stations 
#    with open('Ns_all', 'wb') as file: # all stations
        
        my_pickler = pickle.Pickler(file)
        my_pickler.dump(data) #data (IC and OC) with deviations
        my_pickler.dump(dataNs.time)
        my_pickler.dump(dataNs.stationNames)
        my_pickler.dump(dataIC) #IC series without deviations (for chart design)
        my_pickler.dump(pool) #index of series in the pool 
        my_pickler.dump(level) #ratio smoothed on 27 days


    ##########################################################################
    ### Number of sunspot groups
    
#    with open('datasets/data_35_47', 'rb') as file: # load subset of 21 stations
    with open('datasets/data_all_47', 'rb') as file: # load all stations
    
         my_depickler = pickle.Unpickler(file)
         Ns = my_depickler.load() #number of spots
         Ng = my_depickler.load() #number of sunspot groups
         Nc = my_depickler.load() #Ns+10Ng
         station_names = my_depickler.load()
         time = my_depickler.load()
         
    ### Apply preprocessing
    dataNg = PreProcessing(Ng, time, station_names)  
    
    dataNg.rescaling(period_rescaling=14)
    #Ng = dataNg.obs_rescaled 
    dataNg.commonSignalRemoval() 
    #ratio= dataNg.raw_ratio
    dataNg.movingAverage() 
    level = dataNg.ratio #mu2
    
    dataNg.levelRemoval(wdw=420)
    #dataNg.selectionIC(method='fixed', nIC1=10, nIC2=6) #Ng
    dataNg.selectionIC(method='fixed', nIC1=32, nIC2=18) #Ng all
    pool = dataNg.pool
    #p2 = dataNg.p2
    
    dataNg.outliersRemoval()    
    dataNg.standardisation()
    data = np.copy(dataNg.data)
    dataIC = dataNg.dataIC
        
    plt.rcParams['figure.figsize'] = (10.0, 6.0)
    plt.figure(1)  
    plt.hist(dataIC[~np.isnan(dataIC)], range=[-4,4], bins='auto', density=True, facecolor='b')  
    plt.title("Data IC")
    plt.text(2, 1, 'mean:' '%4f' %np.nanmean(dataIC))
    plt.text(2, 0.8, 'std:' '%4f' %np.nanstd(dataIC))
    plt.text(2, 0.6, 'skewness:' '%4f' %skew(dataIC[~np.isnan(dataIC)]))
    plt.text(2, 0.4, 'kurtosis:' '%4f' %kurtosis(dataIC[~np.isnan(dataIC)]))
    plt.axis([-4, 4, 0, 1.5])
    plt.grid(True)
    plt.show()
    
    #### export data 
#    with open('Ng', 'wb') as file: # subset of 21 stations 
    with open('Ng_all', 'wb') as file: # all stations
        
        my_pickler = pickle.Pickler(file)
        my_pickler.dump(data) #data (IC and OC) with deviations
        my_pickler.dump(dataNg.time)
        my_pickler.dump(dataNg.stationNames)
        my_pickler.dump(dataIC) #IC series without deviations (for chart design)
        my_pickler.dump(pool) #index of series in the pool 
        my_pickler.dump(level) #ratio smoothed on 27 days
        





