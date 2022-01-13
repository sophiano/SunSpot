# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 13:40:19 2021

@author: sopmathieu

This file contains functions to train and validate the feed-forward and recurrent 
neural networks for monitoring the sunspot numbers. 
It also includes a function to select the length of the input vector
of the networks based on the autocorrelation of the data. 

The networks designed for regression purpose aim at predicting the size of the 
deviations in a continuous range. Those constructed for classification purpose
aim at prediction the shape of the deviations among three general classes: 
    - sudden jumps
    - more progressive drifts
    - oscillating shifts.
    
"""

import numpy as np
from sklearn.utils import resample
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import layers
from sklearn import metrics
from statsmodels.tsa.stattools import acf
from sklearn.preprocessing import MinMaxScaler

from SunSpot import bb_methods as bb


def input_in_autocorr(data, thresh, lag_max=150):
    """
    Computes the length of the input vector of the networks using 
    an autocorrelation threshold.
    
    The length of the input vector can be selected based on the 
    autocorrelation of the data, as follows. 
    The autocorrelation may be computed until 'lag_max' in the (IC) series.
    Then, the lag at which the autocorrelation falls below a pre-defined 
    threshold ('thresh') may be computed in those series.
    The length of the input vector may finally be selected as the mean of
    those lags.

    Parameters
    ----------
    data : 2D-array
        in-control (IC) dataset (rows: time, columns: IC series).
    thresh : 0 <= float <= 1,
        The autocorrelation threshold.
    lag_max : int > 0, optional
        The maximum lag until the autocovariance should 
        be computed.  The default is 150.

    Returns
    -------
    m : float >= 0
        The selected length of the input vector.

    """
    assert np.ndim(data) == 2, "Input data must be a 2D array"
    (n_obs, n_series) = data.shape 
    assert thresh >= 0 and thresh <= 1, "thresh must be in [0,1]"
    lag_max = int(lag_max)
    assert lag_max >= 0, "Lag_max must be superior or equal to 0"
    
    #compute autocorrelation until lag_max
    autocorr = np.zeros((n_series, lag_max))
    for i in range(n_series):
        series_i = data[:,i]
        if len(series_i[~np.isnan(series_i)])> 1000:
            autocorr[i,:] = acf(series_i, nlags=lag_max, missing='drop')[1:]
        
    autocorr = autocorr[np.all(autocorr != 0, axis=1)] #remove zeros
        
        
    #find a min value
    knees = np.zeros((len(autocorr)))
    for i in range(len(autocorr)):
        y = autocorr[i,:]
        if len(np.where(y < thresh)[0]) > 0:
            knees[i] = np.where(y < thresh)[0][0]
        else:
            knees[i] = 150
        
    m = np.mean(knees)
    
    return m 

def feed_forward_reg(data, wdw_length, scale=1, n=50000, n_hidden=2,
                  n_neurons=[40,20], activation='sigmoid', n_epochs=30, 
                  batch_size=50, verbose=1):
    """
    Trains and validates a feed-forward neural network for the monitoring.
    
    The network is trained to predict in a continuous range the size of the 
    deviations (i.e. they are designed for regression purposes). 

    Parameters
    ----------
    data : 2D-array
        in-control (IC) dataset (rows: time, columns: IC series).
    wdw_length : int > 0
        The length of the input data. 
    scale : float > 0, optional
        The scale parameter of the normal distribution that is used to simulate
        the size of the deviations. The default is 1.
    n : int > 0, optional
        Number of training and validating instances. This value is 
        typically large. The default is 50000.
    n_hidden : int > 0, optional
        Number of hidden layers. The defaults is 2. 
    n_neurons : list of int, optional
        Number of neurons in each hidden layer. The default is 
        [40, 20].
    activation : string, optional
        The activation function. The default is 'sigmoid'.
        Other values are 'relu', 'softmax', 'softplus', 'tanh' etc. 
        (see keras documentation)
    n_epochs : int > 0, optional
        The number of epochs. The default is 30.
    batch-size : int > 0, optional
        The batch size. The default is 50.
    verbose : int, optional 
        Verbosity mode. 0=silent, 1=progress bar, 2=one line per epoch.
        The default is 1. 

    Returns
    -------
    model : (keras) model
        The trained neural network.
    scores : list
        The performances of the model.
        It contains the mean squared error (loss function), the mean absolute
        error and the mean absolute percentage error (metrics) 
        of the network on the validation set.

    """
    wdw_length = int(wdw_length)
    assert wdw_length > 0, "wdw_length must be superior to zero"
    assert scale > 0, "scale must be superior to zero"
    n = int(n)
    assert n > 0, "n must be superior to zero"
    n_hidden = int(n_hidden)
    assert n_hidden > 0, "n_hidden must be superior to zero"
    assert n_hidden == len(n_neurons), 'The neurons do not match the number of hidden layers'
    #assert all(isinstance(item, int) for item in n_neurons), 'The number of neurons should be integer>0'
    n_epochs = int(n_epochs)
    assert n_epochs > 0, "n_epochs must be superior to zero"
    batch_size = int(batch_size )
    assert batch_size  > 0, "batch_size must be superior to zero"
    
    n_test = int(n/5) #n testing instances
    n_train = n - n_test #n training instances
    blocks = bb.MBB(data, wdw_length)     
        
    ### training set 
    X_train = np.zeros((n_train, wdw_length))
    Y_train = np.zeros((n_train))
    S_train = np.zeros((n_train)) #shapes
    rnd_shifts = np.random.normal(0, scale, n_train) #shift sizes
    
    
    for b in range(0, n_train-2, 3):
        
        shift = rnd_shifts[b]
        series = resample(blocks, replace=True, n_samples=1).flatten()
        
        for shape in range(3):
        
            boot = np.copy(series)
            
            if shape == 0:
                delay = np.random.randint(0, wdw_length)
                boot[delay:] = boot[delay:] + shift
                S_train[b] = 0
            elif shape == 1:
                power = np.random.uniform(1, 2)
                boot = shift/(500) * (np.arange(wdw_length)**power) + boot
                S_train[b] = 1
            else:
                eta = np.random.uniform(np.pi/(2*wdw_length), 2*np.pi/wdw_length)
                phi = np.random.randint(0, int(wdw_length/4))
                boot = np.sin(eta*np.pi*np.arange(wdw_length)+phi)*shift + boot
                S_train[b] = 2
                
            X_train[b] = boot
            Y_train[b] = shift
            b += 1
            
        
    ### testing set 
    X_test = np.zeros((n_test, wdw_length))
    Y_test = np.zeros((n_test))
    S_test = np.zeros((n_test))
    rnd_shifts = np.random.normal(0, scale, n_test)
    
    for b in range(0, n_test-2, 3):
        
        shift = rnd_shifts[b]
    
        series = resample(blocks, replace=True, n_samples=1).flatten()
        
        for shape in range(3):
        
            boot = np.copy(series)
            
            if shape == 0:
                delay = np.random.randint(0, wdw_length)
                boot[delay:] = boot[delay:] + shift
                S_test[b] = 0
            elif shape == 1:
                power = np.random.uniform(1, 2)
                boot = shift/(500) * (np.arange(wdw_length)**power) + boot
                S_test[b] = 1
            else:
                eta = np.random.uniform(np.pi/(2*wdw_length), 2*np.pi/wdw_length)
                phi = np.random.randint(0, int(wdw_length/4))
                boot = np.sin(eta*np.pi*np.arange(wdw_length) + phi)*shift + boot
                S_test[b] = 2
                
            X_test[b] = boot
            Y_test[b] = shift
            b += 1
            
    ### scaling input and output data
    # if norm: 
    #     input_scaler = MinMaxScaler(feature_range=(-1,1))     
    #     input_scaler.fit(X_train)
    #     X_train = input_scaler.transform(X_train)
    #     X_test = input_scaler.transform(X_test)
        
    #     Y_train = Y_train.reshape(-1,1)
    #     Y_test = Y_test.reshape(-1,1)
    #     output_scaler = MinMaxScaler(feature_range=(-1,1))     
    #     output_scaler.fit(Y_train)
    #     Y_train = output_scaler.transform(Y_train)
    #     Y_test = output_scaler.transform(Y_test)
    
    ### Neural network Architecture 
    model = tf.keras.Sequential() #linear stack of layers 
    for i in range(n_hidden):
        model.add(layers.Dense(n_neurons[i], input_dim=wdw_length, activation=activation, use_bias=True)) 
    model.add(layers.Dense(1, use_bias=True))# no activation in the output layer since regression
    if verbose > 0:
        model.summary()
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    
    model.fit(X_train, Y_train, epochs=n_epochs, batch_size=batch_size, 
              verbose=verbose)
    
    # make predictions with the model
    predictions = model.predict(X_test)
    
    # if norm:
    #     X_test = input_scaler.inverse_transform(X_test)
    #     Y_test = output_scaler.inverse_transform(Y_test)
    #     predictions = output_scaler.inverse_transform(predictions)
    #     Y_test = Y_test.reshape(-1)
        
    predictions = predictions.reshape(-1)
    ind = np.where(abs(Y_test) > 0.1)
    mape = (1/len(Y_test[ind])) * sum(abs((abs(Y_test[ind]) - abs(predictions[ind]))/abs(Y_test[ind])))*100
    scores = model.evaluate(X_test, Y_test, verbose=verbose)
    scores.append(mape)
    
    return model, scores
    
    
    
def feed_forward_class(data, wdw_length, scale=1, n=50000, n_hidden=1,
                  n_neurons=[40], activation='sigmoid', 
                  n_epochs=30, batch_size=50, verbose=1):
    """
    Trains and validates a feed-forward neural network for
    the monitoring.
    
    The network is trained to classify the shape of the deviations into 
    three different classes:
        - sudden jumps
        - more progressive drifts
        - oscillating shifts.

    Parameters
    ----------
    data : 2D-array
        in-control (IC) dataset (rows: time, columns: IC series).
    wdw_length : int > 0
        The length of the input data. 
    scale : float > 0, optional
        The scale parameter of the normal distribution that is used to simulate
        the size of the deviations. The default is 1.
    n : int > 0, optional
        Number of training and validating instances. This value is 
        typically large. The default is 50000.
    n_hidden : int > 0, optional
        Number of hidden layers. The defaults is 1. 
    n_neurons : list of int, optional
        Number of neurons in each hidden layer. The default is [40].
    activation : string, optional
        The activation function. The default is 'sigmoid'.
        Other values are 'relu', 'softmax', 'softplus', 'tanh' etc. 
        (see keras documentation)
    n_epochs : int > 0, optional
        The number of epochs. The default is 30.
    batch-size : int > 0, optional
        The batch size. The default is 50.
    verbose : int, optional 
        Verbosity mode. 0=silent, 1=progress bar, 2=one line per epoch.
        The default is 1.

    Returns
    -------
    model : (keras) model
        The trained neural network.
    scores : list
        The performances of the model.
        It contains the mean squared error (loss function) and the
        accuracy (metrics) of the network on the validation set.
    matrix : 2D-array (float)
        The confusion matrix of the classifier. 

    """
    wdw_length = int(wdw_length)
    assert wdw_length > 0, "wdw_length must be superior to zero"
    assert scale > 0, "scale must be superior to zero"
    n = int(n)
    assert n > 0, "n must be superior to zero"
    n_hidden = int(n_hidden)
    assert n_hidden > 0, "n_hidden must be superior to zero"
    assert n_hidden == len(n_neurons), 'The neurons do not match the number of hidden layers'
    #assert all(isinstance(item, int) for item in n_neurons), 'The number of neurons should be integer>0'
    n_epochs = int(n_epochs)
    assert n_epochs > 0, "n_epochs must be superior to zero"
    batch_size = int(batch_size )
    assert batch_size  > 0, "batch_size must be superior to zero"
    
    n_test = int(n/5) #n testing instances
    n_train = n - n_test #n training instances
    blocks = bb.MBB(data, wdw_length) 
    
    ### training set
    X_train = np.zeros((n_train, wdw_length))
    Y_train = np.zeros((n_train))
    S_train = np.zeros((n_train)) #shapes
    rnd_shifts = np.random.normal(0, scale, n_train)
    
    for b in range(0, n_train-2, 3):
        
        shift = rnd_shifts[b]
        series = resample(blocks, replace=True, n_samples=1).flatten()
        
        for shape in range(3):
        
            boot = np.copy(series)
            
            if shape == 0:
                delay = np.random.randint(0, wdw_length)
                boot[delay:] = boot[delay:] + shift
                S_train[b] = 0
            elif shape == 1:
                power = np.random.uniform(1, 2)
                boot = shift/(100) * (np.arange(wdw_length)**power) + boot
                S_train[b] = 1
            else:
                eta = np.random.uniform(np.pi/(2*wdw_length), 2*np.pi/wdw_length)
                phi = np.random.randint(0, int(wdw_length/4))
                boot = np.sin(eta*np.pi*np.arange(wdw_length)+phi)*shift * boot
                S_train[b] = 2
                
            X_train[b] = boot
            Y_train[b] = shift
            b += 1
            
        
    ### testing set 
    X_test = np.zeros((n_test, wdw_length))
    Y_test = np.zeros((n_test))
    S_test = np.zeros((n_test))
    rnd_shifts = np.random.normal(0, scale, n_test)
    
    for b in range(0, n_test-2, 3):
        
        shift = rnd_shifts[b]
        series = resample(blocks, replace=True, n_samples=1).flatten()
        
        for shape in range(3):
        
            boot = np.copy(series)
            
            if shape == 0:
                delay = np.random.randint(0, wdw_length)
                boot[delay:] = boot[delay:] + shift
                S_test[b] = 0
            elif shape == 1:
                power = np.random.uniform(1, 2)
                boot = shift/(100) * (np.arange(wdw_length)**power) + boot
                S_test[b] = 1
            else:
                eta = np.random.uniform(np.pi/(2*wdw_length), 2*np.pi/wdw_length)
                phi = np.random.randint(0, int(wdw_length/4))
                boot = np.sin(eta*np.pi*np.arange(wdw_length) + phi)*shift * boot
                S_test[b] = 2
                
            X_test[b] = boot
            Y_test[b] = shift
            b += 1
    
    ### Neural network Architecture (classification) 
    model = tf.keras.Sequential() #linear stack of layers 
    for i in range(n_hidden):
        model.add(layers.Dense(n_neurons[i], input_dim=wdw_length, activation=activation, use_bias=True)) 
    model.add(layers.Dense(3, activation='softmax', use_bias=True))# three output classes 
    if verbose > 0:
        model.summary()
    
    #compile model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    S_train = to_categorical(S_train) #hot encoding
    model.fit(X_train, S_train, epochs=n_epochs, batch_size=batch_size, 
              verbose=verbose)
    
    # make class predictions with the model
    S_test = to_categorical(S_test)
    #predictions = model.predict_classes(X_test) 
    predictions = np.argmax(model.predict(X_test), axis=-1)
    predictions = to_categorical(predictions)
    scores = model.evaluate(X_test, S_test, verbose=verbose)
    #confusion matrix
    matrix = metrics.confusion_matrix(predictions.argmax(axis=1), 
                            S_test.argmax(axis=1), normalize='true')

    return model, scores, matrix

#=======================================================================
#=======================================================================

def recurrent_reg(data, wdw_length, scale=1, n=50000, n_neurons=[10, 10],
                  n_epochs=30, batch_size=50, verbose=1):
    """
    Trains and validates a recurrent neural network (composed of one recurrent 
    layer and one fully-connected layer) for the monitoring.
    
    The network is trained to predict in a continuous range the size of the 
    deviations (i.e. they are designed for regression purposes). 

    Parameters
    ----------
    data : 2D-array
        in-control (IC) dataset (rows: time, columns: IC series).
    wdw_length : int > 0
        The length of the input data. 
    scale : float > 0, optional
        The scale parameter of the normal distribution that is used to simulate
        the size of the deviations. The default is 1.
    n : int > 0, optional
        Number of training and validating instances. This value is 
        typically large. The default is 50000.
    n_neurons : list of int, optional
        Number of neurons in each hidden layer. The default is 
        [10, 10].
    n_epochs : int > 0, optional
        The number of epochs. The default is 30.
    batch-size : int > 0, optional
        The batch size. The default is 50.
    verbose : int, optional 
        Verbosity mode. 0=silent, 1=progress bar, 2=one line per epoch.
        The default is 1. 

    Returns
    -------
    model : (keras) model
        The trained neural network.
    scores : list
        The performances of the model.
        It contains the mean squared error (loss function), the mean absolute
        error and the mean absolute percentage error (metrics) 
        of the network on the validation set.

    """
    wdw_length = int(wdw_length)
    assert wdw_length > 0, "wdw_length must be superior to zero"
    assert scale > 0, "scale must be superior to zero"
    n = int(n)
    assert n > 0, "n must be superior to zero"
    n_epochs = int(n_epochs)
    assert n_epochs > 0, "n_epochs must be superior to zero"
    batch_size = int(batch_size )
    assert batch_size  > 0, "batch_size must be superior to zero"
    assert len(n_neurons) == 2, "n_neurons must be composed of 2 elements"
    #assert all(isinstance(item, int) for item in n_neurons), 'The number of neurons should be integer>0'
    
    n_test = int(n/5) #n testing instances
    n_train = n - n_test #n training instances
    blocks = bb.MBB(data, wdw_length) 
        
    ### training set 
    X_train = np.zeros((n_train, wdw_length))
    Y_train = np.zeros((n_train))
    S_train = np.zeros((n_train)) #shapes
    rnd_shifts = np.random.normal(0, scale, n_train) #shift sizes
    
    
    for b in range(0, n_train-2, 3):
        
        shift = rnd_shifts[b]
        series = resample(blocks, replace=True, n_samples=1).flatten()
        
        for shape in range(3):
        
            boot = np.copy(series)
            
            if shape == 0:
                delay = np.random.randint(0, wdw_length)
                boot[delay:] = boot[delay:] + shift
                S_train[b] = 0
            elif shape == 1:
                power = np.random.uniform(1, 2)
                boot = shift/(500) * (np.arange(wdw_length)**power) + boot
                S_train[b] = 1
            else:
                eta = np.random.uniform(np.pi/(2*wdw_length), 2*np.pi/wdw_length)
                phi = np.random.randint(0, int(wdw_length/4))
                boot = np.sin(eta*np.pi*np.arange(wdw_length)+phi)*shift + boot
                S_train[b] = 2
                
            X_train[b] = boot
            Y_train[b] = shift
            b += 1
            
        
    ### testing set 
    X_test = np.zeros((n_test, wdw_length))
    Y_test = np.zeros((n_test))
    S_test = np.zeros((n_test))
    rnd_shifts = np.random.normal(0, scale, n_test)
    
    for b in range(0, n_test-2, 3):
        
        shift = rnd_shifts[b]
    
        series = resample(blocks, replace=True, n_samples=1).flatten()
        
        for shape in range(3):
        
            boot = np.copy(series)
            
            if shape == 0:
                delay = np.random.randint(0, wdw_length)
                boot[delay:] = boot[delay:] + shift
                S_test[b] = 0
            elif shape == 1:
                power = np.random.uniform(1, 2)
                boot = shift/(500) * (np.arange(wdw_length)**power) + boot
                S_test[b] = 1
            else:
                eta = np.random.uniform(np.pi/(2*wdw_length), 2*np.pi/wdw_length)
                phi = np.random.randint(0, int(wdw_length/4))
                boot = np.sin(eta*np.pi*np.arange(wdw_length) + phi)*shift + boot
                S_test[b] = 2
                
            X_test[b] = boot
            Y_test[b] = shift
            b += 1
            
    
    ### Neural network Architecture 
    model = tf.keras.Sequential() #linear stack of layers 
    model.add(layers.SimpleRNN(n_neurons[0], input_dim=wdw_length)) 
    model.add(layers.Dense(n_neurons[1], activation='sigmoid')) 
    model.add(layers.Dense(1))
    if verbose > 0:
        model.summary()
        
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    model.fit(X_train, Y_train, epochs=n_epochs, batch_size=batch_size,
              verbose=verbose)
    
    # make predictions with the model
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    predictions = model.predict(X_test).reshape(-1)
    ind = np.where(abs(Y_test) > 0.1)
    mape = (1/len(Y_test[ind])) * sum(abs((abs(Y_test[ind]) - abs(predictions[ind]))/abs(Y_test[ind])))*100
    scores = model.evaluate(X_test, Y_test, verbose=verbose)
    scores.append(mape)

    return model, scores


def recurrent_class(data, wdw_length, scale=1, n=50000, n_neurons=[10],
                    n_epochs=30, batch_size=50, verbose=1):
    """
    Trains and validates a recurrent neural network (composed of one 
    recurrent hidden layer) for the monitoring.
    
    The network is trained to classify the shape of the deviations into 
    three different classes:
        - sudden jumps
        - more progressive drifts
        - oscillating shifts.

    Parameters
    ----------
    data : 2D-array
        in-control (IC) dataset (rows: time, columns: IC series).
    wdw_length : int > 0
        The length of the input data. 
    scale : float > 0, optional
        The scale parameter of the normal distribution that is used to simulate
        the size of the deviations. The default is 1.
    n : int > 0, optional
        Number of training and validating instances. This value is 
        typically large. The default is 50000.
    n_neurons : list of int, optional
        Number of neurons in the hidden layer. The default is 
        [10].
    n_epochs : int > 0, optional
        The number of epochs. The default is 30.
    batch-size : int > 0, optional
        The batch size. The default is 50.
    verbose : int, optional 
        Verbosity mode. 0=silent, 1=progress bar, 2=one line per epoch.
        The default is 1. 
        

    Returns
    -------
    model : (keras) model
        The trained neural network.
    scores : list
        The performances of the model.
        It contains the mean squared error (loss function) and the
        accuracy (metrics) of the network on the validation set.
    matrix : 2D-array (float)
        The confusion matrix of the classifier. 

    """
    wdw_length = int(wdw_length)
    assert wdw_length > 0, "wdw_length must be superior to zero"
    assert scale > 0, "scale must be superior to zero"
    n = int(n)
    assert n > 0, "n must be superior to zero"
    n_epochs = int(n_epochs)
    assert n_epochs > 0, "n_epochs must be superior to zero"
    batch_size = int(batch_size )
    assert batch_size  > 0, "batch_size must be superior to zero"
    assert len(n_neurons) == 1, "n_neurons must be composed of 2 elements"
    #assert all(isinstance(item, int) for item in n_neurons), 'The number of neurons should be integer>0'
    
    
    n_test = int(n/5) #n testing instances
    n_train = n - n_test #n training instances
    blocks = bb.MBB(data, wdw_length) 
    #blocks_training = bb.MBB(dataIC[:,:75],wdw_length) 
    #blocks_testing = bb.MBB(dataIC[:,75:], wdw_length) 
    
    ### training set
    X_train = np.zeros((n_train, wdw_length))
    Y_train = np.zeros((n_train))
    S_train = np.zeros((n_train)) #shapes
    rnd_shifts = np.random.normal(0, scale, n_train)
    
    for b in range(0, n_train-2, 3):
        
        shift = rnd_shifts[b]
        series = resample(blocks, replace=True, n_samples=1).flatten()
        
        for shape in range(3):
        
            boot = np.copy(series)
            
            if shape == 0:
                delay = np.random.randint(0, wdw_length)
                boot[delay:] = boot[delay:] + shift
                S_train[b] = 0
            elif shape == 1:
                power = np.random.uniform(1, 2)
                boot = shift/(100) * (np.arange(wdw_length)**power) + boot
                S_train[b] = 1
            else:
                eta = np.random.uniform(np.pi/(2*wdw_length), 2*np.pi/wdw_length)
                phi = np.random.randint(0, int(wdw_length/4))
                boot = np.sin(eta*np.pi*np.arange(wdw_length)+phi)*shift * boot
                #plt.plot(boot); plt.show()
                S_train[b] = 2
                
            X_train[b] = boot
            Y_train[b] = shift
            b += 1
            
        
    ### testing set 
    X_test = np.zeros((n_test, wdw_length))
    Y_test = np.zeros((n_test))
    S_test = np.zeros((n_test))
    rnd_shifts = np.random.normal(0, scale, n_test)
    
    for b in range(0, n_test-2, 3):
        
        shift = rnd_shifts[b]
        series = resample(blocks, replace=True, n_samples=1).flatten()
        
        for shape in range(3):
        
            boot = np.copy(series)
            
            if shape == 0:
                delay = np.random.randint(0, wdw_length)
                boot[delay:] = boot[delay:] + shift
                S_test[b] = 0
            elif shape == 1:
                power = np.random.uniform(1, 2)
                boot = shift/(100) * (np.arange(wdw_length)**power) + boot
                S_test[b] = 1
            else:
                eta = np.random.uniform(np.pi/(2*wdw_length), 2*np.pi/wdw_length)
                phi = np.random.randint(0, int(wdw_length/4))
                boot = np.sin(eta*np.pi*np.arange(wdw_length) + phi)*shift * boot
                S_test[b] = 2
                
            X_test[b] = boot
            Y_test[b] = shift
            b += 1
    
    ### Neural network Architecture (classification) 
    model = tf.keras.Sequential() #linear stack of layers 
    model.add(layers.SimpleRNN(n_neurons[0], input_dim=wdw_length))  
    model.add(layers.Dense(3, activation='softmax'))# three output classes 
    if verbose > 0:
        model.summary()

    
    #compile model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    S_train = to_categorical(S_train) #hot encoding
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    model.fit(X_train, S_train, epochs=n_epochs, batch_size=batch_size, 
              verbose=verbose)
    
    # make class predictions with the model
    S_test = to_categorical(S_test)
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    #predictions = model.predict_classes(X_test) 
    predictions = np.argmax(model.predict(X_test), axis=-1)
    predictions = to_categorical(predictions)
    scores = model.evaluate(X_test, S_test, verbose=verbose)
    #confusion matrix
    matrix = metrics.confusion_matrix(predictions.argmax(axis=1), 
                             S_test.argmax(axis=1), normalize='true')

    return model, scores, matrix

#=======================================================================
#=======================================================================


def feed_forward_4reg(data, wdw_length, scale=1, n=50000, n_hidden=2,
                  n_neurons=[40,20], activation='sigmoid', n_epochs=30, 
                  batch_size=50, verbose=1):
    """
    Trains and validates a feed-forward neural network for the monitoring.
    
    The network is trained to predict in a continuous range the size of the 
    deviations (i.e. they are designed for regression purposes). 

    Parameters
    ----------
    data : 2D-array
        in-control (IC) dataset (rows: time, columns: IC series).
    wdw_length : int > 0
        The length of the input data. 
    scale : float > 0, optional
        The scale parameter of the normal distribution that is used to simulate
        the size of the deviations. The default is 1.
    n : int > 0, optional
        Number of training and validating instances. This value is 
        typically large. The default is 50000.
    n_hidden : int > 0, optional
        Number of hidden layers. The defaults is 2. 
    n_neurons : list of int, optional
        Number of neurons in each hidden layer. The default is 
        [40, 20].
    activation : string, optional
        The activation function. The default is 'sigmoid'.
        Other values are 'relu', 'softmax', 'softplus', 'tanh' etc. 
        (see keras documentation)
    n_epochs : int > 0, optional
        The number of epochs. The default is 30.
    batch-size : int > 0, optional
        The batch size. The default is 50.
    verbose : int, optional 
        Verbosity mode. 0=silent, 1=progress bar, 2=one line per epoch.
        The default is 1. 

    Returns
    -------
    model : (keras) model
        The trained neural network.
    scores : list
        The performances of the model.
        It contains the mean squared error (loss function), the mean absolute
        error and the mean absolute percentage error (metrics) 
        of the network on the validation set.

    """
    wdw_length = int(wdw_length)
    assert wdw_length > 0, "wdw_length must be superior to zero"
    assert scale > 0, "scale must be superior to zero"
    n = int(n)
    assert n > 0, "n must be superior to zero"
    n_hidden = int(n_hidden)
    assert n_hidden > 0, "n_hidden must be superior to zero"
    assert n_hidden == len(n_neurons), 'The neurons do not match the number of hidden layers'
    #assert all(isinstance(item, int) for item in n_neurons), 'The number of neurons should be integer>0'
    n_epochs = int(n_epochs)
    assert n_epochs > 0, "n_epochs must be superior to zero"
    batch_size = int(batch_size )
    assert batch_size  > 0, "batch_size must be superior to zero"
    
    n_test = int(n/5) #n testing instances
    n_train = n - n_test #n training instances
    blocks = bb.MBB(data, wdw_length)     
        
    ### training set 
    X_train = np.zeros((n_train, wdw_length))
    Y_train = np.zeros((n_train))
    S_train = np.zeros((n_train)) #shapes
    rnd_shifts = np.random.normal(0, scale, n_train) #shift sizes
    
    
    for b in range(0, n_train-3, 4):
        
        shift = rnd_shifts[b]
        series = resample(blocks, replace=True, n_samples=1).flatten()
        
        for shape in range(4):
        
            boot = np.copy(series)
            
            if shape == 0:
                S_train[b] = 0
            elif shape == 1:
                delay = np.random.randint(0, wdw_length)
                boot[delay:] = boot[delay:] + shift
                S_train[b] = 1
            elif shape == 2:
                power = np.random.uniform(1, 2)
                boot = shift/(500) * (np.arange(wdw_length)**power) + boot
                S_train[b] = 2
            else:
                eta = np.random.uniform(np.pi/(2*wdw_length), 2*np.pi/wdw_length)
                phi = np.random.randint(0, int(wdw_length/4))
                boot = np.sin(eta*np.pi*np.arange(wdw_length)+phi)*shift + boot
                S_train[b] = 3
                
            X_train[b] = boot
            Y_train[b] = shift
            b += 1
            
        
    ### testing set 
    X_test = np.zeros((n_test, wdw_length))
    Y_test = np.zeros((n_test))
    S_test = np.zeros((n_test))
    rnd_shifts = np.random.normal(0, scale, n_test)
    
    for b in range(0, n_test-3, 4):
        
        shift = rnd_shifts[b]
    
        series = resample(blocks, replace=True, n_samples=1).flatten()
        
        for shape in range(4):
        
            boot = np.copy(series)
            
            if shape == 0:
                S_test[b] = 0
            elif shape == 1:
                delay = np.random.randint(0, wdw_length)
                boot[delay:] = boot[delay:] + shift
                S_test[b] = 1
            elif shape == 2:
                power = np.random.uniform(1, 2)
                boot = shift/(500) * (np.arange(wdw_length)**power) + boot
                S_test[b] = 2
            else:
                eta = np.random.uniform(np.pi/(2*wdw_length), 2*np.pi/wdw_length)
                phi = np.random.randint(0, int(wdw_length/4))
                boot = np.sin(eta*np.pi*np.arange(wdw_length) + phi)*shift + boot
                S_test[b] = 3
                
            X_test[b] = boot
            Y_test[b] = shift
            b += 1
            
    
    ### Neural network Architecture 
    model = tf.keras.Sequential() #linear stack of layers 
    for i in range(n_hidden):
        model.add(layers.Dense(n_neurons[i], input_dim=wdw_length, activation=activation, use_bias=True)) 
    model.add(layers.Dense(1, use_bias=True))# no activation in the output layer since regression
    if verbose > 0:
        model.summary()
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    
    model.fit(X_train, Y_train, epochs=n_epochs, batch_size=batch_size, 
              verbose=verbose)
    
    # make predictions with the model
    predictions = model.predict(X_test)    
    predictions = predictions.reshape(-1)
    ind = np.where(abs(Y_test) > 0.1)
    mape = (1/len(Y_test[ind])) * sum(abs((abs(Y_test[ind]) - abs(predictions[ind]))/abs(Y_test[ind])))*100
    scores = model.evaluate(X_test, Y_test, verbose=verbose)
    scores.append(mape)
    
    return model, scores


def feed_forward_4class(data, wdw_length, scale=1, n=50000, n_hidden=1,
                  n_neurons=[40], activation='sigmoid', 
                  n_epochs=30, batch_size=50, verbose=1):
    """
    Trains and validates a feed-forward neural network for
    the monitoring.
    
    The network is trained to classify the shape of the deviations into 
    four different classes:
        - sudden jumps
        - more progressive drifts
        - oscillating shifts
        - no shifts.

    Parameters
    ----------
    data : 2D-array
        in-control (IC) dataset (rows: time, columns: IC series).
    wdw_length : int > 0
        The length of the input data. 
    scale : float > 0, optional
        The scale parameter of the normal distribution that is used to simulate
        the size of the deviations. The default is 1.
    n : int > 0, optional
        Number of training and validating instances. This value is 
        typically large. The default is 50000.
    n_hidden : int > 0, optional
        Number of hidden layers. The defaults is 1. 
    n_neurons : list of int, optional
        Number of neurons in each hidden layer. The default is [40].
    activation : string, optional
        The activation function. The default is 'sigmoid'.
        Other values are 'relu', 'softmax', 'softplus', 'tanh' etc. 
        (see keras documentation)
    n_epochs : int > 0, optional
        The number of epochs. The default is 30.
    batch-size : int > 0, optional
        The batch size. The default is 50.
    verbose : int, optional 
        Verbosity mode. 0=silent, 1=progress bar, 2=one line per epoch.
        The default is 1.

    Returns
    -------
    model : (keras) model
        The trained neural network.
    scores : list
        The performances of the model.
        It contains the mean squared error (loss function) and the
        accuracy (metrics) of the network on the validation set.
    matrix : 2D-array (float)
        The confusion matrix of the classifier. 

    """
    wdw_length = int(wdw_length)
    assert wdw_length > 0, "wdw_length must be superior to zero"
    assert scale > 0, "scale must be superior to zero"
    n = int(n)
    assert n > 0, "n must be superior to zero"
    n_hidden = int(n_hidden)
    assert n_hidden > 0, "n_hidden must be superior to zero"
    assert n_hidden == len(n_neurons), 'The neurons do not match the number of hidden layers'
    #assert all(isinstance(item, int) for item in n_neurons), 'The number of neurons should be integer>0'
    n_epochs = int(n_epochs)
    assert n_epochs > 0, "n_epochs must be superior to zero"
    batch_size = int(batch_size )
    assert batch_size  > 0, "batch_size must be superior to zero"
    
    n_test = int(n/5) #n testing instances
    n_train = n - n_test #n training instances
    blocks = bb.MBB(data, wdw_length) 
    
    ### training set
    X_train = np.zeros((n_train, wdw_length))
    Y_train = np.zeros((n_train))
    S_train = np.zeros((n_train)) #shapes
    rnd_shifts = np.random.normal(0, scale, n_train)
    
    for b in range(0, n_train-3, 4):
        
        shift = rnd_shifts[b]
        series = resample(blocks, replace=True, n_samples=1).flatten()
        
        for shape in range(4):
        
            boot = np.copy(series)
            
            if shape == 0:
                S_train[b] = 0
            elif shape == 1:
                delay = np.random.randint(0, wdw_length)
                boot[delay:] = boot[delay:] + shift
                S_train[b] = 1
            elif shape == 2:
                power = np.random.uniform(1, 2)
                boot = shift/(100) * (np.arange(wdw_length)**power) + boot
                S_train[b] = 2
            else:
                eta = np.random.uniform(np.pi/(2*wdw_length), 2*np.pi/wdw_length)
                phi = np.random.randint(0, int(wdw_length/4))
                boot = np.sin(eta*np.pi*np.arange(wdw_length)+phi)*shift * boot
                #plt.plot(boot); plt.show()
                S_train[b] = 3
                
            X_train[b] = boot
            Y_train[b] = shift
            b += 1
            
        
    ### testing set 
    X_test = np.zeros((n_test, wdw_length))
    Y_test = np.zeros((n_test))
    S_test = np.zeros((n_test))
    rnd_shifts = np.random.normal(0, scale, n_test)
    
    for b in range(0, n_test-3, 4):
        
        shift = rnd_shifts[b]
        series = resample(blocks, replace=True, n_samples=1).flatten()
        
        for shape in range(4):
        
            boot = np.copy(series)
            
            if shape == 0:
                S_test[b] = 0
            elif shape == 1:
                delay = np.random.randint(0, wdw_length)
                boot[delay:] = boot[delay:] + shift
                S_test[b] = 1
            elif shape == 2:
                power = np.random.uniform(1, 2)
                boot = shift/(100) * (np.arange(wdw_length)**power) + boot
                S_test[b] = 2
            else:
                eta = np.random.uniform(np.pi/(2*wdw_length), 2*np.pi/wdw_length)
                phi = np.random.randint(0, int(wdw_length/4))
                boot = np.sin(eta*np.pi*np.arange(wdw_length) + phi)*shift * boot
                S_test[b] = 3
                
            X_test[b] = boot
            Y_test[b] = shift
            b += 1
    
    ### Neural network Architecture (classification) 
    model = tf.keras.Sequential() #linear stack of layers 
    for i in range(n_hidden):
        model.add(layers.Dense(n_neurons[i], input_dim=wdw_length, activation=activation, use_bias=True)) 
    model.add(layers.Dense(4, activation='softmax', use_bias=True))# three output classes 
    if verbose > 0:
        model.summary()
    
    #compile model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    S_train = to_categorical(S_train) #hot encoding
    model.fit(X_train, S_train, epochs=n_epochs, batch_size=batch_size, 
              verbose=verbose)
    
    # make class predictions with the model
    S_test = to_categorical(S_test)
    #predictions = model.predict_classes(X_test) 
    predictions = np.argmax(model.predict(X_test), axis=-1)
    predictions = to_categorical(predictions)
    scores = model.evaluate(X_test, S_test, verbose=verbose)
    
    #confusion matrix
    matrix = metrics.confusion_matrix(predictions.argmax(axis=1), 
                        S_test.argmax(axis=1), normalize='true')

    return model, scores, matrix