===================
SunSpot Module
===================

** This module contains a robust non-parametric quality control procedure, based on the block boostrap and the CUSUM chart, to monitor a panel of non-normal autocorrelated observations prone to noise and potential missing values. The method also provides predictions of the shapes and the sizes of the deviations encountered using support vector machine classifier and regressor.**

The module contains different files to preprocess and monitor a panel of autocorrelated series with potential missing values in presence of strong noise.
The method does not require any prior knowledge about the data such as information about the non deviating processes, parametric model for the distribution or the autocorrelation structure of the data, to correctly operate.
The complete procedure is applied on the sunspot number data as examples. 

=======================================
Structure of the package:

**SunSpot** 
The main functions (codes) are located inside the folder **SunSpot**.

**data**
The data are composed of the number of sunspots (Ns), the number of sunspot groups (Ng) and the composite (Nc=Ns+10Ng). They are located inside **data**.
The monitoring procedure is applied on these data as examples of how to use the package. 
The files are provided into .zip and serialized format (pickling).

**notebooks**
Jupyter notebooks are provided in the folder **notebooks**. 
In the notebooks, the monitoring procedure is applied to the long-term errors of Ns, Ng and Nc on two different time-scales with detailed explanations. 

**scripts**
The scripts associated to the notebooks are located inside the folder **scripts**.

**svm_models**
The trained support vector machine classifiers and regressors are also saved into **svm_models**. They are conserved for reproducibility purposes. 

====================================
Installation: 

The package can be installed using 'pip' with the following command: 

>>>pip install git+https://github.com/sophiano/SunSpot





