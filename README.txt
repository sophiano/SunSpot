===================
SunSpot Module
===================

**A robust non-parametric quality control procedure, based on the block boostrap and the CUSUM chart, to monitor a panel of non-Normal autocorrelated observations prone to noise and potential missing values. The method also provides predictions of the forms and the sizes of the deviations using support vector machine procedures.**

The module contains different files to preprocess, train and monitor a panel of autocorrelated series with potential missing values in presence of strong noise.
The method does not require any a priori knowledge about the data such as information about the in-control (IC), i.e. non deviating processes.  
The main functions are located inside the folder **SunSpot**.

The raw and preprocessed observations are located inside the folder **data**.

The methodology, though generic, is also applied on the number of sunspots (Ns) and the number of sunspot groups (Ng) as examples.
Results for a subset of 21 observatories (also called 'stations') and a larger database containing 59 stations are available in jupyter notebooks in the folder **notebooks**. 

The support vector machine classifiers and regressors are also saved into **models** since they take few minutes to be executed. They are also conserved for reproducibility purposes. 





