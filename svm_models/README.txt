===================
svm_models
===================

This folder contains the trained support vector machine classifiers (svc) and regressors (svr) for the composites 'Nc'. 
Files with '365' in their extension indicate that the models are trained to detect deviations in the data smoothed on 365 days. 
Whereas files with '27' in their extension indicate that the models are trained to detect deviations in the data smoothed on 27 days. 
They are saved for reproducibility purpose.

======================================

These models can be loaded with the python module 'pickle': 

reg = pickle.load(open('svm_models/svr_Nc_27.sav', 'rb'))
clf = pickle.load(open('svm_models/svc_Nc_27.sav', 'rb'))

(with appropriate paths). 


