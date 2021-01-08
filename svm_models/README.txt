===================
models
===================

This folder contains the trained support vector machine classifiers (svc) and regressors (svr) for the composites 'Nc'. The confusion matrices (normalized and not normalized) and the prediction performances of the models are also saved. 
Files with '365' in their extension indicate that the models are trained to detect deviations in the data smoothed on 365 days. 
Whereas files with '27' in their extension indicate that the models are trained to detect deviations in the data smoothed on 27 days. 

======================================

These files may be opened with the python module 'pickle' such as: 

reg = pickle.load(open('../svm_models/svr_Ns_27.sav', 'rb'))
clf = pickle.load(open('../svm_models/svc_Ns_27.sav', 'rb'))

(with appropriate paths). 


