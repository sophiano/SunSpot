
# svm_models

This folder contains the trained support vector machine classifiers (**svc**) and regressors (**svr**) for monitoring the composites 'Nc'. 

The confusion matrices (normalized and not normalized) and the prediction performances of the models are also saved (in PNG images). 

Files with **365** (resp. **27**) in their extension indicate that the models are trained to detect deviations in the data smoothed on 365 days (resp. 27 days). 

Those models are saved for reproducibility purpose.

## Instructions

These models can be opened with the module 
[pickle](https://docs.python.org/3/library/pickle.html) (with appropriate path):  

````
reg = pickle.load(open('svm_models/svr_Nc_27.sav', 'rb'))
clf = pickle.load(open('svm_models/svc_Nc_27.sav', 'rb'))

````
