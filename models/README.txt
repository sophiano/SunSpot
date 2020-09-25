===================
models
===================

This folder contains the trained support vector machine classifiers (svc) and regressors (svr) for the number of groups 'Ng' and the number of spots 'Ns'. 
Files with 'all' in their extension indicates that the entire database (59 stations) was used while the other files are only based on a subset of 21 stations. 

======================================

These files should be opened with the python module 'pickle' such as: 

reg = pickle.load(open('../models/svr_Ns.sav', 'rb'))
clf = pickle.load(open('../models/svc_Ns.sav', 'rb'))

(with appropriate paths). 


