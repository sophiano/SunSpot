===================
SunSpot 
===================

**errors** 
This file contains functions to estimate the short-term error, long-term error and error at solar minima. It also includes functions to estimate the reference of the network (median and transformed median), the additive errors (individual errors) and other desired products that are not directly related to the monitoring. 

**preprocessing**
This file contains functions to remove the individual levels of the series, to select a pool of stable series from the panel, to estimate the longitudinal IC patterns of the data and to standardize the data by these patterns.

**bb_methods**
This file contains functions related to the block boostrap procedures. Four methods are implemented: moving block bootstrap (MBB), circular block bootstrap (CBB), non-overlapping block bootstrap (NBB) and matched block bootstrap (MABB). 

**block_length**
This file contains a procedure to estimate automatically an appropriate value for the block length based on the data. 

**cusum_design_bb**
This file contains a set of functions to design the CUSUM chart using the block bootstrap procedures. In particular, functions for computing the ARL0 or ARL1 of the chart are implemented as well as an algorithm to adjust the control limits of the chart.

**svr_svc_training**
This file includes several functions related to the support vector machine (SVM) procedures. 
It is composed of a function that automatically estimates an appropriate value for the length of the input vector, functions to train and valide the SVMs and to remove missing values from the input vectors.

**alerts**
This file contains a function that actually applies the monitoring to the series and returns the durations, the predicted sizes and shapes of the deviations after each alert. 
Other functions are also dedicated to graphically display the main features of the monitoring procedure and the alerts. 






