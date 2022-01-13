
# SunSpot Module

SunSpot is a module written in Python to monitor the sunspot numbers.

The package is composed of a robust non-parametric quality control based on the block bootstrap and the CUSUM chart. The method also provides predictions of the shapes and the sizes of the deviations encountered, using support vector machine (SVM) procedures.

Moreover, a neural network based procedure is included in the package, to allow the monitoring 
of the sunspot numbers using an alternative method. 

This package was written by Sophie Mathieu (UCLouvain/Royal Observatory of Belgium). 

## Installation 

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install SunSpot:

````
pip install git+https://github.com/sophiano/SunSpot
````
Tips : We strongly adsive the user to install the package in a new Python environment (with a version of Python ranging between 3.6 and 3.8) to avoid any conflict with existing packages. For more information about the installation, we refer to the setup. 

## Folder contents

* **SunSpot** <br>
The main functions of the package are contained in the folder SunSpot.

* **data** <br>
The data are composed of the number of sunspots (Ns), the number of sunspot groups (Ng) and the composite (Nc=Ns+10Ng). 
The data are provided into zip, txt and serialized format (pickling).

* **docs** <br>
This folder contains the documentation of the package, in the form of jupyter notebooks. 

* **scripts** <br>
This folder contains the scripts of the package.

* **svm_models** <br>
The trained support vector machine classifiers and regressors are saved into the folder svm_models. They are conserved for reproducibility purposes. 

* **nn_models** <br>
The trained neural networks are saved into the folder nn_models. They are conserved for reproducibility purposes. 

## References

* Mathieu, S., von Sachs, R., Delouille, V., Lefevre, L. & Ritter, C. (2019).
_Uncertainty quantification in sunspot counts_. The Astrophysical Journal, 886(1):7. Available on [arXiv](https://arxiv.org/abs/2009.09810).

* Mathieu, S., Lefevre, L, von Sachs, R., Delouille, V., Ritter, C. & Clette, F. (2021).
_Nonparametric monitoring of sunspot number observations: a case study_.
Available on [arXiv](https://arxiv.org/abs/2106.13535).

## License
[MIT](https://choosealicense.com/licenses/mit/)
