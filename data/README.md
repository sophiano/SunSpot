# Data

This folder contains the datafiles. <br>
They are provided into txt, zip or serialized format (pickling).

## Folder contents


**data_1981** contains the observations of the entire network (of ~278 stations) over the years 1981-2019. 

This file is composed of:

* _Ns_ : number of sunspots
* _Ng_ : number of sunspot groups
* _Nc_ : composites (Ns + 10Ng)
* _station_names_ : codenames of the stations
* _time_ : index of time, expressed in fraction of years 

**Nc_27** and **Nc_365** contain the long-term errors of the composites (Nc) for the entire network (~278 stations) over the years 1981-2019.
Those errors are smoothed on respectively 27 and 365 days. 

The file is composed of:

* _data_ :  standardized long-term errors (variable $\hat \epsilon_{\mu_2}$)
* _time_ : index of time, expressed in fraction of years
* _station_names_ : codenames of the stations
* _dataIC_ : standardized long-term errors in the in-control (IC) stations. These data contain only the disparities (but not the deviations) of the IC stations. Hence, data that do not fall into one standard deviation around the (cross-sectional) mean are removed from _dataIC_. These data serves to train the CUSVM procedure.
* _pool_ : index of the IC stations  
* _level_ : long-term errors with levels (variable $\widehat{eh}$)
* _data_nn_ : long-term errors without levels (variable $\hat \mu_2$). Those data are used by the neural networks. 
 

**kisl_wolf** contains the data of the station KS (Kislovodsk) over the years 1954-2020, which are partly missing from the database. 
The three first columns of the file correspond to the datetime of the observations. 
The fourth column (labelled 'W') contains the composite (Nc) of the station. The other columns are not used in this analysis. <br>

## Instructions

The files in serialized format (**data_1981**, **Nc_27** and **Nc_365**)  can be opened with the module 
[pickle](https://docs.python.org/3/library/pickle.html) (with appropriate path): 

````
with open('data_1981', 'rb') as file:
     my_depickler = pickle.Unpickler(file)
     Ns = my_depickler.load() 
     Ng = my_depickler.load() 
     Nc = my_depickler.load() 
     station_names = my_depickler.load() 
     time = my_depickler.load() 
     
with open('Nc_27', 'rb') as file:
      my_depickler = pickle.Unpickler(file)
      data = my_depickler.load() 
      time = my_depickler.load()
      station_names = my_depickler.load()
      dataIC = my_depickler.load()
      pool = my_depickler.load()
      level = my_depickler.load()  
      data_nn = my_depickler.load() 
     
with open('Nc_365', 'rb') as file:
      my_depickler = pickle.Unpickler(file)
      data = my_depickler.load() 
      time = my_depickler.load()
      station_names = my_depickler.load()
      dataIC = my_depickler.load()
      pool = = my_depickler.load()
      level = my_depickler.load()  
      data_nn = my_depickler.load() 
     
````




