===================
data
===================

**data_35_47** contains the observations of 21 stations over the period 1947-2013.

The datafile is composed of:
'Ns' == number of sunspots
'Ng' == number of sunspot groups
'Nc' == composite (Ns + 10Ng)
'station_names' == code names of the stations
'time' == index of the time, expressed in fraction of years

====================================

**data_all_47** contains the observations of 59 stations over the period 1947-2013.

The datafile is composed of:
'Ns' == number of sunspots
'Ng' == number of sunspot groups
'Nc' == composite (Ns + 10Ng)
'station_names' == code names of the stations
'time' == index of the time, expressed in fraction of years

=====================================

**Ns** and **Ng** contain the preprocessed observations of 21 stations over the period 1947-2013. 

The datafile is composed of:
'data' ==  standardized residuals (without levels). All stations contain deviations. 
'time' == time index, expressed in fraction of years.
'station_names' == code names the stations
'dataIC' == IC stations, included in P1. These data contain only the disparities (but not the deviations) of the IC stations. 
'IC_stations' == indexes of the stations included in P1 
'level' == long-term error of the stations, with levels (mu2)

=====================================

**Ns_all** and **Ng_all** contain the preprocessed observations of 59 stations over the period 1947-2013. 

The datafile is composed of:
'data' ==  standardized residuals (without levels). All stations contain deviations. 
'time' == time index, expressed in fraction of years.
'station_names' == code names the stations
'dataIC' == IC stations, included in P1. These data contain only the disparities (but not the deviations) of the IC stations. 
'IC_stations' == indexes of the stations included in P1 
'level' == long-term error of the stations, with levels (mu2)

======================================

These files should be opened with the python module 'pickle' such as: 

with open('data', 'rb') as file:
      my_depickler = pickle.Unpickler(file)
      data = my_depickler.load() 
      time = my_depickler.load()
      station_names = my_depickler.load()
      dataIC = my_depickler.load()
      IC_stations= my_depickler.load()
      level = my_depickler.load()  

(with appropriate paths).


