===================
data
===================



**data_1981** contains the observations of the entire database (~278 stations) over the years 1981-2019.

The datafile is composed of:
'Ns' == number of sunspots
'Ng' == number of sunspot groups
'Nc' == composites (Ns + 10Ng)
'station_names' == code names of the stations
'time' == index of the time, expressed in fraction of years

=====================================

**Nc_27** and **Nc_365** contain the standardized long-term errors of the composites for the entire database over the years 1981-2019, smoothed on 27 and 365 days respectively. 

The datafile is composed of:
'data' ==  standardized errors (without levels). All stations contain deviations. 
'time' == time index, expressed in fraction of years.
'station_names' == code names the stations
'dataIC' == IC stations, included in the pool. These data contain only the disparities (but not the deviations) of the IC stations. 
'IC_stations' == indexes of the stations included in the pool 
'level' == long-term error of the stations, with levels 

======================================

**kisl_wolf** contains the composites (Nc) of the station KS (these data are partly missing from the database). 

======================================

The files are provided into .zip files. After extraction, the different objects are contained into csv files. The files are also available in a serialized format, which may be opened with the python module 'pickle' with the following python code: 

with open('data', 'rb') as file:
      my_depickler = pickle.Unpickler(file)
      data = my_depickler.load() 
      time = my_depickler.load()
      station_names = my_depickler.load()
      dataIC = my_depickler.load()
      IC_stations= my_depickler.load()
      level = my_depickler.load()  

(with appropriate paths).


