===================
data
===================

**data_1981** contains the observations of the entire network (of ~278 stations) over the years 1981-2019.

The datafile is composed of:
'Ns': number of sunspots
'Ng': number of sunspot groups
'Nc': composites (Ns + 10Ng)
'station_names': codenames of the stations
'time': index of time, expressed in fraction of years

=====================================

**kisl_wolf** contains the data of the station KS (these data are partly missing from the database). 
The three first columns of the file correspond to the datetime of the observations. 
The fourth column (labelled 'W') contains the composite (Nc) of the station. The other columns are not used in this analysis. 

====================================

The files are provided into txt and serialized format, which may be opened with the python module 'pickle'.


