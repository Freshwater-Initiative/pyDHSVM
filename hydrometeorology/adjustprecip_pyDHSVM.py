# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 12:36:37 2018

@author: cband
"""
import os
import pandas as pd
import numpy as np

#import numpy as np
#import pandas as pd
#Read in forcing file

homedir='D:/UW_PhD/PreeventsProject/Hydrometeorology and Floods/ForcingData/HydrologySensitivityForcingData'
header=['Date','T_min','T_max','Relative_Humidity','ShortWave','LongWave','Precip_m']
data=pd.read_csv('C:/Users/jisheng/pyDHSVM/data_48.34375_-121.34375.csv',sep=' ',names=header)
T=(data.T_min+data.T_max)/2
P=data.Precip_m
RH=data.Relative_Humidity
T.shape
z=np.repeat(167,T.shape[0])
plapse=0.0000006
Precip_z = np.zeros(shape = (len(P),1))

for i in range (len(P)):
    if P[i] > 0.0:                 
        Precip_z[i]=P[i] + plapse*(z[i])
    else:
        Precip_z[i]=P[i]

#print some combination of old and new data
data_lapsed=pd.DataFrame()
data_lapsed['Date']=data.Date.values
data_lapsed['T_min']=np.around(data.T_min.values,2)
data_lapsed['T_max']=np.around(data.T_max.values,2)
data_lapsed['Relative_Humidity']=np.around(data.Relative_Humidity.values,1)
data_lapsed['ShortWave']=np.around(data.ShortWave.values,3)
data_lapsed['LongWave']=np.around(data.LongWave.values,3)
data_lapsed['Precip_m']=np.around(Precip_z,5)

os.chdir(homedir)
data_lapsed.to_csv('sauk_base_station.data_48.34375_-121.34375_test.csv',index=False,header=None,sep=' ')
