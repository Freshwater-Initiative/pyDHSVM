# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 12:36:37 2018

@author: cband
"""
import numpy as np
import pandas as pd

def lapsemodel(T,P,RH,z,plapse):
    if P > 0.0:
    mass_air=mass_wet_air
    Precip_z=P + plapse*(z)
    else:
    mass_air=mass_dry_air
    Precip_z=P
    
    mass_air=RH*mass_wet_air/100 + (100-RH)*mass_dry_air/100
    H=R_mol*1000*(T+273.15)/(mass_air*g)  # result in m\n",
    pressure_z=P_sea_level*np.exp(-1*(z/H))
    density_sealevel=P_sea_level/((T+273.15)*R_kg)
    dPdz=(pressure_z-P_sea_level)/(z-0)
    dTdz=dPdz/(cp*density_sealevel)
    return(Precip_z,dTdz)
    
#Read in forcing file
header=['Date','T_min','T_max','Relative_Humidity','ShortWave','LongWave','Precip_m']
data=pd.read_csv('C:\Users\cband\Skagit\data_48.34375_-121.34375',sep=' ',names=header)
T=(data.T_min+data.T_max)/2
P=data.Precip_m
RH=data.Relative_Humidity
T.shape
z=np.repeat(167,T.shape[0])
plapse=0.0000006
Precip_z = []
dTdz=[]\n",
for i in range(len(T)):
    Precip_temp,dTdz_temp = lapsemodel(T[i],P[i],RH[i],z[i],plapse)
    Precip_z.append(Precip_temp)
    dTdz.append(dTdz_temp)

#print some combination of old and new data
data_lapsed=pd.DataFrame()
data_lapsed['Date']=data.Date.values
data_lapsed['T_min']=np.around(data.T_min.values,2)
data_lapsed['T_max']=np.around(data.T_max.values,2)
data_lapsed['Relative_Humidity']=np.around(data.Relative_Humidity.values,1)
data_lapsed['ShortWave']=np.around(data.ShortWave.values,3)
data_lapsed['LongWave']=np.around(data.LongWave.values,3)
data_lapsed['Precip_m']=np.around(Precip_z,5)
data_lapsed.to_csv('sauk_base_station.data_48.34375_-121.34375_test',index=False,header=None,sep=' ')