# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 11:56:15 2019

@author: keckj
"""
#set up workspace
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
#import HydroFunctions as HF

ddir = 'D:/UW_PhD/PreeventsProject/Hydrometeorology and Floods/FlowModeling/GridMetDataComparison_WY2007/'
os.chdir(ddir)

o = pd.read_csv('USGSgage12189500SaukatSaukWy2007_d.txt', sep='\t')#observed flow from USGS ft3/s
o['2006-10-01 00:00'] = pd.to_datetime(o['2006-10-01 00:00'])
o = o.set_index('2006-10-01 00:00')
Qo = o['908'] #observed flow, ft3/s
Qo = Qo/(3.281**3) #m3/s
Qoh = Qo.resample('H').mean() #hourly instantaneous

### STORM 2

#parameter set 1
#model results for all grid points, PNNL, horly
m1 = pd.read_csv('PNNL1HStorm2Streamflow.Only', delim_whitespace=True)#, delim_whitespace=True) m3/h
m1['DATE'] = pd.to_datetime(m1['DATE'])
m1 = m1.set_index('DATE')
Qm1 = (m1['12189500']*3.281**3)/3600 #convert m3/h to ft3/s
Qm1 = (m1['12189500'])/3600 #convert m3/h to m3/s

#parameter set 2
#model results for all grid points, PNNL, horly
m2 = pd.read_csv('PNNL1HStorm2t2_Streamflow.Only', delim_whitespace=True)#, delim_whitespace=True) m3/h
m2['DATE'] = pd.to_datetime(m2['DATE'])
m2 = m2.set_index('DATE')
Qm2 = (m2['12189500']*3.281**3)/3600 #convert m3/h to ft3/s
Qm2 = (m2['12189500'])/3600 #convert m3/h to m3/s

#parameter set 3
#model results for all grid points, PNNL, horly
m3 = pd.read_csv('PNNL1HStorm2t3_Streamflow.Only', delim_whitespace=True)#, delim_whitespace=True) m3/h
m3['DATE'] = pd.to_datetime(m3['DATE'])
m3 = m3.set_index('DATE')
Qm3 = (m3['12189500']*3.281**3)/3600 #convert m3/h to ft3/s
Qm3 = (m3['12189500'])/3600 #convert m3/h to m3/s


##parameter set 6
##model results for all grid points, PNNL, horly
#m6 = pd.read_csv('PNNL1HStorm2t5_Streamflow.Only', delim_whitespace=True)#, delim_whitespace=True) m3/h
#m6['DATE'] = pd.to_datetime(m6['DATE'])
#m6 = m6.set_index('DATE')
#Qm6 = (m6['12189500']*3.281**3)/3600 #convert m3/h to ft3/s
#Qm6 = (m6['12189500'])/3600 #convert m3/h to m3/s

#parameter set 6v2
#model results for all grid points, PNNL, horly
m6 = pd.read_csv('PNNL1HStorm2t6v2_Streamflow.Only', delim_whitespace=True)#, delim_whitespace=True) m3/h
m6['DATE'] = pd.to_datetime(m6['DATE'])
m6 = m6.set_index('DATE')
Qm6 = (m6['12189500']*3.281**3)/3600 #convert m3/h to ft3/s
Qm6 = (m6['12189500'])/3600 #convert m3/h to m3/s

#parameter set 7
#model results for all grid points, PNNL, horly
m7 = pd.read_csv('PNNL1HStorm2t7_Streamflow.Only', delim_whitespace=True)#, delim_whitespace=True) m3/h
m7['DATE'] = pd.to_datetime(m7['DATE'])
m7 = m7.set_index('DATE')
Qm7 = (m7['12189500']*3.281**3)/3600 #convert m3/h to ft3/s
Qm7 = (m7['12189500'])/3600 #convert m3/h to m3/s

#parameter set 8
#model results for all grid points, PNNL, horly
m8 = pd.read_csv('PNNL1HStorm2t8_Streamflow.Only', delim_whitespace=True)#, delim_whitespace=True) m3/h
m8['DATE'] = pd.to_datetime(m8['DATE'])
m8 = m8.set_index('DATE')
Qm8 = (m8['12189500']*3.281**3)/3600 #convert m3/h to ft3/s
Qm8 = (m8['12189500'])/3600 #convert m3/h to m3/s


#parameter set 9
#model results for all grid points, PNNL, horly
m9 = pd.read_csv('PNNL1HStorm2t9_Streamflow.Only', delim_whitespace=True)#, delim_whitespace=True) m3/h
m9['DATE'] = pd.to_datetime(m9['DATE'])
m9 = m9.set_index('DATE')
Qm9 = (m9['12189500']*3.281**3)/3600 #convert m3/h to ft3/s
Qm9 = (m9['12189500'])/3600 #convert m3/h to m3/s

#parameter set 10
#model results for all grid points, PNNL, horly
m10 = pd.read_csv('PNNL1HStorm2t10_Streamflow.Only', delim_whitespace=True)#, delim_whitespace=True) m3/h
m10['DATE'] = pd.to_datetime(m10['DATE'])
m10 = m10.set_index('DATE')
Qm10 = (m10['12189500']*3.281**3)/3600 #convert m3/h to ft3/s
Qm10 = (m10['12189500'])/3600 #convert m3/h to m3/s

#parameter set 11
#model results for all grid points, PNNL, horly
m11 = pd.read_csv('PNNL1HStorm2t11_Streamflow.Only', delim_whitespace=True)#, delim_whitespace=True) m3/h
m11['DATE'] = pd.to_datetime(m11['DATE'])
m11 = m11.set_index('DATE')
Qm11 = (m11['12189500']*3.281**3)/3600 #convert m3/h to ft3/s
Qm11 = (m11['12189500'])/3600 #convert m3/h to m3/s


#parameter set 12
#model results for all grid points, PNNL, horly
m12 = pd.read_csv('PNNL1HStorm2t12_Streamflow.Only', delim_whitespace=True)#, delim_whitespace=True) m3/h
m12['DATE'] = pd.to_datetime(m12['DATE'])
m12 = m12.set_index('DATE')
Qm12 = (m12['12189500']*3.281**3)/3600 #convert m3/h to ft3/s
Qm12 = (m12['12189500'])/3600 #convert m3/h to m3/s

#parameter set 12
#model results for all grid points, PNNL, horly
m12 = pd.read_csv('PNNL1HStorm2t12_Streamflow.Only', delim_whitespace=True)#, delim_whitespace=True) m3/h
m12['DATE'] = pd.to_datetime(m12['DATE'])
m12 = m12.set_index('DATE')
Qm12 = (m12['12189500']*3.281**3)/3600 #convert m3/h to ft3/s
Qm12 = (m12['12189500'])/3600 #convert m3/h to m3/s

#parameter set 13
#model results for all grid points, PNNL, horly
m13 = pd.read_csv('PNNL1HStorm2t13_Streamflow.Only', delim_whitespace=True)#, delim_whitespace=True) m3/h
m13['DATE'] = pd.to_datetime(m13['DATE'])
m13 = m13.set_index('DATE')
Qm13 = (m13['12189500']*3.281**3)/3600 #convert m3/h to ft3/s
Qm13 = (m13['12189500'])/3600 #convert m3/h to m3/s

#parameter set 14
#model results for all grid points, PNNL, horly
m14 = pd.read_csv('PNNL1HStorm2t14_Streamflow.Only', delim_whitespace=True)#, delim_whitespace=True) m3/h
m14['DATE'] = pd.to_datetime(m14['DATE'])
m14 = m14.set_index('DATE')
Qm14 = (m14['12189500']*3.281**3)/3600 #convert m3/h to ft3/s
Qm14 = (m14['12189500'])/3600 #convert m3/h to m3/s

#parameter set 15
#model results for all grid points, PNNL, horly
m15 = pd.read_csv('PNNL1HStorm2t15_Streamflow.Only', delim_whitespace=True)#, delim_whitespace=True) m3/h
m15['DATE'] = pd.to_datetime(m15['DATE'])
m15 = m15.set_index('DATE')
Qm15 = (m15['12189500']*3.281**3)/3600 #convert m3/h to ft3/s
Qm15 = (m15['12189500'])/3600 #convert m3/h to m3/s

#parameter set 16
#model results for all grid points, PNNL, horly
m16 = pd.read_csv('PNNL1HStorm2t16_Streamflow.Only', delim_whitespace=True)#, delim_whitespace=True) m3/h
m16['DATE'] = pd.to_datetime(m16['DATE'])
m16 = m16.set_index('DATE')
Qm16 = (m16['12189500']*3.281**3)/3600 #convert m3/h to ft3/s
Qm16 = (m16['12189500'])/3600 #convert m3/h to m3/s

#parameter set 17
#model results for all grid points, PNNL, horly
m17 = pd.read_csv('PNNL1HStorm2t17_Streamflow.Only', delim_whitespace=True)#, delim_whitespace=True) m3/h
m17['DATE'] = pd.to_datetime(m17['DATE'])
m17 = m17.set_index('DATE')
Qm17 = (m17['12189500']*3.281**3)/3600 #convert m3/h to ft3/s
Qm17 = (m17['12189500'])/3600 #convert m3/h to m3/s


m = pd.read_csv('NovToFev_p21.Only', delim_whitespace=True)
m['DATE'] = pd.to_datetime(m['DATE'])
m = m.set_index('DATE')
Qm = (m['12189500']*3.281**3)/3600 #convert m3/h to ft3/s
Qm21 = (m['12189500'])/3600 #convert m3/h to m3/s

m = pd.read_csv('NovToFev_p22.Only', delim_whitespace=True)
m['DATE'] = pd.to_datetime(m['DATE'])
m = m.set_index('DATE')
Qm = (m['12189500']*3.281**3)/3600 #convert m3/h to ft3/s
Qm22 = (m['12189500'])/3600 #convert m3/h to m3/s

m = pd.read_csv('NovToFev_p23.Only', delim_whitespace=True)
m['DATE'] = pd.to_datetime(m['DATE'])
m = m.set_index('DATE')
Qm = (m['12189500']*3.281**3)/3600 #convert m3/h to ft3/s
Qm23 = (m['12189500'])/3600 #convert m3/h to m3/s

m = pd.read_csv('WY2007_p51.Only', delim_whitespace=True)
m['DATE'] = pd.to_datetime(m['DATE'])
m = m.set_index('DATE')
Qm = (m['12189500']*3.281**3)/3600 #convert m3/h to ft3/s
Qm51 = (m['12189500'])/3600 #convert m3/h to m3/s

fig, ax = plt.subplots(figsize=(24, 10))

ax.plot(Qm1,'r-',linewidth=1, alpha = 1, label = 'PNNL Hourly P1')
ax.plot(Qm2,'y--',linewidth=3.5, alpha = .6,  label = 'PNNL Hourly P2')
ax.plot(Qm3,'g--',linewidth=2, alpha = .6,  label = 'PNNL Hourly P3')
ax.plot(Qm6,'b--',linewidth=2, alpha = .6,  label = 'PNNL Hourly P6')
ax.plot(Qm7,'b:',linewidth=2, alpha = .6,  label = 'PNNL Hourly P7')
#ax.plot(Qm8,'r:',linewidth=2, alpha = .6,  label = 'PNNL Hourly P8')
#ax.plot(Qm9,'k:',linewidth=2, alpha = .6,  label = 'PNNL Hourly P9')
#ax.plot(Qm10,'c:',linewidth=2, alpha = .6,  label = 'PNNL Hourly P10')
#ax.plot(Qm11,'m:',linewidth=2, alpha = .6,  label = 'PNNL Hourly P11')
ax.plot(Qm12,'r:',linewidth=4, alpha = .6,  label = 'PNNL Hourly P12')
#ax.plot(Qm13,'y:',linewidth=4, alpha = .6,  label = 'PNNL Hourly P13')
#ax.plot(Qm14,'c:',linewidth=4, alpha = .6,  label = 'PNNL Hourly P14')
#ax.plot(Qm15,'g:',linewidth=4, alpha = .6,  label = 'PNNL Hourly P15')
ax.plot(Qm16,'g:',linewidth=4, alpha = .6,  label = 'PNNL Hourly P16')
ax.plot(Qm17,'r-',linewidth=4, alpha = .2,  label = 'PNNL Hourly P17')
ax.plot(Qm21,'c-',linewidth=4, alpha = .2,  label = 'PNNL Hourly P21')
ax.plot(Qm22,'b-',linewidth=4, alpha = .2,  label = 'PNNL Hourly P22')
ax.plot(Qm23,'y-',linewidth=4, alpha = .8,  label = 'PNNL Hourly P23')

ax.plot(Qm51,'k:',linewidth=4, alpha = .8,  label = 'PNNL Hourly P51')

ax.plot(Qo,'k-',linewidth=1, alpha = 1, label = 'Observed')

plt.xlim(['2006-11-01 00:00:00','2007-6-13 00:00:00'])
plt.ylim([0,2000])
plt.ylabel('Instantaneous flow [m3/s]')
legend = ax.legend(loc = 'upper right')
#plt.savefig('DecJan2007FlowCompDaily.jpg',dpi=300, bbox_inches = 'tight')
plt.show()

def NashSutcliffe(Qobs,Qmod):
    '''
    Computes NashSutcliffe - metric used to evaluate model performance. 
    Same computation as the correlation coefficiene, ie: residual variance ((obs-mod)^2) 
    normalized by observation variance (obs-obs_mean)^2.
    Qobs - pd series of observations
    Qmod - pd series of modeled values
    
    Qobs and Qmod must  have the same index and be the same lenth =>
    
    Use pd methods such as .resample('H').mean() and fillna(method = 'pad') to 
    make sure index matches.
    '''
    SqDif = (Qmod-Qobs)**2
    SSqDif = SqDif.sum()
    MnDif = (Qobs-Qobs.mean())**2
    SMnDif = MnDif.sum()
    
    NS = 1-(SSqDif/SMnDif)
    
    return NS

#adjust time series so that same number of entries and begin and end date match
Qor = Qo.loc['2006-10-01 01:00:00':'2007-09-30 00:00:00'].resample('H').mean()
Qm51 = Qm51.loc['2006-10-01 01:00:00':'2007-09-30 00:00:00']


#compare mean values
Qord = Qor.resample('M').mean()
Qm51d = Qm51.resample('M').mean()

fig, ax = plt.subplots(figsize=(24, 10))

ax.plot(Qm51,'k:',linewidth=4, alpha = .8,  label = 'PNNL Hourly P51')

ax.plot(Qor,'k-',linewidth=1, alpha = 1, label = 'Observed')

plt.xlim(['2006-11-01 00:00:00','2007-6-13 00:00:00'])
plt.ylim([0,3000])
plt.ylabel('Instantaneous flow [m3/s]')
legend = ax.legend(loc = 'upper right')
#plt.savefig('DecJan2007FlowCompDaily.jpg',dpi=300, bbox_inches = 'tight')
plt.show()



NashSutcliffe(Qor,Qm51)
