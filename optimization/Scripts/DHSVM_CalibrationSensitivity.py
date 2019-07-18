# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:38:08 2019

@author: keckj
"""
#visualize and quantify fast calibration results


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import subprocess
from distutils.dir_util import remove_tree
import shutil
import time
import sys
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# # Create your connection.
# cnx = sqlite3.connect('fast_dhsvm_parallel_run_pnnl_Oct2005toNov2007.db')
# simulations = cnx.execute('''SELECT * FROM fast_dhsvm_parallel_run_pnnl_Oct2005toNov2007''')
# simulations = pd.read_sql_query("SELECT * FROM fast_dhsvm_parallel_run_pnnl_Oct2005toNov2007", cnx)

os.chdir('C:/Users/keckj')
simulations = pd.read_pickle("simulation.pkl")
validation = pd.read_csv('validation_1hr_Oct2005toSep2007.csv')
print("Number of Runs:", len(simulations))

#extract best simulation
simulation_subset = simulations.loc[simulations['like1'].idxmax()]

#add to observation dataframe
validation['best'] = simulation_subset[simulation_subset.index.str.contains('simulation')].values
validation.columns = ['Date','Observed', 'Simulation_best']
validation = validation.set_index(pd.to_datetime(validation.Date))


#extract all simulations with NS greater than 0.49
simulations.head()
NSt = simulations[simulations['like1']>.475]

#extract simulations form dataframe to make new dataframe, each column is a model run, index is date
NS = pd.DataFrame()
for index, row in NSt.iterrows():
    NS[index] = row[row.index.str.contains('simulation')].values

NS = NS.set_index(pd.to_datetime(validation.Date))

 
#plot all model runs
fig= plt.figure(figsize=(16,9))
plt.plot(pd.to_datetime(validation.Date), validation.Observed)
for i in NS.columns:
    print(i)
    NS[i].plot()

plt.xlim('2006-11-05','2006-12-01')

plt.show()

#compute mean and interquartile range
mn = []
PercL = []
PercU = []
mini = []
maxi =[]

for index, row in NS.iterrows():
    mn.append(row.mean())
    mini.append(row.min())
    maxi.append(row.max())
    PercL.append(row.quantile(q=0.05))
    PercU.append(row.quantile(q=.95))

#add to NS dataframe
NS['5p'] = PercL
NS['95p'] = PercU
NS['mean'] = mn
NS['min'] = mini
NS['max'] = maxi


#plot best model run with observations and interquartile range
fig= plt.figure(figsize=(16,9))
os.chdir('D:/UW_PhD/PreeventsProject/Paper_1/figures')
#plt.fill_between(NS.index,NS['5p'].values,NS['95p'].values,color='grey', alpha=0.5,label = 'modeled range')
plt.fill_between(NS.index,NS['min'].values,NS['max'].values,color='grey', alpha=0.5,label = 'modeled range')
plt.plot(pd.to_datetime(validation.Date), validation.Observed,'k--',alpha=0.5, label = 'observed')
plt.plot(pd.to_datetime(validation.Date), validation.Simulation_best,label='modeled best')
#NS['75p'].plot(style = 'r:')
NS['mean'].plot(style = 'k-',linewidth=2,label = 'modeled mean')


plt.xlim('2006-11-04','2006-11-22')
#plt.xlim('2007-01-01 12:00','2007-01-12 12:00')
plt.legend(loc='best',fontsize=16)
plt.xlabel ('Date',fontsize=16)
plt.ylabel('Flow [m3/s]',fontsize=16)
plt.tick_params(labelsize=16)
plt.grid(which='both')
#plt.savefig('calibration_illustration.jpg',dpi=300, bbox_inches = 'tight')  

plt.show()


#compute ns for daily values
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
    SqDif = (Qobs-Qmod)**2
    SSqDif = SqDif.sum()
    MnDif = (Qobs-Qobs.mean())**2
    SMnDif = MnDif.sum()
    
    NS = 1-(SSqDif/SMnDif)
    
    return NS



#convert all hourly time series to daily

NSd = pd.DataFrame()

#model results
for i in NS.columns:
    NSd[i] = pd.to_numeric(NS[i]).resample('D').mean()

#observations
obs_d = pd.to_numeric(validation['Observed']).resample('D').mean()

NSdsum = {}   
for i in NS.columns:
    print(i)
    NSdsum[i] = NashSutcliffe(obs_d,NSd[i])


