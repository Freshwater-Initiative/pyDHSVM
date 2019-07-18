# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 20:59:08 2019

@author: keckj
"""

import os
import pandas as pd
import numpy as np

os.chdir('D:/UW_PhD/QualifyingExam/DHSVM_Routing_Check/')

clsvals = pd.read_excel('Sauk_DetermineRoutingParameter.xlsx',sheet_name = 'dhsvm_classes', header = 0)
                        

f = open('StreamClassDat.txt','w+')

g = open('StreamClassCode.txt','w+')

f.write('#ID W D n inf \r')
                        
                        
ca = clsvals['ca [m2]'].iloc[::-1].reset_index(drop=True)
slp = clsvals['slp [m/m]']#.iloc[::-1]
wdt = clsvals['width [m]'].iloc[::-1].reset_index(drop=True)
dep = clsvals['depth [m]'].iloc[::-1].reset_index(drop=True)
mn =  clsvals['mannings n']#.iloc[::-1]
# ID W  D   n    inf

c=1
for c1,i in enumerate(slp):
    for c2,j in enumerate(ca):
        print(c1)
        r2 = c #channel class
        r3 = dep.loc[c2] #depth
        r4 = wdt.loc[c2] #width
        r5 = wdt.loc[c2] #width
        r6 = mn.loc[c1] #mannings n

        #write stream.class.dat file
        f.write('%d' % r2 + ' %.2f' % r4 +' %.2f' % r3 +' %.4f' % r6 +  ' 0' +'\r' ) #d - distplays value of "sn", r- return, or end command
        
        #write stream class code that can be copied into channelclass.py
        if c1 == 0 and c2 ==0 : #if (row[0] <= 0.002 and row[1] <= 1000000):
            g.write('if (row[0] <= ' ' %.4f' % i + ' and row[1] <= ' '%d' % j + '):\r')# % ca + '+):' + '\r')        

        elif c1 == 0 and c2 == ca.shape[0]-1: #elif (row[0] <= 0.002 and row[1] > 40000000) :
            g.write('elif (row[0] <= ' ' %.4f' % i + ' and row[1] > ' '%d' % ca[c2-1] + '):\r')# % ca + '+):' + '\r')
    
        elif c1 == 0: #elif (row[0] <= 0.002 and (row[1] > 1000000 and row[1] <= 10000000)):
            g.write('elif (row[0] <= ' ' %.4f' % i + ' and (row[1] > ' ' %d' % ca[c2-1] +' and row[1] <= ''%d' % j + ')):\r')# % ca + '+):' + '\r')
        
        elif c1 == slp.shape[0]-1 and c2 == ca.shape[0]-1: #elif (row[0] > 0.01 and row[1] > 40000000) :
            g.write('elif (row[0] > ' ' %.4f' % slp[c1-1]  +' and row[1] > ' '%d' % ca[c2-1] + '):\r')# % ca + '+):' + '\r')
               
        elif c2 == 0 and c1 == slp.shape[0]-1: #elif ((row[0] > 0.002 and row[0] <= 0.1) and row[1] <= 1000000):
            g.write('elif (row[0] > ' ' %.4f' % slp[c1-1] + ' and row[1] <= ' '%d' % j + '):\r')# % ca + '+):' + '\r')

        elif c2 == ca.shape[0]-1: #elif ((row[0] > 0.002 and row[0] <= 0.1) and row[1] <= 1000000):
            g.write('elif ((row[0] > ' ' %.4f' % slp[c1-1] + ' and row[0] <= ' ' %.4f' % i +') and row[1] > ' '%d' % ca[c2-1] + '):\r')# % ca + '+):' + '\r')
       
        elif c2 == 0: #elif ((row[0] > 0.002 and row[0] <= 0.1) and row[1] <= 1000000):
            g.write('elif ((row[0] > ' ' %.4f' % slp[c1-1] + ' and row[0] <= ' ' %.4f' % i +') and row[1] <= ' '%d' % j + '):\r')# % ca + '+):' + '\r')
        
        elif c1  == slp.shape[0]-1: #elif (row[0] > 0.01 and (row[1] > 30000000 and row[1] <= 40000000)):
            g.write('elif (row[0] > ' ' %.4f' % slp[c1-1] + ' and (row[1] > ' '%d' %ca[c2-1] + ' and row[1] <= ' ' %d' % j + ')):\r')# % ca + '+):' + '\r')
#            
        else: # elif ((row[0] > 0.002 and row[0] <= 0.1) and (row[1] > 1000000 and row[1] <= 10000000)):
            g.write('elif ((row[0] > ' ' %.4f' % slp[c1-1] + ' and row[0] <= ' ' %.4f' % slp[c1] +') and (row[1] > ' '%d' % ca[c2-1] + ' and row[1] <= ' ' %d' % j + ')):\r')# % ca + '+):' + '\r')
                
        g.write('    row[2] = ' '%d' %c + '\r')
        g.write('    row[3] = ' '%.2f' % r3 + '\r')
        g.write('    row[4] = ' '20\r')
        #g.write('    row[4] = ' '%.2f' % r4 + '\r')
        g.write('    row[5] = ' '20\r')
        #g.write('    row[5] = ' '%.2f' % r4 + '\r')
        
        c +=1
        
f.close()
g.close()