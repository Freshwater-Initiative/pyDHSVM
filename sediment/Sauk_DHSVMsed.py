# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 16:30:00 2018

@author: cband
Coded to be interoperable with HydroShare Utilities or run local
"""
## -*- coding: utf-8 -*-
#"""
#Created on Sat Jun 10 14:44:30 2017
#
#@author: cbev
#"""
#
#%% Processing Options-- 'yes' or 'no'
setup_mg='no'
process_met='no'
plot_met='no'
process_obs='no'
process_qmod='yes' #modeled streamflow
process_network='yes' 

#%% Import modules and define functions
import os
import csv
import pandas as pd
import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
#import geopandas as gpd
#from shapely.geometry import Point
#import xarray as xr
import pickle
import scipy.stats

# Landlab modules
#from landlab import RasterModelGrid
#from landlab.plot.imshow import imshow_grid
#from landlab.io import read_esri_ascii
#%% Parameters
# Universal constants
g=9.81                     # acceleration of gravity (m/s2)
rho_w=1000                 # density of water (kg/m3)
v=10**-6                   # kinematic viscosity of water (m2/s)
sg=2.65                    # specific gravity 
bd_fine=1.13                # bulk density of fine sediments in reservoir 
bd_coarse=1.71              # bulk density of coarse sediments in reservoir 

# Directories
## sediment model
model_dir='C:/Users/cband/Sauk/sediment/SaukSediment-develop/SaukSediment-develop/'
model_input_folder=model_dir+'input/'
model_data_folder=model_dir+'data/'

# Files
## model_input_folder files
network_input='streamfile_1km2_sauk.txt'

## dhsvm_network_folder files
networkdat_input='stream.network.dat'
map_input='stream.map.dat'

## dhsvm_output_folder
streamflow_input='Outflow.Only'

# Parameterization
## time parameters
forcing_start_date=datetime.date(2006,10,1)
forcing_end_date=datetime.date(2006,12,1)

## hydraulic geometry (HG) parameters (note: a is constant, b is exponent)
ref_stream= 144 # DHSVM stream number where observations are collected that is used to scale other streams. For Elwha, this is the Lake Mills stream.

# stream width @ ref stream
w_const_ref=41.6
## stream depth HG parameters @ ref stream
a_d_ref=0.24
b_d_ref=0.41
### stream velocity HG parameters @ ref stream
a_u_ref=0.10
b_u_ref=0.58

## stream width HG upstream exponent
exp_w_us=0.5
## stream depth HG upstream exponent
exp_d_us=0.4
## stream velocity HG upstream exponent
exp_u_us=0.1

# roughness parameterization
a_n=1.08
b_n=-0.44

## geomorphology
D=0.2/365.25 # denudation rate [mm/day]- based on computations/literature
beta_mw=20*365.25 # lag time [days] between mass wasting events- exponenetial distribution parameter
init_depth=1 # Set initial depth of sediment in channels [m] 
abrasion_alpha=0.027
abrasion_alpha_Fg=0.027
mw_pcnt_g=0.7
mw_pcnt_s=1-mw_pcnt_g

## sediment transport parameterization per Wilcock and Crowe, 2003
# Channel grain size properties at Lake mills gage stream
Ch_Fs_LM=0.37
Ch_Fg_LM=1-Ch_Fs_LM
d90ch_LM=0.0275 # m -  d90 of channel bed= 27.5 mm=0.0275 m
dsandch_LM=0.00093 # m- d_sand of channel bed= 0.93 mm = 0.00093 m
dgravelch_LM=0.0132 # m- d_gravel of channel bed= 13.2 mm = 0.0132 m
dmeanch_LM=0.0125 # m- d_mean of channel bed= 12.5 mm = 0.0125 m

# Wilcock- Crowe Equation Parameters
A_gravel=14
chi_gravel=0.894
exp_gravel=0.5
phi_prime_gravel=1.35

A_sand=14
chi_sand=0.894
exp_sand=0.5
phi_prime_sand=1.35

# Suspended Sediment Equation Parameters
a_s=1.17*10**-4 # regression coefficient from Curran et al., 2009
b_s_c=3        # regression coefficient for SS concentration from Curran et al., 2009
b_s_l=4        # regression coefficient for SS load from Curran et al., 2009
cf_s=1.07      # log-regression correction factor from Curran et al., 2009
K_ss=0.0864       # unit conversion factor from Curran et al., 2009  


# Network suspended sediment parametrization
# Key references: Patil et al., 2012
# for sand equations
c0_ss=1.1038
c1_ss=2.6626
c2_ss=5.6497
c3_ss=0.3822
c4_ss=-0.6174
c5_ss=0.1315
c6_ss=-0.0091

# for silt equations
tau_c_fines=0.015*(bd_fine-1)
a_w_m=0.08
n_w_m=1.65
b_w_m=3.5
m_w_m=1.88
c1_m=0.15
c2_m=b_w_m/((2*m_w_m-1)**0.5)
#%% Functions
#  "Find nearest" function
def find_nearest(array,value):
    val = (np.abs(array-value)).argmin()
    return array[val]

# Upload observed data
def create_q_obs_df(file_name, drainage_area):
    q=pd.read_excel(file_name, sheetname='data', skiprows=[0], header=None, usecols='A:D')
    q.columns=['year','month','day','flow_cfs']
    q_dates=pd.to_datetime(q.loc[:,['year','month','day']])
    q.set_index(q_dates, inplace=True)
    q.drop(['year','month','day'],axis=1, inplace=True)
    q_cms=q.flow_cfs/(3.28084**3)
    q_mmday=q_cms*1000*3600*24/drainage_area
    q=pd.concat([q_cms, q, q_mmday],axis=1)
    q.columns=['flow_cms','flow_cfs', 'flow_mmday']
    return q

def import_obs_folders(obs_folder,streamflow_obs_input_txt):
    os.chdir(obs_folder)
    LM_usgs= np.genfromtxt(streamflow_obs_input_txt,skip_header=1,dtype=str)                   
    # Extract Dates- get year, month, day into datetime objects:
    n_1=len(LM_usgs[:,0]) # n is number of days in the record
    date_LM=np.full(n_1,'', dtype=object) # Preallocate date_1 matrix
    for x in range(0,n_1): # Cycle through all days of the year
        date_LM_temp=datetime.date(int(LM_usgs[x,0]),int(LM_usgs[x,1]),
                                               int(LM_usgs[x,2]))
        # make numpy array of individual temporary datetime objects
        date_LM[x]=date_LM_temp # enter temporary object into preallocated date matrix
    del(date_LM_temp) # delete temporay object
    
    # Extract remaining variables and convert to standard units:
    Q_day=np.array((LM_usgs[:,3]), dtype='float64')
    Q_day=Q_day/(3.28084**3) # convert from ft^3/s to m^3/s
    SSC_day=np.array(LM_usgs[:,7], dtype='float64')
    SSL_day=np.array(LM_usgs[:,9], dtype='float64')
    T_day=np.array(LM_usgs[:,11], dtype='float64')
    
    LM_data=pd.DataFrame({"Q_m3s": Q_day, "SSC_mgL": SSC_day,
                          "SSL_tonsday":SSL_day, "T_fnu":T_day},index=pd.to_datetime(date_LM))
    
    # Now find the 50 % exceedance flow at Lakem Mills gage- use complete water years only
    Q_curve=LM_data.loc[datetime.date(1994, 10, 1):datetime.date(1997, 9, 30), 'Q_m3s'].values
    Q_curve=np.append(Q_curve,LM_data.loc[datetime.date(2004, 10, 1):datetime.date(2011, 9, 30), 'Q_m3s'].values)
    
    # FUTURE:  Use calibrated DHSVM outputs instead?
    Q_RC=np.sort(Q_curve)
    n_curve=len(Q_curve)
    ep_Q_curve=np.array(range(1,n_curve+1))/(n_curve+1)
    cum_pcntl=ep_Q_curve
    Q_LM_50EP=Q_RC[np.where(cum_pcntl==find_nearest(cum_pcntl, 0.50))] # 50th percentile
            
    return (LM_data, Q_LM_50EP)

def compute_NSE_rs (modeled, observed):
    NSE=1-((np.sum((modeled-observed)**2))/(np.sum((modeled-np.mean(observed))**2)))
    WC,WC_0,WC_r, WC_p, WC_err=scipy.stats.linregress(modeled, observed)
    r2=WC_r**2
    print('r2=',r2)
    print('NSE=',NSE)
    return NSE, r2


def setup_network(input_folder):                      
    os.chdir(model_input_folder)  
    network=pd.read_table(network_input, delimiter='\t', index_col=0,\
                          usecols=[0, 1, 2, 3, 4, 5, 6, 7])
    network.columns=['segment_length_m','local_ca','dest_channel_id',\
                     'segment_slope','total_ca_mean','segment_order',\
                     'channel_class_id']  
    return (network)



def run_stochastic_mass_wasting(ref_stream, a_u_ref, b_u_ref, a_d_ref, b_d_ref, a_n, b_n, ng_obs_bar, S, total_ca_ref, total_ca, Qref, Q):  
    rho_w=1000
    # ref stream values for given flow
    Uref=a_u_ref*Qref**b_u_ref
    Dref=a_d_ref*Qref**b_d_ref
    # stream-of-interest values for given flow
    U=Uref*(total_ca**exp_w_us)/(total_ca_ref**exp_w_us)
    D=Dref*(total_ca**exp_d_us)/(total_ca_ref**exp_d_us)
    ng=ng_obs_bar*a_n*D**b_n
    tau=rho_w*g*((ng*U)**(3/2))*S**(1/4)
    u=(tau/rho_w)**0.5
    return tau, u         

def compute_channel_properties(ref_stream, a_u_ref, b_u_ref, a_d_ref, b_d_ref, a_n, b_n, ng_obs_bar, S, total_ca_ref, total_ca, Qref, Q):  
    rho_w=1000
    # ref stream values for given flow
    Uref=a_u_ref*Qref**b_u_ref
    Dref=a_d_ref*Qref**b_d_ref
    # stream-of-interest values for given flow
    U=Uref*(total_ca**exp_u_us)/(total_ca_ref**exp_u_us)
    D=Dref*(total_ca**exp_d_us)/(total_ca_ref**exp_d_us)
    ng=ng_obs_bar*a_n*D**b_n
    tau=rho_w*g*((ng*U)**(3/2))*S**(1/4)
    u_star=(tau/rho_w)**0.5
    return tau, u_star         
        
def run_wc2003_2F_model (tau, tau_r_sand,tau_r_gravel):
    # Constants
    A=14
    chi=0.894
    exp=0.5
    phi_prime=1.35 
             
    # Run WC 2003 Two-Fraction Model
    phi_gravel=tau/tau_r_gravel
    phi_sand=tau/tau_r_sand
    
    if phi_gravel<phi_prime:
        Wstar_gravel=0.002*(phi_gravel)**7.5
    elif chi/((phi_gravel)**exp)>=1: # Checka that term in paraentheses is not negative
       Wstar_gravel=0
    else:
        Wstar_gravel=A*((1-(chi/((phi_gravel)**exp)))**(4.5))

    if phi_sand<phi_prime:
        Wstar_sand=0.002*(phi_sand)**7.5
    elif chi/((phi_sand)**exp)>=1: # Checka that term in paraentheses is not negative
       Wstar_sand=0
    else:
        Wstar_sand=A*((1-(chi/((phi_sand)**exp)))**(4.5))        
    return (Wstar_gravel, Wstar_sand)    
#%%
if process_qmod=='yes':
    os.chdir(dhsvm_output_folder)
    stream_columns=['Date',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439]
    
    forcing_date_range=pd.date_range(forcing_start_date, forcing_end_date)
    streamflow=pd.DataFrame(index=forcing_date_range, columns=stream_columns)
    chunksize = 8
    i=0
    for chunk in pd.read_table(streamflow_input, skiprows=[0,1], header=None,\
                               sep='\s+', chunksize=chunksize):
        chunk.columns=stream_columns
        streamflow.loc[forcing_date_range[i],:]=chunk.sum(axis=0)[1::]/86400
        i=i+1
    last_day=pd.read_table(streamflow_input, skiprows=np.arange(0,(i-1)*8+2), 
                           header=None,sep='\s+')
    last_day.columns=stream_columns
    streamflow.loc[forcing_date_range[i],:]=last_day.sum(axis=0)[1::]/86400
    Qmod_median=streamflow.median(axis=0)
    os.chdir(model_data_folder)
    pickle.dump(streamflow, open("streamflow.py", "wb"))
    pickle.dump(Qmod_median, open("Qmod_median.py", "wb"))


#%% Setup network file
if process_network=='yes':
    network=setup_network(model_input_folder)
    
    # Downstream links and distances
    ds=pd.DataFrame(index=network.index, columns=['ds_strms','ds_dist_array','ds_dist_m']) # establish array of downstream stream numbers
    strm_orders_rev=np.unique(network.segment_order)[::-1] # get all segement orders in the watershed and sort from highest to lowest
    for st_o in strm_orders_rev:
        for i in network.index:
            if network.segment_order.loc[i]==st_o: # go in order of stream orders
                j=network.loc[i, 'dest_channel_id']
                if j in network.index:
                    ds.loc[i,'ds_strms']=np.append(j, ds.loc[j,'ds_strms'])
                    ds.loc[i,'ds_dist_array']=np.append(network.loc[j, 'segment_length_m'], ds.loc[j,'ds_strms'])
                    ds.loc[i,'ds_dist_m']=np.nansum(ds.loc[i,'ds_dist_array'])
                else:
                    ds.loc[i,'ds_strms']=[]
                    ds.loc[i,'ds_dist_array']=[]
                    ds.loc[i,'ds_dist_m']=0
    
    network=pd.concat([network, ds], axis=1, join_axes=[network.index]) # add these to network array 
    network[['ds_dist_m']]=network[['ds_dist_m']].astype(float)
    network = network.loc[:,~network.columns.duplicated()] # Ensure that there are no duplicate columns!
    
    # Stream Width- assume constant
    width=pd.DataFrame(index=network.index, columns=['width'],
                       data=w_const_ref*((network['total_ca_mean'].values)**exp_w_us)/(((network['total_ca_mean'][ref_stream])**exp_w_us)))
    network=pd.concat([network, width],axis=1, join_axes=[network.index]) # add these to network array
    
    # Grain Size
    os.chdir(model_data_folder)
    strm_link_vals=pickle.load(open('strm_link_vals.py', 'rb')) # stream link values of interest
    n_strms=len(strm_link_vals) # number of streams/tributary areas
    strm_orders=np.unique(network.segment_order)
    
    ## Upstream grain size diameters from Downstream Fining Equation
    d90_ch_us=d90ch_LM/np.exp(-abrasion_alpha*(network.ds_dist_m-network.ds_dist_m[ref_stream])/1000)
    dsand_ch_us=dsandch_LM/np.exp(-abrasion_alpha*(network.ds_dist_m-network.ds_dist_m[ref_stream])/1000)
    dgravel_ch_us=dgravelch_LM/np.exp(-abrasion_alpha*(network.ds_dist_m-network.ds_dist_m[ref_stream])/1000)
    dmean_ch_us=dmeanch_LM/np.exp(-abrasion_alpha*(network.ds_dist_m-network.ds_dist_m[ref_stream])/1000)
    
    
    # Roughness
    if 'Qmod_median' not in locals():
        os.chdir(model_data_folder)
        Qmod_median=pickle.load(open('Qmod_median.py', 'rb'))
    H_median_lm=a_d_ref*(Qmod_median[ref_stream])**b_d_ref   
    H_median_all=H_median_lm*((network['total_ca_mean'].values)**exp_d_us)/(((network['total_ca_mean'][ref_stream])**exp_d_us))
    ng_bar=(1/(np.sqrt(8*g)))*(H_median_all**(1/6))/(1.26-2.16*np.log10(d90_ch_us/H_median_all))
    
    network=pd.concat([network,pd.DataFrame({"d90_ch_m": d90_ch_us,
                                          "dsand_ch_m": dsand_ch_us,
                                          "dgravel_ch_m": dgravel_ch_us,
                                          "dmean_ch_m": dmean_ch_us,
                                          "Ch_Fs_strms": Ch_Fs_LM, # start out with all the same as LM and will let evolve
                                          "ng_bar":ng_bar},index=strm_link_vals)], axis=1, join_axes=[network.index]) 
    
    network=network.loc[strm_link_vals,:]
    del(dsand_ch_us, dgravel_ch_us, dmean_ch_us)
    os.chdir(model_data_folder)
    pickle.dump(network, open("network.py", "wb"))



#%% Upload model data 
    
os.chdir(model_data_folder)
network=pickle.load(open('network.py', 'rb'))
streamflow=pickle.load(open('streamflow.py', 'rb'))

#Setup for Elwha with upper network values only
strm_link_vals=pickle.load(open('strm_link_vals.py', 'rb')) # stream link values of interest

#Setup for Sauk will all values in network
#strm_link_vals=network.index # stream link values of interest

        
#%%
# INITIAL COMPUTATIONS/VARIABLE ALLOCATIONS
n_strms=len(strm_link_vals) # number of streams/tributary areas
strm_orders=np.unique(network.segment_order) # array of unique stream orders
start_year_len=len(streamflow.index[streamflow.index.year==streamflow.index[0].year]) # number of days in first year

# Preallocate sediment volume data frames representing sediment "buckets" at the end of each timestep (day)
# Vg=gravel; Vs=sand; Vm=mud (silt, clay)
# Volumes stored on bed [m3]
Vg_b=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
Vs_b=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
Vm_b=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))

# Volumes deposited from mass wasting event [m3]
Vg_mw=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
Vs_mw=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
Vm_mw=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))

# Volume of bedload that stream had capacity to transport  [m3]
Vg_cap=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
Vs_cap=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))

# Volume transported out of stream  [m3]
Vg_t=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
Vs_t=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
Vsb_t=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
Vss_t=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
Vm_t=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))

# Preallocate sediment volume data frames representing sediment "buckets" at the end of each year
# Volumes stored on bed [m3]
Vg_b_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
Vs_b_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
Vm_b_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
# Volumes deposited from mass wasting event [m3]
Vg_mw_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
Vs_mw_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
Vm_mw_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
# Volume of bedload  that stream had capacity to transport  [m3]
Vg_cap_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year)) 
Vs_cap_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
# Volume transported out of stream  [m3]
Vg_t_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
Vs_t_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
Vsb_t_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
Vss_t_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))
Vm_t_annual=pd.DataFrame(0, index=strm_link_vals, columns=np.unique(streamflow.index.year))

# Other- temporary
C_ss=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))
Vs_dep=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,start_year_len+1))

#Set any initial (time=0) conditions in streams
# Volumes stored on bed [m3]
Vg_b.loc[:,0]=0.33*init_depth*network.segment_length_m*network.width # set initial volume of sediment stored in channel
Vs_b.loc[:,0]=0.34*init_depth*network.segment_length_m*network.width # set initial volume of sediment stored in channel
Vg_b.loc[:,0]=0.33*init_depth*network.segment_length_m*network.width # set initial volume of sediment stored in channel
# Volumes deposited from mass wasting event [m3]
Vg_mw.loc[:,0]=0
Vs_mw.loc[:,0]=0
Vm_mw.loc[:,0]=0

# Volume that stream had capacity to transport  [m3]
Vg_cap.loc[:,0]=0
Vs_cap.loc[:,0]=0

# Volume transported out of stream  [m3]
Vg_t.loc[:,0]=0
Vs_t.loc[:,0]=0
Vm_t.loc[:,0]=0

# Set arrays for times between mass wasting events
time_mw=pd.DataFrame(0, index=strm_link_vals, columns=['output']) # time since last mass wasting event for each stream link
tL_mw=np.ceil(np.random.exponential(scale=beta_mw, size=len(strm_link_vals))) # time (years) until first mass wasting event- randomly sampled from exponential distribution
tL_mw=pd.DataFrame(data=tL_mw, index=strm_link_vals, columns=['output']) # turn into a data frame that will grow

#%%
start = time.time()
date_string=time.strftime("%Y-%m-%d_%H%M")
os.mkdir(date_string)
day_of_year=0
## RUN MODEL
for i in range (0,len(streamflow.index)):
    print(streamflow.index[i])
    date=streamflow.index[i]
    day_of_year=day_of_year+1
    time_mw=time_mw+1 # start new year
   # run stochastic mass wasting generator for all stream links
    for st_o in strm_orders:
        for j in network.index.values:
            if network.segment_order.loc[j]==st_o: # go in order of stream orders
                if time_mw.loc[j,'output']==tL_mw.loc[j,'output']: # if time since the last mass wasting event is the same as randomly generated lag time
                    # Generate random values for percent gravel, sand, and mud,
                    temp_rand=np.random.rand(3)
                    mw_pcnts=temp_rand/sum(temp_rand)
                    # Generate mass wasting volumes based on denudation rate
                    Vg_mw.loc[j,day_of_year]=mw_pcnts[0]*Drate*network.loc[j,'local_ca']*(1/1000)*tL_mw.loc[j,'output']
                    Vs_mw.loc[j,day_of_year]=mw_pcnts[1]*Drate*network.loc[j,'local_ca']*(1/1000)*tL_mw.loc[j,'output']                   
                    Vm_mw.loc[j,day_of_year]=mw_pcnts[2]*Drate*network.loc[j,'local_ca']*(1/1000)*tL_mw.loc[j,'output']                   
                    # Update mass wasting clocks
                    time_mw.loc[j,'output']=0 # Reset counter until next mass wasting event
                    tL_mw.loc[j,'output']=np.ceil(np.random.exponential(scale=beta_mw))  # generate a new lag time until next mass wasting event and add it to array of lag times#
                else: # If have not reach the lag time for a mass wasting event at the stream then the volume will be the same as last year
                    # Not mass wasting volume is added
                    Vg_mw.loc[j,day_of_year]=0
                    Vs_mw.loc[j,day_of_year]=0
                    Vm_mw.loc[j,day_of_year]=0

                # Add mass wasting volume to the bed from previous timestep
                Vg_b.loc[j,day_of_year]=Vg_b.loc[j,day_of_year-1]+Vg_mw.loc[j,day_of_year]
                Vs_b.loc[j,day_of_year]=Vs_b.loc[j,day_of_year-1]+Vs_mw.loc[j,day_of_year] 
                Vm_b.loc[j,day_of_year]=Vm_b.loc[j,day_of_year-1]+Vm_mw.loc[j,day_of_year] 

                # Add in sediment from upstream tributaries
                if st_o!=1: # if not at a headwater stream (i.e., stream order >1), need to add volume of sediment coming from upstream
                    feeder_links=network.loc[network['dest_channel_id']==j].index.values # find the stream links that feed into the link which are immediately upstream
                    # sum up the volume transported out of the upstream link and add to current volume
                    Vg_b.loc[j,day_of_year]=Vg_b.loc[j,day_of_year]+np.nansum(Vg_t.loc[feeder_links,day_of_year])  
                    Vs_b.loc[j,day_of_year]=Vs_b.loc[j,day_of_year]+np.nansum(Vs_t.loc[feeder_links,day_of_year])
                    Vm_b.loc[j,day_of_year]=Vm_b.loc[j,day_of_year]+np.nansum(Vm_t.loc[feeder_links,day_of_year])
                
                # Compute shear stress and shear velocity from modeled flow
                Q=streamflow.loc[date,j]
                Qvol=Q*3600*24 # m3 of streamflow throughout the day [m3]
                tau_strm, u_strm= compute_channel_properties(ref_stream, a_u_ref, b_u_ref, a_d_ref, b_d_ref, a_n, b_n, network['ng_bar'][j], 
                                                             network.segment_slope[j], network['total_ca_mean'][ref_stream], network['total_ca_mean'][j], 
                                                             streamflow.loc[date,ref_stream], Q)

                # Suspended Sediment
                C_ss.loc[j,day_of_year]=(Vs_b.loc[j,day_of_year]+Vm_b.loc[j,day_of_year])/Qvol # Concentration of suspended sediment (unitless)
                
                # Suspended sediment- silt: Compute deposition and transport
                if tau_strm>tau_c_fines:
                    Vm_d=0
                else: 
                    if C_ss.loc[j,day_of_year]<=c1_m:
                        v_ss_m=a_w_m*(c1_m**n_w_m)/((c1_m**2+b_w_m**2)**m_w_m)
                    if C_ss.loc[j,day_of_year]>c2_m:
                        v_ss_m=a_w_m*(c2_m**n_w_m)/((c2_m**2+b_w_m**2)**m_w_m)
                    else:
                        v_ss_m=a_w_m*(C_ss.loc[j,day_of_year]**n_w_m)/((C_ss.loc[j,day_of_year]**2+b_w_m**2)**m_w_m)                        
                    Vm_d=(1-(tau_strm/tau_c_fines))*v_ss_m*C_ss.loc[j,day_of_year]*network.loc[j,'segment_length_m']*network.loc[j,'width']*3600*24
                Vm_t.loc[j,day_of_year]=Vm_b.loc[j,day_of_year]-min(Vm_d, Vm_b.loc[j,day_of_year])
                Vm_b.loc[j,day_of_year]=max(Vm_b.loc[j,day_of_year]-Vm_t.loc[j,day_of_year],0)
                
                # Suspended sediment- sand: Compute deposition/erosion
                #v_ss_s=g*(network.loc[j,'dsand_ch_m']**2)*(sg-1)/(18*v) # settling velocity of sand
                d_star_ss=(((sg-1)*g*network.loc[j,'dsand_ch_m']**3)/(v**2))**(1/3)
                v_ss_s=(v/network.loc[j,'dsand_ch_m'])*(np.sqrt((1/4)*(24/1.5)**(2/1)+((4*d_star_ss**3)/(3*1.5))**(1/1))-(1/2)*(24/1.5)**(1/1))**1                
                Z_R=v_ss_s/(0.41*u_strm)
                int_Z_R=1/(c0_ss+c1_ss*Z_R+c2_ss*Z_R**2+c3_ss*Z_R**3+c4_ss*Z_R**4+c5_ss*Z_R**5+c6_ss*Z_R**6)
                Vs_dep_comp=v_ss_s*C_ss.loc[j,day_of_year]/int_Z_R*network.loc[j,'segment_length_m']*network.loc[j,'width']*3600*24 #m3/day- volume of deposited sand
                if Vs_dep_comp<0:
                    Vs_dep.loc[j,day_of_year]=0
                elif Vs_dep_comp>Vs_b.loc[j,day_of_year]:
                    Vs_dep.loc[j,day_of_year]=Vs_b.loc[j,day_of_year]
                else:
                    Vs_dep.loc[j,day_of_year]=Vs_dep_comp

                # Sand transported out as suspended sediment
                Vss_t.loc[j,day_of_year]=Vs_b.loc[j,day_of_year]-Vs_dep.loc[j,day_of_year]
                
                # Sand remaining on bed, with potential to be transferred as bedload
                Vs_b.loc[j,day_of_year]=Vs_dep.loc[j,day_of_year]
                
                # Recompute W&C 2003 parameterizaion of sand and gravel on the channel bed
                network.loc[j,'Ch_Fs_strms']=Vs_b.loc[j,day_of_year]/(Vs_b.loc[j,day_of_year]+Vg_b.loc[j,day_of_year])
                tau_star_rsm=0.021+0.015*np.exp(-20*network.loc[j,'Ch_Fs_strms']) # dimensionless reference shear stress for mean grain size
                tau_rsm=tau_star_rsm*(sg-1)*rho_w*g*(network.loc[j,'dmean_ch_m']) # reference shear stress for mean grain size [N/m2]
                b_sand=0.67/(1+np.exp(1.5-(network.loc[j,'dsand_ch_m']/network.loc[j,'dmean_ch_m']))) # b parameter for sand
                b_gravel=0.67/(1+np.exp(1.5-(network.loc[j,'dgravel_ch_m']/network.loc[j,'dmean_ch_m']))) # b parameter for gravel
                tau_r_sand=tau_rsm*(network.loc[j,'dsand_ch_m']/network.loc[j,'dmean_ch_m'])**b_sand # reference tau for sand [N/m2]
                tau_r_gravel=tau_rsm*(network.loc[j,'dgravel_ch_m']/network.loc[j,'dmean_ch_m'])**b_gravel # reference tau for gravel [N/m2]
                tau_star_r_sand=tau_r_sand/(rho_w*g*(sg-1)*network.loc[j,'dsand_ch_m'])
                tau_star_r_gravel=tau_r_gravel/(rho_w*g*(sg-1)*network.loc[j,'dgravel_ch_m'])
                         
                # Compute Bedload sediment transport capacity with calibrated Wilcock and Crowe equation
                Wstar_gravel, Wstar_sand=run_wc2003_2F_model (tau_strm, tau_star_r_sand, tau_star_r_gravel)
                
                # Compute gravel transport capacity, volume transported, and volume remaining
                Vg_cap.loc[j,day_of_year]=3600*24*((u_strm)**3)*Wstar_gravel*(1-network.loc[j,'Ch_Fs_strms'])/((sg-1)*g)*network.loc[j,'width'] #  m3 (unit is m3/day; timestep is 1 day)
                Vg_t.loc[j,day_of_year]=np.min([Vg_cap.loc[j,day_of_year],Vg_b.loc[j,day_of_year]]) # gravel: volume transported is the minimum between available volume and capacity of flow
                Vg_b.loc[j,day_of_year]=Vg_b.loc[j,day_of_year]-Vg_t.loc[j,day_of_year] # new volume deposited in stream- difference between existing volume and volume transported

                # Compute sand bedload transport capacity, volume transported, and volume remaining
                Vs_cap.loc[j,day_of_year]=3600*24*((u_strm)**3)*Wstar_sand*(network.loc[j,'Ch_Fs_strms'])/((sg-1)*g)*network.loc[j,'width'] # m3 (unit is m3/day; timestep is 1 day)
                Vsb_t.loc[j,day_of_year]=np.min([Vs_cap.loc[j,day_of_year],Vs_b.loc[j,day_of_year]]) # sand: volume transported is the minimum between available volume and capacity of flow
                Vs_t.loc[j,day_of_year]=Vsb_t.loc[j,day_of_year]+Vss_t.loc[j,day_of_year] # add volumes of sand transported as bedload and suspended load 
                Vs_b.loc[j,day_of_year]=Vs_b.loc[j,day_of_year]-Vsb_t.loc[j,day_of_year] # new volume deposited in stream- difference between existing volume and volume transported                             
                
    if date.month==12 and date.day==31:
        if date.year==streamflow.index[streamflow.index.year==streamflow.index[-1].year][0].year:
            next_year_len=0
        else: next_year_len=len(streamflow.index[streamflow.index.year==date.year+1])
        
        Vg_b_last=Vg_b.loc[:, Vg_b.columns[-1]]
        Vs_b_last=Vs_b.loc[:, Vs_b.columns[-1]]
        Vm_b_last=Vm_b.loc[:, Vm_b.columns[-1]]
        Vg_mw_last=Vg_mw.loc[:, Vg_mw.columns[-1]]
        Vs_mw_last=Vs_mw.loc[:, Vs_mw.columns[-1]]
        Vm_mw_last=Vm_mw.loc[:, Vm_mw.columns[-1]]
        Vg_cap_last=Vg_cap.loc[:, Vg_cap.columns[-1]]
        Vs_cap_last=Vs_cap.loc[:, Vs_cap.columns[-1]]
        Vg_t_last=Vg_t.loc[:, Vg_t.columns[-1]]
        Vs_t_last=Vs_t.loc[:, Vs_t.columns[-1]]
        Vsb_t_last=Vsb_t.loc[:, Vsb_t.columns[-1]]
        Vss_t_last=Vss_t.loc[:, Vss_t.columns[-1]]
        Vm_t_last=Vm_t.loc[:, Vm_t.columns[-1]]
        C_ss_last=C_ss.loc[:, C_ss.columns[-1]]
        Vs_dep_last=Vs_dep.loc[:, C_ss.columns[-1]]

        Vg_b_annual.loc[:, date.year]=Vg_b_last
        Vs_b_annual.loc[:, date.year]=Vs_b_last
        Vg_mw_annual.loc[:, date.year]=Vg_mw[1::].sum(axis=1)
        Vs_mw_annual.loc[:, date.year]=Vs_mw[1::].sum(axis=1)
        Vg_cap_annual.loc[:, date.year]=Vg_cap[1::].sum(axis=1)
        Vs_cap_annual.loc[:, date.year]=Vs_cap[1::].sum(axis=1)
        Vg_t_annual.loc[:, date.year]=Vg_t[1::].sum(axis=1)
        Vs_t_annual.loc[:, date.year]=Vs_t[1::].sum(axis=1)
        Vsb_t_annual.loc[:, date.year]=Vsb_t[1::].sum(axis=1)
        Vss_t_annual.loc[:, date.year]=Vss_t[1::].sum(axis=1)
        Vm_t_annual.loc[:, date.year]=Vm_t[1::].sum(axis=1)
        
        np.save('Vg_b_annual', Vg_b_annual)
        np.save('Vs_b_annual', Vs_b_annual)
        np.save('Vg_mw_annual', Vg_mw_annual)   
        np.save('Vs_mw_annual', Vs_mw_annual)   
        np.save('Vg_cap_annual', Vg_cap_annual)
        np.save('Vs_cap_annual', Vs_cap_annual)
        np.save('Vg_t_annual', Vg_t_annual)
        np.save('Vs_t_annual', Vs_t_annual)
        np.save('Vsb_t_annual', Vsb_t_annual)
        np.save('Vss_t_annual', Vss_t_annual)
        np.save('Vm_t_annual', Vm_t_annual)
        np.save('time_mw', time_mw)
        np.save('tL_mw', tL_mw)
                
        Vg_b=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
        Vs_b=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
        Vg_mw=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
        Vs_mw=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
        Vg_cap=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
        Vs_cap=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
        Vg_t=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
        Vs_t=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
        Vsb_t=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
        Vss_t=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
        Vm_t=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1))
        C_ss=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1)) 
        Vs_dep=pd.DataFrame(0, index=strm_link_vals, columns=np.arange(0,next_year_len+1)) 

        Vg_b.loc[:, Vg_b.columns[0]]=Vg_b_last
        Vs_b.loc[:, Vs_b.columns[0]]=Vs_b_last
        Vg_mw.loc[:, Vg_mw.columns[0]]=Vg_mw_last
        Vs_mw.loc[:, Vs_mw.columns[0]]=Vs_mw_last
        Vg_cap.loc[:, Vg_cap.columns[0]]=Vg_cap_last
        Vs_cap.loc[:, Vs_cap.columns[0]]=Vs_cap_last
        Vg_t.loc[:, Vg_t.columns[0]]=Vg_t_last
        Vs_t.loc[:, Vs_t.columns[0]]=Vs_t_last
        Vsb_t.loc[:, Vsb_t.columns[0]]=Vsb_t_last
        Vss_t.loc[:, Vss_t.columns[0]]=Vss_t_last
        Vm_t.loc[:, Vm_t.columns[0]]=Vm_t_last
        C_ss.loc[:, C_ss.columns[0]]=C_ss_last
        Vs_dep.loc[:, Vs_dep.columns[0]]=Vs_dep_last
        
        day_of_year=0


# Save results!
end = time.time()
print(end - start)

np.save('Vg_b_annual', Vg_b_annual)
np.save('Vs_b_annual', Vs_b_annual)
np.save('Vg_mw_annual', Vg_mw_annual)
np.save('Vs_mw_annual', Vs_mw_annual)
np.save('Vg_cap_annual', Vg_cap_annual)
np.save('Vs_cap_annual', Vs_cap_annual)
np.save('Vg_t_annual', Vg_t_annual)
np.save('Vs_t_annual', Vs_t_annual)
np.save('Vsb_t_annual', Vsb_t_annual)
np.save('Vss_t_annual', Vss_t_annual)
np.save('Vm_t_annual', Vm_t_annual)
np.save('Vs_dep', Vs_dep)
np.save('C_ss',  C_ss)
np.save('time_mw', time_mw)
np.save('tL_mw', tL_mw)

#%% PLOT RESUlTS
streams_to_plot=[144, 153, 174]
evaluation_years_plot=np.arange(1927,2012) #np.unique(streamflow.index.year)

for stream_index in streams_to_plot:  
    plot_title=''
    fig, (ax1, ax2, ax3, ax4)=plt.subplots(4,1, sharex=True, sharey=False)        
    
    ax1.plot(evaluation_years_plot,
             Vg_mw_annual.loc[stream_index,evaluation_years_plot]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'b-', linewidth=3, label='gravel')
    ax1.plot(evaluation_years_plot,
             Vs_mw_annual.loc[stream_index,evaluation_years_plot]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'r--', linewidth=3, label='sand')
    ax1.set_title('Stream Number '+str(stream_index)+'\nSediment Deposited from Landslide',fontsize=14)
    ax1.set_ylabel('Depth (m)',fontsize=12)
    ax1.legend()
    
    ax2.plot(evaluation_years_plot, Vg_cap_annual.loc[stream_index,evaluation_years_plot],'b-', linewidth=3)  
    ax2.plot(evaluation_years_plot, Vs_cap_annual.loc[stream_index,evaluation_years_plot],'r--', linewidth=3)  
    ax2.set_title('Bedload Transport Capacity of Stream',fontsize=14)
    ax2.set_ylabel('Depth (m)',fontsize=12)
    #ax2.set_ylabel('Sediment\nDischarge\n(m3/s)',fontsize=12)
    
    ax3.plot(evaluation_years_plot,Vg_t_annual.loc[stream_index,evaluation_years_plot], 'b-',linewidth=3)
    ax3.plot(evaluation_years_plot,Vs_t_annual.loc[stream_index,evaluation_years_plot], 'r--',linewidth=3)
    ax3.set_title('Sediment Transported Out of Channel',fontsize=14)
    ax3.set_ylabel('Depth (m)',fontsize=12)
    #ax3.set_ylabel('Sediment\nDischarge\n(m3/s)',fontsize=12)
    
    ax4.plot(evaluation_years_plot,
             Vg_b_annual.loc[stream_index,evaluation_years_plot]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'b-', linewidth=3)
    ax4.plot(evaluation_years_plot,
             Vs_b_annual.loc[stream_index,evaluation_years_plot]/(network.loc[stream_index,'width']*network.loc[stream_index,'segment_length_m']),
             'r--', linewidth=3)
    ax4.set_title('Sediment Accumulated on Channel Bed',fontsize=14)
    ax4.set_ylabel('Depth (m)',fontsize=12)
    ax4.set_xlabel('Year',fontsize=16)
    
    sum_transported_gravel=np.sum(Vg_t_annual.loc[stream_index,evaluation_years_plot])
    sum_transported_sand=np.sum(Vs_t_annual.loc[stream_index,evaluation_years_plot])
    print('Stream Number', stream_index, ', Total gravel transported (10^6 m3)=',sum_transported_gravel/10**6)
    print('Stream Number', stream_index, ', Total sand transported(10^6 m3)=',sum_transported_sand/10**6)
    print('Stream Number', stream_index, ', Total sediment transported (10^6 m3)=', (sum_transported_gravel+sum_transported_sand)/10**6)
    if stream_index==ref_stream:
        sum_dep_gravel=np.sum(Vg_b_annual.loc[stream_index,evaluation_years_plot[-1]])
        sum_dep_sand=np.sum(Vs_b_annual.loc[stream_index,evaluation_years_plot[-1]])           
        print('Stream Number', stream_index, ', Reservoir Sedimentation (10^6 m3)=', (sum_dep_gravel+sum_dep_sand+sum_transported_gravel+sum_transported_sand)*sg/(bd_coarse*10**6))
        print('Stream Number', stream_index, ', Reservoir Percent Gravel=', (sum_dep_gravel+sum_transported_gravel)/(sum_dep_gravel+sum_dep_sand+sum_transported_gravel+sum_transported_sand))
        print('Stream Number', stream_index, ', ReservoirPercent Sand=', (sum_dep_sand+sum_transported_sand)/(sum_dep_gravel+sum_dep_sand+sum_transported_gravel+sum_transported_sand))
    print('--------------------')

