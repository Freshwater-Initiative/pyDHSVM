from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import subprocess
from distutils.dir_util import remove_tree
import shutil
import time
import spotpy
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def plot_results(results, threshold=0.2):
    names = []
    values = []
    no_names = []
    no_values = []
    index = []
    no_index = []

    parnames = spotpy.analyser.get_parameternames(results)
    sensitivity_data = spotpy.analyser.get_sensitivity_of_fast(results,M=4)
    sensitivity_data = list(sensitivity_data.values())[1]
    
    with open('sensitivity_data_run_pnnlOct2005toSep2007.txt', 'w') as f:
        f.writelines([str(pname) + ", " + str(s) + "\n" for (pname, s) in zip(parnames, sensitivity_data)])

    for j in range(len(sensitivity_data)):
        if sensitivity_data[j] > threshold:
            names.append(parnames[j])
            values.append(sensitivity_data[j])
            index.append(j)
        else:
            no_names.append(parnames[j])
            no_values.append(sensitivity_data[j])
            no_index.append(j)

    fig = plt.figure(figsize=(16,6))
    ax = plt.subplot(1,1,1)
    ax.bar(index, values, align='center')
    ax.bar(no_index, no_values, color='orange', label = 'Insensitive parameter')

    ax.plot(np.arange(-1,len(parnames)+1,1),[threshold]*(len(parnames)+2),'r--')
    plt.xticks(list(range(len(sensitivity_data))), parnames,rotation = 15)
    plt.setp(ax.get_xticklabels(), fontsize=10)
    fig.savefig('FAST_sensitivity_run_pnnlOct2005toSep2007.png',dpi=300)

#Function to write to DHSVM config file simulated inputs from DREAM.
def change_setting(config_file, setting_name, new_value, occurrence_loc='g'):
    sed_cmd = "sed -i 's:{setting_name} = .*:{setting_name} = {new_value}:{occurrence_loc}' {config_file}"
    sed_cmd = sed_cmd.format(setting_name = setting_name, new_value = new_value
                             , config_file = config_file
                             , occurrence_loc = occurrence_loc)
    return subprocess.call(sed_cmd, shell=True)

#Files and their input paths

config_file = 'Input.sauk.dynG.pnnl.Oct2005toSep2007'
streamflow_only = 'output/PNNLWRF/Streamflow.Only'
validation_csv = 'validation_1hr_Oct2005toSep2007.csv'
dhsvm_cmd = '/home/ubuntu/DHSVM-PNNL/DHSVM/sourcecode/DHSVM3.1.3 ' + config_file 
DIR_PREFIX = "dhsvm_run_pnnl_data_pid_"

# Cheap logging
def log(s):
    print(time.asctime() + " ({})".format(os.getpid()) + ": " + str(s))

class fast_run_setup(object):
    def __init__(self, parallel='seq'):
        
        self.params = [spotpy.parameter.Uniform('exponential_decrease_62',low=0.5, high=3,  optguess=1.5),
                       spotpy.parameter.Uniform('lateral_conductivity_62',low=0.0002, high=0.0015,  optguess=0.0008),
                       spotpy.parameter.Uniform('Snow Threshold',low=-6, high=6,  optguess=2),
                       ]

        evaluation = pd.read_csv(validation_csv)
        self.evals = evaluation['value'].values
        self.parallel = parallel

    def parameters(self):
        return spotpy.parameter.generate(self.params)
    
    #setting up simulation for location:12189500 with predefined params and writing to config file 
    def simulation(self, x):
        pid = str(os.getpid())
        log("Initiating Copy for Process {}".format(pid))
        child_dir = "./" + DIR_PREFIX + pid
        shutil.copytree(".", child_dir, ignore=shutil.ignore_patterns(DIR_PREFIX + "*", "fast_dhsvm_parallel_run_pnnlOct2005toSep2007*"))
        log("Copy for Process {} completed".format(pid))
        log("Forking into " + child_dir)
        os.chdir(child_dir)
        #write DREAM parameter input to config file.

        #precipitation parameters
        change_setting(config_file, "Snow Threshold       ", str(round(x[2],5)))
        change_setting(config_file, "Rain Threshold       ", str(round(x[2]+2,5)))       
        
        #soil parameters
        #loam and roccky colluvium a function of sandy loam based on values in Rawls et al., 1982
        #sandy loam, soil type 62, sat K and exponential decrease determined by script
        change_setting(config_file, "Exponential Decrease 62", str(round(x[0],5)))
        change_setting(config_file, "Lateral Conductivity 62", str(round(x[1],5)))
        change_setting(config_file, "Maximum Infiltration 62", str(round(x[1]*2,5))) #assume equalt to 2*saturated hydraulic conductivity
        change_setting(config_file, "Vertical Conductivity 62"," ".join([str(round(x[1],5)),str(round(x[1],5)),str(round(x[1],5))]))
        
        #loam - sat K and exponential decrease 5 times less than sandy loam
        change_setting(config_file, "Exponential Decrease 61", str(round(x[0]/5,5)))
        change_setting(config_file, "Lateral Conductivity 61", str(round(x[1]/5,5)))
        change_setting(config_file, "Maximum Infiltration 61", str(round(x[1]/5*2,5)))
        change_setting(config_file, "Vertical Conductivity 61"," ".join([str(round(x[1]/5,5)),str(round(x[1]/5,5)),str(round(x[1]/5,5))]))
        
        #rocky colluvium -treat as coarse sand - sat K and exponential decrease are 2 to 3 times greater than sandy loam
        change_setting(config_file, "Exponential Decrease 65", str(round(x[0]*2,5)))
        change_setting(config_file, "Lateral Conductivity 65", str(round(x[1]*3,5)))
        change_setting(config_file, "Maximum Infiltration 65", str(round(x[1]*3,5)))
        change_setting(config_file, "Vertical Conductivity 65"," ".join([str(round(x[1]*3,5)),str(round(x[1]*3,5)),str(round(x[1]*3,5))]))
                
        #run DHSVM with modified parameters in config file
        subprocess.call(dhsvm_cmd, shell=True, stdout=False, stderr=False)
        simulations=[]
        #read streamflow data from DHSVM output file
        with open(streamflow_only, 'r') as file_output:
            header_name = file_output.readlines()[0].split(' ')

        with open(streamflow_only) as inf:
            next(inf)
            date_q = []
            q_12189500 = []
            for line in inf:
                parts = line.split()
                if len(parts) > 1:
                    date_q.append(parts[0])
                    q_12189500.append(float(parts[2])/(3600*1))
                    
        os.chdir("..")
        log("Removing copied directory: " + str(child_dir))
        remove_tree(child_dir)
        log("Removed directory: " + str(child_dir))

        simulation_streamflow = pd.DataFrame({'x[0]':date_q, 'x[2]':q_12189500})
        simulation_streamflow.columns = [header_name[0], header_name[2]]
        log('len of simulation is:' + str(len(simulation_streamflow)))
        simulation_streamflow.to_csv('simulations.csv', index=False)
        simulations = simulation_streamflow['12189500'].values
        return simulations
    
    def evaluation(self):
        return self.evals.tolist()
    
    def objectivefunction(self, simulation, evaluation, params=None):
        try:
            model_fit = spotpy.objectivefunctions.nashsutcliffe(evaluation,simulation)
            log('Nashsutcliffe: ' + str(model_fit))
        except Exception as e:
            log(e)
        return model_fit

# Initialize the Dream Class
fast_run = fast_run_setup()

log("Starting...")
# N = 1284s
# Create the Dream sampler of spotpy, al_objfun is set to None to force SPOTPY
# to jump into the def objectivefunction in the Dream_run_setup class with 
# nashsutcliffe as objectivefunction.
sampler=spotpy.algorithms.fast(fast_run, dbname='fast_dhsvm_parallel_run_pnnlOct2005toSep2007', dbformat='sql', parallel='mpc')

sampler.sample(8)

results = sampler.getdata()
results_data = pd.DataFrame(results)
results_data.to_csv('fast_sensitive_params_run_pnnlOct2005toSep2007.csv')

plot_results(results)

