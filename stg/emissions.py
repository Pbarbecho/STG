import pandas as pd
import sys, os
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend
import math
import time
import multiprocessing
import xml.etree.ElementTree as ET
import psutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib import rc
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d
from utils import filter_emission_traffic_jams, process_emissions_file


# import sumo tool xmltocsv
if 'SUMO_HOME' in os.environ:
    tools = os.path.join('/opt/sumo-1.8.0/', 'tools')
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(tools))
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")



plt.rcParams.update({'font.size': 16})
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=False)

rlist = ['dua', 'duaiterate','ma','od2']
rrlist = ['0_0', '0_1']
basedir = '/root/Desktop/MSWIM/Revista/sim_files/Taz'
output_dir = '/root/Desktop/MSWIM/Revista/sim_files/emissions'
emissiones_dir = '/root/Desktop/MSWIM/Revista/sim_files/emissiones'
random_0 = '/root/Desktop/MSWIM/Revista/sim_files/RandomTrips/0_0/outputs/Hospitalet_SanAdria_emission_0.xml'
random_1 = '/root/Desktop/MSWIM/Revista/sim_files/RandomTrips/0_1/outputs/Hospitalet_SanAdria_emission_0.xml'

dua_emission = "/root/Desktop/MSWIM/Revista/sim_files/Taz/dua/0_0/xmltocsv/Hospitalet_SanAdria_emission_0.csv"
ma_emission = "/root/Desktop/MSWIM/Revista/sim_files/Taz/ma/0_0/xmltocsv/Hospitalet_SanAdria_emission_0.csv"
duai_emission = "/root/Desktop/MSWIM/Revista/sim_files/Taz/duaiterate/0_0/xmltocsv/Hospitalet_SanAdria_emission_0.csv"
od2_emission = "/root/Desktop/MSWIM/Revista/sim_files/Taz/od2/0_0/xmltocsv/Hospitalet_SanAdria_emission_0.csv"
random_emission = "/root/Desktop/MSWIM/Revista/sim_files/RandomTrips/0_0/xmltocsv/Hospitalet_SanAdria_emission_0.csv"


def emissions_full_day():
    DUA = process_emissions_file(dua_emission, 'DUAR')
    MA = process_emissions_file(ma_emission, 'MAR')
    DUAI = process_emissions_file(duai_emission, 'DUAI')
    OD2 = process_emissions_file(od2_emission, 'OD2')
    RT = process_emissions_file(random_emission, 'RT')
        
    routing_list =[MA,DUA,DUAI,RT,OD2]
    new_df = pd.DataFrame()   
    new_df['Hour'] = range(24) # puede estar desordenado
    for rou in routing_list:
        rou = rou.sort_values(by=['Hour'])
        r_name = rou["Routing"].unique()
        # originalmente NOx en [mg/s] https://sumo.dlr.de/docs/Simulation/Output/EmissionOutput.html
        # como ya lo he agrupado por hora queda g/hr
        new_df[r_name] = rou['vehicle_NOx']/1000 # de mg/hr a gr/hr
    return new_df



def plot_emissions_full_day(emission_df):    
    # traffic tools
    cols = ['MAR', 'DUAR', 'DUAI', 'RT', 'OD2']
    linestyle_list = ['-', '--','-.',':','-','--'] 
    # plot traffic intensity
    fig, ax = plt.subplots(figsize=(6,4))
    for i, col in enumerate(cols):
        emission_df.plot(kind='line', linewidth=2, linestyle=linestyle_list[i] , x='Hour',y=col, ax=ax)
    plt.legend(prop={'size': 15})
    plt.ylabel('Total NOx emissions [g/hr]')
    #plt.title('Traffic intensity')
    plt.xticks(np.arange(min(emission_df['Hour']), max(emission_df['Hour'])+1, 2.0))
    #plt.yticks(np.arange(0, 300, 50))
    plt.grid(True, linewidth=0.5, linestyle='--')  
    
    
    
    
def process_sumo_emissions_randomTrips():
    # iterate over routing directories 
    emisions_dic = {}
    randomlist = ['RandomTrips']
    random_dir = os.path.join(basedir, '..')
    for routing in randomlist:
        for reroute in rrlist:
            path = os.path.join(random_dir, routing, reroute, 'outputs/Hospitalet_SanAdria_tripinfo_0.xml')
            print(f'Routing = {routing}_{reroute}')
            df = emissions_tripinfo(path)
            
            #emisions_dic[f'{routing}_{reroute}'] = [df['NOx/km'].sum()]
            emisions_dic[f'{routing}_{reroute}'] = df['NOx/km']
            
            out_file = os.path.join(output_dir, f'{routing}_{reroute}.csv')
            df.to_csv(out_file, index=False, header=True )
    return emisions_dic
   


def process_sumo_emissions():
    # iterate over routing directories 
    emisions_dic = {}
    for routing in rlist:
        for reroute in rrlist:
            path = os.path.join(basedir, routing, reroute, 'outputs/Hospitalet_SanAdria_tripinfo_0.xml')
            print(f'Routing = {routing}_{reroute}')
            df = emissions_tripinfo(path)
          
            
            #[emisions_dic][f'{routing}_{reroute}'] = [df['NOx/km'].sum()]
            emisions_dic[f'{routing}_{reroute}'] = df['NOx/km']
                       
            out_file = os.path.join(output_dir, f'{routing}_{reroute}.csv')
            df.to_csv(out_file, index=False, header=True )
    return emisions_dic

            
            
def emissions_tripinfo(emissionsfile):
    # Open original file
    tree = ET.parse(emissionsfile)
    root = tree.getroot()
    children = list(root.getchildren()) # time
    emissions_results = []
    
    for child in children:
        veh_attrib= child.attrib #returnt a dic
        emissions_attrib = child.getchildren()[0].attrib
        data_list = [veh_attrib['id'], veh_attrib['routeLength'], emissions_attrib['NOx_abs']]    
        emissions_results.append(data_list)    

    df = pd.DataFrame(emissions_results, columns=['id', 'routeLength', 'NOx_abs'], dtype=('float'))
    
    
    # filter emissions outage points
    df = filter_emission_traffic_jams(df)
       
    # compute emissions mg/m = g/km
    # NOx_abs = the complete amount during the simulation
    # en mg -> g/km
    df['NOx/km'] = (df['NOx_abs']/1000)/(df['routeLength']/1000) # por metro
    
    return df
    

def rename_cols(results):
    #results = results.T
    results.rename(columns={'ma_0_0':'MAR',
                           'ma_0_1':'MAR-R',
                           'dua_0_0':'DUAR',
                           'dua_0_1':'DUAR-R',
                           'duaiterate_0_0':'DUAI',
                           'duaiterate_0_1':'DUAI-R',
                           'RandomTrips_0_0':'RT',
                           'RandomTrips_0_1':'RT-R',
                           'od2_0_0':'OD2',
                           'od2_0_1':'OD2-R'
                           },inplace=True)
    return results


def filter_no_reroute(results_df):
        
    results_df.to_csv('/root/Desktop/test.csv')
    return results_df.filter(items=['MAR', 'DUAR','DUAI','RT','OD2'])
 
    #return results_df.filter(items=['MAR-R', 'DUAR-R','DUAI-R','RT-R','OD2-R'])


def plot_emissions(emissions_df):
    fig, ax = plt.subplots(figsize=(8,4))
    emissions_df.plot.bar(width=0.5, color=mcolors.TABLEAU_COLORS, ax=ax)
    
    ymin = 60 
    ymax = 90
    plt.axhline(ymin,linestyle='dashed', color='tab:gray')
    plt.axhline(ymax,linestyle='dashed', color='tab:gray')
    style = dict(size=20, color='black')
    margin = 3
    ax.text(3.5,ymax+margin, "EURO 6", **style)
            
    
    #plt.ylabel('Total NOx emissions (g/km)')
    plt.title('Total NOx emissions (g/km)')
    plt.xticks(rotation=0)
    
    
def singlexml2csv(em_file, output):
       
    # SUMO tool xml into csv
    sumo_tool = os.path.join(tools, 'xml', 'xml2csv.py')
    # Run sumo tool with sumo output file as input
    cmd = 'python {} {} -s , -o {}'.format(sumo_tool, em_file, output)
   
    os.system(cmd)


def process_emissions_df(mdf):
    # Prepare dataframe
    mdf = mdf.T.reset_index()
    # axis=1  for compute mean of rows
    mdf['Mean'] = mdf.mean(axis=1, skipna=True)
    mdf['SD'] = mdf.std(axis=1, skipna=True)
    mdf = mdf.filter(['index','Mean','SD'])
    return mdf


def plot_metric(df, ylabel):
    # Plot current metric 
    fig, ax = plt.subplots(figsize=(8,4))
    plt.errorbar(df['index'], df['Mean'], yerr=df['SD'], fmt="None", color='Black', elinewidth=1, capthick=1,errorevery=1, alpha=1, ms=4, capsize = 2)
    plt.bar(df['index'], df['Mean'], width=0.4, color=mcolors.TABLEAU_COLORS)
    #plt.ylabel(f'{ylabel}')
 
    # ayuntament de catalunya limits for C card  
    ymin = 0.08 #[g/km]
    ymax = 0.57 #[g/km]
    
    plt.axhline(ymin,linestyle='dashed', color='tab:gray')
    plt.axhline(ymax,linestyle='dashed', color='tab:gray')
    style = dict(size=16, color='k')
    
    ax.text(3, ymax+0.05, "Real-world NOx", **style)
   
    ax.text(3.7, 0.1, "EURO 6", **style)
    #48 50 46 52 49  = 49
   
    plt.ylabel('NOx emissions [g/km]')
    #plt.title('Total NOx emissions (g/km)')
    plt.xticks(rotation=0)

#singlexml2csv("/root/Desktop/MSWIM/Revista/sim_files/new_emissions/xml/edgedump.xml", "/root/Desktop/MSWIM/Revista/sim_files/new_emissions/csv/test.csv")
  

#plot emissions of a full day
emission_df = emissions_full_day()
plot_emissions_full_day(emission_df)


# read emission outputs 
dic_A = process_sumo_emissions()    
dic_B = process_sumo_emissions_randomTrips()
# prepare dataframe
dic_A.update(dic_B)

new_df = pd.DataFrame(dic_A)
new_df = rename_cols(new_df)
# filter reroute no reroute
new_df = filter_no_reroute(new_df)
# Prepare dataframe
new_df = process_emissions_df(new_df)

print(new_df)
plot_metric(new_df, 'ylabel')
#plot_emissions(new_df)
