import os, sys, glob
import xml.etree.ElementTree as ET
import multiprocessing
import sumolib
import pandas as pd
import numpy as np
import time
from datetime import datetime
import shutil
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend
import subprocess
from stg.utils import create_folder, cpu_mem_folders, detector_cfg

# import sumo tool xmltocsv
#os.environ['SUMO_HOME']='/opt/sumo-1.8.0'

from stg.utils import SUMO_preprocess, parallel_batch_size, detector_cfg


factor = 1
"""

class folders:
    traffic = traffic
    dua = dua
    config = config
    outputs = outputs
    detector = detector
    csv = csv
    parsed = parsed
    reroute = reroute
    O = O
    trips=trips 
    ma=ma
    cpu=cpu
    mem=mem
    disk=disk
"""
    
def clean_folder(folder):
    files = glob.glob(os.path.join(folder,'*'))
    [os.remove(f) for f in files]
    #print(f'Cleanned: {folder}')
    

def gen_routes(O, k, O_files, folders):
    """
    Generate configuration files for od2 trips
    """
    # Generate od2trips cfg
    cfg_name, output_name = gen_od2trips(O,k, folders)
    
    # Execute od2trips
    output_name = exec_od2trips(cfg_name, output_name, folders)
    
    # Generate sumo cfg
    od2_sim_cfg_file = gen_sumo_cfg('od2trips', output_name, k, folders, '0') # last element reroute probability
    
    return od2_sim_cfg_file
    
        
        
def gen_route_files(folders, k, repetitions, end_hour):
    """
    Generate O files given the real traffic in csv format. 
    Args:
    folder: (path class) .
    max_processors: (int) The max number of cpus to use. By default, all cpus are used.
    repetitios: number of repetitions
    end hour: The simulation time is the end time of the simulations 
    """
    # generate cfg files
    for h in [folders.O_district]:
        print(f'\nGenerating cfg files for TAZ: {h}')
        for sd in tqdm([folders.D_district]):
            # build O file    
            O_name = os.path.join(folders.O, f'{h}_{sd}')
            create_O_file(folders, O_name, f'{h}', f'{sd}', end_hour)
                       
            # Generate cfg files 
            for k in range(repetitions):
                # backup O files
                O_files = os.listdir(folders.O)
                # Gen Od2trips
                od2_sim_cfg_file = gen_routes(O_name, k, O_files, folders)
               
    return od2_sim_cfg_file
    
    

def create_O_file(folders, fname, origin_district, destination_distric, end_hour):
    """
    Generate O files given the real traffic in csv format and oriding/destination districs names as in TAZ file. 
    An O file is generated each 15 minutes.
    Args:
    folder: (path class) .
    O distric name: origin distric
    D distric name: Destination distric
    repetitios: number of repetitions
    end
    """
    #create 24 hour files
    traffic = pd.read_csv(folders.realtraffic)
 
    df = pd.DataFrame(traffic)
    #traffic_24 = traffic_df['Total'].values
    name = os.path.basename(fname)
     
    col = list(df)
    col = col[1:-1]
    for hour in tqdm(range(end_hour)):  #hora
        for minute in col:    # minuto
            vehicles = df[minute][hour]
            
            h = hour
            m = str(minute)
            until = int(minute) + 15
            
            O_file_name = os.path.join(folders.O,f'{h}_{m}_{name}')
            O = open(f"{O_file_name}", "w")
         
            #print(f'{h}:{m} ->  {vehicles}')
            #num_vehicles = traffic_24[h] * 1.1 # margin of duarouter checkroutes
            text_list = ['$OR;D2\n',               # O format
                     f'{h}.{m} {h}.{until}\n',  # Time 0-48 hours
                     f'{factor}\n',         # Multiplication factor
                     f'{origin_district} '     # Origin
                 	 f'{destination_distric} ',   # Destination
                     f'{vehicles}']            # NUmber of vehicles x multiplication factor
            O.writelines(text_list)
            O.close()

        
def gen_od2trips(O,k, folders):
    """
    Generate the od2trips configutation file
    
    Parameters
    ----------
    O : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    folders : path class
        Contains all paths for the simulation.

    Returns
    -------
    cfg_name : TYPE
        DESCRIPTION.
    output_name : TYPE
        DESCRIPTION.

    """
    
    # read O files
    O_files_list = os.listdir(folders.O)
    O_listToStr = ','.join([f'{os.path.join(folders.O, elem)}' for elem in O_files_list]) 
    TAZ = os.path.join(folders.parents_dir, 'templates', 'TAZ.xml')
    od2trips_conf =  os.path.join(folders.parents_dir,'templates', 'od2trips.cfg.xml')
    # Open original file
    tree = ET.parse(od2trips_conf)
    
    # Update O input
    parent = tree.find('input')
    ET.SubElement(parent, 'od-matrix-files').set('value', f'{O_listToStr}')    
    ET.SubElement(parent, 'taz-files').set('value', f'{TAZ}')    
        
    # Update output
    parent = tree.find('output')
    output_name = f'{O}_od2_{k}.trip.xml'
    ET.SubElement(parent, 'output-file').set('value', output_name)    
    
    # Update seed number
    parent = tree.find('random_number')
    ET.SubElement(parent, 'seed').set('value', f'{k}')    
    
    # Write xml
    cfg_name = f'{O}_trips_{k}.cfg.xml'
    tree.write(cfg_name)
    
    return cfg_name, output_name    


def gen_sumo_cfg(routing, dua, k, folders, rr_prob):
    """
    Generate the sumo cfg file to execute the simulation

    Parameters
    ----------
    routing : TYPE
        DESCRIPTION.
    dua : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.

    Returns
    -------
    output_dir : TYPE
        DESCRIPTION.

    """
    sumo_cfg = os.path.join(folders.parents_dir,'templates', 'osm.sumo.cfg')
    vtype = os.path.join(folders.parents_dir,'templates', 'vtype.xml')
    new_emissions = os.path.join(folders.parents_dir,'templates', 'emissions.add.xml')
    TAZ = os.path.join(folders.parents_dir, 'templates', 'TAZ.xml')
    net_file = os.path.join(folders.parents_dir, 'templates', 'osm.net.xml')
    
    # Create detector file
    detector_dir = os.path.join(folders.parents_dir,'templates','detector.add.xml')
    detector_cfg(os.path.join(folders.parents_dir,'templates', 'detector.add.xml'),detector_dir, os.path.join(folders.SUMO_tool,'detector', 'detector.xml')) 


    # Open original file
    tree = ET.parse(sumo_cfg)
      
    
    # Update rou input
    parent = tree.find('input')
    ET.SubElement(parent, 'net-file').set('value', f'{net_file}') 
    ET.SubElement(parent, 'route-files').set('value', f'{dua}')    
    
    add_list = [TAZ, detector_dir, vtype]
    
    additionals = ','.join([elem for elem in add_list]) 
    
    # Update detector
    ET.SubElement(parent, 'additional-files').set('value', f'{additionals}')    

    # Routing
    parent = tree.find('routing')
    ET.SubElement(parent, 'device.rerouting.probability').set('value', f'{rr_prob}')   
    ET.SubElement(parent, 'device.rerouting.output').set('value', f'{os.path.join(folders.reroute, "reroute.xml")}')   
      

    # Update outputs
    parent = tree.find('output')
    curr_name = os.path.basename(dua).split('_')
    curr_name = curr_name[0] + '_' + curr_name[1]
    
    # outputs 
    outputs = ['emission', 'summary', 'tripinfo']
    for out in outputs:
        ET.SubElement(parent, f'{out}-output').set('value', os.path.join(
            folders.outputs, f'{curr_name}_{out}_{k}.xml'))    
     
    # Write xml
    output_dir = os.path.join(folders.cfg, f'{curr_name}_{routing}_{k}.sumo.cfg')
    tree.write(output_dir)
   
    return output_dir
    
    
    
def exec_od2trips(fname, tripfile, folders):
    print('\n OD2Trips running .............\n')
    cmd = f'od2trips -v -c {fname}'
    os.system(cmd)
    # remove fromtotaz
    output_file = f'{tripfile}.xml'
    rm_taz = f"sed 's/fromTaz=\"{folders.O_district}\" toTaz=\"{folders.D_district}\"//' {tripfile} > {output_file}"
    os.system(rm_taz)
    return output_file
    
    

def exec_duarouter_cmd(fname):
    print('\nRouting  .......')
    cmd = f'duarouter -c {fname}'
    os.system(cmd)    

def exec_marouter_cmd(fname):
    print('\nRouting  .......')
    cmd = f'marouter -c {fname}'
    os.system(cmd) 


def summary():
    # Count generated files
    expected_files = len(origin_district)*len(destination_distric)*n_repetitions
  
    if routing=='dua':
        output_files = os.listdir(folders.dua)   
        generated_files = len(output_files)/2 # /2 due to alt files
    
        print(f'\nExpected files: {expected_files}   Generated files: {generated_files}\n')
        if generated_files != expected_files:
            print('Missing files, check log at console')
        else:
            # Count routes 
            measure = ['id','fromTaz', 'toTaz']
            out_list = []
            for f in output_files:    
                if 'alt' not in f.split('.'):
                    summary_list = sumolib.output.parse_sax__asList(os.path.join(folders.dua,f), "vehicle", measure)
                    temp_df = pd.DataFrame(summary_list).groupby(['fromTaz', 'toTaz']).count().reset_index()
                    temp_df['Repetition'] = f.split('.')[0].split('_')[-1]
                    out_list.append(temp_df.to_numpy()[0])
                    
            summary = pd.DataFrame(out_list, columns=['Origin', 'Destination','Routes', 'Repetition']).sort_values(by=['Repetition', 'Origin', 'Destination'])
            save_to = os.path.join(new_dir,'route_files.csv')
            summary.to_csv(save_to, index=False, header=True)
            print(summary)
            print(f'\nSummary saved to: {save_to}\n')
            
        
def clean_memory():
    #Clean memory cache at the end of simulation execution
    if os.name != 'nt':  # Linux system
        os.system('sync')
        os.system('echo 3 > /proc/sys/vm/drop_caches')
        os.system('swapoff -a && swapon -a')
    # print("Memory cleaned")
    
    
def simulate(folders, processors):
    simulations = os.listdir(folders.cfg)
    if simulations: 
        batch = parallel_batch_size(simulations)
        # Execute simulations
        print('\nExecuting simulations ....')
        with parallel_backend("loky"):
                Parallel(n_jobs=processors, verbose=0, batch_size=batch)(delayed(exec_sim_cmd)(s, folders) for s in simulations)
        clean_memory()
        print(f'\n{len(os.listdir(folders.outputs))} outputs generated: {folders.outputs}')
    else:
       sys.exit('No sumo.cfg files}')
    print_time('End simulations ')
                
       
def exec_sim_cmd(cfg_file, folders):
    #print('\n Simulating ............\n')
    full_path = os.path.join(folders.cfg, cfg_file)
    cmd = f'sumo -c {full_path}'
    os.system(cmd)
  
    
def SUMO_outputs_process(folders):
    class options:
        sumofiles = folders.outputs
        xmltocsv = folders.xmltocsv
        parsed = folders.parsed
        detector = folders.detector
    SUMO_preprocess(options)
      
def print_time(process_name):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"\n{process_name} Time =", current_time)
    
    
def od2(config,sim_time,repetitions, end_hour, processors):
    """
    OD2Trips funcions

    Parameters
    ----------
    config : TYPE
        DESCRIPTION.
    sim_time : TYPE
        DESCRIPTION.
    repetitions : TYPE
        DESCRIPTION.
    end_hour : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    od2_sim_cfg_file = gen_route_files(config, 0, repetitions, end_hour)
    simulate(config, processors)
    # Outputs preprocess
    SUMO_outputs_process(config)
    
    
    """
    class options:
        sumofiles = outputs
        xmltocsv = csv
        parsed = parsed
        detector = detector
    SUMO_preprocess(options)
    """



