import os, sys, glob
import xml.etree.ElementTree as ET
import multiprocessing
import sumolib
import pandas as pd
import numpy as np
import time
import shutil
from tqdm import tqdm
from datetime import datetime
from joblib import Parallel, delayed, parallel_backend
import subprocess


# import sumo tool xmltocsv
os.environ['SUMO_HOME']='/opt/sumo-1.8.0'

from utils import SUMO_preprocess, parallel_batch_size, detector_cfg, create_folder, cpu_mem_folders
# number of cpus
processors = multiprocessing.cpu_count() # due to memory lack -> Catalunya  map = 2GB

    
# General settings
veh_num = 10  # number of vehicles in O file
n_repetitions = 1 # number of repetitions 
k = 0
sim_time = 24 # in hours # TO DO parameter of time in files
end_hour = 24
factor = 1.0 # multiplied by the number of vehicles
# Vehicles equiped with a device Reroute probability
rr_prob = 0
# routing dua / ma
routing = 'duaiterate'
iterations = 10
net_update = 600 #cada cuantos segundos se actualiza la network status triptimes defailt 900


# Informacion de origen / destino como aparece en TAZ file 
origin_district = ['Hospitalet']
destination_distric = ['SanAdria']

 
# Static paths 
sim_dir = '/root/Desktop/MSWIM/Revista/sim_files'   # directory of sumo cfg files
base_dir =  os.path.join(sim_dir,'Taz',f'{routing}') # directorio base
traffic = os.path.join(sim_dir, '..', 'TrafficPgSanJoan.csv')
new_dir = os.path.join(base_dir, f'{k}_{rr_prob}')
# Sumo templates
duarouter_conf = os.path.join(sim_dir,'templates','duarouter.cfg.xml') # duaroter.cfg file location
marouter_conf = os.path.join(sim_dir,'templates','marouter.cfg.xml') # duaroter.cfg file location
sumo_cfg = os.path.join(sim_dir,'templates', 'osm.sumo.cfg')
od2trips_conf =  os.path.join(sim_dir,'templates', 'od2trips_duaiterate.cfg.xml')

TAZ = os.path.join(sim_dir,'templates', 'TAZ.xml')
vtype = os.path.join(sim_dir,'templates', 'vtype.xml')

# New paths
create_folder(new_dir)  
folders = ['emissions','duaiterate','trips', 'O', 'dua', 'ma', 'cfg', 'outputs', 'detector', 'xmltocsv', 'parsed', 'reroute']
[create_folder(os.path.join(new_dir, f)) for f in folders]

# create folders for cpu mem check
cpu, mem, disk = cpu_mem_folders(new_dir)

# Static paths
dua = os.path.join(new_dir, 'dua')
O = os.path.join(new_dir, 'O')
config = os.path.join(new_dir, 'cfg')
outputs = os.path.join(new_dir, 'outputs') 
detector = os.path.join(new_dir, 'detector') 
csv = os.path.join(new_dir, 'xmltocsv')
parsed = os.path.join(new_dir, 'parsed')
reroute = os.path.join(new_dir, 'reroute')
trips = os.path.join(new_dir, 'trips')
ma = os.path.join(new_dir, 'ma')
duaiterate = os.path.join(new_dir, 'duaiterate')
emissions = os.path.join(new_dir, 'emissions')

# Create detector file
detector_dir = os.path.join(new_dir,'detector.add.xml')
detector_cfg(os.path.join(sim_dir,'templates', 'detector.add.xml'),detector_dir, os.path.join(detector, 'detector.xml')) 


# Custom routes via
#route_0 = '237659862#23 208568871#3 208568871#4 208568871#5'
route_0 = '208568871#5'


os.chdir(duaiterate)

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
    duaiterate=duaiterate
    emissions = emissions
    cpu=cpu
    mem=mem
    disk=disk
    
    
def clean_folder(folder):
    files = glob.glob(os.path.join(folder,'*'))
    [os.remove(f) for f in files]
    #print(f'Cleanned: {folder}')
    

def gen_routes(O, k, O_files):
    
    # Generate od2trips cfg
    cfg_name, output_name = gen_od2trips(O,k)
    
    # Execute od2trips
    output_name = exec_od2trips(cfg_name, output_name)
    
    # Custom route via='edges'
    via_trip = custom_routes(output_name, k)
   
    
    if routing == 'duaiterate':
    
        # Generate DUArouter cfg
        cfg_name, output_name = gen_DUArouter(via_trip, k)
        #cfg_name, output_name = gen_DUArouter(output_name, k)
      
        # Generate sumo cfg
        gen_sumo_cfg('dua', output_name, k)
        
              
    else:
        SystemExit('Routing not found')
            
    return via_trip 
    #return output_name 
        

        
def gen_route_files():
    # generate cfg files
    for h in origin_district:
        print(f'\nGenerating cfg files for TAZ: {h}')
        for sd in tqdm(destination_distric):
            time.sleep(1)
            
            # build O file    
            O_name = os.path.join(folders.O, f'{h}_{sd}')
            create_O_file(O_name, f'{h}', f'{sd}', veh_num)
            
            # Generate cfg files 
            for k in tqdm(range(n_repetitions)):
                time.sleep(1) 
                # backup O files
                O_files = os.listdir(folders.O)
                # Gen Od2trips/MArouter
                via_trip = gen_routes(O_name, k, O_files)
    return via_trip

    

def create_O_file(fname, origin_district, destination_distric, vehicles):
    #create 24 hour files
    traffic = pd.read_csv('/root/Desktop/MSWIM/Revista/TrafficPgSanJoan.csv')
 
    df = pd.DataFrame(traffic)
    #traffic_24 = traffic_df['Total'].values
    name = os.path.basename(fname)
     
    col = list(df)
    col = col[1:-1]
    for hour in range(end_hour):  #hora
        for minute in col:    # minuto
            vehicles = df[minute][hour]
            
            h = hour
            m = str(minute)
            until = int(minute) + 15
            
            O_file_name = os.path.join(folders.O,f'{h}_{m}_{name}')
            O = open(f"{O_file_name}", "w")
            
            #num_vehicles = traffic_24[h] * 1.1 # margin of duarouter checkroutes
            text_list = ['$OR;D2\n',               # O format
                     f'{h}.{m} {h}.{until}\n',  # Time 0-48 hours
                     f'{factor}\n',         # Multiplication factor
                     f'{origin_district} '     # Origin
                 	 f'{destination_distric} ',   # Destination
                     f'{vehicles}']            # NUmber of vehicles x multiplication factor
            O.writelines(text_list)
            O.close()


def custom_routes(trips, k):
    trip = os.path.join(folders.O, trips)
    
    # Open original file
    tree = ET.parse(trip)
    root = tree.getroot()
     
    # Update via route in xml
    [child.set('via', route_0) for child in root]

    # name    
    curr_name = os.path.basename(trips).split('_')
    curr_name = curr_name[0] + '_' + curr_name[1]
    output_name = os.path.join(folders.O, f'{curr_name}_trips_{k}.rou.xml')
           
    # Write xml
    cfg_name = os.path.join(folders.O, output_name)
    tree.write(cfg_name) 
    return output_name
    

def gen_MArouter(O, i, O_files, trips):
    # read O files
    O_listToStr = ','.join([f'{os.path.join(folders.O, elem)}' for elem in O_files]) 
 
    # Open original file
    tree = ET.parse(marouter_conf)
    
    # Update trip input
    parent = tree.find('input')
    #ET.SubElement(parent, 'route-files').set('value', f'{trips}')    
    ET.SubElement(parent, 'od-matrix-files').set('value', f'{O_listToStr}')    
  
    # update additionals 
    add_list = [TAZ,vtype]
    additionals = ','.join([elem for elem in add_list]) 
    
    # Update detector
    ET.SubElement(parent, 'additional-files').set('value', f'{additionals}')    

     
    # Update output
    parent = tree.find('output')
    curr_name = os.path.basename(O)
    output_name = os.path.join(folders.ma, f'{curr_name}_ma_{i}.rou.xml')
    ET.SubElement(parent, 'output-file').set('value', output_name)    
    
    # Update seed number
    parent = tree.find('random_number')
    ET.SubElement(parent, 'seed').set('value', f'{i}')    
    
    # Write xml
    cfg_name = os.path.join(folders.O, f'{curr_name}_marouter_{i}.cfg.xml')
    tree.write(cfg_name) 
    return cfg_name, output_name

    
def gen_DUArouter(trips, i):
    # Open original file
    tree = ET.parse(duarouter_conf)
    
    # Update trip input
    parent = tree.find('input')
    ET.SubElement(parent, 'route-files').set('value', f'{trips}')    
     
    # Update output
    parent = tree.find('output')
    curr_name = os.path.basename(trips).split('_')
    curr_name = curr_name[0] + '_' + curr_name[1]
    output_name = os.path.join(folders.dua, f'{curr_name}_dua_{i}.rou.xml')
    ET.SubElement(parent, 'output-file').set('value', output_name)    
    
    # Update seed number
    parent = tree.find('random_number')
    ET.SubElement(parent, 'seed').set('value', f'{i}')    
    
    # Write xml
    original_path = os.path.dirname(trips)
    cfg_name = os.path.join(original_path, f'{curr_name}_duarouter_{i}.cfg.xml')
    tree.write(cfg_name) 
    return cfg_name, output_name
        
    
    
def gen_od2trips(O,k):
    # read O files
    O_files_list = os.listdir(folders.O)
    O_listToStr = ','.join([f'{os.path.join(folders.O, elem)}' for elem in O_files_list]) 
    
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


def gen_sumo_cfg(routing,dua, k):
    # Open original file
    tree = ET.parse(sumo_cfg)
    
    # Update rou input
    parent = tree.find('input')
    ET.SubElement(parent, 'route-files').set('value', f'{dua}')    
    
    
    if routing =='duaiterate':
        add_list = [detector_dir]
    else:    
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
    output_dir = os.path.join(folders.config, f'{curr_name}_{routing}_{k}.sumo.cfg')
    tree.write(output_dir)
    
    
    
def exec_od2trips(fname, tripfile):
    print('\nRouting .......')
    cmd = f'od2trips -c {fname}'
    os.system(cmd)
    # remove fromtotaz
    output_file = f'{tripfile}.xml'
    rm_taz = f"sed 's/fromTaz=\"Hospitalet\" toTaz=\"SanAdria\"//' {tripfile} > {output_file}"
    os.system(rm_taz)
    return output_file
    
    


def exec_duarouter_cmd(fname):
    print('\nSimulando .......')
    cmd = f'duarouter -c {fname}'
    os.system(cmd)    

def exec_marouter_cmd(fname):
    print('\nSimulando .......')
    cmd = f'marouter -c {fname}'
    os.system(cmd) 


    
def exec_DUArouter(via_trips):
    # duaiterate for iterative assigment
    sumo_tool = '/opt/sumo-1.8.0/tools/assign/duaIterate.py'
    net_file = '/root/Desktop/MSWIM/Revista/sim_files/templates/osm.net.xml'
    add = detector_dir
    
    reroute_path = os.path.join(folders.reroute, "reroute.xml")
    
    # duaiterate command 
    cmd = f'python {sumo_tool} --time-to-teleport 84600 \
                               --time-to-teleport.highways 84600 \
                               -a {net_update} \
                               -n {net_file} \
                               -+ {vtype},{add} \
                               -t {via_trips} \
                               -l {iterations} \
                               sumo--device.rerouting.probability {rr_prob} \
                               sumo--device.rerouting.output {reroute_path}'
    os.system(cmd)
    
    # regresa el path al ultimo sumo iterate y copia los ouputs
    # base name
    curr_name = os.path.basename(via_trips).split('_')
    curr_name = curr_name[0] + '_' + curr_name[1]
    
    # last iteration folder
    liter =  iterations-1 # begin with 0
    last_iter_path = os.path.join(folders.duaiterate, f'{liter}')
    # sumo last iteration outputs summmary/tripinfo
    
    # complete name    
    fill_0_name = 3-len(str(liter))
    name = ''
    
    if fill_0_name!=0:
        name = name.join(['0' for i in range(fill_0_name)])
   
    summary_liter = os.path.join(last_iter_path, f'summary_{name}{liter}.xml')
    tripinfo_liter = os.path.join(last_iter_path, f'tripinfo_{name}{liter}.xml')
    fcd_liter = os.path.join(last_iter_path, f'fcd_{name}{liter}.xml')
    emission_liter = os.path.join(last_iter_path, f'emission_{name}{liter}.xml')
    
    # copy last iteration outputs to original folders
    cmd = f'cp {summary_liter} {folders.outputs}/{curr_name}_summary_{k}.xml'
    os.system(cmd)
    cmd = f'cp {tripinfo_liter} {folders.outputs}/{curr_name}_tripinfo_{k}.xml'
    os.system(cmd)
    cmd = f'cp {fcd_liter} {folders.outputs}/{curr_name}_fcd_{k}.xml'
    os.system(cmd)
    cmd = f'cp {emission_liter} {folders.outputs}/{curr_name}_emission_{k}.xml'
    os.system(cmd)

    
 
def exec_MArouter():
    cfg_files = os.listdir(folders.O)
  
    # Get ma.cfg files list
    ma_cfg_list = []
    [ma_cfg_list.append(cf) for cf in cfg_files if 'marouter' in cf.split('_')]
    
    if ma_cfg_list:
        batch = parallel_batch_size(ma_cfg_list)
        
        # Generate dua routes
        print(f'\nGenerating MAroutes ({len(ma_cfg_list)} files) ...........\n')
        with parallel_backend("loky"):
            Parallel(n_jobs=processors, verbose=0, batch_size=batch)(delayed(exec_marouter_cmd)(
                     os.path.join(folders.O, cfg)) for cfg in ma_cfg_list)
    else:
       sys.exit('No ma.cfg files}')
       

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
    
    
def simulate():
    simulations = os.listdir(folders.config)
    if simulations: 
        batch = parallel_batch_size(simulations)
        # Execute simulations
        print('\nExecuting simulations ....')
        with parallel_backend("loky"):
                Parallel(n_jobs=processors, verbose=0, batch_size=batch)(delayed(exec_sim_cmd)(s) for s in simulations)
        clean_memory()
        print(f'\n{len(os.listdir(folders.outputs))} outputs generated: {folders.outputs}')
    else:
       sys.exit('No sumo.cfg files}')
  
                
       
def exec_sim_cmd(cfg_file):
    full_path = os.path.join(folders.config, cfg_file)
    cmd = f'sumo -c {full_path}'
    os.system(cmd)
  
    
  
def SUMO_outputs_process():
    class options:
        sumofiles = outputs
        xmltocsv = csv
        parsed = parsed
        detector = detector
        emissions = emissions
    SUMO_preprocess(options)

def print_time(process_name):      
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"{process_name} Time =", current_time)

"""
########################################################
print('CPU/MEM/DISC check fix time.....')
cmd = ['/root/CPU/disk.sh', f'{new_dir}', f'{folders.disk}']
print(cmd)
subprocess.Popen(cmd)
#cpu mem scripts
cmd = ['/root/CPU/cpu.sh', f'{folders.cpu}']
subprocess.Popen(cmd)
cmd = ['/root/CPU/memory.sh', f'{folders.mem}']
subprocess.Popen(cmd)
########################################################
"""        
# Generate cfg files
print_time('Cfg files generation')
via_trip = gen_route_files()
# Exceute duaiterate
print_time('Begin simulation')
exec_DUArouter(via_trip)
print_time('End simulation')

# Outputs preprocess
SUMO_outputs_process()
