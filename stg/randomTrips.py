# -*- coding: utf-8 -*-
import os, sys, glob
import xml.etree.ElementTree as ET
import multiprocessing
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import shutil
import subprocess
from utils import SUMO_preprocess, detector_cfg, kill_cpu_pid, create_folder, cpu_mem_folders
from tqdm import tqdm


edge_weights = '/root/Desktop/MSWIM/Revista/sim_files/templates/randomTripsWeigths.xml'

# number of cpus
processors = multiprocessing.cpu_count() # due to memory lack -> Catalunya  map = 2GB
# import sumo tool xmltocsv
os.environ['SUMO_HOME']='/opt/sumo-1.8.0'
# import sumo tool xmltocsv
if 'SUMO_HOME' in os.environ:
    tools = os.path.join('/opt/sumo-1.8.0/', 'tools')
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(tools))
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
    

ERASE_FOLDERS = True
    
# General settings
end_hour = 24
seed = 0
k = 0
# reroute probability
rr_prob = 0


# Informacion de origen / destino como aparece en TAZ file 
origin_district = ['Hospitalet']
destination_distric = ['SanAdria']

# Static paths 
sim_dir = '/root/Desktop/MSWIM/Revista/sim_files'   # directory of sumo cfg files
base_dir =  os.path.join(sim_dir, 'RandomTrips') # directorio base
traffic = os.path.join(sim_dir, '..', 'TrafficPgSanJoan.csv')
new_dir = os.path.join(base_dir, f'{k}_{rr_prob}')

# Sumo templates
duarouter_conf = os.path.join(sim_dir,'templates','duarouter.cfg.xml') # duaroter.cfg file location
sumo_cfg = os.path.join(sim_dir,'templates','osm.sumo.cfg')
net_file = os.path.join(sim_dir,'templates', 'osm.net.xml')
add_file = os.path.join(sim_dir,'templates', 'vtype.xml')

        
# New paths
if ERASE_FOLDERS:
    create_folder(new_dir)  
    folders = ['CPU', 'trips', 'dua', 'cfg', 'outputs', 'detector', 'xmltocsv', 'parsed', 'reroute']
    [create_folder(os.path.join(new_dir, f)) for f in folders]


# Static paths
random_dir =  os.path.join(new_dir, 'trips') # xml directoryRandomTrips
dua = os.path.join(new_dir, 'dua')
config = os.path.join(new_dir, 'cfg')
outputs = os.path.join(new_dir, 'outputs') 
detector = os.path.join(new_dir, 'detector') 
csv = os.path.join(new_dir, 'xmltocsv')
parsed = os.path.join(new_dir, 'parsed')
reroute = os.path.join(new_dir, 'reroute')
cpu = os.path.join(new_dir, 'CPU')
 

# Create detector file
if ERASE_FOLDERS:
    detector_dir = os.path.join(new_dir,'detector.add.xml')
    detector_cfg(os.path.join(sim_dir,'templates', 'detector.add.xml'),detector_dir, os.path.join(detector, 'detector.xml')) 
# create folders for cpu mem check
cpu, mem, disk = cpu_mem_folders(new_dir)


# Custom routes via
route_0 = '237659862#23 208568871#3 208568871#4 208568871#5'
#route_0 = '208568871#5'


class folders:
    random_dir = random_dir
    traffic = traffic
    dua = dua
    config = config
    outputs = outputs
    detector = detector
    csv = csv
    cpu = cpu
    parsed = parsed
    reroute = reroute
    cpu=cpu
    mem=mem
    disk=disk
    
    
def clean_folder(folder):
    files = glob.glob(os.path.join(folder,'*'))
    [os.remove(f) for f in files]
    #print(f'Cleanned: {folder}')
    
def RandomTrips():
    def exec_randomTrips(fname, ini_time, veh_number):
        # SUMO Tool randomtrips
        sumo_tool = os.path.join(tools, 'randomTrips.py')
            
        # output directory
        output = os.path.join(folders.random_dir,  f'{fname}.xml')
          
        vtype = "car"
        iter_seed = seed + 1
        begin = ini_time
        end =  begin + 15*60  # 15 minutes 
        
        # vehicles arrival 
        period = (end - begin) / veh_number
        # Execute random trips
        cmd = f"python {sumo_tool} -v \
        --weights-prefix='RT'\
        -n {net_file} \
        -a {add_file}  \
        --edge-permission passenger  \
        -b {begin} -e {end} -p {period} \
        --trip-attributes 'type=\"{vtype}\" departSpeed=\"0\"' \
        -s {seed}  \
        -o {output} \
        --validate"
        
        os.system(cmd)
    #--validate \
    #--edge-permission {vtype}  \
    def custom_routes():
        randomtrips = os.listdir(folders.random_dir)
        
        for trip in randomtrips:
            # file locations
            trip_loc = os.path.join(folders.random_dir, trip)
            
            # Open original file
            tree = ET.parse(trip_loc)
            root = tree.getroot()
             
            # Update via route in xml
            for i, child in enumerate(root):
                if i != 0:
                    child.set('via', route_0)
            
            # Write xml
            cfg_name = os.path.join(folders.random_dir, trip)
            tree.write(cfg_name) 
                
    def trips_for_traffic():
        traffic_df = pd.read_csv(folders.traffic)
        
        # generate randomtrips file each 15 min
        col = list(traffic_df)
        col = col[1:-1]
        
        print('CPU MEM check fix time.....')
        
        #cpu script
        #subprocess.Popen(['/root/cpu_mem_check.sh', f'{folders.cpu}'])
        
        print(f'\nGenerating {len(col) * end_hour} randomTrips ......')
        for hour in tqdm(range(end_hour)):  #hora
            for minute in col:    # minuto
                vehicles = traffic_df[minute][hour]
                name = f'{hour}_{minute}_randomTrips'
                # convert to sec            
                ini_time = hour*3600 + (int(minute)) * 60
                exec_randomTrips(name, ini_time, vehicles)
        
        #kill_cpu_pid()
        # verify generated trip files
        if len(os.listdir(folders.random_dir)) == len(col)*end_hour:
            print('OK')
        else:
            sys.exit(f'Missing randomTrips files in {folders.random_dir}')
            
    
    def change_veh_ID(trip_file, veh_id_number):
        # full path 
        file = os.path.join(folders.random_dir, trip_file)
        # Open original file
        tree = ET.parse(file)
        root = tree.getroot()
           
        # Update via route in xml
        veh_id = veh_id_number
        for child in root:
            veh_id += 1
            child.set('id',str(veh_id))
               
        # Write xml
        #new_file = str(os.path.join(folders.dua_trips, trip_file))
        tree.write(file) 
        return veh_id
              
    
    def update_vehicle_ID():
        trips = os.listdir(folders.random_dir)
        veh_id = 0
        print('Update vehicle IDs......\n')
         
        for f in tqdm(trips):
            veh_id = change_veh_ID(f, veh_id)
        
    
    def gen_DUArouter():
        print('\nGenerate DUA config...')
        #trip files
        trip_list = os.listdir(folders.random_dir)
        trip_list_str = ','.join([f'{os.path.join(folders.random_dir, elem)}' for elem in trip_list]) 
        
        # Open original file
        tree = ET.parse(duarouter_conf)
        
        # Update trip input
        parent = tree.find('input')
        ET.SubElement(parent, 'route-files').set('value', f'{trip_list_str}')    
         
        # Update output
        parent = tree.find('output')
        output_name = os.path.join(folders.dua, 'dua.rou.xml')
        ET.SubElement(parent, 'output-file').set('value', output_name)    
        
        # Update seed number
        parent = tree.find('random_number')
        ET.SubElement(parent, 'seed').set('value', str(seed))    
        
        # Write xml
        cfg_name = os.path.join(folders.config, 'duarouter.cfg.xml')
        tree.write(cfg_name) 
        print('OK')
    
    
    def exec_duarouter_cmd(fname):
        cmd = f'duarouter -c {fname}'
        os.system(cmd)   
    
    
    def gen_sumo_cfg():
        print('\nGenerate sumo config file')
        
        
        # read trips
        read_trips = os.listdir(folders.random_dir)
        
        trips = ','.join([f'{os.path.join(folders.random_dir, elem)}' for elem in read_trips]) 
        
        # Open original file
        tree = ET.parse(sumo_cfg)
        
        # Update rou input
        parent = tree.find('input')
        #ET.SubElement(parent, 'route-files').set('value', f'{os.path.join(folders.dua, "dua.rou.xml")}')   
        ET.SubElement(parent, 'route-files').set('value', f'{trips}')   
        
        
        # Update additional
        add_list = [detector_dir, add_file]
        additionals = ','.join([elem for elem in add_list]) 
        ET.SubElement(parent, 'additional-files').set('value', f'{additionals}')    
    
        # Routing
        parent = tree.find('routing')
        ET.SubElement(parent, 'device.rerouting.probability').set('value', f'{rr_prob}')   
        ET.SubElement(parent, 'device.rerouting.output').set('value', f'{os.path.join(folders.reroute, "reroute.xml")}')   
             
        # Update outputs
        curr_name = origin_district[0] + '_' + destination_distric[0]
        outputs = ['emission', 'summary','tripinfo']
        parent = tree.find('output')
        for out in outputs:
            ET.SubElement(parent, f'{out}-output').set('value', os.path.join(
                folders.outputs, f'{curr_name}_{out}_{k}.xml'))    
         
        # Write xml
        output_dir = os.path.join(folders.config, f'sumo_{rr_prob}_{k}.sumocfg.xml')
        tree.write(output_dir)
        return output_dir
    
    
    
    def exec_sumo_sim(cfg_full_name):
        print('\nSimulating ......')
        cmd = f'sumo -c {cfg_full_name}'
        os.system(cmd)
	


    def singlexml2csv(f):
        # output directory
        output = os.path.join(folders.detector, f'{f.strip(".xml")}.csv')
        # SUMO tool xml into csv
        sumo_tool = os.path.join(tools, 'xml', 'xml2csv.py')
        # Run sumo tool with sumo output file as input
        
        print('\nConvert to csv detector ! \nExecute outside...........\n')
        cmd = 'python {} {} -s , -o {}'.format(sumo_tool, os.path.join(folders.detector,f), output)
        print(cmd)
        #os.system(cmd)
    
    def SUMO_outputs_process():
        class options:
            sumofiles = outputs
            xmltocsv = csv
            parsed = parsed
            detector = detector
        SUMO_preprocess(options)


    def print_time(process_time):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"\n{process_time} Time =", current_time)

    
    if ERASE_FOLDERS:
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
        # trips
        print_time('Cfg files generation')
        trips_for_traffic()
        
        # via route Travessera
        #custom_routes()
        
        update_vehicle_ID()
        
        # sumo sim
        cfg_full_name = gen_sumo_cfg()
        print_time('Begin simulation')
        exec_sumo_sim(cfg_full_name)
        print_time('End simulation')
        # detectors
        #singlexml2csv('detector.xml')
        
    # process sumo outputs  
    SUMO_outputs_process() 
    

RandomTrips()
