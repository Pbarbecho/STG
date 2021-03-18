import pandas as pd
import sys, os
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend
import math
import time
import multiprocessing
import xml.etree.ElementTree as ET
import psutil
import shutil
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed, parallel_backend


# import sumo tool xmltocsv
os.environ['SUMO_HOME']='/opt/sumo-1.8.0'

# number of cpus
processors = multiprocessing.cpu_count() # due to memory lack -> Catalunya  map = 2GB

# import sumo tool xmltocsv
if 'SUMO_HOME' in os.environ:
    tools = os.path.join('/opt/sumo-1.8.0/', 'tools')
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(tools))
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


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
    #new_emissions = os.path.join(folders.parents_dir,'templates', 'emissions.add.xml')
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
    
    
    if routing =='dua':add_list = [detector_dir, vtype]
    elif routing =='ma':add_list = [TAZ, detector_dir, vtype]
    elif routing =='od2':add_list = [TAZ, detector_dir, vtype]
    
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


def clean_memory():
    #Clean memory cache at the end of simulation execution
    if os.name != 'nt':  # Linux system
        os.system('sync')
        os.system('echo 3 > /proc/sys/vm/drop_caches')
        os.system('swapoff -a && swapon -a')
    # print("Memory cleaned")
    
    
def exec_od2trips(fname, tripfile, folders):
    print('\n OD2Trips running .............\n')
    cmd = f'od2trips -v -c {fname}'
    os.system(cmd)
    # remove fromtotaz
    output_file = f'{tripfile}.xml'
    rm_taz = f"sed 's/fromTaz=\"{folders.O_district}\" toTaz=\"{folders.D_district}\"//' {tripfile} > {output_file}"
    os.system(rm_taz)
    return output_file


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


def create_O_file(folders, fname, origin_district, destination_distric, end_hour, factor):
    
    """
        Generate O files given the real traffic in csv format and origing/destination districs names as in TAZ file. 
        An O file is generated each 15 minutes.
        Args:
        folder: (path class) .
        O distric name: origin distric
        D distric name: Destination distric
        repetitios: number of repetitions
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
            
            
def process_emissions_file(path_csv,routing):
    # process sumo emissions ouput file for a full day simulation 24 hrs
    df = pd.read_csv(path_csv)
    df = df.filter(items=['timestep_time','vehicle_NOx'])
    df = df.groupby(['timestep_time']).sum().reset_index()
    df = df.groupby(pd.cut(df["timestep_time"], np.arange(0, 86400+1, 3600))).sum().reset_index(drop=True)
    df['Hour'] = range(24)
    df = df.filter(items=['Hour','vehicle_NOx'])
    df['Routing'] = f'{routing}'
    return df
    

def filter_emission_traffic_jams(df):
    print(df.shape)
    df = df[df['routeLength']>=500] # filter distances < 500m
    print(df.shape)
    return df



def parallel_batch_size(plist):
    if len(plist) < processors:
        batch = 1
    else:
        batch = int(math.ceil(len(plist)/processors))
    return batch


def detector_cfg(detector_template, detector_cfg, output):
    # full path 
    tree = ET.parse(detector_template)
    root = tree.getroot()
    for child in root:
        child.set('file', f'{output}')  
    
    tree.write(detector_cfg) 



def kill_cpu_pid():
    process_name = "cpu_mem_check.s"
    for proc in psutil.process_iter():
        if process_name in proc.name():
           pid = proc.pid
           os.system(f'kill {pid}')
           print(f'\nCPU pid={pid} killed')
           
       
def create_folder(path):
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
            os.mkdir(path)
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)



def merge_detector_lanes(dtor_df, tool, routing):
    # dataframe, routing tool, re routing capabilities 1, 0
    
    # separete detector lanes L0 ... L5 en pares
    dtor_df_L0 = dtor_df.loc[dtor_df['interval_id'] == 'L0']
    dtor_df_L1 = dtor_df.loc[dtor_df['interval_id'] == 'L1']
    dtor_df_L2 = dtor_df.loc[dtor_df['interval_id'] == 'L2']
    dtor_df_L3 = dtor_df.loc[dtor_df['interval_id'] == 'L3']
    dtor_df_L4 = dtor_df.loc[dtor_df['interval_id'] == 'L4']
    dtor_df_L5 = dtor_df.loc[dtor_df['interval_id'] == 'L5']
    
    
    # ordena por tiempo
    dtor_df_L0.sort_values(by ='interval_begin', inplace=True)
    dtor_df_L1.sort_values(by ='interval_begin', inplace=True)
    dtor_df_L2.sort_values(by ='interval_begin', inplace=True)
    dtor_df_L3.sort_values(by ='interval_begin', inplace=True)
    dtor_df_L4.sort_values(by ='interval_begin', inplace=True)
    dtor_df_L5.sort_values(by ='interval_begin', inplace=True)
    
    s_L0 = dtor_df_L0.size
    s_L1 = dtor_df_L1.size
    s_L2 = dtor_df_L2.size
    s_L3 = dtor_df_L3.size
    s_L4 = dtor_df_L4.size
    s_L5 = dtor_df_L5.size
        
    # la dimension debe ser la misma
    if s_L0 == s_L1 ==s_L2 == s_L3 == s_L4 == s_L5:
        
        L0_L1 = dtor_df_L0.merge(dtor_df_L1, on=['interval_begin', 'interval_end'], suffixes=['_L0','_L1'])
        L2_L3 = dtor_df_L2.merge(dtor_df_L3, on=['interval_begin', 'interval_end'], suffixes=['_L2','_L3'])
        L4_L5 = dtor_df_L4.merge(dtor_df_L4, on=['interval_begin', 'interval_end'], suffixes=['_L4','_L5'])
        new_df = L0_L1.merge(L2_L3, on=['interval_begin', 'interval_end']).merge(L4_L5, on=['interval_begin', 'interval_end'])
           
        #Add L0 .. L5 and filter conteo vehiculos
        new_df[f'{tool}'] = new_df['interval_nVehContrib_L0']\
                           + new_df['interval_nVehContrib_L1']\
                           + new_df['interval_nVehContrib_L2']\
                           + new_df['interval_nVehContrib_L3']\
                           + new_df['interval_nVehContrib_L4']\
                           + new_df['interval_nVehContrib_L5']\
                            
        new_df['interval_occupancy'] = new_df['interval_occupancy_L0'] + new_df['interval_occupancy_L1'] + new_df['interval_occupancy_L2']
        
        
        """
        #Add L0 ..L6 and filter ocupacion de vehiculos
        new_df[f'{tool}'] = new_df['interval_occupancy_L0']\
                           + new_df['interval_occupancy_L1']\
                           + new_df['interval_occupancy_L2']\
                           + new_df['interval_occupancy_L3']\
                           + new_df['interval_occupancy_L4']\
                           + new_df['interval_occupancy_L5']\
        """
                
        new_df = new_df.filter(['interval_begin',
                                'interval_end',
                                tool,
                                #'interval_occupancy'
                                ])
        # Add new fields
        new_df['Hour'] = range(24)
        new_df['RP'] = routing
        return new_df
    else:
        sys.exit('L0 and L1 are different size')



# CPU
def cpu_mem_folders(new_dir):
    parent_path = os.path.join(new_dir, "CPU")
    statistics_tool_Path = os.path.join(parent_path, 'statistics')
    create_folder(parent_path)
    create_folder(statistics_tool_Path)
    tools = ['cpu', 'memory', 'disk']
    # create tools subfolders
    [create_folder(os.path.join(statistics_tool_Path, f)) for f in tools]
    cpu_path = os.path.join(statistics_tool_Path, 'cpu')
    mem_path = os.path.join(statistics_tool_Path, 'memory')
    disk_path = os.path.join(statistics_tool_Path, 'disk')

    # New paths
    curr_subdir = os.path.join(parent_path, 'data')
    create_folder(curr_subdir)
    folders = ["cpu", 'memory', 'disk', 'results', 'plots']
    [create_folder(os.path.join(curr_subdir, f)) for f in folders]
    [create_folder(os.path.join(os.path.join(curr_subdir, 'results'), f)) for f in tools]

    return cpu_path, mem_path, disk_path
        

def SUMO_preprocess(options):
    # Process SUMO output files to build the dataset
    def save_file(df, name, parsed_dir):
        #print(f'Parsed --> {name}')
        if 'ID' in df.keys():
            df.sort_values(by=['ID'], inplace=True)
        df.to_csv(os.path.join(parsed_dir, f'{name}.csv'), index=False, header=True)
        
    
    def parallelxml2csv(f, options):
        # output directory
        output = os.path.join(options.xmltocsv, f'{f.strip(".xml")}.csv')
        # SUMO tool xml into csv
        sumo_tool = os.path.join(tools, 'xml', 'xml2csv.py')
        # Run sumo tool with sumo output file as input
        cmd = 'python {} {} -s , -o {}'.format(sumo_tool, os.path.join(options.sumofiles,f), output)
        os.system(cmd)
        
        """
        # get emissions from tripinfo file
        if 'emission' in f.split('_'):
            emissions_df = emissions_tripinfo(os.path.join(options.sumofiles,f))
            out_name = os.path.join(options.emissions,'emissions.csv')
            emissions_df.to_csv(out_name, index=False, header=True)
        """
        

    def singlexml2csv(f, options):
        # output directory
        output = os.path.join(options.detector, f'{f.strip(".xml")}.csv')
        # SUMO tool xml into csv
        sumo_tool = os.path.join(tools, 'xml', 'xml2csv.py')
        # Run sumo tool with sumo output file as input
        cmd = 'python {} {} -s , -o {}'.format(sumo_tool, os.path.join(options.detector,f), output)
       
        os.system(cmd)


    def bulid_list_of_df(csv):
        time.sleep(1)
        dname = csv.split('.')[0].split('_')
        data = pd.read_csv(os.path.join(options.xmltocsv, csv))
        dname.append(data)
        return dname


    def convert_xml2csv(files_list):
        if files_list:
            print(f'\nGenerating {len(files_list)} csv files. This may take a while .........')
            batch = parallel_batch_size(files_list)
           
            # Parallelize files generation
            with parallel_backend("loky"):
                Parallel(n_jobs=processors, verbose=0, batch_size=batch)(delayed(parallelxml2csv)(
                     f, options) for f in files_list)
        else:
            sys.exit(f"Empty or missing output data files: {options.sumofiles}")

               
      
    def xml2csv(options): 
        # convert xml to csv
        files_list = os.listdir(options.sumofiles)
        
        # convert xmls to csvs
        convert_xml2csv(files_list)
        
        # Read generated csvs
        csvs_list = os.listdir(options.xmltocsv)
        
        if len(csvs_list) == len(files_list):
            data_list = []
            print(f'\nBuilding {len(csvs_list)} dataframes from sumo outputs ......\n')
            
            # build list of dataframes        
            [data_list.append(bulid_list_of_df(csv)) for csv in tqdm(csvs_list)]
           
            # convert to a single dataframe
            result_df = pd.DataFrame(data_list, columns=['Origin', 'Destination', 'Output','Repetition','Dataframe'])
            return result_df           
        else:
            sys.exit(f'Missing csv files: {options.xmltocsv}')
            
 
              
    def merge_files(options):
        #combine all files in the parsed dir
        parsed_files = os.listdir(options.parsed)
        print(f'\nCombining {len(parsed_files)} files')
        combined_csv = pd.concat([pd.read_csv(os.path.join(options.parsed, pf)) for pf in parsed_files ])
        #export to csv
        combined_csv.to_csv(os.path.join(options.parsed, "data.csv"), index=False, header=True)
        return combined_csv
                
    
    
    def emissions_tripinfo(tripfile):
        # Open original file
        tree = ET.parse(tripfile)
        root = tree.getroot()
        tuple_values = []
        # child -> timestep ! 
        for i, child in enumerate(root.getchildren()):
            tuple_values.append((child.get('time'), root[i][0].attrib['CO2'], root[i][0].attrib['x'], root[i][0].attrib['y']))
        df = pd.DataFrame(tuple_values, columns =['Time', 'CO2', 'x', 'y']) 
        return df
        
              
                
    def veh_trip_info(df):
        # filter know features
        df = df.filter(items=['tripinfo_duration', 
                              'tripinfo_routeLength', 
                              'tripinfo_timeLoss', 
                              'tripinfo_waitingCount', 
                              'tripinfo_waitingTime', 
                              'tripinfo_arrivalLane', 
                              'tripinfo_departLane',
                              'tripinfo_id']).rename(columns={'tripinfo_id':'ID'})
        return df
    
                
    def parallel_parse_output_files(key, group_df):
        # save parsed files
        df_name = f'{key[0]}_{key[1]}_{key[2]}'
        
        # Process sumo outputs
        #vehroute = group_df.loc[group_df['Output'] == 'vehroute', 'Dataframe'].iloc[0]
        #vehroute = vehroute.filter(['route_edges','vehicle_arrival','vehicle_depart','vehicle_departSpeed','vehicle_id','vehicle_routeLength'])
        #vehroute['route_edges'] = vehroute['route_edges'].fillna(0)
               
        #fcd = group_df.loc[group_df['Output'] == 'fcd', 'Dataframe'].iloc[0]
        
        tripinfo = group_df.loc[group_df['Output'] == 'tripinfo', 'Dataframe'].iloc[0]
        #taz_locations_edgenum_df = lanes_counter_taz_locations(vehroute)                  # Count edges on route from vehroute file and get from/to TAZ locations
        
        #veh_speed_positions_df = avrg_speed_and_geo_positions(fcd)                        # Get average speed and initial/end positions (x,y)
        tripinfo_df = veh_trip_info(tripinfo) 
        
        # merge dataframes
        #sdata = taz_locations_edgenum_df.merge(veh_speed_positions_df,on='ID').merge(tripinfo_df,on='ID')
        #sdata = veh_speed_positions_df.merge(tripinfo_df,on='ID')
        # save each scenario in parsed files
        #save_file(sdata, f'{df_name}', options.parsed)
       
        save_file(tripinfo_df, f'{df_name}', options.parsed)


       
    def lanes_counter_taz_locations(df):
        # contador de edges en ruta
        df['lane_count'] = df['route_edges'].apply(lambda x: len(str(x).split()))
        df = df.filter(items=['vehicle_id', 'lane_count', 'vehicle_fromTaz', 'vehicle_toTaz']).rename(columns={'vehicle_id':'ID'})
        return df


    def get_positions(df, id,min,max):
        # get initial and end positions based on ini/end time of vehicle id
        ini_pos =  df.loc[(df['vehicle_id'] == id) & (df['timestep_time'] == min)].iloc[0]
        end_pos =  df.loc[(df['vehicle_id'] == id) & (df['timestep_time'] == max)].iloc[0]
        return id, min, max, ini_pos.vehicle_x, ini_pos.vehicle_y, end_pos.vehicle_x, end_pos.vehicle_y 
        
       
    def avrg_speed_and_geo_positions(df):
        # get average speed
        speed_df = df.groupby(['vehicle_id']).mean().reset_index()
        speed_df = speed_df.filter(items=['vehicle_id','vehicle_speed']).rename(columns={'vehicle_id':'ID','vehicle_speed':'avrg_speed'})
        # Prepare df with ini end times of vehicles 
        df = df.filter(items=['vehicle_id', 'timestep_time','vehicle_x', 'vehicle_y'])
        df.dropna(subset = ["vehicle_id"], inplace=True)
        # get initial end times of vechiel
        time_df = df.groupby(['vehicle_id']).timestep_time.agg(['min','max']).reset_index()
        # Get positions df
        positions_list = [get_positions(df,id,min,max) for id,min,max in zip(time_df['vehicle_id'], time_df['min'], time_df['max'])]
        positions_df = pd.DataFrame(positions_list, columns=['ID', 'ini_time', 'end_time', 'ini_x_pos', 'ini_y_pos','end_x_pos', 'end_y_pos'])
        # Merge speed and positions df
        speed_and_positions = speed_df.merge(positions_df, on='ID')
        return speed_and_positions
    
    
    def parse_df(df):
        # process dataframe    
        nfiles = len(df.index)
        efiles = nfiles/len(df["Output"].unique()) # total files / number of output sumo (tripinfo, fcd, vehroute)
       
        print(f'\nReading {nfiles} dataframes. Expected {efiles} parsed files. This may take a while .........')        

        # group df by features
        grouped_df = df.groupby(['Origin', 'Destination', 'Repetition'])
        # parse dataframe
        [parallel_parse_output_files(key, group_df) for key, group_df in tqdm(grouped_df)]

    # converte dector to csv
    singlexml2csv(os.listdir(options.detector)[0], options) 
    
    
    # Execute functions               
    df = xml2csv(options)   # Convert outputs to csv 
    parse_df(df)            # Convert csv to dataframes and filter files fileds
    merge_files(options)    # Merge dataframes into a single file 'data.csv'
    
    
    
    
    
    
    
    