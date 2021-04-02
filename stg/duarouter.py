import os, sys, glob
import xml.etree.ElementTree as ET
from joblib import Parallel, delayed, parallel_backend
from stg.utils import SUMO_outputs_process, simulate, gen_sumo_cfg, exec_od2trips, gen_od2trips, create_O_file, parallel_batch_size, gen_DUArouter


def clean_folder(folder):
    files = glob.glob(os.path.join(folder,'*'))
    [os.remove(f) for f in files]
    #print(f'Cleanned: {folder}')
    

def gen_routes(O, k, O_files, folders, routing):
     """
     Generate configuration files for dua / ma router
     """
     # Generate od2trips cfg
     cfg_name, output_name = gen_od2trips(O,k, folders)
    
     # Execute od2trips
     output_name = exec_od2trips(cfg_name, output_name, folders)
    
     if routing == 'dua':
        # Generate DUArouter cfg
        cfg_name, output_name = gen_DUArouter(output_name, k, folders)
                          
     elif routing == 'ma':
        # Generate MArouter cfg
        cfg_name, output_name = gen_MArouter(O, k, O_files, output_name, folders)
  
     else:
        SystemExit('Routing name not found')

     # Generate sumo cfg
     return gen_sumo_cfg(routing, output_name, k, folders, folders.reroute_probability) # last element reroute probability
     
     
def gen_route_files(folders, k, repetitions, end_hour, routing):
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
        for sd in [folders.D_district]:
            print(f'\n Generating cfg files for TAZ  From:{h} -> To:{sd}')
            # build O file    
            O_name = os.path.join(folders.O, f'{h}_{sd}')
            create_O_file(folders, O_name, h, sd, end_hour, 1) # factor = 1
                 
            # Generate cfg files 
            for k in range(repetitions):
                # backup O files
                O_files = os.listdir(folders.O)
                # Gen DUArouter/MArouter
                cfg_file_loc = gen_routes(O_name, k, O_files, folders, routing)
    return cfg_file_loc                    
    

def gen_MArouter(O, i, O_files, trips, folders):
    net_file = os.path.join(folders.parents_dir, 'templates', 'osm.net.xml')
    # read O files
    O_listToStr = ','.join([f'{os.path.join(folders.O, elem)}' for elem in O_files]) 
 
    marouter_conf = os.path.join(folders.parents_dir,'templates','marouter.cfg.xml') # duaroter.cfg file location
    
    # Open original file
    tree = ET.parse(marouter_conf)
    
    # Update trip input
    parent = tree.find('input')
    #ET.SubElement(parent, 'route-files').set('value', f'{trips}')    
    ET.SubElement(parent, 'net-file').set('value', f'{net_file}') 
    ET.SubElement(parent, 'od-matrix-files').set('value', f'{O_listToStr}')    
  
    # update additionals 
    TAZ = os.path.join(folders.parents_dir, 'templates', 'TAZ.xml')
    add_list = [TAZ]
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
        
    
def exec_duarouter_cmd(fname):
    print('\Generating DUArouter.......')
    cmd = f'duarouter -c {fname}'
    os.system(cmd)    

def exec_marouter_cmd(fname):
    print('\Generating MArouter.......')
    cmd = f'marouter -c {fname}'
    os.system(cmd) 


def exec_DUArouter(folders,processors):
    cfg_files = os.listdir(folders.O)
  
    # Get dua.cfg files list
    dua_cfg_list = []
    [dua_cfg_list.append(cf) for cf in cfg_files if 'duarouter' in cf.split('_')]
 
    if dua_cfg_list:
        batch = parallel_batch_size(dua_cfg_list)
        
        # Generate dua routes
        print(f'\nGenerating duaroutes ({len(dua_cfg_list)} files) ...........\n')
        with parallel_backend("loky"):
            Parallel(n_jobs=processors, verbose=0, batch_size=batch)(delayed(exec_duarouter_cmd)(
                     os.path.join(folders.O, cfg)) for cfg in dua_cfg_list)
    else:
       sys.exit('No dua.cfg files}')
    
 
def exec_MArouter(folders,processors):
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
                                   

def dua_ma(config,k,repetitions, end_hour, processors, routing, gui):
    """
    DUARouter / MARouter  funcions

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
    # Generate cfg files
    gen_route_files(config, k, repetitions, end_hour, routing)

    if routing  == 'dua':
        # Execute DUArouter 
        exec_DUArouter(config,processors)
    elif routing  == 'ma':          
        # Execute MArouter 
        exec_MArouter(config,processors)
    
    simulate(config, processors, gui)
    # Outputs preprocess
    SUMO_outputs_process(config)
    
 




