import os, glob
from stg.utils import SUMO_outputs_process, simulate, gen_sumo_cfg, exec_od2trips, gen_od2trips, create_O_file


def clean_folder(folder):
    files = glob.glob(os.path.join(folder,'*'))
    [os.remove(f) for f in files]
    #print(f'Cleanned: {folder}')
    

def gen_routes(O, k, O_files, folders, routing):
    """
    Generate configuration files for od2 trips
    """
    if routing == 'od2':
        # Generate od2trips cfg
        cfg_name, output_name = gen_od2trips(O,k, folders)
        
        # Execute od2trips
        output_name = exec_od2trips(cfg_name, output_name, folders)
        
        # Generate sumo cfg
        return gen_sumo_cfg(routing, output_name, k, folders, 0) # last element reroute probability
        
    else:
        SystemExit('Routing name not found')
            
          
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
            print(f'\n Generating cfg files for TAZ  Origin:{h}, Destination:{sd}')
            # build O file    
            O_name = os.path.join(folders.O, f'{h}_{sd}')
            create_O_file(folders, O_name, h, sd, end_hour, 1) # factor =1
                       
            # Generate cfg files 
            for k in range(repetitions):
                # backup O files
                O_files = os.listdir(folders.O)
                # Gen Od2trips
                cfg_file_loc = gen_routes(O_name, k, O_files, folders, routing)
    return cfg_file_loc
    

def exec_duarouter_cmd(fname):
    print('\nRouting  .......')
    cmd = f'duarouter -c {fname}'
    os.system(cmd)    

def exec_marouter_cmd(fname):
    print('\nRouting  .......')
    cmd = f'marouter -c {fname}'
    os.system(cmd)   
    
    
def od2(config,k,repetitions, end_hour, processors, routing, gui):
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
    # Generate configurtion files
    gen_route_files(config, k, repetitions, end_hour, routing)
    # Execute OD@Trips simulations
    simulate(config, processors, gui)
    # Outputs preprocess
    SUMO_outputs_process(config)
