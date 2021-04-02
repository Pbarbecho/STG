import os, glob
import xml.etree.ElementTree as ET
from stg.utils import SUMO_outputs_process, gen_sumo_cfg, exec_od2trips, gen_od2trips, create_O_file, gen_DUArouter

route_0 = '208568871#5'


def clean_folder(folder):
    files = glob.glob(os.path.join(folder,'*'))
    [os.remove(f) for f in files]
    #print(f'Cleanned: {folder}')
    

def gen_routes(O, k, O_files, folders, routing):
    """
    Generate configuration files for duaiterate
    """    
    # Generate od2trips cfg
    cfg_name, output_name = gen_od2trips(O,k, folders)
    
    # Execute od2trips
    output_name = exec_od2trips(cfg_name, output_name, folders)
    
    # Custom route via='edges'
    via_trip = custom_routes(output_name, k, folders)
   
    if routing == 'duai':
        #Generate DUArouter cfg
        cfg_name, output_name = gen_DUArouter(via_trip, k, folders)
        # Generate sumo cfg
        gen_sumo_cfg(routing, output_name, k, folders, 0)  # last element reroute probability 
        return  via_trip
    else:
        SystemExit('Routing not found')


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
                trips = gen_routes(O_name, k, O_files, folders, routing)
    return trips  


def custom_routes(trips, k, folders):
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


def exec_duarouter_cmd(fname):
    print('\nSimulando .......')
    cmd = f'duarouter -c {fname}'
    os.system(cmd)    

def exec_marouter_cmd(fname):
    print('\nSimulando .......')
    cmd = f'marouter -c {fname}'
    os.system(cmd) 


    
def exec_DUAIterate(folders, via_trips, processors,end_hour, k):
    # duaiterate for iterative assigment
    
    # update paths
    sumo_tool = os.path.join(folders.SUMO_exec, '..', 'tools/assign/duaIterate.py')
    net_file = os.path.join(folders.parents_dir, 'templates', 'osm.net.xml')
    vtype = os.path.join(folders.parents_dir,'templates', 'vtype.xml')
    
    # Path to detector file
    detector_file = os.path.join(folders.SUMO_tool, 'detector.xml')
        
    reroute_path = os.path.join(folders.reroute, "reroute.xml")
    # update options
    iterations = folders.iterations
    rr_prob = int(folders.reroute_probability)
    net_update = 600 #netwrok update (i.e., verify netwrok status) each 600s
    edges = os.path.join(folders.edges, 'edges.add.xml')
    # duaiterate command 
    cmd = f'python {sumo_tool} --router-verbose --time-to-teleport 84600 \
                               --time-to-teleport.highways 84600 \
                               -+ {vtype},{detector_file},{edges} \
                               -a {net_update} \
                               -n {net_file} \
                               -t {via_trips} \
                               -l {iterations} \
                               sumo--device.rerouting.probability {rr_prob} \
                               sumo--device.rerouting.output {reroute_path}'
   
    os.chdir(os.path.join(folders.SUMO_tool, 'duaiterate'))  # Create detector file
    os.system(cmd)
    
    # regresa el path al ultimo sumo iterate y copia los ouputs
    # base name
    curr_name = os.path.basename(via_trips).split('_')
    curr_name = curr_name[0] + '_' + curr_name[1]
    
    # last iteration folder
    liter =  iterations-1 # begin with 0
    last_iter_path = os.path.join(folders.SUMO_tool,'duaiterate', f'{liter}')
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
    cmd = f'cp {summary_liter} {folders.SUMO_tool}/outputs/{curr_name}_summary_{k}.xml'
    os.system(cmd)
    cmd = f'cp {tripinfo_liter} {folders.SUMO_tool}/outputs/{curr_name}_tripinfo_{k}.xml'
    os.system(cmd)
    cmd = f'cp {fcd_liter} {folders.SUMO_tool}/outputs/{curr_name}_fcd_{k}.xml'
    os.system(cmd)
    cmd = f'cp {emission_liter} {folders.SUMO_tool}/outputs/{curr_name}_emission_{k}.xml'
    os.system(cmd)
    


def duai(config,k,repetitions, end_hour, processors, routing, gui):
    """
    

    Parameters
    ----------
    config : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    repetitions : TYPE
        DESCRIPTION.
    end_hour : TYPE
        DESCRIPTION.
    processors : TYPE
        DESCRIPTION.
    routing : TYPE
        DESCRIPTION.
    gui : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # Generate cfg files
    trips = gen_route_files(config, k, repetitions, end_hour, routing)
 
    # Exceute duaiterate
    exec_DUAIterate(config, trips, processors, end_hour, k)
       
    # Outputs preprocess
    SUMO_outputs_process(config)
