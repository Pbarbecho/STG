import multiprocessing
import os, sys
import subprocess
import stg
import click
import shutil
from tqdm import tqdm


class Config(object):
    def __init__(self):
        self.verbose = False
        self.parents_dir = os.path.dirname(os.path.abspath('{}/..'.format(__file__)))
        self.processors = multiprocessing.cpu_count()
        self.SUMO_exec = ""
        self.SUMO_templates = ""
        self.SUMO_tool = ""
        self.tool = ""
        self.cfg = ""
        self.detector = ""
        self.dua = ""
        self.ma = ""
        self.O = ""
        self.outputs = ""
        self.trips = ""
        self.xmltocsv = ""
        self.parsed = ""
        self.reroute = ""
        self.reroute_probability = ""
        self.realtraffic = ""
        self.O_district = ""
        self.D_district = ""
        self.edges = ""
        self.net =""
        self.iterations =""
         
pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group()
@click.option('-v', is_flag=True, help=' verbose')
@pass_config
def cli(config, v):
    """

    CLI STG SUMO Traffic Generator.

    """
    config.verbose = v # verbose


##########
# Launch #
##########
@cli.command()
@click.option('-s', '--sumo-bin',
              type=click.Path(exists=True, resolve_path=True),
              help='SUMO bin directory.')
@click.option('-net',
              type=click.Path(exists=True, resolve_path=True),
              help='Net file converted to SUMO readable file (.net)')
@click.option('-T', '--real-traffic',
              type=click.Path(exists=True),
              help='Path to real traffic file with .csv format. ')
@click.option('-O', '--O-district-name',
              help='Origin district name as in TAZ file.')
@click.option('-D', '--D-district-name',
              help='Destination district name as in TAZ file.')
@click.option('-o','--outputs',
              type=click.Path(exists=True, resolve_path=True),
              help="Output directory (route traces, statistics).")
@click.option('-ma',
              default= False, is_flag=True,
              help='MARouter SUMO tool')
@click.option('-dua',
              default= False, is_flag=True,
              help='DUARouter SUMO tool')
@click.option('-duai',
              default= False, is_flag=True,
              help='DUAIterate SUMO tool')
@click.option('-rt',
              default= False, is_flag=True,
              help='RandomTrips SUMO tool')
@click.option('-od2',
              default= False, is_flag=True,
              help='OD2Trips SUMO tool')
@click.option('-i', '--iterations',
              default= 1, show_default=True,
              help='DUAIterate number of iterations')
@click.option('-reroute',
              default= False, is_flag=True, show_default=True,
              help='Enable SUMO reroute capabilities.')
@click.option('-gui',
              default= False, is_flag=True,
              help='Graffical interface for SUMO simulations')
@click.option('-p', '--max-processes',
              default=1,
              help='The maximum number of parallel simulations. [ default available cpus are used ]')
@click.option('--sim-time', '-t',
              default=1, show_default=True,
              help='Number of hours to simulate  (e.g., 24 hours)')
@click.option('-n', '--repetitions',
              default=1,
              show_default=True,
              help='Number of repetitions.')

    
@pass_config
def run(config, real_traffic, iterations, o_district_name, gui, d_district_name, sumo_bin, outputs, net, ma, dua, duai, rt, od2, max_processes, sim_time, repetitions, reroute):
    """
    STG SUMO Traffic generator. Required options: tool, -O, -D, 
    """
    if dua:
        config.tool='dua' 
    elif ma:
        config.tool='ma' 
    elif duai:
        config.tool='duai' 
    elif od2:
        config.tool='od2' 
    elif rt:
        config.tool='rt' 
    else:
        sys.exit('No SUMO routing tool selected.')
        
    # set path to templates 
    cfg_templates = os.path.join(config.parents_dir, 'templates')  
    
    if real_traffic is None or o_district_name is None or d_district_name is None:
        click.echo('\n Empty arguments [-T, -O, -D, -cfg]. Use \x1B[3mstg trafficgenerator --help\x1B[23m')
    else :
        if config.verbose: click.echo(f'\n SUMO Installation: {sumo_bin} \n SUMO Templates: {cfg_templates}\n\n Setting program paths.... ')
        #sumo = get_sumo_path('sumo') # try to get sumo installation dir 
        
        # Create/Update paths  
        config.SUMO_exec = sumo_bin
        config.SUMO_outputs = os.path.join(config.parents_dir, 'outputs')
        if not os.path.lexists(config.SUMO_outputs):os.makedirs(config.SUMO_outputs)
             
        
        config.SUMO_tool = os.path.join(config.SUMO_outputs, config.tool)
        config.O_district = o_district_name
        config.D_district = d_district_name
        config.realtraffic = real_traffic
        config.net = net # TO DO look for need file in templates folder
        config.iterations = iterations
        config.reroute_probability = reroute
        update_paths(config)
        cpu_processes = get_MAX_PROCESS(config, max_processes)
        
        # SUMO Tools
        k = 0
        if config.tool =='od2':stg.od2(config, k, repetitions, sim_time, cpu_processes, config.tool, gui)
        elif config.tool =='dua':stg.dua_ma(config, k, repetitions, sim_time, cpu_processes, config.tool, gui)
        elif config.tool =='ma':stg.dua_ma(config, k, repetitions, sim_time, cpu_processes, config.tool, gui)
        elif config.tool =='duai':stg.duai(config, k, repetitions, sim_time, cpu_processes, config.tool, gui)
        elif config.tool =='rt':stg.rt(config, k, repetitions, sim_time, cpu_processes, config.tool, gui)
          
      
        
def update_paths(config):
    create_folder(config.SUMO_tool)   
    subfolders = ['trips', 'O', 'dua', 'ma', 'cfg', 'outputs', 'detector', 'xmltocsv', 'parsed', 'reroute', 'edges', 'duaiterate']
    for sf in tqdm(subfolders):
        create_folder(os.path.join(config.SUMO_tool, sf)) 
    config.trips = os.path.join(config.SUMO_tool, 'trips')
    config.O = os.path.join(config.SUMO_tool, 'O')
    config.dua = os.path.join(config.SUMO_tool, 'dua')
    config.ma = os.path.join(config.SUMO_tool, 'ma')
    config.cfg = os.path.join(config.SUMO_tool, 'cfg')
    config.outputs = os.path.join(config.SUMO_tool, 'outputs')
    config.detector = os.path.join(config.SUMO_tool, 'detector')
    config.xmltocsv = os.path.join(config.SUMO_tool, 'xmltocsv')
    config.parsed = os.path.join(config.SUMO_tool, 'parsed')
    config.reroute = os.path.join(config.SUMO_tool, 'reroute')
    config.edges = os.path.join(config.SUMO_tool, 'edges')
        
    
       
def create_folder(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    
    

def get_sumo_path(app):
    """
    If no path to SUMO installation is pass. The script try to find SUMO installation path. 
    Args:
        application: 'sumo'
    """
   
    
    command = 'whereis' if os.name != 'nt' else 'which'
    r = subprocess.getoutput('{0} {1}'.format(command, app))
    app_instance = (r.strip('sumo:').strip()).split(' ')
    if len(app_instance) > 1:
        click.echo('\n {} SUMO installations found !!!'.format(len(app_instance)))
        # TO DO menu to select OMNet++ instance
        return app_instance[1].strip('sumo')  # default first installation instance
    else:
        return app_instance[0].strip('sumo')      
    
    
def get_MAX_PROCESS(config, max_processes):
    if max_processes == 1:
        # By default the max number of cpus are try to used in simulations
        max_processes = config.processors
    else:
        if config.processors < max_processes:
            max_processes = config.processors
    return max_processes
