import multiprocessing
import os
import subprocess
import stg
import click


class Config(object):
    def __init__(self):
        self.verbose = False
        self.parents_dir = os.path.dirname(os.path.abspath('{}/..'.format(__file__)))
        self.processors = multiprocessing.cpu_count()
        self.SUMO_exec = ""
        self.SUMO_templates = ""
        self.SUMO_xmltocsv = ""
        self.SUMO_outputs = ""
        self.SUMO_parsed = ""
        self.SUMO_detector = ""
        self.SUMO_tool = ""
          

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
@click.option('-sp', '--sumo-path',
              type=click.Path(exists=True, resolve_path=True),
              help='SUMO bin directory.')
@click.option('-cfg', '--cfg-templates-path',
              type=click.Path(exists=True, resolve_path=True),
              help='Templates of SUMO configuration files.')
@click.option('--output-dir',
              type=click.Path(exists=True, resolve_path=True),
              help="Vehicles' traces output directory (routes,trips,flows).")
@click.option('-st', '--SUMO-tool',
              default="OD2Trips",
              help='SUMO demand generation tool [OD2Trips, DUArouter, DUAIterate, MArouter, RandomTrips]')
@click.option('-p', '--max-processes',
              default=1,
              help='The maximum number of parallel simulations. [ default available cpus are used ]')
@click.option('--sim-time', '-t',
              default=300,
              show_default=True,
              help='SUMO simulation time.')
@click.option('-r', '--repetitions',
              default=1,
              show_default=True,
              help='Number of repetitions.')

    
@pass_config
def generator(config, SUMO_tool, sumo_path):
    """
    Traffic generator
    """
    if config.verbose: click.echo('\n Setting program paths....')
    
    # Update paths  
    config.SUMO_exec = sumo_path
    config.SUMO_outputs = os.path.join(config.parents_dir, 'outputs')
    os.mkdir(config.SUMO_outputs)
    if SUMO_tool:
        config.SUMO_tool = os.path.join(config.SUMO_outputs, SUMO_tool)
        os.mkdir(config.SUMO_tool)
        
        
        
        
        
    #if SUMO_tool == "OD2Trips":
    #    stg.od2run(config)
    #else:
    #    click.echo('\nSUMO tool not supported.')

    #if output_dir is None: output_dir = os.path.join(config.parents_dir, 'results', config.mac)
    #if sumo_path is None: sumo_path = get_sumo_path('sumo')  # Try to get SUMO installation
    #if os.path.exists(sumo_path) and os.path.exists(cfg_templates_path):
    #    if config.verbose:
    #        click.echo(' SUMO installation: {}'.format(sumo_path))
    #        click.echo(' SUMO Templates: {}'.format(cfg_templates_path))
    #    updated_max_processes = get_MAX_PROCESS(config, max_processes)
        # Execute OMNET simulation campaign
    #    stg.run(output_dir, updated_max_processes, sim_time, repetitions,
    #            config.verbose)
    #else:
    #    click.echo('\nNo such file or directory or is empty: {sumo_path} or {cfg_templates_path}')


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
