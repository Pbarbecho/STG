import os, glob
import pandas as pd
import xml.etree.ElementTree as ET
import multiprocessing
import traci
import sumolib
import pyproj
from stg.utils import SUMO_outputs_process, simulate, gen_sumo_cfg, exec_od2trips

os.environ['SUMO_HOME']='/opt/sumo-1.8.0'
processors = multiprocessing.cpu_count() # due to memory lack -> Catalunya  map = 2GB

class folders(object):
    def __init__(self):
        self.verbose = False
        self.parents_dir = os.path.dirname(os.path.abspath('{}/..'.format(__file__)))
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
        self.end_hour = 0
        self.factor = "1"
        self.repetitions = 1

def clean_folder(folder):
    files = glob.glob(os.path.join(folder,'*'))
    [os.remove(f) for f in files]

def gen_route_files(traffic_df, folders):

    for h in folders.O_district:
        #for sd in folders.D_district:
            O_name = os.path.join(folders.O, f'{h}')
            O_files_list = create_O_file(traffic_df, folders, O_name, h)
            # Generate cfg files
            cfg_file_loc = gen_routes(O_name, O_files_list, folders, 'od2')
    return cfg_file_loc



def gen_od2trips(O_files, O, folders):

    # read O files
    #O_files_list = os.listdir(folders.O)
    O_listToStr = ','.join([f'{elem}' for elem in O_files])
    TAZ = os.path.join(folders.parents_dir, 'templates', 'TAZ.xml')
    od2trips_conf = os.path.join(folders.parents_dir, 'templates', 'od2trips.cfg.xml')
    # Open original file
    tree = ET.parse(od2trips_conf)

    # Update O input
    parent = tree.find('input')
    ET.SubElement(parent, 'od-matrix-files').set('value', f'{O_listToStr}')
    ET.SubElement(parent, 'taz-files').set('value', f'{TAZ}')

    # Update output
    parent = tree.find('output')
    output_name = f'{O}_od2.trip.xml'
    ET.SubElement(parent, 'output-file').set('value', output_name)

    # Update seed number
    parent = tree.find('random_number')
    ET.SubElement(parent, 'seed').set('value', f'{1}')

    # Write xml
    cfg_name = f'{O}_trips.cfg.xml'
    tree.write(cfg_name)
    return cfg_name, output_name



def gen_routes(O, O_files_list, folders, routing):
    """
    Generate configuration files for od2 trips
    """
    if routing == 'od2':
        # Generate od2trips cfg
        cfg_name, output_name = gen_od2trips(O_files_list, O, folders)
        # Execute od2trips
        output_name = exec_od2trips(cfg_name, output_name, folders)
        # Generate sumo cfg
        return gen_sumo_cfg(routing, output_name, 'r', folders,
                            folders.reroute_probability)  # last element reroute probability

    else:
        SystemExit('Routing name not found')

def clean_folders_ini(folders):
    clean_folder(folders.O)
    clean_folder(folders.cfg)
    clean_folder(folders.outputs)


def create_O_file(traffic_df, folders, fname, origin):
    O_files_saved = []
    for hour in range(folders.end_hour):
        # Build traffic demand from traffic file per hour
        O_tail_list = []
        O_tail_list.append([f'  {origin}    {destination}   {traffic_df.loc[hour,destination]}.00\n' for destination in folders.D_district])
        O_text = ['$OR;D2\n',  # O format
                  f'{hour}.00 {hour + 1}.00\n',  # Time 0-23 hours
                  f'{folders.factor}.00\n']   # Multiplication factor
        [O_text.append(e) for e in O_tail_list[0]]
        O_files_saved.append(write_O_file(O_text,hour,os.path.basename(fname)))
    return O_files_saved


def write_O_file(text,hour,name):
    O_file_name = os.path.join(folders.O, f'{hour}_{name}')
    O = open(f"{O_file_name}", "w")
    O.writelines(text)
    O.close()
    return(O_file_name)


def convertXYtoLonLat():
    net = sumolib.net.readNet('/root/STG/templates/osm.net.xml')
    #x, y = net.convertLonLat2XY(0.508087,40.817967)
    #can be used to convert statistics sumo outputs
    lon, lat = net.convertXY2LonLat(x, y)


def read_traffic(folders):
    traffic_df = pd.DataFrame(pd.read_csv(folders.realtraffic))
    folders.end_hour = int(traffic_df.shape[0])
    return traffic_df


def ini_paths(folders, factor, repetitions):
    folders.reroute_probability = '1'
    folders.parents_dir = os.path.dirname(os.path.abspath('{}/..'.format(__file__)))
    #folders.O_district = ['H_1','H_2','H_3','H_4']
    folders.O_district = ['H_1', 'H_2']
    folders.D_district = ['baix','montsia','terra','ribera']
    folders.O = "/root/Desktop/SEM/Torres_del_Ebre/O_files"
    folders.cfg = "/root/Desktop/SEM/Torres_del_Ebre/cfg"
    folders.outputs = "/root/Desktop/SEM/Torres_del_Ebre/outputs"
    folders.realtraffic = "/root/Desktop/SEM/Torres_del_Ebre/traffic.csv"
    folders.factor = "{}".format(factor)
    folders.repetitions = repetitions

    clean_folders_ini(folders)
    traffic_df = read_traffic(folders)
    sumo_cfg_file = gen_route_files(traffic_df, folders)
    return sumo_cfg_file


# Generate simulation files
sumo_cfg_file = ini_paths(folders,1,1) #folders, end_hour, factor, repetitions
# execute simulations OD2Trips
#simulate(folders, processors, 0)
