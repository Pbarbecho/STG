import os, sys, glob
import pandas as pd
import xml.etree.ElementTree as ET
import multiprocessing
import sumolib
from stg.utils import SUMO_outputs_process, simulate, gen_sumo_cfg, exec_od2trips
processors = multiprocessing.cpu_count() # due to memory lack -> Catalunya  map = 2GB
import timeit



# import sumo tool xmltocsv
if 'SUMO_HOME' in os.environ:
    tools = os.path.join('/opt/sumo-1.8.0/', 'tools')
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(tools))
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


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
        self.factor = 1
        self.repetitions = 1


def clean_folder(folder):
    files = glob.glob(os.path.join(folder,'*'))
    [os.remove(f) for f in files]


def gen_route_files(traffic_df, folders):
    for h in folders.O_district:
        O_name = os.path.join(folders.O, f'{h}')
        O_files_list = create_O_file(traffic_df, folders, O_name, h)
        # Generate cfg files
        cfg_file_loc = gen_routes(O_name, O_files_list, folders, 'od2')
    return cfg_file_loc


def gen_od2trips(O_files, O, folders):
    # create od2trips config file
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

    #update end time
    parent = tree.find('time')
    end_time = f'{int(folders.end_hour) * 3600}' # ain seconds
    ET.SubElement(parent, 'end').set('value', end_time)

    # processing options
    parent = tree.find('processing')
    scale = f'1.{folders.factor}'  # scale 1.X
    ET.SubElement(parent, 'scale').set('value', scale)

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
        return gen_sumo_cfg(routing, output_name, 'r', folders, folders.reroute_probability)  # last element reroute probability
    else:
        SystemExit('Routing name not found')


def clean_folders_ini(folders):
    clean_folder(folders.O)
    clean_folder(folders.cfg)
    clean_folder(folders.outputs)
    clean_folder(folders.xmltocsv)


def create_O_file(traffic_df, folders, fname, origin):
    O_files_saved = []
    for hour in range(folders.end_hour):
        # O file header
        O_text = ['$OR;D2\n',  # O format
                  f'{hour}.00 {hour + 1}.00\n',  # Time 0-23 hours
                  f'1.{folders.factor}\n']  # Multiplication factor

        O_tail_list = []
        direct_destination = ['H_5','H_6'] # no tiene que generar para tpdps los taz sino entre ellos

        Save_O_file = False
        for destination in folders.D_district:
            if destination not in direct_destination and origin not in direct_destination:
                O_tail_list.append(f'  {origin}    {destination}   {traffic_df.loc[hour, destination]}.00\n')
                Save_O_file = True
            elif origin == 'H_5' and destination== 'H_6':
                O_tail_list.append(f'  {origin}    {destination}   {traffic_df.loc[hour, "H5_To_H6"]}.00\n')
                Save_O_file = True
            elif origin == 'H_6' and destination == 'H_5':
                O_tail_list.append(f'  {origin}    {destination}   {traffic_df.loc[hour, "H6_To_H5"]}.00\n')
                Save_O_file = True

        if Save_O_file: # save only filtered files
            [O_text.append(e) for e in O_tail_list]
            O_files_saved.append(write_O_file(O_text,hour,os.path.basename(fname)))

    return O_files_saved


def write_O_file(text,hour,name):
    O_file_name = os.path.join(folders.O, f'{hour}_{name}')
    O = open(f"{O_file_name}", "w")
    O.writelines(text)
    O.close()
    return(O_file_name)


def read_traffic(folders):
    traffic_df = pd.DataFrame(pd.read_csv(folders.realtraffic))
    folders.end_hour = int(traffic_df.shape[0])
    return traffic_df

def ini_paths(folders, factor, repetitions):
    folders.reroute_probability = '-1'
    folders.parents_dir = os.path.dirname(os.path.abspath('{}/..'.format(__file__)))
    folders.O_district = ['H_1','H_2','H_3','H_4','H_5','H_6']
    #folders.O_district = ['H_1']
    #folders.D_district = ['camp']
    folders.D_district = ['baix','montsia','terra','ribera','camp','H_5','H_6']
    folders.O = "/root/Desktop/SEM/Torres_del_Ebre/O_files"
    folders.cfg = "/root/Desktop/SEM/Torres_del_Ebre/cfg"
    folders.outputs = "/media/newdisk/SEM/outputs"
    folders.realtraffic = "/root/Desktop/SEM/Torres_del_Ebre/traffic.csv"
    folders.xmltocsv = "/media/newdisk/SEM/xmltocsv"
    folders.factor = "{}".format(factor)
    folders.repetitions = repetitions

def generate_simulation_files(folders):
    clean_folders_ini(folders)
    traffic_df = read_traffic(folders)
    sumo_cfg_file = gen_route_files(traffic_df, folders)
    return sumo_cfg_file

def convertXYtoLonLat():
    net = sumolib.net.readNet('/root/STG/templates/osm.net.xml')
    #x, y = net.convertLonLat2XY(0.508087,40.817967)
    #can be used to convert statistics sumo outputs
    lon, lat = net.convertXY2LonLat(x, y)

def sort_csv_files(folders):
    sumoOutFiles_list = ['tripinfo', 'fcd']
    sumo_out_list = os.listdir(folders.outputs)

    h_list = []
    [h_list.append(int(f.split('_')[1])) for f in sumo_out_list]
    max_h = max(h_list)

    dic = {}
    for h in range(1, max_h + 1):
        h_temp_list = []
        for f in sumo_out_list:
            if f.split('_')[2] in sumoOutFiles_list and h == int(f.split('_')[1]):
                h_temp_list.append(f'{f.strip(".xml")}.csv')
        dic[f"{h}"] = h_temp_list
    return dic


def xml_to_csv(folders):
   for f in os.listdir(folders.outputs):
        # output directory
        output = os.path.join(folders.xmltocsv, f'{f.strip(".xml")}.csv')
        # SUMO tool xml into csv
        sumo_tool = os.path.join(tools, 'xml', 'xml2csv.py')
        # Run sumo tool with sumo output file as input
        cmd = 'python {} {} -s , -o {}'.format(sumo_tool, os.path.join(folders.outputs, f), output)
        print(f'Convirtiendo {f} to csv ....')
        os.system(cmd)
   dic_csv = sort_csv_files(folders)
   return dic_csv


def merge_function(trip,fcd, fname):
    print('\nMerging ... ',trip, fcd)
    df_trip = pd.read_csv(trip)
    df_fcd = pd.read_csv(fcd)
    df_fcd.rename(columns={'vehicle_id' :'tripinfo_id'}, inplace=True)
    df = pd.merge(df_trip, df_fcd, on='tripinfo_id')
    df = df.filter(['tripinfo_id','timestep_time','vehicle_x','vehicle_y',
                    'tripinfo_duration','tripinfo_routeLength', 'tripinfo_departDelay','tripinfo_timeloss', 'tripinfo_departLane', 'tripinfo_arrivalLane'])
    df.sort_values(by=['tripinfo_id','timestep_time'],inplace=True)
    df = df[df['tripinfo_duration'] > 60] # remove entries with triptime less than a minute
    df['Origin'] = fname

    data_list = []
    for id in df['tripinfo_id'].unique():
        temp_df = df[df['tripinfo_id']==id]
        temp_df = temp_df[(temp_df['timestep_time']== min(temp_df['timestep_time'])) | (temp_df['timestep_time']==max(temp_df['timestep_time']))]
        data_list.append(temp_df)
    new_df = pd.concat(data_list)
    new_df.to_csv(f'/media/newdisk/SEM/results/{fname}.csv')



def merge_outputs(folders, csv_dic):
    for e in range(1,len(csv_dic)+1):
        trip_file = ''
        fcd_file = ''
        for ename in csv_dic[f'{e}']:
            if 'tripinfo' in ename.split('_'):
                trip_file = os.path.join(folders.xmltocsv,ename)
            elif 'fcd' in ename.split('_'):
                fcd_file = os.path.join(folders.xmltocsv, ename)
        merge_function(trip_file, fcd_file, f'H{e}')


def print_time(n,begin,end):
    print('**'*20)
    print(f'Step {n}:','B:',begin,'E:',end )
    print('**'*20)


# Initialize paths
start = timeit.timeit()
ini_paths(folders,1,1) #folders, factor, repetitions  ## horu updates at traffic roww entry
end = timeit.timeit()
print_time(1,start,end)

#2. Generate simulation files
start = timeit.timeit()
sumo_cfg_file = generate_simulation_files(folders)
end = timeit.timeit()
print_time(2,start,end)

#3. Execute simulations OD2Trips
start = timeit.timeit()
simulate(folders, processors, 0) #gui
end = timeit.timeit()
print_time(3,start,end)

#4. Merge outputs
start = timeit.timeit()
csv_dic = xml_to_csv(folders)
merge_outputs(folders, csv_dic)
end = timeit.timeit()
print_time(4,start,end)
#5. Filter outputs

# pendiente script para localizar el lane en algun taz