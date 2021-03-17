import os, sys
import shutil
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import subprocess


def run(output_dir, max_processors, omnet_path, sim_time, repetitions, analyze, iter_path, inifile, makefile, verbose, NED_files_dir):
    """
    Return the results relative to the desired parameter space in the form of an xarray data structure.csv.

    Args:
    output_dir: (path) The space of structure.csv to export.
    max_processors: (int) The max number of cpus to use. By default, all cpus are used.
    omnet_path: (path) Path to the OMNET++ installation.
    By default the script try to find the installation path.
    sim_time: (int) The simulation time. Common for all scenarios.
    repetitions: (int) The number of runs for each iteration parameter.
    analyze: (bool) If true execute simulation campaign. Otherwise analyze result files
    from a simulation campaign to find missing simulations.
    inifile: (path) Path to .ini file of veins project.
    makefile: (path) Path to executable veins project.

    """

    # Try to read scenarios from .ini file in veins_ini_file_path
    sim_scenarios_list = get_scenarios(inifile)

    print("\n Configuration file ({0}):\n".format(inifile))

    # List scenarios found in .ini file
    for i, scenario in enumerate(sim_scenarios_list):
        print(' {0}) {1} '.format(i, scenario))

    # Ask for scenarios to simulate or analyze
    scenarios_to_sim = ask_for_scenarios_to_simulate(sim_scenarios_list)

    # Build the simulation campaign
    build_simulation_campaign(max_processors, output_dir, omnet_path, sim_time, repetitions, scenarios_to_sim,
                              iter_path, inifile, analyze, makefile, verbose, NED_files_dir)

    # clear memory, swap after simulations
    clear_memory()


def get_scenarios(veins_ini_file):
    """
     Try to read simulation scenarios in VEINs project .ini file

     OMNET++ defines [Config ] as the structure.csv in ini file to declare an scenario.
     This function return a list with [Config ] declarations found in ini file.

    """
    scenarios = []
    with open(veins_ini_file, 'r') as ini_file:
        for line in ini_file:
            if isNotBlank(line):
                if '[Config' in line:
                    s_name = line.strip('\n[Config')
                    s_name = s_name.strip(']')
                    if isNotBlank(s_name):
                        scenarios.append(s_name.strip())
    return scenarios


def read_iteration_variables_from_file(scenarios_to_sim, iter_parameters_file_path):
    """
    Get the number of iteration structure.csv (defined as specified in OMNET++ Simulation manual 10.4 Parameter Studies)
    of each scenario (sim_scenarios) defined in .ini file.

    e.g. OMNET++ study parameter definition -> *.numHosts = ${1, 2, 5, 10..50 step 10}

    Args:
    sim_scenarios (list): List of selected scenarios to include in the campaign
    veins_ini_file_path (path): Path to .ini file of veins project.

    Return a dictionary with scenarios as dic keys and read_iteration_variables_from_file as values.
    Values in read_iteration_variables_from_file dictionary are defined as follows:

    scenario1: [parameters1, parameter2, ... , # of read_iteration_variables_from_file]
    scenario2: [parameters1, parameter2, ... , # of read_iteration_variables_from_file]
    """
    dic_iter_vars = {}
    dic_scenario_iter_vars = {}

    iter_parameters_file_path = os.path.join(iter_parameters_file_path, 'variables.txt')

    for s in scenarios_to_sim:
        with open(iter_parameters_file_path, 'r') as ini_file:
            for line in ini_file:
                if isNotBlank(line):
                    ivar, value = line.split('=')
                    temp, units = value.split('}')
                    temp = temp.strip().strip('{')
                    vlist = temp.strip().split(',')
                    iter_var_tuple = (vlist, units.strip())
                    dic_iter_vars[ivar.strip()] = iter_var_tuple
        dic_scenario_iter_vars[s] = dic_iter_vars
    return dic_iter_vars, dic_scenario_iter_vars


def scenario_runs_set(scenario_iteration_variables_dictionary, repetitions):
    """
     Generate runs list per scenario in OMNET++ format (opp_run all OMNET++ simulation manual  11.20 Running Simulation Campaigns)
     for create batches e.g. -r 0,1,2,3.

     Args:
        scenario_iteration_variables_dictionary (dict): Dictionary with scenarios as keys and the number of iteration variables.txt as values
        repetitions (int): The number of runs for each iteration parameter.

    """
    runs_dic_command = {}
    runs_dic_compute = {}
    simulations_count = 0

    for scenario in scenario_iteration_variables_dictionary:
        values_list = [len(scenario_iteration_variables_dictionary[scenario][x][0]) for x in scenario_iteration_variables_dictionary[scenario]]
        scenario_runs = np.prod(values_list) * repetitions
        simulations_count += scenario_runs
        runs_range = list(range(int(scenario_runs)))
        # compute
        runs_dic_compute[scenario] = runs_range
        # opp_runall syntax
        runs_dic_command[scenario] = str(runs_range).strip('[]').replace(" ", "")
    total_simulations = scenario_runs * len(scenario_iteration_variables_dictionary)
    return total_simulations, runs_dic_command, runs_dic_compute


def sim_campaign_info(scenarios_to_sim, iteration_variables_dictionary, repetitions, simtime, total_sims):
    """
     Print simulation campaign summary:

     Args:

    """
    iter_var = [len(iteration_variables_dictionary[k][0]) for k in iteration_variables_dictionary.keys()]
    banner = ' Campaign Info'
    print('\n{}\n'.format(banner), '-' * len(banner))
    print("\n Scenarios to simulate: {0}"
          "\n Iteration variables.txt: {1} = {2}"
          "\n Repetitions per scenario: {3}"
          "\n Simulation time: {4}s"
          "\n Total Runs: {5}".format(scenarios_to_sim, len(list(iteration_variables_dictionary.keys())), iter_var, repetitions,
                                           simtime, total_sims))
    print('', '-' * len(banner), '\n')


def run_simulations():
    try:
        s_build_campaign = str(input(' Run simulations (*Y/N): '))
    except:
        print('Wrong input!')
        return False
    else:
        if s_build_campaign in ['y', 'Y', '']:
            return True
        else:
            return False


def build_simulation_campaign(max_processors, output_dir, omnet_path, sim_time, repetitions, sim_scenarios_list,
                              variables_path, inifile, analyze, makefile, verbose, NED_files_dir):
    """
     Execute parallel simulations of simulation campaign elements. If there is not enough processors,
     a bath is used for queue simulations and distribute among processors.

     Args:

    @param max_processors: (int) The max number of cpus to use. By default, all cpus are used.
    @param output_dir: (path) The space of structure.csv to export.
    @param omnet_path: (path) The OMNET++ installation path
    @param sim_time: (int) The simulation time. Common for all scenarios.
    @param repetitions: (int) The number of runs for each iteration parameter.
    @param sim_scenarios: (list) List of selected scenarios to include in the campaign
    @param inifile: (path) Path to .ini file of veins project.
    @param analyze: (bool) If true execute simulation campaign. Otherwise analyze result files
    @param makefile: (path) Path to executable veins project.
    @return:
    """
    # Return a dictionary with scenarios as dic keys and read_iteration_variables_from_file as values
    iteration_variables_dictionary, scenario_iteration_variables_dictionary = read_iteration_variables_from_file(
        sim_scenarios_list, variables_path)

    # Total number of runs = scenarios * iter variable * repetitions_per_scenario
    # return -r 0,1,2,3,4 parameter per scenario according to iter variables.txt
    total_sims, runs_dic_command, runs_dic_compute = scenario_runs_set(scenario_iteration_variables_dictionary, repetitions)

    # simulation campaign information
    sim_campaign_info(sim_scenarios_list, iteration_variables_dictionary, repetitions, sim_time, total_sims)

    if analyze:
        # check missing files of simulation campaign
        missing_files(total_sims, output_dir)

    else:

        if run_simulations():
            # script memory
            #program = '/root/cpu_mem.sh'
            #subprocess.Popen(program)

            # create temp ini file for the simulation campaign
            temp_ini_name = create_temp_ini_file(output_dir, repetitions, inifile, iteration_variables_dictionary)

            # create results folder
            new_folder(output_dir)

            if total_sims <= max_processors:
                batch = 1  # default when b <= 1 enough cpus

            else:
                #b = total_sims / max_processors
                #batch = math.ceil(b)
                batch = allocate_processors(runs_dic_compute, max_processors)
            # execute parallel simulations
            parallel(max_processors, omnet_path, batch, sim_time, runs_dic_command, temp_ini_name, sim_scenarios_list, makefile, verbose, NED_files_dir)


def allocate_processors(df_scenarios, Processors):
    len_scenarios = []
    processors_per_scenario_tmp = []

    [len_scenarios.append(len(df_scenarios[s])) for s in df_scenarios]
    processors_per_scenario = (np.true_divide(len_scenarios, sum(len_scenarios))) * Processors

    [processors_per_scenario_tmp.append(1 if s < 1 else np.math.floor(s)) for s in processors_per_scenario]
    batch_per_scenario = np.ceil(np.divide(len_scenarios, processors_per_scenario_tmp))
    #np.repeat(batch_per_scenario, processors_per_scenario_tmp)
    return batch_per_scenario


def missing_files(total_sims, output_dir):
    """
    Check in results folder if there are missing files of simulation campaign
    @param total_sims: (int) Total number of runs = scenarios * iter variable * repetitions_per_scenario
    @param output_dir: (path) The space of structure.csv to export.
    @return:
    """

    # Analyze results looking for missing result simulation files
    files_in_results_folder = len(os.listdir(output_dir))
    if total_sims == files_in_results_folder:
        print('\n Files successfully generated: %s' % total_sims)
    else:
        print('\n Missing files in results folder: %s' % (total_sims - files_in_results_folder))


def parallel(max_processors, omnet_path, batch, sim_time, runs_bundle, temp_ini_name, sim_scenarios_list,
             makefile, verbose, NED_files_dir):
    """
    Execute parallel summary. If the number of cpus < # of summary a batch of runs is set

    Args:
    @param max_processors: (int) The max number of cpus to use. By default, all cpus are used.
    @param omnet_path: (path) Path to the OMNET++ installation.
    @param batch: (int) Number of simulations per cpu
    @param sim_time: (int) The simulation time. Common for all scenarios.
    @param runs_bundle: (int) Bundle of runs (e.g. 0,1,2,3..)
    @param temp_ini_name: (string) VEINs ini configuration file name
    @param iter_var_per_scenario: (dict) Dictionary with scenarios as keys and iterations as values
    @param makefile: (path) Path to executable veins project.
    @return:
    """
    # Parallelize simulations
    with parallel_backend("loky"):
        Parallel(n_jobs=max_processors, verbose=10)(delayed(execute_sim)(makefile, max_processors, omnet_path, k, 1 if isinstance(batch, int) else int(batch[i]), sim_time, runs_bundle[k], temp_ini_name, verbose, NED_files_dir) for i, k in enumerate(sim_scenarios_list))


def new_folder(new_directory):
    """
     Create new folder and replace if it exists. Used to creates the results folder where results files are saved.

     Args:
        new_directory (path): Path of the new folder
    """
    if os.path.exists(new_directory):
        shutil.rmtree(new_directory)  # Removes all the subdirectories!
    os.makedirs(new_directory)


def folder_permissions(veins_exec_project_path):
    # TO DO assign folder permissions
    pass
    # os.chmod(PROJECT_EXECUTABLE, stat.S_IXGRP)
    # os.system('chmod -R +x {0}'.format(os.path.dirname(PROJECT_EXECUTABLE)))
    # os.system('chmod -R +x {0}'.format(os.path.dirname(VEINS_INI_PATH)))
    # exec = '{0}/../../src/{1}'.format(temp_ini_name,)  # TO DO find executable


def execute_sim(veins_exec_project_path, max_processors, omnet_path, scenario, batch, sim_time, runs, temp_ini_name,
                verbose, NED_files_dir):
    """
    Execute scenario simulation using OMNET++ funcionality (opp_run all OMNET++ simulation manual  11.20 Running Simulation Campaigns)

    Args:
    @param veins_exec_project_path: (path) Path to executable veins project.
    @param max_processors: (int) The max number of cpus to use. By default, all cpus are used.
    @param omnet_path: (path) Path to the OMNET++ installation.
    @param scenario: Scenario to simulate
    @param batch: Batch of simulations
    @param sim_time: (int) The simulation time. Common for all scenarios.
    @param runs: Bundle of runs (e.g. 0,1,2,3....)
    @param temp_ini_name: (string) VEINs ini configuration file name
    @return:
    :param verbose:
    """

    '-n .:../../src/veins '
    # Change directory before execute simulation
    os.chdir(os.path.dirname(temp_ini_name))

    # Allow executions and file creation in simulations paths
    folder_permissions(veins_exec_project_path)
    if isNotBlank(scenario):
        cmd = '{0}opp_runall -j{1} -b{2} {3} -u Cmdenv -c {4} -r {5} ' \
              '-n {6} ' \
              '--cmdenv-performance-display=false ' \
              '--sim-time-limit={7}s ' \
              '--cmdenv-redirect-output=true ' \
              '--cmdenv-express-mode=true ' \
              '{8}'.format(omnet_path, max_processors, batch, veins_exec_project_path, scenario, runs, NED_files_dir,
                           sim_time, temp_ini_name)
        print(cmd)
        # Execute command
        if verbose:
            os.system(cmd)
        else:
            to_log_file = subprocess.check_output(cmd, shell=True)  # TO DO save in log file


def isNotBlank(myString):
    """
    Check if string is empty or null.

    Args:

    @param myString: (string) Any string
    @return: (bool)
    """

    if myString and myString.strip():
        return True
    return False


def clear_memory():
    """Clean memory cache at the end of simulation execution"""
    if os.name != 'nt':  # Linux system
        os.system('sync')
        os.system('echo 3 > /proc/sys/vm/drop_caches')
        os.system('swapoff -a && swapon -a')
    # print("Memory cleaned")


def create_temp_ini_file(output_results, repetitions, veins_ini_file_name, iteration_variables_dictionary):
    """
     Instantiates a temp.ini file with simulation campaign configurations.

     Args:
          output_results (path): The space of structure.csv to export.
          repetitions (int): The number of runs for each iteration parameter.
          veins_ini_file_name (path): Path to .ini file of veins project.


     """
    file_name = veins_ini_file_name.strip('.ini')
    temp_file_name = "{0}.temp.ini".format(file_name)
    # declare variable where results will be save
    var_save_result_in_ini = '*.*.appl.filename'  # TO DO check in ini file a common variable
    filename = '${configname},${iterationvarsf},${repetition}'
    log_output_file = 'cmdenv-output-file'  # TO DO check in ini file a common variable

    with open(veins_ini_file_name, 'r') as ini_file:
        with open(temp_file_name, 'w') as temp_ini_file:
            for line in ini_file:
                if 'repeat =' in line:
                    pass
                elif 'repeat=' in line:
                    pass
                elif log_output_file in line:
                    scenario_log_name = '{}.out'.format(filename)
                    temp_ini_file.write(
                        "{0} = {1}/../logs/{2}\n".format(log_output_file, output_results, scenario_log_name))
                elif '[Config' in line:
                    temp_ini_file.writelines(line)
                    temp_ini_file.write("repeat = {0}\n".format(repetitions))
                    add_iteration_variables_to_scenarions_ini_file(temp_ini_file, iteration_variables_dictionary)
                    results_file_name = os.path.join(output_results, filename)
                    temp_ini_file.write('{0} = "{1}"\n'.format(var_save_result_in_ini, results_file_name))
                else:
                    temp_ini_file.writelines(line)
    return temp_file_name


def add_iteration_variables_to_scenarions_ini_file(temp_ini_file, iteration_variables_dictionary):
    """
    Scenarios with same iteration variables.txt

    """
    for iteration_variable in iteration_variables_dictionary:
        values = '{' + ','.join(iteration_variables_dictionary[iteration_variable][0]) + '}'
        temp_ini_file.write('{0} = ${1}{2}\n'.format(iteration_variable, values, iteration_variables_dictionary[iteration_variable][1]))


def ask_for_scenarios_to_simulate(sim_scenarios_list):
    """
     Select scenarios to include in the simulation campaign

     Args:
        sim_scenarios_list(list): List of scenarios found in .ini file specified as argument

    @param sim_scenarios_list:
    @return:
    """
    # TO DO REPEATED INPUTs
    try:
        input_scenarios = input('\n Select the scenario number to simulate separated by space (e.g. 1 5 20): ')
        scenarios_sim_list = input_scenarios.split()
        scenarios_to_simulate = list(map(int, scenarios_sim_list))
        sim_list = list(range(len(sim_scenarios_list)))
        if scenarios_to_simulate:
            for s in scenarios_to_simulate:
                if s not in sim_list:
                    print('\n Scenario {} not in simulation list'.format(s))
                    sys.exit(os.EX_SOFTWARE)
        else:
            sys.exit(os.EX_SOFTWARE)
    except:
        print('\n !!! Wrong input !!!\n')
        sys.exit(os.EX_SOFTWARE)
    else:
        return [sim_scenarios_list[i] for i in scenarios_to_simulate]


