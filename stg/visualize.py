#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 21:59:07 2020

@author: root
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib import rc
import matplotlib.colors as mcolors
from utils import merge_detector_lanes

plt.rcParams.update({'font.size': 20})
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=False)


output_dir = '/root/Desktop/MSWIM/Revista/sim_files'
####################### Real traffic on Travessera de Gracia  ####################################
actual_traffic = pd.read_csv('/root/Desktop/MSWIM/Revista/TrafficPgSanJoan.csv')
actual_traffic = actual_traffic.filter(['Hour','Actual'])

origin = 'Hospitalet'
destination = 'SanAdria'
#################################### Reroute 0 ####################################
taz_ma_df_0 = pd.read_csv('/root/Desktop/MSWIM/Revista/sim_files/Taz/ma/0_0/detector/detector.csv')
taz_dua_df_0 = pd.read_csv('/root/Desktop/MSWIM/Revista/sim_files/Taz/dua/0_0/detector/detector.csv')
random_df_0 = pd.read_csv('/root/Desktop/MSWIM/Revista/sim_files/RandomTrips/0_0/detector/detector.csv')
taz_od2_df_0 = pd.read_csv('/root/Desktop/MSWIM/Revista/sim_files/Taz/od2/0_0/detector/detector.csv')
taz_duaiter_df_0 = pd.read_csv('/root/Desktop/MSWIM/Revista/sim_files/Taz/duaiterate/0_0/detector/detector.csv')

# SUMO outputs
random_traffic_metrics_df_0 = pd.read_csv('/root/Desktop/MSWIM/Revista/sim_files/RandomTrips/0_0/parsed/data.csv')
taz_ma_traffic_metrics_df_0 =  pd.read_csv('/root/Desktop/MSWIM/Revista/sim_files/Taz/ma/0_0/parsed/data.csv')
taz_dua_traffic_metrics_df_0 =  pd.read_csv('/root/Desktop/MSWIM/Revista/sim_files/Taz/dua/0_0/parsed/data.csv')
taz_od2_traffic_metrics_df_0 =  pd.read_csv('/root/Desktop/MSWIM/Revista/sim_files/Taz/od2/0_0/parsed/data.csv')
taz_duaiter_traffic_metrics_df_0 =  pd.read_csv('/root/Desktop/MSWIM/Revista/sim_files/Taz/duaiterate/0_0/parsed/data.csv')

# SUMO summary
taz_dua_summary_df_0 =  pd.read_csv(f'/root/Desktop/MSWIM/Revista/sim_files/Taz/dua/0_0/xmltocsv/{origin}_{destination}_summary_0.csv')
taz_ma_summary_df_0 =  pd.read_csv(f'/root/Desktop/MSWIM/Revista/sim_files/Taz/ma/0_0/xmltocsv/{origin}_{destination}_summary_0.csv')
taz_od2_summary_df_0 =  pd.read_csv(f'/root/Desktop/MSWIM/Revista/sim_files/Taz/od2/0_0/xmltocsv/{origin}_{destination}_summary_0.csv')
random_summary_df_0 =  pd.read_csv(f'/root/Desktop/MSWIM/Revista/sim_files/RandomTrips/0_0/xmltocsv/{origin}_{destination}_summary_0.csv')
taz_duaiter_summary_df_0 =  pd.read_csv(f'/root/Desktop/MSWIM/Revista/sim_files/Taz/duaiterate/0_0/xmltocsv/{origin}_{destination}_summary_0.csv')
###################################################################################

# Process detector -> number of vehicles L0 and L1
random_df = merge_detector_lanes(random_df_0, 'RT',0)
taz_ma_df = merge_detector_lanes(taz_ma_df_0, 'MAR',0)
taz_dua_df = merge_detector_lanes(taz_dua_df_0, 'DUAR',0)
taz_duaiter_df = merge_detector_lanes(taz_duaiter_df_0, 'DUAI',0)
taz_od2_df = merge_detector_lanes(taz_od2_df_0, 'OD2',0)


# Merge all tools
merge_on = ['RP','Hour','interval_begin','interval_end']
#traffic_df = random_df.merge(taz_ma_df, on='Hour').merge(taz_dua_df, on='Hour').merge(taz_duaiter_df, on='Hour').merge(taz_od2_df, on='Hour').merge(actual_traffic, on='Hour')
traffic_df = random_df.merge(taz_ma_df, on=merge_on).merge(taz_dua_df, on=merge_on).merge(taz_duaiter_df, on=merge_on).merge(taz_od2_df, on=merge_on)
traffic_df = traffic_df.merge(actual_traffic, on='Hour')


# Save to sim folder
traffic_df.to_csv('/root/Desktop/MSWIM/Revista/sim_files/data.csv')


# Sort traffic df
#traffic_df = pd.melt(traffic_df, id_vars=['Hour'], value_vars=['MAR','DUAR','DUAI','RT','OD2','Actual'], 
#             var_name='Routing',
#             value_name='Traffic')



# Files list
f_list = [taz_ma_traffic_metrics_df_0, 
        taz_dua_traffic_metrics_df_0, 
        taz_duaiter_traffic_metrics_df_0, 
        taz_od2_traffic_metrics_df_0, 
        random_traffic_metrics_df_0]

# Files list
sum_list = [taz_ma_summary_df_0, 
            taz_dua_summary_df_0, 
            taz_duaiter_summary_df_0, 
            taz_od2_summary_df_0, 
            random_summary_df_0]
            
f_names = ['MAR', 'DUAR', 'DUAI', 'OD2', 'RT']
markers=['+','s','o','^','x','v','D','.','<','>',',']



def plot_actual_traffic():
    fig, ax = plt.subplots(figsize=(12,4))
    new_df = traffic_df.copy()
    # smoth
    #f2 = interp1d(x='Hour', y=col, kind='cubic')
    new_df.rename(columns={'Actual':'Workin day'}, inplace=True)
    col = 'Workin day'
    # plot actual traffic
    new_df.plot(kind='line',x='Hour',y=col, marker=markers[0], ax=ax)
    # Plot peak hour line
    ymax = new_df['Workin day'].max() 
    xmax = new_df.loc[new_df['Workin day'] == ymax, 'Hour']
    plt.axhline(ymax, color='tab:orange')
    style = dict(size=10, color='black')
    margin = 10
    ax.text(xmax,ymax+margin, "Peak hour", **style)
    # plot settings    
    plt.ylabel('# of vehicles')
    #plt.title('Traffic intensity')
    plt.xticks(np.arange(min(traffic_df['Hour']), max(traffic_df['Hour'])+1, 2.0))
    plt.yticks(np.arange(min(traffic_df['Hour']), ymax+50, 50.0))
    plt.grid(True, linewidth=1, linestyle='--')    
    
    
    
def plot_tools_traffic():
    # traffic tools
    cols = ['MAR', 'DUAR', 'DUAI', 'RT', 'OD2', 'Actual']
    linestyle_list = ['-', '--','-.',':','-','--'] 
   
    # plot traffic intensity
    fig, ax = plt.subplots(figsize=(8,5))
    for i, col in enumerate(cols):
        traffic_df.plot(kind='line', linewidth=2, linestyle=linestyle_list[i] , x='Hour',y=col, ax=ax)
    plt.legend(prop={'size': 16})
    plt.ylabel('Traffic intensity [veh/hour]')
    #plt.title('Traffic intensity')
    plt.xticks(np.arange(min(traffic_df['Hour']), max(traffic_df['Hour'])+1, 2.0))
    plt.yticks(np.arange(0, 300, 50))
    plt.grid(True, linewidth=0.5, linestyle='--')  
    
def plot_tools_traffic():
    # traffic tools
    cols = ['MAR', 'DUAR', 'DUAI', 'RT', 'OD2', 'Actual']
    linestyle_list = ['-', '--','-.',':','-','--'] 
   
    # plot traffic intensity
    fig, ax = plt.subplots(figsize=(8,5))
    for i, col in enumerate(cols):
        traffic_df.plot(kind='line', linewidth=2, linestyle=linestyle_list[i] , x='Hour',y=col, ax=ax)
    plt.legend(prop={'size': 16})
    plt.ylabel('Traffic intensity [veh/hour]')
    #plt.title('Traffic intensity')
    plt.xticks(np.arange(min(traffic_df['Hour']), max(traffic_df['Hour'])+1, 2.0))
    plt.yticks(np.arange(0, 300, 50))
    plt.grid(True, linewidth=0.5, linestyle='--')  
    

def traffic_metrics(df, tittle):
    
    # Plot origin destination (longitud latitude)
    # receives df and title
    mpl.style.use('default')
    df = df.filter(['ini_x_pos', 'ini_y_pos','end_x_pos', 'end_y_pos' ])
    
    fig, ax = plt.subplots(figsize=(10,8))
    df.plot.scatter(x='ini_x_pos',y='ini_y_pos', c='tab:orange', label='Origen', ax=ax)
    df.plot.scatter(x='end_x_pos',y='end_y_pos', c='tab:blue', label='Destination', ax=ax)
    plt.ylabel('Logitud')
    plt.xlabel('Latitud')
    plt.title(f'{tittle}')
    plt.grid(True, linewidth=1, linestyle='--')      
    ax.legend()


def build_metrics_df(metric):
    metric_dic = {'MAR':taz_ma_traffic_metrics_df_0[f'{metric}'],
             #'MAR-R':taz_ma_traffic_metrics_df_1[f'{metric}'],
             'DUAR':taz_dua_traffic_metrics_df_0[f'{metric}'],
             #'DUAR-R':taz_dua_traffic_metrics_df_1[f'{metric}'],
             'DUAI':taz_duaiter_traffic_metrics_df_0[f'{metric}'],
             #'DUAI-R':taz_dua_traffic_metrics_df_1[f'{metric}'],
             'RT':random_traffic_metrics_df_0[f'{metric}'],
             #'RT-R':random_traffic_metrics_df_1[f'{metric}']
             'OD2':taz_od2_traffic_metrics_df_0[f'{metric}']
             #'OD2-R':taz_od2_traffic_metrics_df_1[f'{metric}'],
            }
    return pd.DataFrame(metric_dic)
    
       
def prepare_fundamental_traffic_metrics(mdf):
    # Prepare dataframe
    mdf = mdf.T.reset_index()
    # axis 1 rows  *****
    mdf['Mean'] = mdf.mean(axis=1, skipna=True)
    mdf['SD'] = mdf.std(axis=1, skipna=True)
    mdf = mdf.filter(['index','Mean','SD'])
    return mdf



def OD_plots(df, tittle):
    
    # Plot origin destination (longitud latitude)
    # receives df and title
    mpl.style.use('default')
    df = df.filter(['ini_x_pos', 'ini_y_pos','end_x_pos', 'end_y_pos' ])
    
  
    
    fig, ax = plt.subplots(figsize=(10,8))
    df.plot.scatter(x='ini_x_pos',y='ini_y_pos', c='tab:orange', label='Origen', ax=ax)
    df.plot.scatter(x='end_x_pos',y='end_y_pos', c='tab:blue', label='Destination', ax=ax)
    plt.ylabel('Logitud')
    plt.xlabel('Latitud')
    plt.title(f'{tittle}')
    plt.grid(True, linewidth=1, linestyle='--')      
    ax.legend()
    
    

def plot_metric(df, ylabel):
     # Plot current metric 
    fig, ax = plt.subplots(figsize=(8,4))
    plt.errorbar(df['index'], df['Mean'], yerr=df['SD'], fmt="None", color='Black', elinewidth=1, capthick=1,errorevery=1, alpha=1, ms=4, capsize = 2)
    plt.bar(df['index'], df['Mean'], width=0.5, color=mcolors.TABLEAU_COLORS)
    plt.ylabel(f'{ylabel}')


def boxplot_metric(df, ylabel):
    box_width = 0.3
    df.plot.box(figsize=(4,2), showfliers=False,
                whiskerprops=dict(color='tab:blue', linestyle='--', linewidth=1),
                color=dict(boxes='tab:blue', whiskers='tab:blue', medians='tab:green', caps='tab:blue'),
                widths=(box_width, box_width, box_width, box_width, box_width))
    plt.ylabel(f'{ylabel}')
    #plt.grid(axis='y', linewidth=1, linestyle='--')
    plt.grid(axis='y', linewidth=1)
    """
    df.plot.box(showfliers=False, 
         color=dict(boxes='r', whiskers='r', medians='r', caps='r'),
         boxprops=dict(linestyle='-', linewidth=1.2),
         flierprops=dict(linestyle='-', linewidth=1.2),
         medianprops=dict(linestyle='-', linewidth=1.2),
         whiskerprops=dict(linestyle='--', linewidth=1.2),
         capprops=dict(linestyle='-', linewidth=1.2),
         widths=(box_width, box_width, box_width, box_width, box_width))
    """
 
def fundamental_metric_plots():   
    # Plot  mean distance/triptime/speed
    # receives df and title
 
    
    # traffic metrics
    cols = ['avrg_speed','tripinfo_duration','tripinfo_routeLength','tripinfo_timeLoss','tripinfo_waitingCount','tripinfo_waitingTime']
    for f in f_list:
        f = f.filter(cols)
    #######################################
    # Prepare dataframe of passed metric  #
    #######################################
    
    # Tripinfo duration
    mdf = build_metrics_df('tripinfo_duration')/60 # minutes
    plot_df = prepare_fundamental_traffic_metrics(mdf)
    
    print('time',mdf.describe())
    plot_metric(plot_df, "Trip time [min]" )
    boxplot_metric(mdf, "Trip time [min]")
    
    # Trip length
    mdf = build_metrics_df('tripinfo_routeLength')/1000 # minutes
    
    
    plot_df = prepare_fundamental_traffic_metrics(mdf)
    plot_metric(plot_df, "Trip length [km]" )
    boxplot_metric(mdf, "Trip length [km]" )
    print('lenght',mdf.describe())
    """
    # Mean Speed
    mdf = build_metrics_df('avrg_speed') # minutes
    plot_df = prepare_fundamental_traffic_metrics(mdf)
    plot_metric(plot_df, "Mean speed [m/s]" )
    """
    
    

def fundamental_metrics_hist():   
    # Plot  mean distance/triptime/speed
    # receives df and title
    
  # filter columns
    cols = ['avrg_speed','tripinfo_duration','tripinfo_routeLength','tripinfo_timeLoss','tripinfo_waitingCount','tripinfo_waitingTime']
    for f in f_list:
        f = f.filter(cols)
    
    n_bins, i, p_x, p_y = 5,0,3,2
    fig, axs = plt.subplots(p_x,p_y, figsize=(8,8), sharex=False, sharey=False)
    
    for x in range(p_x):
        for y in range(p_y):    
          if i < len(cols):
                metric = cols[i]
                d_dic = {'MAR':taz_ma_traffic_metrics_df_0[f'{metric}'],
                         'MAR-R':taz_ma_traffic_metrics_df_1[f'{metric}'],
                         'DUAR':taz_dua_traffic_metrics_df_0[f'{metric}'],
                         'DUAR-R':taz_dua_traffic_metrics_df_1[f'{metric}'],
                         'DUAI':taz_dua_traffic_metrics_df_0[f'{metric}'],
                         'DUAI-R':taz_dua_traffic_metrics_df_1[f'{metric}'],
                         'OD2':taz_od2_traffic_metrics_df_0[f'{metric}'],
                         'OD2-R':taz_od2_traffic_metrics_df_1[f'{metric}'],
                         'RT':random_traffic_metrics_df_0[f'{metric}'],
                         'RT-R':random_traffic_metrics_df_1[f'{metric}']
                         }
                df = pd.DataFrame(d_dic)
                #df.plot.hist(n_bins, density=True, histtype='step', legend=False, stacked=False, edgecolor='black', ax=axs[x,y])
                df.plot.hist(n_bins, density=True, histtype='step', legend=False, stacked=False, ax=axs[x,y])
                #axs[x,y].legend(prop={'size': 5})                
                axs[x,y].set_title(f'Avrg. {metric}')
                i+=1
    axs[x,y].legend(loc='upper center', bbox_to_anchor=(-0.2, -0.3),
          ncol=3, fancybox=True, shadow=True)
                
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.3,
                    wspace=0.35)    
    
    
    
def summary_plot():

    # tag dfs
    taz_ma_summary_df_0['Routing'] = 'MAR'
    taz_dua_summary_df_0['Routing'] = 'DUAR'
    taz_duaiter_summary_df_0['Routing'] = 'DUAI'
    taz_od2_summary_df_0['Routing'] = 'OD2'
    random_summary_df_0['Routing'] = 'RT'
    actual_traffic['Routing'] = 'Actual'
    
    # filter cols   
    # summmaty sumo output metric  step_running
    cols = ['step_time', 'step_inserted', 'Routing']
    df = sum_list[0].append(sum_list[1:], ignore_index=True)    
   
    print(df)
    df = df.filter(cols)
   
    
   
    # traffic filtes
    f_names_temp = ['MAR', 'DUAR', 'DUAI', 'RT', 'OD2'] # real va luego
    linestyle_list = ['-', '--','-.',':','-','--'] 
    # plot step running vehicles
    fig, ax = plt.subplots(figsize=(4,2))
    
    
    for i, name in enumerate(f_names_temp):
       
        temp_df = df[df['Routing']==name]
        # group by hour mean nu,ber of vehicles
        temp_df = temp_df.groupby(pd.cut(temp_df["step_time"], np.arange(0, 1+24*3600,3600))).max()
        # obtain veh/hour
        temp_df['Hour'] = range(24) 
        temp_df['shift'] = temp_df['step_inserted'].shift().fillna(0)
        temp_df['vehicles'] = temp_df['step_inserted'] - temp_df['shift'] 
        # plot veh/hour
        
        temp_df.plot(kind='line',x='Hour', y='vehicles', linewidth=2, linestyle=linestyle_list[i], label=name, ax=ax)
    
    actual_traffic.plot(kind='line',x='Hour', y='Actual',  linestyle=linestyle_list[5], label='Actual', ax=ax)
    
    plt.legend(prop={'size': 10})
    plt.ylabel('# of vehicles')
    plt.xlabel('Hour')
    #plt.title('Traffic intensity')
    #plt.xticks(np.arange(min(traffic_df['Hour']), max(traffic['Hour'])+1, 2.0))
    plt.grid(True, linewidth=1, linestyle='--')    
    
   

#[OD_plots(f, name) for f,name in zip(f_list,f_names)]
plot_actual_traffic() 
plot_tools_traffic()
fundamental_metric_plots()

#summary_plot()

"""
# points O/D
# Histograma fundamental metrics
#fundamental_metrics_hist()
# Summary SUMO output
summary_plot()

"""

   
