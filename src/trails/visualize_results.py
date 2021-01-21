import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import pygeos
import geopandas as gpd
import contextily as cx
import traceback
import seaborn as sns
from matplotlib.ticker import MaxNLocator

def main():

    # set data paths to results
    data_path_perc = r'C:\Data\percolation_results'
    data_path_met= r'C:\Data\percolation_metrics'
    data_path_net = r'C:\Data\percolation_networks'

    # file to get full country names
    glob_info = pd.read_excel(r'C:\Projects\trails\data\global_information.xlsx')

    # get all files from data paths
    perc_files = os.listdir(data_path_perc)
    met_files = os.listdir(data_path_met)
    net_files = os.listdir(data_path_net)

    # save the failed ones, so we can check them later
    save_failed = []

    # set x-axis
    x = np.arange(1,100,1)

    # create figure
    fig, axs = plt.subplots(3,2,figsize=(15,20))

    for iter1,file in enumerate(perc_files):
        
        # get name of percolation analysis
        net_name = file[:5]
        
        try:
            if os.path.isfile(os.path.join('..','..','figures','{}_results.png'.format(net_name))):
                #print(net_name+" already finished!")
                continue  
                
            # load metrics
            df_metrics = pd.read_csv(os.path.join(data_path_met,[x for x in met_files if file[:5] in x][0]))

            # load percolation results
            df = pd.read_csv(os.path.join(data_path_perc,file),index_col=[0])
            df.frac_counter = df.frac_counter*100
            
            # remove all results where it is pretty much done, so we can zoom onto the interesting part
            df = df.loc[df.pct_isolated < 99.5]   
            max_frac_counter = df.frac_counter.max()       
            
            df_isolated = pd.DataFrame([df.frac_counter.values,df.pct_isolated.values,df.pct_unaffected.values,df.pct_delayed.values]).T
            df_isolated.columns = ['frac_counter','pct_isolated','pct_unaffected','pct_delayed']
            df_sloss = pd.DataFrame([df.frac_counter.values,df.total_pct_surplus_loss_e1.values,df.total_pct_surplus_loss_e2.values]).T
            df_sloss.columns = ['frac_counter','total_pct_surplus_loss_e1','total_pct_surplus_loss_e2']
            
            # load network
            network = pd.read_feather(os.path.join(data_path_net,[x for x in net_files if file[:5] in x][0]))
            network.geometry = pygeos.from_wkb(network.geometry)
            network = gpd.GeoDataFrame(network)
            network.crs = 4326

            # get mean,max,min values
            y_unaff = df_isolated.groupby('frac_counter').max()['pct_unaffected'].values
            y_del = df_isolated.groupby('frac_counter').mean()['pct_delayed'].values
            y_iso = df_isolated.groupby('frac_counter').min()['pct_isolated'].values

            mainnet = 'yes'
            if net_name[4] != '0':
                mainnet = 'no, #{}'.format(net_name)

            if iter1 > 0:
                for iter2,ax in enumerate(axs.flatten()):
                    ax.clear()

            #and plot
            for iter2,ax in enumerate(axs.flatten()):
                
                if iter2 == 0:
                    sns.boxplot(x="frac_counter", y="pct_isolated", data=df_isolated,ax=ax,fliersize=0,order=np.arange(max_frac_counter+2),palette="rocket_r",linewidth=0.5)
                    ax.set_ylabel('Percentage of isolated trips', fontsize=13)
                    ax.set_xlabel('Percentage of network removed', fontsize=13)    
                    ax.set_title('Boxplots of isolated trips', fontsize=15, fontweight='bold')      
                    ax.set_xlim([0, max_frac_counter+2])
                    ax.set_xticks(np.arange(0, max_frac_counter+2, 5)) 
                    ax.set_ylim([0, 102])
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

                elif iter2 == 1:
                    network.plot(column='highway',legend=True,ax=ax)
                    try:
                        cx.add_basemap(ax, crs=network.crs.to_string(),alpha=0.5)
                    except:
                        cx.add_basemap(ax, crs=network.crs.to_string(),alpha=0.5,zoom=10)

                    ax.set_title('Road network', fontsize=15, fontweight='bold')      

                elif iter2 == 2:
                    sns.boxplot(x="frac_counter", y="pct_unaffected", data=df_isolated,ax=ax,fliersize=0,order=np.arange(max_frac_counter+2),palette="rocket_r",linewidth=0.5)
                    ax.set_ylabel('Percentage of unaffected trips', fontsize=13)
                    ax.set_xlabel('Percentage of network removed', fontsize=13)    
                    ax.set_title('Boxplots of unaffected trips', fontsize=15, fontweight='bold')      
                    ax.set_xlim([0, max_frac_counter+2])
                    ax.set_xticks(np.arange(0, max_frac_counter+2, 5)) 
                    ax.set_ylim([0, 102])
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                
                elif iter2 == 3:
                    sns.boxplot(x="frac_counter", y="pct_delayed", data=df_isolated,ax=ax,fliersize=0,order=np.arange(max_frac_counter+2),palette="rocket_r",linewidth=0.5)
                    ax.set_ylabel('Percentage of delayed trips', fontsize=13)
                    ax.set_xlabel('Percentage of network removed', fontsize=13)    
                    ax.set_title('Boxplots of delayed trips', fontsize=15, fontweight='bold')      
                    ax.set_xlim([0, max_frac_counter+2])
                    ax.set_xticks(np.arange(0, max_frac_counter+2, 5)) 
                    ax.set_ylim([0, 102])
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                
                elif iter2 == 4:
                    sns.boxplot(x="frac_counter", y="total_pct_surplus_loss_e1", data=df_sloss,ax=ax,fliersize=0,order=np.arange(max_frac_counter+2),palette="rocket_r",linewidth=0.5)
                    ax.set_ylabel('Percentage of surpluss loss (e1)', fontsize=13)
                    ax.set_xlabel('Percentage of network removed', fontsize=13)                   
                    ax.set_xlim([0, max_frac_counter+2])
                    ax.set_xticks(np.arange(0, max_frac_counter+2, 5)) 
                    ax.set_ylim([0, 102])
                    ax.set_title('Boxplots of surpluss loss e1', fontsize=15, fontweight='bold')      
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                

                elif iter2 == 5:
                    sns.boxplot(x="frac_counter", y="total_pct_surplus_loss_e2", data=df_sloss,ax=ax,fliersize=0,order=np.arange(max_frac_counter+2),palette="rocket_r",linewidth=0.5)
                    ax.set_ylabel('Percentage of surpluss loss (e2)', fontsize=13)
                    ax.set_xlabel('Percentage of network removed', fontsize=13)                   
                    ax.set_xlim([0, max_frac_counter+2])
                    ax.set_xticks(np.arange(0, max_frac_counter+2, 5)) 
                    ax.set_ylim([0, 102])
                    ax.set_title('Boxplots of surpluss loss e2', fontsize=15, fontweight='bold')      
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                                
                    ax.text(max_frac_counter*0.5, 10.5,'Main Network: {} \nEdges: {} \nDensity: {} \nClique_No: {} \nAssortativity: {} \nDiameter: {} \nMax_Degree: {}'.format( 
                                                                                                mainnet,
                                                                                                    df_metrics.Edge_No.values[0],
                                                                                                    np.round(df_metrics.Density.values[0],7),
                                                                                                    df_metrics.Clique_No.values[0],
                                                                                                    np.round(df_metrics.Assortativity.values[0],7),
                                                                                                    df_metrics.Diameter.values[0],
                                                                                                    df_metrics.Max_Degree.values[0],                                                                                               
                                                                                                    ), fontsize=15)   
                  
            if net_name[4] == '0':    
                if net_name[:3] in ['HKG','TWN','MNP','MAC','MHL','GUM']:
                    name_dict_errors = {'HKG': "Hong Kong",
                    'TWN': "Taiwan",
                    'MNP': "Northern Mariana Islands",
                    'MAC': "Macau",
                    'MHL': "Marshall Islands",
                    'GUM' : "Guam"
                    }
                    plt.suptitle('Main network of {}'.format(name_dict_errors[net_name[:3]]), fontsize=20, fontweight='bold',y=0.92)
                else:
                    plt.suptitle('Main network of {}'.format(dict(zip(glob_info.ISO_3digit,glob_info.Country))[net_name[:3]]), fontsize=20, fontweight='bold',y=0.92)
            else:
                plt.suptitle('Subnetwork of {}'.format(dict(zip(glob_info.ISO_3digit,glob_info.Country))[net_name[:3]]), fontsize=20, fontweight='bold',y=0.92)
            
            plt.savefig(os.path.join('..','..','figures','{}_results.png'.format(net_name)))
                    
        except Exception as e: 
            print(net_name+" failed because of {}".format(e))
            print(traceback.format_exc())
            save_failed.append(net_name)
        
    print(save_failed)
if __name__ == '__main__':       
    main()
    
