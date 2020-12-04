import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import pygeos
import geopandas as gpd
import contextily as cx
import seaborn as sns

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
    fig, axs = plt.subplots(2,2,figsize=(15,15))

    for iter1,file in enumerate(perc_files):

        # get name of percolation analysis
        net_name = file[:5]

        if os.path.isfile(os.path.join('..','..','figures','{}_results.png'.format(net_name))):
            print(net_name+" already finished!")
            continue  

        try:
            # load metrics
            df_metrics = pd.read_csv(os.path.join(data_path_met,[x for x in met_files if file[:5] in x][0]))
            
            # load percolation results
            df = pd.read_csv(os.path.join(data_path_perc,file),index_col=[0])
            df_isolated = pd.DataFrame([df.frac_counter.values,df.pct_isolated.values]).T
            df_isolated.columns = ['frac_counter','pct_isolated']
            df_isolated = df_isolated*100
            df_sloss = pd.DataFrame([df.frac_counter.values,df.total_pct_surplus_loss_e1.values]).T
            df_sloss.columns = ['frac_counter','total_pct_surplus_loss_e1']
            df_sloss.total_pct_surplus_loss_e1 = df_sloss.total_pct_surplus_loss_e1
            df_sloss = df_sloss*100

            # load network
            network = pd.read_feather(os.path.join(data_path_net,[x for x in net_files if file[:5] in x][0]))
            network.geometry = pygeos.from_wkb(network.geometry)
            network = gpd.GeoDataFrame(network)
            network.crs = 4326

            # get mean,max,min values
            max_iso = df_isolated.groupby('frac_counter').max()['pct_isolated'].values
            y_iso = df_isolated.groupby('frac_counter').mean()['pct_isolated'].values
            min_iso = df_isolated.groupby('frac_counter').min()['pct_isolated'].values

            mainnet = 'yes'
            if net_name[4] != '0':
                mainnet = 'no, #{}'.format(net_name)
            
            if iter1 > 0:
                for iter2,ax in enumerate(axs.flatten()):
                    ax.clear()
            
            #and plot
            for iter2,ax in enumerate(axs.flatten()):
                if iter2 == 0:
                    ax.plot(df_isolated.frac_counter.unique(), y_iso, 'r-')
                    ax.fill_between(df_isolated.frac_counter.unique(), min_iso, max_iso)

                    ax.text(50, 0.2,'Main Network: {} \nEdges: {} \nDensity: {} \nClique_No: {} \nAssortativity: {} \nDiameter: {} \nMax_Degree: {}'.format( 
                                                                                                mainnet,
                                                                                                    df_metrics.Edge_No.values[0],
                                                                                                    np.round(df_metrics.Density.values[0],7),
                                                                                                    df_metrics.Clique_No.values[0],
                                                                                                    np.round(df_metrics.Assortativity.values[0],7),
                                                                                                    df_metrics.Diameter.values[0],
                                                                                                    df_metrics.Max_Degree.values[0],                                                                                               
                                                                                                    ), fontsize=15)
                    ax.set_ylabel('Percentage of isolated trips', fontsize=13, fontweight='bold')
                    ax.set_xlabel('Percentage of network removed', fontsize=13, fontweight='bold')     
                    ax.set_xticks(np.arange(0, 100+2, 10))
                    ax.set_title('Max,mean and min isolated trips', fontsize=15, fontweight='bold')      

                elif iter2 == 1:

                    network.plot(column='highway',legend=True,ax=ax)
                    cx.add_basemap(ax, crs=network.crs.to_string(),alpha=0.5)
                    ax.set_title('Road network', fontsize=15, fontweight='bold')      

                elif iter2 == 2:

                    sns.boxplot(x="frac_counter", y="pct_isolated", data=df_isolated,ax=ax,fliersize=0)
                    ax.set_ylabel('Percentage of isolated trips', fontsize=13, fontweight='bold')
                    ax.set_xlabel('Percentage of network removed', fontsize=13, fontweight='bold')    
                    ax.set_xticks(np.arange(-1, 100+2, 10)) 
                    ax.set_title('Boxplots of isolated trips', fontsize=15, fontweight='bold')      
                
                elif iter2 == 3:
                    #ax.plot(x, y_sloss, 'r-')
                    #ax.fill_between(x, min_sloss, max_sloss)     
                    sns.boxplot(x="frac_counter", y="total_pct_surplus_loss_e1", data=df_sloss,ax=ax,fliersize=0)
                    ax.set_ylabel('Percentage of surpluss loss', fontsize=13, fontweight='bold')
                    ax.set_xlabel('Percentage of network removed', fontsize=13, fontweight='bold')                   
                    ax.set_xticks(np.arange(-1, 100+2, 10)) 
                    ax.set_title('Boxplots of surpluss loss e1', fontsize=15, fontweight='bold')      

            plt.suptitle(dict(zip(glob_info.ISO_3digit,glob_info.Country))[net_name[:3]], fontsize=20, fontweight='bold')
            plt.savefig(os.path.join('..','..','figures','{}_results.png'.format(net_name)))
        #plt.pause(5)
                
        except Exception as e: 
            print(net_name+" failed because of {}".format(e))
            save_failed.append(net_name)
        
    print(save_failed)
if __name__ == '__main__':       
    main()
    
