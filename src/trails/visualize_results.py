import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import pygeos
import geopandas as gpd
import contextily as cx

def main():
    
    data_path_perc = r'C:\Data\percolation_results'
    data_path_met= r'C:\Data\percolation_metrics'
    data_path_net = r'C:\Data\percolation_networks'

    glob_info = pd.read_excel(r'C:\Projects\trails\data\global_information.xlsx')
    
    perc_files = os.listdir(data_path_perc)
    met_files = os.listdir(data_path_met)
    net_files = os.listdir(data_path_net)

    save_failed = []
    
    x = np.arange(1,100,1)
    
    fig, axs = plt.subplots(1,2,figsize=(20,10))
    for iter1,file in enumerate(perc_files[:50]):

        net_name = file[:5]
        try:
            df_metrics = pd.read_csv(os.path.join(data_path_met,[x for x in met_files if file[:5] in x][0]))
            df = pd.read_csv(os.path.join(data_path_perc,file),index_col=[0])
            df_isolated = pd.DataFrame([df.frac_counter.values,df.pct_isolated.values]).T
            df_isolated.columns = ['frac_counter','pct_isolated']

            network = pd.read_feather(os.path.join(data_path_net,[x for x in net_files if file[:5] in x][0]))
            network.geometry = pygeos.from_wkb(network.geometry)
            network = gpd.GeoDataFrame(network)
            network.crs = 4326
            error = df_isolated.groupby('frac_counter').std()['pct_isolated'].values
            y = df_isolated.groupby('frac_counter').mean()['pct_isolated'].values
                
            mainnet = 'yes'
            if net_name[4] != '0':
                mainnet = 'no, #{}'.format(net_name)
            
            if iter1 > 0:
                for iter2,ax in enumerate(axs.flatten()):
                    ax.clear()
            #and plot
            for iter2,ax in enumerate(axs.flatten()):

        
                if iter2 == 0:
                    ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma\
                    ax.fill_between(x, y-error, y+error)

                    ax.text(50, 0.2,'Main Network: {} \nEdges: {} \nDensity: {} \nClique_No: {} \nAssortativity: {} \nDiameter: {} \nMax_Degree: {}'.format( 
                                                                                                mainnet,
                                                                                                    df_metrics.Edge_No.values[0],
                                                                                                    np.round(df_metrics.Density.values[0],7),
                                                                                                    df_metrics.Clique_No.values[0],
                                                                                                    np.round(df_metrics.Assortativity.values[0],7),
                                                                                                    df_metrics.Diameter.values[0],
                                                                                                    df_metrics.Max_Degree.values[0],                                                                                               
                                                                                                    ), fontsize=15)
                else:
                    network.plot(column='highway',legend=True,ax=ax)
                    cx.add_basemap(ax, crs=network.crs.to_string(),alpha=0.5)
            
            plt.suptitle(dict(zip(glob_info.ISO_3digit,glob_info.Country))[net_name[:3]], fontsize=20, fontweight='bold')
            plt.draw()
            plt.pause(5)
            
        except:
            save_failed.append(net_name)
        
if __name__ == '__main__':       
    main()
