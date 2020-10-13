import os,sys
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from extract import retrieve, mainRoads,roads,ferries,railway#, mainRoads_pyg
import gdal
import pygeos as pyg
from timeit import default_timer as timer
import shapely as shp
from pgpkg import Geopackage
import prepare as prep
import simplify as simp
import network as net
import matplotlib.pyplot as plt
import igraph as ig
import feather
import math
import random
import pygeos as pyg 
from numpy.ma import masked
gdal.SetConfigOption("OSM_CONFIG_FILE", os.path.join(".","osmconf.ini"))


wb = ['JAM','AFG']

def filename(country):
    osm_prefix = ''
    osm_suffix = ".osm.pbf"
    return osm_prefix + country + osm_suffix


def create_results_from_osm(countries = [], OD_no = 100, runs=100, mean = True, seed = []):
    for x in countries:
        print(x.capitalize())
        cGDF = mainRoads(filename(x))
        cGDF = simp.simplified_network(cGDF)
        edges, nodes = net.largest_component_df(cGDF.edges,cGDF.nodes)
        results = net.run_percolation(x,edges,nodes,OD_no, runs, seed)
        if mean:
            results = results.groupby(['frac_counter']).mean()
            
            results.to_csv(x + "_results_mean.csv")

        else :
            results.to_csv(x + "_results_all.csv")

def plot_column(column, countries = []):
    for x in countries:
        res = pd.read_csv(x+"_results_mean.csv")
        print(res.columns)
        plt.plot(res.frac_counter,res[column], label = x)
    plt.ylabel(column)
    plt.title("Simple percolation - " + column)
    plt.xlabel("Fraction of edges destroyed")
    plt.legend(title="Countries")
    plt.show()


def plot_from_results_mean(countries = []):
    columns = pd.read_csv(countries[0]+"_results_mean.csv").columns

    for column in columns:
        plot_column(column,countries)
    

if __name__ == '__main__':       
    #create_results_from_osm(wb)
    plot_from_results_mean(wb)
   