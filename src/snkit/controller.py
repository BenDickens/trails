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
gdal.SetConfigOption("OSM_CONFIG_FILE", os.path.join(".","osmconf.ini"))

countries = ['azerbaijan','bulgaria','jamaica']
tiny = ['monaco','tonga','djibouti']

def filename(country):
    osm_prefix = './osm_files/'
    osm_suffix = "-latest.osm.pbf"
    return osm_prefix + country + osm_suffix

if __name__ == '__main__':       
    #geometry = 'lines'
    #keyCol = ['highway']#['highway','oneway','lanes','maxspeed']
    #valConstraint = {'highway':["='primary' or ","='trunk' or ","='motorway' or ","='trunk_link' or ", "='primary_link' or ", "='secondary' or ","='tertiary' or ","='tertiary_link'"]}
    #cGDF = retrieve(filename('madagascar'), geometry, keyCol, **valConstraint)
    #allCountries = tiny.append(small.append(mid.append(dec)))
    
    for x in tiny:
        print(x.capitalize())
        cGDF = mainRoads(filename(x))
        bob = simp.simplify_network_from_gdf(cGDF)
        a,b = net.largest_component_df(bob.edges,bob.nodes)
        x_ax, isolated_trip_results = net.alt(a)


        plt.plot(x_ax,isolated_trip_results, label=x)

    plt.ylabel("Isolated Trips")
    plt.title("Simple percolation")
    plt.xlabel("Percentage of edges destroyed")
    plt.legend()
    plt.savefig('./results/large_test.png')
        
  