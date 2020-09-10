import os
import pygeos
import geopandas as gpd
import pandas as pd
import numpy
from tqdm import tqdm
from pgpkg import Geopackage
import warnings
from shapely.wkb import loads,dumps
warnings.filterwarnings("ignore")
from multiprocessing import Pool,cpu_count
from rasterstats import zonal_stats

def create_bbox(df):
    """Create bbox around dataframe

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """    
    return pygeos.creation.box(pygeos.total_bounds(df.geometry)[0],
                                  pygeos.total_bounds(df.geometry)[1],
                                  pygeos.total_bounds(df.geometry)[2],
                                  pygeos.total_bounds(df.geometry)[3])

def create_grid(bbox,height):
    """Create a vector-based grid

    Args:
        bbox ([type]): [description]
        height ([type]): [description]

    Returns:
        [type]: [description]
    """    

    # set xmin,ymin,xmax,and ymax of the grid
    xmin, ymin = pygeos.total_bounds(bbox)[0],pygeos.total_bounds(bbox)[1]
    xmax, ymax = pygeos.total_bounds(bbox)[2],pygeos.total_bounds(bbox)[3]
    
    #estimate total rows and columns
    rows = int(numpy.ceil((ymax-ymin) / height))
    cols = int(numpy.ceil((xmax-xmin) / height))

    # set corner points
    x_left_origin = xmin
    x_right_origin = xmin + height
    y_top_origin = ymax
    y_bottom_origin = ymax - height

    # create actual grid
    res_geoms = []
    for countcols in range(cols):
        y_top = y_top_origin
        y_bottom = y_bottom_origin
        for countrows in range(rows):
            res_geoms.append(pygeos.polygons(
                ((x_left_origin, y_top), (x_right_origin, y_top),
                (x_right_origin, y_bottom), (x_left_origin, y_bottom)
                )))
            y_top = y_top - height
            y_bottom = y_bottom - height
        x_left_origin = x_left_origin + height
        x_right_origin = x_right_origin + height

    return res_geoms

def create_country_OD_points(country):
    """Create a list of OD points for the specified country 

    Args:
        country ([type]): [description]

    Returns:
        [type]: [description]
    """    
    
    # set paths to data
    world_pop = 'C:\Data\worldpop\ppp_2018_1km_Aggregated.tif'
            
    #get country ID
    GID_0 = country['GID_0']
    print('{} started!'.format(GID_0))
    
    #create dataframe of country row
    df = pd.DataFrame(country).T
    df.geometry = pygeos.from_wkb(df.geometry.values[0])
    #specify height of cell in the grid and create grid of bbox
    height = numpy.sqrt(pygeos.area(df.geometry)/100).values[0]
    grid = pd.DataFrame(create_grid(create_bbox(df),height),columns=['geometry'])

    #clip grid of bbox to grid of the actual spatial exterior of the country
    clip_grid = pygeos.intersection(grid,df.geometry)
    clip_grid = clip_grid.loc[~pygeos.is_empty(clip_grid.geometry)]

    # turn to shapely geometries again for zonal stats
    clip_grid.geometry = pygeos.to_wkb(clip_grid.geometry)
    clip_grid.geometry = clip_grid.geometry.apply(loads)
    clip_grid = gpd.GeoDataFrame(clip_grid)

    # get total population per grid cell
    clip_grid['tot_pop'] = clip_grid.geometry.apply(lambda x: zonal_stats(x,world_pop,stats="sum"))
    clip_grid['tot_pop'] = clip_grid['tot_pop'].apply(lambda x: x[0]['sum'])    

    # remove cells in the grid that have no population data
    clip_grid = clip_grid.loc[~pd.isna(clip_grid.tot_pop)]
    clip_grid = clip_grid.loc[clip_grid.tot_pop > 100]
    clip_grid.reset_index(inplace=True,drop=True)
    clip_grid.geometry = clip_grid.geometry.centroid
    clip_grid['GID_0'] = GID_0
    
    print('{} finished!'.format(GID_0))
    
    clip_grid.to_csv(os.path.join('..','country_OD_points','{}.csv'.format(GID_0)))
    # return the country 
    return clip_grid 

def create_OD_points():
    """[summary]
    """      
    #load data and convert to pygeos
    gdf = gpd.read_file('C:\Data\GADM\gadm36_levels.gpkg',layer=0)
    tqdm.pandas(desc='Convert geometries to pygeos')
    gdf = pd.DataFrame(gdf)
    gdf['geometry'] = gdf.geometry.progress_apply(dumps)
    
    # create OD points
    save_points_per_country = []
    list_countries = list([x[1] for x in gdf.iterrows()])
    
    #multiprocess all countries
    with Pool(cpu_count()-1) as pool: 
        save_points_per_country = pool.map(create_country_OD_points,list_countries,chunksize=1) 
           
    df_all = pd.concat(save_points_per_country)
    df_all['geometry'] = df_all.geometry.progress_apply(lambda x: pygeos.from_shapely(x))
    df_all.reset_index(inplace=True,drop=True)
    df_all.to_csv('od_points_per_country.csv')
    
if __name__ == "__main__":
    # execute only if run as a script
    create_OD_points()    