import os,sys
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py
import math
import random
import matplotlib.pyplot as plt
import feather
import pandas as pd
import pygeos as pyg

from timeit import default_timer as timer
from tqdm import tqdm
from pathlib import Path
from plotly.graph_objs import *

from numpy import inf

#from pgpkg import Geopackage
from numpy.ma import masked

data_path = Path(__file__).resolve().parents[2].joinpath('data','percolation')

from pathos.multiprocessing import Pool,cpu_count
from itertools import repeat

#import warnings
#warnings.filterwarnings("ignore")

def metrics(graph):
    """This method prints some basic network metrics of an iGraph

    Args:
        graph (iGraph.Graph object): 
    Returns:
        m: 
    """    
    g = graph
    return pd.DataFrame([[g.ecount(),g.vcount(),g.density(),g.omega(),g.average_path_length(directed=False),g.assortativity_degree(False),g.diameter(directed=False),g.edge_connectivity(),g.maxdegree(),np.sum(g.es['distance'])]],columns=["Edge_No","Node_No","Density","Clique_No", "Ave_Path_Length", "Assortativity","Diameter","Edge_Connectivity","Max_Degree","Total_Edge_Length"])


def metrics_Print(graph):
    """This method prints some basic network metrics of an iGraph

    Args:
        graph (iGraph.Graph object): 
    Returns:
        m: 
    """    
    g = graph
    m = []
    print("Number of edges: ", g.ecount())
    print("Number of nodes: ", g.vcount())
    print("Density: ", g.density())
    print("Number of cliques: ", g.omega())#omega or g.clique_number()
    print("Average path length: ", g.average_path_length(directed=False))
    print("Assortativity: ", g.assortativity_degree(False))
    print("Diameter: ",g.diameter(directed=False))
    print("Edge Connectivity: ", g.edge_connectivity())
    print("Maximum degree: ", g.maxdegree())
    print("Total Edge length ", np.sum(g.es['distance']))

#Creates a graph 
def graph_load(edges):
    """Creates 

    Args:
        edges (pandas.DataFrame) : containing road network edges, with from and to ids, and distance / time columns

    Returns:
        igraph.Graph (object) : a graph with distance and time attributes
    """    
    #return ig.Graph.TupleList(gdfNet.edges[['from_id','to_id','distance']].itertuples(index=False),edge_attrs=['distance'])
    graph = ig.Graph(directed=False)
    max_node_id = max(max(edges.from_id),max(edges.to_id))
    graph.add_vertices(max_node_id+1)
    edge_tuples = zip(edges.from_id,edges.to_id)
    graph.add_edges(edge_tuples)
    graph.es['distance'] = edges.distance
    graph.es['time'] = edges.time
    return graph
    
def graph_load_largest(edges):
    """Returns the largest component of a graph given an edge dataframe

    Args:
        edges (pandas.DataFrame): A dataframe containing from, to ids; time and distance attributes for each edge

    Returns:
        igraph.Graph (object) : a graph with distance and time attributes
    """    
    graph = graph_load(gdfNet)
    return graph.clusters().giant()


def largest_component_df(edges,nodes):
    """Returns the largest component of a network object (network.edges pd  
    and network.nodes pd) with reset ids. Uses igraphs built in function, while adding ids as attributes

    Args:
        edges (pandas.DataFrame): A dataframe containing from and to ids
        nodes (pandas.DataFrame): A dataframe containing node ids

    Returns:
        edges, nodes (pandas.DataFrame) : 2 dataframes containing only those edges and nodes belonging to the giant component
    """    
    edges = edges
    nodes = nodes
    edge_tuples = zip(edges['from_id'],edges['to_id'])
    graph = ig.Graph(directed=False)
    graph.add_vertices(len(nodes))
    graph.vs['id'] = nodes['id']
    graph.add_edges(edge_tuples)
    graph.es['id'] = edges['id']
    graph = graph.clusters().giant()
    edges_giant = edges.loc[edges.id.isin(graph.es()['id'])]
    nodes_giant = nodes.loc[nodes.id.isin(graph.vs()['id'])]
    return reset_ids(edges_giant,nodes_giant)


def create_demand(OD_nodes, OD_orig, node_pop):
    """This function creates a demand matrix from the equation:
    
    Demand_a,b = Population_a * Population_b * e^ [-p * Distance_a,b] 
    
    -p is set to 1, populations represent the grid square of the origin, 

    Args:
        OD_nodes (list): a list of nodes to use for the OD, a,b
        OD_orig (np.matrix): A shortest path matrix used for the distance calculation
        node_pop (list): population per OD node

    Returns:
        demand (np.ndarray) : A matrix with demand calculations for each OD pair
    """    
    demand = np.zeros((len(OD_nodes), len(OD_nodes)))

    dist_decay = 1
    maxtrips = 100

    for o in range(0, len(OD_nodes)):
        for d in range(0, len(OD_nodes)):
            if o == d:
                demand[o][d] = 0
            else:
                normalized_dist = OD_orig[o,d] / OD_orig.max()
                demand[o][d] = ((node_pop[o] * node_pop[d]) * np.exp(-1 * dist_decay * normalized_dist))

    demand = ((demand / demand.max()) * maxtrips)
    demand = np.ceil(demand).astype(int)
    return demand

def choose_OD(pos_OD, OD_no):
    """Chooses nodes for OD matrix according to their population size stochastically and probabilistically 

    Args:
        pos_OD (list): a list of tuples representing the nodes and their population
        OD_no (int): Number of OD pairs to create

    Returns:
        OD_nodes [list]: The nodes chosen for the OD
        mapped_pops [list]: Population for nodes chosen
    """    

    #creates 2 tuples of the node ids and their total representative population
    node_ids, tot_pops = zip(*pos_OD)
    #Assigns a probability by population size
    pop_probs = [x/sum(tot_pops) for x in tot_pops]
    #OD nodes chosen
    OD_nodes = list(np.random.choice(node_ids, size=OD_no, replace = False, p=pop_probs))
    #Population counts in a mapped list
    node_positions = [node_ids.index(i) for i in OD_nodes]
    mapped_pops = [tot_pops[j] for j in node_positions]
    #returns the nodes, and their populations, should this be zipped?
    return OD_nodes, mapped_pops


def prepare_possible_OD(gridDF, nodes, tolerance = 1):
    """Returns an array of tuples, with the first value the node ID to consider, and the
       second value the total population associated with this node. 
       The tolerance is the size of the bounding box to search for nodes within

    Args:
        gridDF (pandas.DataFrame): A dataframe with the grid centroids and their population
        nodes (pandas.DataFrame): A dataframe of the road network nodes
        tolerance (float, optional): size of the bounding box . Defaults to 0.1.

    Returns:
        final_possible_pop (list): a list of tuples representing the nodes and their population
    """    
    nodeIDs = []
    sindex = pyg.STRtree(nodes['geometry'])

    pos_OD_nodes = []
    pos_tot_pop = []
    for i in gridDF.itertuples():
        ID = nearest(i.geometry, nodes, sindex, tolerance)
        #If a node was found
        if ID > -1: 
            pos_OD_nodes.append(ID)
            pos_tot_pop.append(i.tot_pop)
    a = nodes.loc[nodes.id.isin(pos_OD_nodes)]
    #Create a geopackage of the possible ODs
    #with Geopackage('nodyBGR.gpkg', 'w') as out:
    #    out.add_layer(a, name='finanod', crs='EPSG:4326')
    nodes = np.array([pos_OD_nodes])
    node_unique = np.unique(nodes)
    count = np.array([pos_tot_pop])
    
    #List comprehension to add total populations of recurring nodes 
    final_possible_pop = [(i, count[nodes==i].sum()) for i in node_unique]
    return final_possible_pop

#WOULD IT BE BETTER TO USE HALF THE MINIMUM DISTANCE BETWEEN GRID POINTS AS THE TOLERANCE INSTEAD

def nearest(geom, gdf,sindex, tolerance):
    """Finds the nearest node

    Args:
        geom (pygeos.Geometry) : Geometry to find nearest
        gdf (pandas.index): Node dataframe to provide possible nodes
        sindex (pygeos.Sindex): Spatial index for faster lookup
        tolerance (float): Size of buffer to use to find nodes

    Returns:
        nearest_geom.id [int]: The node id that is closest to the geom
    """    
    matches_idx = sindex.query(geom)
    if not matches_idx.any():
        buf = pyg.buffer(geom, tolerance)
        matches_idx = sindex.query(buf,'contains').tolist()
    try:
        nearest_geom = min(
            [gdf.iloc[match_idx] for match_idx in matches_idx],
            key=lambda match: pyg.measurement.distance(match.geometry,geom)
        )
    except: 
        #print("Couldn't find node")
        return -1
    return nearest_geom.id


def simple_OD_calc(OD, comparisonOD,pos_trip_no):
    """An alternative OD calculation that counts how many trips exceed threshold length

    Args:
        OD ([type]): [description]
        comparisonOD ([type]): [description]
        pos_trip_no ([type]): [description]

    Returns:
        [type]: [description]
    """    
    compare_thresh = np.greater(OD,comparisonOD)
    over_thresh_no = np.sum(compare_thresh) / 2
    return over_thresh_no / pos_trip_no


def percolation_Final(edges, del_frac=0.01, OD_list=[], pop_list=[], GDP_per_capita=50000):
    """Final version of percolation, runs a simulation on the network provided, to give an indication of network resilience.

    Args:
        edges (pandas.DataFrame): A dataframe containing edge information: the nodes to and from, the time and distance of the edge
        del_frac (float): The fraction to increment the percolation. Defaults to 0.01. e.g.0.01 removes 1 percent of edges at each step
        OD_list (list, optional): OD nodes to use for matrix and calculations.  Defaults to []. 
        pop_list (list, optional): Corresponding population sizes for ODs for demand calculations. Defaults to [].
        GDP_per_capita (int, optional): The GDP of the country/area for surplus cost calculations. Defaults to 50000.

    Returns:
        result_df [pandas.DataFrame]: The results! 'frac_counter', 'pct_isolated', 'average_time_disruption', 'pct_thirty_plus', 'pct_twice_plus', 'pct_thrice_plus','total_surp_loss_e1', 'total_pct_surplus_loss_e1', 'total_surp_loss_e2', 'total_pct_surplus_loss_e2'    """    
    
    result_df = []
    g = graph_load(edges)
    #These if statements allow for an OD and population list to be randomly generated
    if OD_list == []: 
        OD_nodes = random.sample(range(g.vcount()-1),100)
    else: 
        OD_nodes = OD_list
    edge_no = g.ecount() 
    OD_node_no = len(OD_nodes)

    if pop_list == []: 
        node_pop = random.sample(range(4000), OD_node_no)
    else:
         node_pop = pop_list

    #Creates a matrix of shortest path times between OD nodes
    base_shortest_paths = g.shortest_paths_dijkstra(source=OD_nodes,target = OD_nodes,weights='time')
    OD_orig = np.matrix(base_shortest_paths)
    OD_thresh = OD_orig * 10
    
    demand = create_demand(OD_nodes, OD_orig, node_pop)
    exp_g = g.copy()
    trips_possible = True
    pos_trip_no = (((OD_node_no**2) - OD_node_no) / 2) - ((np.count_nonzero(np.isinf(OD_orig)))/2)
    counter = 0
    frac_counter = 0 
    tot_edge_length = np.sum(g.es['distance'])
    tot_edge_time = np.sum(g.es['time'])

    # add frac 0.00 for better figures and results
    result_df.append((0.00, 0, 100, 0, 0.0, 0, 0.0, 0, 0.0, 0.0, 0.0, 
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                    [0.0,0.0, 0.0, 0.0, 0.0, 0.0]))

    while trips_possible:
        if frac_counter > 0.3 and frac_counter <= 0.5: del_frac = 0.02
        if frac_counter > 0.5: del_frac = 0.05
        exp_edge_no = exp_g.ecount()
        #The number of edges to delete
        no_edge_del = math.floor(del_frac * edge_no)
        try:
            edges_del = random.sample(range(exp_edge_no),no_edge_del)
        except:
            print("random sample playing up but its ok")
            edges_del = range(exp_edge_no)
        exp_g.delete_edges(edges_del)

        frac_counter += del_frac
        
        cur_dis_length = 1 - (np.sum(exp_g.es['distance'])/tot_edge_length)
        cur_dis_time = 1 - (np.sum(exp_g.es['time'])/tot_edge_time)
        new_shortest_paths = exp_g.shortest_paths_dijkstra(source=OD_nodes,target = OD_nodes,weights='time')
        perc_matrix = np.matrix(new_shortest_paths)
        perc_matrix[perc_matrix == inf] = 99999999999
        perc_matrix[perc_matrix == 0] = np.nan

        #if ((perc_matrix[perc_matrix != 99999999999]).shape[1]==len(OD_nodes)):
        #    break

        results = SummariseOD(perc_matrix, 99999999999, demand, OD_orig, GDP_per_capita,round(frac_counter,3),cur_dis_length,cur_dis_time)  
        result_df.append(results)

        #If the frac_counter goes past 0.99
        if results[0] >= 0.99: break
        #If there are no edges left to remove
        if exp_edge_no < 1: break

    #'pct_thirty_plus', 'pct_twice_plus', 'pct_thrice_plus','pct_thirty_plus_over2', 'pct_thirty_plus_over6', 'pct_twice_plus_over1'
    result_df = pd.DataFrame(result_df, columns=['frac_counter', 'pct_isolated','pct_unaffected', 'pct_delayed',
                                                'average_time_disruption','total_surp_loss_e1', 
                                                'total_pct_surplus_loss_e1', 'total_surp_loss_e2', 'total_pct_surplus_loss_e2',
                                                'distance_disruption','time_disruption','unaffected_percentiles','delayed_percentiles'])
    result_df = result_df.replace('--',0)
    return result_df

    
def SummariseOD(OD, fail_value, demand, baseline, GDP_per_capita, frac_counter,distance_disruption, time_disruption):
    """Function returns the % of total trips between origins and destinations that exceed fail value
       Almost verbatim from world bank /GOSTnets world_files_criticality_v2.py

    Args:
        OD (np.matrix): Current OD matrix times (during percolation)
        fail_value (int): Came form GOSTNETS , seems just to be a huge int
        demand (np.ndarray): Demand matrix
        baseline (np.matrix): OD matrix before percolation
        GDP_per_capita (int): GDP of relevant area
        frac_counter (float): Keeps track of current fraction for ease of results storage

    Returns:
        frac_counter, pct_isolated, average_time_disruption, pct_thirty_plus, pct_twice_plus, pct_thrice_plus,total_surp_loss_e1, total_pct_surplus_loss_e1, total_surp_loss_e2, total_pct_surplus_loss_e2
    """
    #masked travel times (OD is the percolation matrix)
    masked_OD = np.ma.masked_greater(OD, value = (fail_value - 1))

    #masked baseline shortest paths (but doesnt change now, as there is no fail_value in there)
    masked_baseline = np.ma.masked_greater(baseline, value = (fail_value - 1))

    #masked adjusted time
    #adj_time = np.ma.masked_array(OD,masked_baseline.mask)
    adj_time = OD-baseline
    #masked demand matrix
    masked_demand = np.ma.masked_array(demand, masked_OD.mask)

    total_trips = (baseline.shape[0]*baseline.shape[1])-baseline.shape[0]

    #isolated_trips = np.ma.masked_array(masked_demand,~masked_OD.mask)
    isolated_trips_sum = OD[OD == fail_value].shape[1]

    # get percentage of isolated trips
    pct_isolated = (isolated_trips_sum / total_trips)*100

    potentially_disrupted_trips = np.ma.masked_array(masked_demand,masked_OD.mask)
    potentially_disrupted_trips_sum = potentially_disrupted_trips.sum()
    
    ## get travel times for remaining trips
    #unaffected_trips = np.ma.masked_equal(masked_OD,masked_baseline).compressed()
    #print(OD[OD == baseline])
    time_unaffected_trips = OD[OD == baseline]

    if not (np.isnan(np.array(time_unaffected_trips)).all()):
        unaffected_percentiles = []
        unaffected_percentiles.append(np.nanpercentile(np.array(time_unaffected_trips),10))
        unaffected_percentiles.append(np.nanpercentile(np.array(time_unaffected_trips),25))
        unaffected_percentiles.append(np.nanpercentile(np.array(time_unaffected_trips),50))
        unaffected_percentiles.append(np.nanpercentile(np.array(time_unaffected_trips),75))
        unaffected_percentiles.append(np.nanpercentile(np.array(time_unaffected_trips),90))
        unaffected_percentiles.append(np.nanmean((time_unaffected_trips)))
    else:
        unaffected_percentiles = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
    #print(unaffected_percentiles)

    ## Set up for ratio-based calculations

    fail_ratio = 50.0 #((fail_value-1) / baseline.max()) # headlimit above which trip destroyed

    #potentially_disrupted_trips_original_time = np.ma.masked_array(masked_baseline, masked_OD.mask)
    #delta_time_OD = (masked_OD - potentially_disrupted_trips_original_time)
    #average_time_disruption = (delta_time_OD * potentially_disrupted_trips).sum() / potentially_disrupted_trips.sum()
    #delayed_trips = delta_time_OD[delta_time_OD !=0].compressed()

    delayed_trips_time = adj_time[(OD != baseline) & (np.nan_to_num(np.array(OD),nan=fail_value) != fail_value)]
    #print(np.nan_to_num(np.array(OD),nan=fail_value) 
    #print(np.array(unaffected_trips).shape[1],np.array(delayed_trips).shape[1])

    unaffected_trips = np.array(time_unaffected_trips).shape[1]
    delayed_trips = np.array(delayed_trips_time).shape[1]

    pct_unaffected = (unaffected_trips/total_trips)*100
    pct_delayed = (delayed_trips/total_trips)*100

    if not (np.isnan(np.array(delayed_trips_time)).all()):

        delayed_percentiles = []
        delayed_percentiles.append(np.nanpercentile(np.array(delayed_trips_time),10))
        delayed_percentiles.append(np.nanpercentile(np.array(delayed_trips_time),25))
        delayed_percentiles.append(np.nanpercentile(np.array(delayed_trips_time),50))
        delayed_percentiles.append(np.nanpercentile(np.array(delayed_trips_time),75))
        delayed_percentiles.append(np.nanpercentile(np.array(delayed_trips_time),90))
        delayed_percentiles.append(np.nanmean(np.array(delayed_trips_time)))
        average_time_disruption = np.nanmean(np.array(delayed_trips_time))
        #print(delayed_percentiles)
    else:
        delayed_percentiles = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
        average_time_disruption = np.nan
 
    # Flexing demand with trip cost
    def surplus_loss(e, C2, C1, D1):
        """[summary]

        Args:
            e ([type]): [description]
            C2 ([type]): [description]
            C1 ([type]): [description]
            D1 ([type]): [description]

        Returns:
            [type]: [description]
        """
        Y_intercept_max_cost = C1 - (e * D1)
        #print(np.amax(Y_intercept_max_cost))

        C2 = np.minimum(C2, Y_intercept_max_cost)

        delta_cost = C2 - C1

        #print(np.amin(delta_cost))

        delta_demand = (delta_cost / e)

        D2 = (D1 + delta_demand)

        #print(np.amin(delta_demand))

        surplus_loss_ans = ((delta_cost * D2) + ((delta_cost * -delta_demand) / 2))

        triangle = (D1 * (Y_intercept_max_cost - C1) ) / 2

        total_surp_loss = surplus_loss_ans.sum()

        total_pct_surplus_loss = total_surp_loss / triangle.sum()

        return total_surp_loss, total_pct_surplus_loss*100

    adj_cost = (OD * GDP_per_capita) / (365 * 8 ) #* 3600) time is in hours, so not sure why we do this multiplications with 3600? and minutes would be times 60?
    baseline_cost = (baseline * GDP_per_capita) / (365 * 8 ) #* 3600) time is in hours, so not sure why we do this multiplications with 3600? and minutes would be times 60?

    adj_cost = np.nan_to_num(np.array(adj_cost),nan=np.nanmax(adj_cost))

    total_surp_loss_e1, total_pct_surplus_loss_e1 = surplus_loss(-0.15, adj_cost, baseline_cost, demand)
    total_surp_loss_e2, total_pct_surplus_loss_e2 = surplus_loss(-0.36, adj_cost, baseline_cost, demand)

    #print(pct_unaffected,pct_delayed,pct_isolated,total_pct_surplus_loss_e1,total_pct_surplus_loss_e2)
    #print(pct_unaffected+pct_delayed+pct_isolated)

    return frac_counter, pct_isolated, pct_unaffected, pct_delayed, average_time_disruption, total_surp_loss_e1, total_pct_surplus_loss_e1, total_surp_loss_e2, total_pct_surplus_loss_e2, distance_disruption, time_disruption, unaffected_percentiles, delayed_percentiles

def run_percolation_cluster(x,run_no=100):
    """ This function returns results for a single country's transport network.  Possible OD points are chosen
    then probabilistically selected according to the populations each node counts (higher population more likely).

    Args:
        x : country string

    Returns:
        pd.concat(results) (pandas.DataFrame) : The results of the percolation
    """

    #data_path = Path(r'/scistor/ivm/data_catalogue/open_street_map/')
    data_path = Path(r'C:/data/')

    #get all networks for a country
    get_all_networks = [y.name[4] for y in data_path.joinpath("percolation_networks").iterdir() if (y.name.startswith(x) & y.name.endswith('-edges.feather'))]

    for network in get_all_networks:
    
        try:
            country = x

            if data_path.joinpath('percolation_results','{}_{}_results.csv'.format(country,network)).is_file():
                print(country+' '+network+" already finished!")           
                continue
            
            print(country+' '+network+" started!")
            all_gdp = pd.read_csv(open(data_path.joinpath("percolation_input_data","worldbank_gdp_2019.csv")),error_bad_lines=False)
            gdp = all_gdp.gdp.loc[all_gdp.iso==country].values[0]
            edges = feather.read_dataframe(data_path.joinpath("percolation_networks","{}_{}-edges.feather".format(country,network)))
            nodes = feather.read_dataframe(data_path.joinpath("percolation_networks","{}_{}-nodes.feather".format(country,network)))
            nodes.geometry = pyg.from_wkb(nodes.geometry)

            # Each country has a set of centroids of grid cells with populations for each cell
            possibleOD = pd.read_csv(open(data_path.joinpath("country_OD_points","{}.csv".format(x))))
            grid_height = possibleOD.grid_height.iloc[2]
            #radius of circle to cover entire box,  pythagorus theorem
            h = np.sqrt(((grid_height)**2) *2)
            del possibleOD['Unnamed: 0']
            del possibleOD['GID_0']
            possibleOD['geometry'] = possibleOD['geometry'].apply(pyg.from_wkt)

            seed = sum(map(ord,country))
            random.seed(seed)
            np.random.seed(seed)

            OD_pos = prepare_possible_OD(possibleOD, nodes, h)
            OD_no = min(len(OD_pos),100)
            results = []
            for x in tqdm(range(run_no),total=run_no,desc='percolation for '+country+' '+network):
                OD_nodes, populations = choose_OD(OD_pos, OD_no)
                results.append(percolation_Final(edges, 0.01, OD_nodes, populations,gdp))
            
            res = pd.concat(results)
            res.to_csv(data_path.joinpath('percolation_results','{}_{}_results.csv'.format(country,network)))

        except Exception as e: 
            print(country+' '+network+" failed because of {}".format(e))

def run_percolation(country, edges, nodes, OD_no = 100, run_no = 1, seed = []):
    """ This function returns results for a single country's transport network.  Possible OD points are chosen
    then probabilistically selected according to the populations each node counts (higher population more likely).

    Args:
        country (String): ISO 3 code for country
        edges, nodes (pandas.DataFrame) :  dataframes with the edges and nodes of country
        OD_no (int) :  number of OD pairs to use
        run_no (int) : number of runs
        seed (int) : 
        nodes (pandas.DataFrame): nodes to re-reference ids

    Returns:
        pd.concat(results) (pandas.DataFrame) : The results of the percolation
    """
    #Get GDP for country (world bank values 2019)   
    all_gdp = pd.read_csv(open("worldbank_gdp_2019.csv"),error_bad_lines=False)
    gdp = all_gdp.gdp.loc[all_gdp.iso==country].values[0]
    # Each country has a set of centroids of grid cells with populations for each cell
    all_OD = pd.read_csv(open("od_points_per_country.csv"))
    possibleOD = all_OD.loc[all_OD.GID_0 == country].reset_index(drop=True)
    del possibleOD['Unnamed: 0']
    del possibleOD['GID_0']
    possibleOD['geometry'] = possibleOD['geometry'].apply(pyg.Geometry)

    if seed:
        random.seed(seed)
        np.random.seed(seed)

    OD_pos = prepare_possible_OD(possibleOD, nodes)
    results = []
    for x in tqdm(range(run_no),total=run_no,desc='percolation'):
        OD_nodes, populations = choose_OD(OD_pos, OD_no)
        results.append(percolation_Final(edges, 0.01, OD_nodes, populations,gdp))
    
    return pd.concat(results)

def reset_ids(edges, nodes):
    """Resets the ids of the nodes and edges, editing 
    the references in edge table using dict masking

    Args:
        edges (pandas.DataFrame): edges to re-reference ids
        nodes (pandas.DataFrame): nodes to re-reference ids

    Returns:
        edges, nodes (pandas.DataFrame) : The re-referenced edges and nodes.
    """    
    nodes = nodes.copy()
    edges = edges.copy()
    to_ids =  edges['to_id'].to_numpy()
    from_ids = edges['from_id'].to_numpy()
    new_node_ids = range(len(nodes))
    #creates a dictionary of the node ids and the actual indices
    id_dict = dict(zip(nodes.id,new_node_ids))
    nt = np.copy(to_ids)
    nf = np.copy(from_ids) 
    #updates all from and to ids, because many nodes are effected, this
    #is quite optimal approach for large dataframes
    for k,v in id_dict.items():
        nt[to_ids==k] = v
        nf[from_ids==k] = v
    edges.drop(labels=['to_id','from_id'],axis=1,inplace=True)
    edges['from_id'] = nf
    edges['to_id'] = nt
    nodes.drop(labels=['id'],axis=1,inplace=True)
    nodes['id'] = new_node_ids
    edges['id'] = range(len(edges))
    edges.reset_index(drop=True,inplace=True)
    nodes.reset_index(drop=True,inplace=True)
    return edges,nodes

def get_metrics_and_split(x):
    
    try:
        #data_path = Path(r'/scistor/ivm/data_catalogue/open_street_map/')
        data_path = Path(r'C:/data/')

        print(x+' has started!')
        edges = feather.read_dataframe(data_path.joinpath("road_networks","{}-edges.feather".format(x)))
        nodes = feather.read_dataframe(data_path.joinpath("road_networks","{}-nodes.feather".format(x)))
 
        edges = edges.drop('geometry',axis=1)
        edges = edges.reindex(['from_id','to_id'] + [x for x in list(edges.columns) if x not in ['from_id','to_id']],axis=1)
        graph= ig.Graph.TupleList(edges.itertuples(index=False), edge_attrs=list(edges.columns)[2:],directed=False)
        graph.vs['id'] = graph.vs['name']

        # edge_tuples = zip(edges['from_id'],edges['to_id'])
        # graph = ig.Graph(directed=False)
        # graph.add_vertices(len(nodes))
        # graph.vs['id'] = nodes['id']
        # graph.add_edges(edge_tuples)
        # graph.es['id'] = edges['id']
        # graph.es['distance'] = edges.distance
        all_df = metrics(graph)
        all_df.to_csv(data_path.joinpath("percolation_metrics","{}_all_metrics.csv".format(x)))


        cluster_sizes = graph.clusters().sizes()
        cluster_sizes.sort(reverse=True) 
        cluster_loc = [graph.clusters().sizes().index(x) for x in cluster_sizes[:5]]

        main_cluster = graph.clusters().giant()
        main_df = metrics(main_cluster)
        main_df.to_csv(data_path.joinpath("percolation_metrics","{}_0_metrics.csv".format(x)))
        main_edges = edges.loc[edges.id.isin(main_cluster.es()['id'])]
        main_nodes = nodes.loc[nodes.id.isin(main_cluster.vs()['id'])]
        main_edges, main_nodes = reset_ids(main_edges,main_nodes)
        feather.write_dataframe(main_edges,data_path.joinpath("percolation_networks","{}_0-edges.feather".format(x)))
        feather.write_dataframe(main_nodes,data_path.joinpath("percolation_networks","{}_0-nodes.feather".format(x)))
        skipped_giant = False

        counter = 1
        for y in cluster_loc:
            if not skipped_giant:
                skipped_giant=True
                continue
            if len(graph.clusters().subgraph(y).vs) < 500:
                break
            g = graph.clusters().subgraph(y)
            g_edges = edges.loc[edges.id.isin(g.es()['id'])]
            g_nodes = nodes.loc[nodes.id.isin(g.vs()['id'])]
            g_edges, g_nodes = reset_ids(g_edges,g_nodes)
            feather.write_dataframe(main_edges,data_path.joinpath("percolation_networks","{}_{}-edges.feather".format(x,str(counter))))
            feather.write_dataframe(main_nodes,data_path.joinpath("percolation_networks","{}_{}-nodes.feather".format(x,str(counter))))
            g_df = metrics(g)
            g_df.to_csv("/scistor/ivm/data_catalogue/open_street_map/percolation_metrics/"+x+"_"+str(counter)+"_metrics.csv")
            counter += 1
        print(x+' has finished!')

    except Exception as e: 
        print(x+" failed because of {}".format(e))

if __name__ == '__main__':     

    #data_path = Path("/scistor/ivm/data_catalogue/open_street_map")
    data_path = Path(r'C:/data/')

    #countries is without CHN, DEU, RUS, USA
    countries = [y.name[:3] for y in data_path.joinpath('road_networks').iterdir()]
    fin_countries =  [y.name[:3] for y in data_path.joinpath('percolation_results').iterdir()]
    left_countries = list(set(countries)-set(fin_countries))
    left_countries = [x[:3] for x in left_countries]
    from random import shuffle
    shuffle(left_countries)
    #left_countries = ['ABW', 'AFG', 'AGO', 'AIA', 'ALA', 'ALB', 'AND']
    #left_countries=['ABW']
    #print(left_countries)

    #run_percolation_cluster('ALB')

    with Pool(cpu_count()-1) as pool: 
        pool.map(run_percolation_cluster,left_countries,chunksize=1)   
