import os,sys
import igraph as ig
import numpy as np
from timeit import default_timer as timer
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import plotly.offline as py
from plotly.graph_objs import *
import math
import random
import matplotlib.pyplot as plt
import feather

import pandas as pd
import pygeos as pyg


from numpy import inf

#from pgpkg import Geopackage
from numpy.ma import masked
import pathlib
data_path = os.path.join((pathlib.Path(__file__).resolve().parents[2]),'data','percolation')

from multiprocessing import Pool,cpu_count
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

    while trips_possible:
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
        results = SummariseOD(perc_matrix, 99999999999, demand, OD_orig, GDP_per_capita,round(frac_counter,3),cur_dis_length,cur_dis_time)  
        result_df.append(results)
        
        #If the frac_counter goes past 0.99
        if results[0] >= 0.99: break
        #If there are no edges left to remove
        if exp_edge_no < 1: break

    result_df = pd.DataFrame(result_df, columns=['frac_counter', 'pct_isolated', 'average_time_disruption', 'pct_thirty_plus', 'pct_twice_plus', 'pct_thrice_plus','total_surp_loss_e1', 'total_pct_surplus_loss_e1', 'total_surp_loss_e2', 'total_pct_surplus_loss_e2','distance_disruption','time_disruption','unaffected_percentiles','delayed_percentiles','pct_thirty_plus_over2', 'pct_thirty_plus_over6', 'pct_twice_plus_over1'])
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
    masked_OD = np.ma.masked_greater(OD, value = (fail_value - 1))
    masked_baseline = np.ma.masked_greater(baseline, value = (fail_value - 1))
    adj_time = np.ma.masked_array(OD,masked_baseline.mask)
    masked_demand = np.ma.masked_array(demand, masked_baseline.mask)

    total_trips = masked_demand.sum()

    isolated_trips = np.ma.masked_array(masked_demand,~masked_OD.mask)
    isolated_trips_sum = isolated_trips.sum()

    potentially_disrupted_trips = np.ma.masked_array(masked_demand,masked_OD.mask)
    potentially_disrupted_trips_sum = potentially_disrupted_trips.sum()
    
    unaffected_trips = np.ma.masked_equal(masked_OD,masked_baseline).compressed()

    try:
        unaffected_percentiles = []
        unaffected_percentiles.append(np.percentile(unaffected_trips,10))
        unaffected_percentiles.append(np.percentile(unaffected_trips,25))
        unaffected_percentiles.append(np.percentile(unaffected_trips,50))
        unaffected_percentiles.append(np.percentile(unaffected_trips,75))
        unaffected_percentiles.append(np.percentile(unaffected_trips,90))
        unaffected_percentiles.append(np.mean(unaffected_trips))
    except:
        unaffected_percentiles = []

    try:
        pct_isolated = (isolated_trips_sum / total_trips)
    except:
        pct_isolated  = 0

    ## Set up for ratio-based calculations

    fail_ratio = 50.0 #((fail_value-1) / baseline.max()) # headlimit above which trip destroyed

    potentially_disrupted_trips_original_time = np.ma.masked_array(masked_baseline, masked_OD.mask)
    delta_time_OD = (masked_OD - potentially_disrupted_trips_original_time)
    average_time_disruption = (delta_time_OD * potentially_disrupted_trips).sum() / potentially_disrupted_trips.sum()
    delayed_trips = delta_time_OD[delta_time_OD !=0].compressed()

    try:
        delayed_percentiles = []
        delayed_percentiles.append(np.percentile(delayed_trips,10))
        delayed_percentiles.append(np.percentile(delayed_trips,25))
        delayed_percentiles.append(np.percentile(delayed_trips,50))
        delayed_percentiles.append(np.percentile(delayed_trips,75))
        delayed_percentiles.append(np.percentile(delayed_trips,90))
        delayed_percentiles.append(np.mean(delayed_trips))
    except:
        delayed_percentiles = []

    frac_OD = masked_OD / potentially_disrupted_trips_original_time

    def PctDisrupt(x, frac_OD, demand):
        masked_frac_OD = np.ma.masked_inside(frac_OD, 1, (1+x))
        m_demand = np.ma.masked_array(demand, masked_frac_OD.mask)
        return ((m_demand.sum()) / (demand.sum()))

    def PctDisrupt_with_N(x, frac_OD, demand, n):
        journeys_over_n = np.ma.masked_greater(baseline, n)
        masked_frac_OD = np.ma.masked_inside(frac_OD, 1, (1+x))
        disrupt_and_over = np.logical_and(journeys_over_n.mask,masked_frac_OD.mask)
        m_demand = np.ma.masked_array(demand, disrupt_and_over)
        return ((m_demand.sum()) / (demand.sum()))

    pct_thirty_plus = PctDisrupt(0.3, frac_OD, potentially_disrupted_trips)
    pct_twice_plus = PctDisrupt(1, frac_OD, potentially_disrupted_trips)
    pct_thrice_plus = PctDisrupt(2, frac_OD, potentially_disrupted_trips)

    pct_twice_plus_over1 = PctDisrupt_with_N(1, frac_OD, potentially_disrupted_trips,1)
    pct_thirty_plus_over2 = PctDisrupt_with_N(0.3, frac_OD, potentially_disrupted_trips,2)
    pct_thirty_plus_over6 = PctDisrupt_with_N(0.3, frac_OD, potentially_disrupted_trips,6)
  
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
        (np.amax(Y_intercept_max_cost.shape))

        C2 = np.minimum(C2, Y_intercept_max_cost)
        
        delta_cost = C2 - C1

        delta_demand = (delta_cost / e)

        D2 = (D1 + delta_demand)

        surplus_loss_ans = ((delta_cost * D2) + ((delta_cost * -delta_demand) / 2))

        triangle = (D1 * (Y_intercept_max_cost - C1) ) / 2

        total_surp_loss = surplus_loss_ans.sum()

        total_pct_surplus_loss = total_surp_loss / triangle.sum()

        return total_surp_loss, total_pct_surplus_loss

    adj_cost = (adj_time * GDP_per_capita) / (365 * 8 ) #* 3600) time is in hours, so not sure why we do this multiplications with 3600? and minutes would be times 60?
    baseline_cost = (baseline * GDP_per_capita) / (365 * 8 ) #* 3600) time is in hours, so not sure why we do this multiplications with 3600? and minutes would be times 60?

    total_surp_loss_e1, total_pct_surplus_loss_e1 = surplus_loss(-0.15, adj_cost.data, baseline_cost, demand)
    total_surp_loss_e2, total_pct_surplus_loss_e2 = surplus_loss(-0.36, adj_cost.data, baseline_cost, demand)

    if pct_isolated is masked: pct_isolated = np.nan
    if total_surp_loss_e1 is masked: total_surp_loss_e1 = np.nan
    if total_pct_surplus_loss_e1 is masked: total_pct_surplus_loss_e1 = np.nan
    if total_surp_loss_e2 is masked: total_surp_loss_e2 = np.nan
    if total_pct_surplus_loss_e2 is masked: total_pct_surplus_loss_e2 = np.nan
    if pct_thirty_plus is masked: pct_thirty_plus = np.nan
    if pct_twice_plus is masked: pct_twice_plus = np.nan
    if pct_thrice_plus is masked: pct_thrice_plus = np.nan

    return frac_counter, pct_isolated, average_time_disruption, pct_thirty_plus, pct_twice_plus, pct_thrice_plus,total_surp_loss_e1, total_pct_surplus_loss_e1, total_surp_loss_e2, total_pct_surplus_loss_e2, distance_disruption, time_disruption, unaffected_percentiles, delayed_percentiles,pct_thirty_plus_over2, pct_thirty_plus_over6, pct_twice_plus_over1 

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
    all_gdp = pd.read_csv(open(os.path.join(data_path,"worldbank_gdp_2019.csv")),error_bad_lines=False)
    gdp = all_gdp.gdp.loc[all_gdp.iso==country].values[0]
    # Each country has a set of centroids of grid cells with populations for each cell
    all_OD = pd.read_csv(open(os.path.join(data_path,"od_points_per_country.csv")))
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
    

    # list_ois = []
    # list_pos = []

    # for x in tqdm(range(run_no),total=run_no):
    #     oi, po = choose_OD(OD_pos, OD_no)
    #     list_ois.append(oi)
    #     list_pos.append(po)

    # edges.geometry = pyg.to_wkb((edges.geometry))

    # with Pool(cpu_count()-1) as pool: 
    #       results = pool.starmap(percolation_Final,zip(repeat(edges,run_no),
    #                                                         repeat(0.01,run_no),
    #                                                         list_ois,
    #                                                         list_pos,
    #                                                         repeat(gdp,run_no)),
    #                                                         chunksize=1) 


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


#
#       LEGACY METHODS
#

#Del frac is now a fraction of total edge length rather than of number of edges
def percolation_by_length(graph, OD_nodes, del_frac,isolated_threshold=2.5):
    """[summary]

    Args:
        graph ([type]): [description]
        OD_nodes ([type]): [description]
        del_frac ([type]): [description]
        isolated_threshold (float, optional): [description]. Defaults to 2.5.

    Returns:
        [type]: [description]
    """    
    g = graph
 
    interval_no = 2
    edge_no = g.ecount() #10
    OD_node_no = len(OD_nodes)
    print("Number of Edges: ",edge_no)
    #larg = graph.clusters().giant()
    #print(larg.ecount())
    intervals = np.arange(0.01,1,0.2)
    tot_edge_length = np.sum(g.es['distance'])
    
    base_shortest_paths = g.shortest_paths_dijkstra(source=OD_nodes,target = OD_nodes,weights='distance')
    OD_orig = np.matrix(base_shortest_paths)
    #print(OD_orig)
    isolated_threshold = isolated_threshold
    #If the graph is fully connnected this is the number of possible trips
    theo_pos_trip_no = ((OD_node_no**2) - OD_node_no) / 2
    #counts the trips that are not possible before percolation, this shouldn't happen if a fully connected graph is passed
    imp_trips = (np.count_nonzero(np.isinf(OD_orig)))/2
    #consider passing a fully connected graph or ensure OD are part of giant component
    if imp_trips > 0: print(imp_trips," trips were not possible in the OD matrix")
    pos_trip_no = theo_pos_trip_no - imp_trips
    #print(pos_trip_no)
    
    OD_thresh = OD_orig*isolated_threshold
    del_frac = del_frac
    exp_g = g.copy()
    #print(OD_thresh)
    trips_possible = True
    counter = 0
    counterMax = 1/del_frac
    thresh = del_frac*tot_edge_length
    over_thresh_no = 0

    isolated_trip_results = [0]
    while trips_possible:
        exp_edge_no = exp_g.ecount()
        if exp_edge_no < 1: break
        edges_del =  random.randint(0,exp_edge_no-1)
        exp_g.delete_edges(edges_del)
        if(tot_edge_length-np.sum(exp_g.es['distance'])>thresh):
            new_shortest_paths = exp_g.shortest_paths_dijkstra(source=OD_nodes,target = OD_nodes,weights='distance')
            perc_matrix = np.matrix(new_shortest_paths)
            result = simple_OD_calc(perc_matrix, OD_thresh, pos_trip_no)    
            isolated_trip_results.append(result)
            thresh += del_frac*tot_edge_length
            counter += 1
        if over_thresh_no >= pos_trip_no: break
 
        #showMore(exp_g)
        
        #if counter > counterMax: break
    
    x_ax = []
    frac_inc = 0
    for i in isolated_trip_results:
        x_ax.append(frac_inc)
        frac_inc += del_frac

    return x_ax,isolated_trip_results

def show(graph):
    """Simple 

    Args:
        graph ([type]): [description]
    """    
    g = graph
    layout = g.layout("kk")
    ig.plot(g, layout=layout)

def convert_nx(graph):
    """[summary]

    Args:
        graph ([type]): [description]
    """    
    g = graph
    A = g.get_edgelist()
    print(nx.Graph(A))

def graph_example():
    nodes = range(8)
    OD_nodes = [1,2,3,7]
    ed_id =   [0,1,2,3,4,5,6,7,8,9]
    ed_from = [0,1,4,5,2,2,0,1,6,7]
    ed_to =   [1,3,3,0,1,3,4,6,7,2]
    ed_dist = [4,3,2,5,6,7,6,1,1.5,2]
    dict = {'from_id':ed_from,'to_id':ed_to,'distance':ed_dist}
    edges = pd.DataFrame(dict)
    print(edges)

    zippy = zip(edges['from_id'],edges['to_id'])
    g = ig.Graph(directed=False)
    g.add_vertices(len(nodes))
    g.vs['name'] = nodes
    g.add_edges(zippy)
    g.es['distance'] = ed_dist

    #intervals = np.arange(0.01,1,0.2)
    del_frac = 0.2
    percolation_by_length(g, OD_nodes,del_frac)

#https://plotly.com/python/v3/igraph-networkx-comparison/
def showMore(graph):
    """[summary]

    Args:
        graph ([type]): [description]
    """    
    g = graph
  
    layt=g.layout('kk')
    N = g.vcount()
    labels = list(g.vs['id'])
    E = [e.tuple for e in g.es]
    Xn=[layt[k][0] for k in range(N)]
    Yn=[layt[k][1] for k in range(N)]
    Xe=[]
    Ye=[]
    for e in E:
        Xe+=[layt[e[0]][0],layt[e[1]][0], None]
        Ye+=[layt[e[0]][1],layt[e[1]][1], None]

    trace1=Scatter(x=Xe,
                   y=Ye,
                   mode='lines',
                   line= dict(color='rgb(210,210,210)', width=1),
                   hoverinfo='none'
                   )
    trace2=Scatter(x=Xn,
                   y=Yn,
                   mode='markers',
                   name='ntw',
                   marker=dict(symbol='circle-dot',
                                            size=5,
                                            color='#6959CD',
                                            line=dict(color='rgb(50,50,50)', width=0.5)
                                            ),
                   text=labels,
                   hoverinfo='text'
                   )

    axis=dict(showline=False, # hide axis line, grid, ticklabels and  title
              zeroline=False,
              showgrid=False,
              showticklabels=False,
              title=''
              )

    width=800
    height=800
    layout=Layout(title= "Closer look at network",
        font= dict(size=12),
        showlegend=False,
        autosize=False,
        width=width,
        height=height,
        hovermode='closest',
        annotations=[
               dict(
               showarrow=False,
                text='This igraph.Graph has the Kamada-Kawai layout',
                xref='paper',
                yref='paper',
                x=0,
                y=-0.1,
                xanchor='left',
                yanchor='bottom',
                font=dict(
                size=14
                )
                )
            ]
        )

    data=[trace1, trace2]
    fig=Figure(data=data, layout=layout)
    py.iplot(fig, filename='network-igraph')
    
'''
LEGACY METHODS, MAY BE DELETED SOON
def alt(edges, isolated_threshold=3, OD_no = 100, del_frac=0.02):
    """[summary]

    Args:
        edges ([type]): [description]
        isolated_threshold (int, optional): [description]. Defaults to 3.
        OD_no (int, optional): [description]. Defaults to 100.
        del_frac (float, optional): [description]. Defaults to 0.02.

    Returns:
        [type]: [description]
    """    
    g = ig.Graph(directed=False)
    max_node_id = max(max(edges.from_id),max(edges.to_id))
    g.add_vertices(max_node_id+1)
    zippy = zip(edges.from_id,edges.to_id)
    g.add_edges(zippy)
    g.es['distance'] = edges.distance
    randomOD = random.sample(range(g.vcount()-1),OD_no)
    return percolation(g,randomOD,del_frac, isolated_threshold)

def edge_to_perc(edges, isolated_threshold = 3, OD_no=100, del_frac=0.02):
    """[summary]

    Args:
        edges ([type]): [description]
        isolated_threshold (int, optional): [description]. Defaults to 3.
        OD_no (int, optional): [description]. Defaults to 100.
        del_frac (float, optional): [description]. Defaults to 0.02.

    Returns:
        [type]: [description]
    """    
    g = ig.Graph(directed=False)
    max_node_id = max(max(edges.from_id),max(edges.to_id))
    g.add_vertices(max_node_id+1)
    zippy = zip(edges.from_id,edges.to_id)
    g.add_edges(zippy)
    g.es['distance'] = edges.distance
    randomOD = random.sample(range(g.vcount()-1),OD_no)
    return percolation_by_length(g,randomOD,del_frac, isolated_threshold)


def percolation(graph, OD_nodes, del_frac, isolated_threshold=2):
    """[summary]

    Args:
        graph ([type]): [description]
        OD_nodes ([type]): [description]
        del_frac ([type]): [description]
        isolated_threshold (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """    
    g = graph
 
    interval_no = 2
    edge_no = g.ecount() #10
    OD_node_no = len(OD_nodes)
    print("Number of Edges: ",edge_no)
    intervals = np.arange(0.01,1,0.2)

    base_shortest_paths = g.shortest_paths_dijkstra(source=OD_nodes,target = OD_nodes,weights='distance')
    OD_orig = np.matrix(base_shortest_paths)
    #print(OD_orig)
    isolated_threshold = isolated_threshold
    #If the graph is fully connnected this is the number of possible trips
    theo_pos_trip_no = ((OD_node_no**2) - OD_node_no) / 2
    #counts the trips that are not possible before percolation, this shouldn't happen if a fully connected graph is passed
    imp_trips = (np.count_nonzero(np.isinf(OD_orig)))/2
    #consider passing a fully connected graph or ensure OD are part of giant component
    if imp_trips > 0: print(imp_trips," trips were not possible in the OD matrix")
    pos_trip_no = theo_pos_trip_no - imp_trips
    #print(pos_trip_no)
    
    OD_thresh = OD_orig*isolated_threshold
    del_frac = del_frac
    exp_g = g.copy()
    #print(OD_thresh)
    trips_possible = True
    counter = 0
    counterMax = 1/del_frac
    #showMore(exp_g)
    isolated_trip_results = [0]
    while trips_possible:
        exp_edge_no = exp_g.ecount()
        no_edge_del = math.floor(del_frac * edge_no)
        try:
            edges_del = random.sample(range(exp_edge_no),no_edge_del)
        except:
            print("random sample playing up but its ok")
            edges_del = range(exp_edge_no)
        exp_g.delete_edges(edges_del)
        new_shortest_paths = exp_g.shortest_paths_dijkstra(source=OD_nodes,target = OD_nodes,weights='distance')
        perc_matrix = np.matrix(new_shortest_paths)
        result = simple_OD_calc(perc_matrix, OD_thresh, pos_trip_no)    
        isolated_trip_results.append(result)
        if result >= 1: break
 
        #showMore(exp_g)
        counter += 1
        if counter > counterMax: break
    
    x_ax = []
    frac_inc = 0
    for i in isolated_trip_results:
        x_ax.append(frac_inc)
        frac_inc += del_frac

    return x_ax,isolated_trip_results

    plt.plot(x_ax,isolated_trip_results)
    plt.ylabel("Isolated Trips")
    plt.title("Simple percolation")
    plt.xlabel("Percentage of edges destroyed")
    plt.show()
    print(isolated_trip_results)
'''

def split_record(x):
    edges = feather.read_dataframe("/scistor/ivm/data_catalogue/open_street_map/road_networks/"+x+"-edges.feather")
    nodes = feather.read_dataframe("/scistor/ivm/data_catalogue/open_street_map/road_networks/"+x+"-nodes.feather")

    edge_tuples = zip(edges['from_id'],edges['to_id'])
    graph = ig.Graph(directed=False)
    graph.add_vertices(len(nodes))
    graph.vs['id'] = nodes['id']
    graph.add_edges(edge_tuples)
    graph.es['id'] = edges['id']
    graph.es['distance'] = edges.distance
    all_df = metrics(graph)
    all_df.to_csv("/scistor/ivm/data_catalogue/open_street_map/percolation_metrics/"+x+"_all_metrics.csv")


    cluster_sizes = graph.clusters().sizes()
    cluster_sizes.sort(reverse=True) 
    cluster_loc = [graph.clusters().sizes().index(x) for x in cluster_sizes[:5]]

    main_cluster = graph.clusters().giant()
    main_df = metrics(main_cluster)
    main_df.to_csv("/scistor/ivm/data_catalogue/open_street_map/percolation_metrics/"+x+"_0_metrics.csv")
    main_edges = edges.loc[edges.id.isin(main_cluster.es()['id'])]
    main_nodes = nodes.loc[nodes.id.isin(main_cluster.vs()['id'])]
    main_edges, main_nodes = reset_ids(main_edges,main_nodes)
    feather.write_dataframe(main_edges,"/scistor/ivm/data_catalogue/open_street_map/percolation_networks/"+x+"_0-edges.feather")
    feather.write_dataframe(main_nodes,"/scistor/ivm/data_catalogue/open_street_map/percolation_networks/"+x+"_0-nodes.feather")
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
        feather.write_dataframe(g_edges,"/scistor/ivm/data_catalogue/open_street_map/percolation_networks/"+x+"_"+str(counter)+"-edges.feather")
        feather.write_dataframe(g_nodes,"/scistor/ivm/data_catalogue/open_street_map/percolation_networks/"+x+"_"+str(counter)+"-nodes.feather")
        g_df = metrics(g)
        g_df.to_csv("/scistor/ivm/data_catalogue/open_street_map/percolation_metrics/"+x+"_"+str(counter)+"_metrics.csv")
        counter += 1

if __name__ == '__main__':     
    countries = ['ABW', 'AFG', 'AGO', 'AIA', 'ALA', 'ALB', 'AND', 'ARE', 'ARG', 'ARM', 'ASM', 'ATG', 'AUS', 'AUT', 'AZE', 'BDI', 'BEL', 'BEN', 'BES', 'BFA', 'BGD', 'BGR', 'BHR', 'BHS', 'BIH', 'BLM', 'BLR', 'BLZ', 'BMU', 'BOL', 'BRA', 'BRB', 'BRN', 'BTN', 'BWA', 'CAF', 'CAN', 'CCK', 'CHE', 'CHL', 'CIV', 'CMR', 'COD', 'COG', 'COK', 'COL', 'COM', 'CPV', 'CRI', 'CUB', 'CUW', 'CXR', 'CYM', 'CYP', 'CZE', 'DJI', 'DMA', 'DNK', 'DOM', 'DZA', 'ECU', 'EGY', 'ERI', 'ESH', 'ESP', 'EST', 'ETH', 'FIN', 'FJI', 'FLK', 'FRA', 'FRO', 'FSM', 'GAB', 'GBR', 'GEO', 'GGY', 'GHA', 'GIB', 'GIN', 'GLP', 'GMB', 'GNB', 'GNQ', 'GRC', 'GRD', 'GRL', 'GTM', 'GUF', 'GUM', 'GUY', 'HKG', 'HND', 'HRV', 'HTI', 'HUN', 'IDN', 'IMN', 'IND', 'IRL', 'IRN', 'IRQ', 'ISL', 'ISR', 'ITA', 'JAM', 'JEY', 'JOR', 'JPN', 'KAZ', 'KEN', 'KGZ', 'KHM', 'KIR', 'KNA', 'KOR', 'KWT', 'LAO', 'LBN', 'LBR', 'LBY', 'LCA', 'LIE', 'LKA', 'LSO', 'LTU', 'LUX', 'LVA', 'MAC', 'MAF', 'MAR', 'MCO', 'MDA', 'MDG', 'MDV', 'MEX', 'MHL', 'MKD', 'MLI', 'MLT', 'MMR', 'MNE', 'MNG', 'MNP', 'MOZ', 'MRT', 'MSR', 'MTQ', 'MUS', 'MWI', 'MYS', 'MYT', 'NAM', 'NCL', 'NER', 'NFK', 'NGA', 'NIC', 'NIU', 'NLD', 'NOR', 'NPL', 'NRU', 'NZL', 'OMN', 'PAK', 'PAN', 'PER', 'PHL', 'PLW', 'PNG', 'POL', 'PRI', 'PRK', 'PRT', 'PRY', 'PSE', 'PYF', 'QAT', 'REU', 'ROU', 'RWA', 'SAU', 'SDN', 'SEN', 'SGP', 'SHN', 'SLB', 'SLE', 'SLV', 'SMR', 'SOM', 'SPM', 'SRB', 'SSD', 'STP', 'SUR', 'SVK', 'SVN', 'SWE', 'SWZ', 'SXM', 'SYC', 'SYR', 'TCA', 'TCD', 'TGO', 'THA', 'TJK', 'TKM', 'TLS', 'TON', 'TTO', 'TUN', 'TUR', 'TUV', 'TWN', 'TZA', 'UGA', 'UKR', 'URY', 'UZB', 'VAT', 'VCT', 'VEN', 'VGB', 'VIR', 'VNM', 'VUT', 'WLF', 'WSM', 'XAD', 'XCA', 'XKO', 'XNC', 'YEM', 'ZAF', 'ZMB', 'ZWE']
    #countries is without CHN, DEU, RUS, USA
    with Pool(cpu_count()) as pool: 
         pool.map(split_record,countries,chunksize=1)   
