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

import pandas as pd
import pygeos as pyg

def metrics(graph):
    g = graph
    print("Number of edges: ", g.ecount())
    print("Number of nodes: ", g.vcount())
    print("Density: ", g.density())
    print("Number of cliques: ", g.omega())#or g.clique_number()
    print("Average path length: ", g.average_path_length(directed=False))
    print("Assortativity: ", g.assortativity_degree(False))
    print("Diameter: ",g.diameter(directed=False))
    print("Edge Connectivity: ", g.edge_connectivity())
    print("Graph is simple", g.is_simple())
    print("Maximum degree: ", g.maxdegree())
    print("Total Edge length ", np.sum(g.es['distance']))
    convert_nx(g)

def show(graph):
    g = graph
    layout = g.layout("kk")
    ig.plot(g, layout=layout)

#Creates a graph 
def graph_load(edges):
    #return ig.Graph.TupleList(gdfNet.edges[['from_id','to_id','distance']].itertuples(index=False),edge_attrs=['distance'])
    g = ig.Graph(directed=False)
    max_node_id = max(max(edges.from_id),max(edges.to_id))
    g.add_vertices(max_node_id+1)
    zippy = zip(edges.from_id,edges.to_id)
    g.add_edges(zippy)
    g.es['distance'] = edges.distance
    g.es['time'] = edges.time
    return g
    
def graph_load_largest(gdfNet):
    graph = graph_load(gdfNet)
    return graph.clusters().giant()

#returns the largest component of a network object (network.edges pd  and network.nodes pd) with reset ids
def largest_component_df(edges,nodes):
    edges = edges
    nodes = nodes
    zippy = zip(edges['from_id'],edges['to_id'])
    g = ig.Graph(directed=False)
    g.add_vertices(len(nodes))
    g.vs['id'] = nodes['id']
    g.add_edges(zippy)
    g.es['id'] = edges['id']
    g = g.clusters().giant()
    a = edges.loc[edges.id.isin(g.es()['id'])]
    b = nodes.loc[nodes.id.isin(g.vs()['id'])]
    return reset_ids(a,b)

# This function creates a demand matrix from the equation Demand_a,b = Population_a * Population_b * e^ [-p * Distance_a,b] , see mthodology.doc
#p is set to 1, populations represent the grid square of the origin, 
#OD_nodes is a list of nodes to use for the OD, a,b
#OD_orig is a shortest path matrix used for the distance calculation
#node_pop gives the population per OD node
def create_demand(OD_nodes, OD_orig, node_pop):
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

def chooseOD(gridDF, nodes):
    #possible_OD = np.random.choice(pop_density, size=len(pop_density), p=pop_density/pop_density.max())
    nodeIDs = []
    sindex = pyg.STRtree(nodes['geometry'])

    for i in gridDF.itertuples():
        print(i.geometry)
        print(type(i.geometry))
        ID = nearest(i.geometry, nodes, sindex)
        print(ID)


def nearest(geom, gdf,sindex):
    """Find the element of a GeoDataFrame nearest a shapely geometry
    """
    #sindex = pygeos.STRtree(gdf['geometry'])
    matches_idx = sindex.query(geom)
    #pygeos.measurement.bounds(geom)
    #matches_idx = gdf.sindex.nearest(geom.bounds)
    nearest_geom = min(
        [gdf.iloc[match_idx] for match_idx in matches_idx],
        key=lambda match: pyg.measurement.distance(match.geometry,geom)
    )
    return nearest_geom

def perc_Final(edges, del_frac, OD_list=[]):
    result_df = []
    OD_no = 100
    g = graph_load(edges)
    if OD_list == []: OD_nodes = random.sample(range(g.vcount()-1),OD_no)
    else: OD_nodes = OD_list

    edge_no = g.ecount() 
    OD_node_no = len(OD_nodes)

    node_pop = random.sample(range(4000), OD_no)


    base_shortest_paths = g.shortest_paths_dijkstra(source=OD_nodes,target = OD_nodes,weights='time')
    OD_orig = np.matrix(base_shortest_paths)
    OD_thresh = OD_orig * 10
    
    demand = create_demand(OD_nodes, OD_orig, node_pop)
    print(demand)

    exp_g = g.copy()
    trips_possible = True
    pos_trip_no = (((OD_node_no**2) - OD_node_no) / 2) - ((np.count_nonzero(np.isinf(OD_orig)))/2)
    counter = 0
    frac_counter = 0 

    while trips_possible:
        exp_edge_no = exp_g.ecount()

        no_edge_del = math.floor(del_frac * edge_no)
        try:
            edges_del = random.sample(range(exp_edge_no),no_edge_del)
        except:
            print("random sample playing up but its ok")
            edges_del = range(exp_edge_no)
        exp_g.delete_edges(edges_del)
        frac_counter += del_frac
        new_shortest_paths = exp_g.shortest_paths_dijkstra(source=OD_nodes,target = OD_nodes,weights='distance')
        perc_matrix = np.matrix(new_shortest_paths)
        results = SummariseOD(perc_matrix, 99999999999, demand, OD_orig, 50000,round(frac_counter,3))  
        result_df.append(results)
        
        
        if results[0] >= 0.99: break
        if exp_edge_no < 1: break
        #if counter==4: break
    result_df = pd.DataFrame(result_df, columns=['frac_counter', 'pct_isolated', 'average_time_disruption', 'pct_thirty_plus', 'pct_twice_plus', 'pct_thrice_plus','total_surp_loss_e1', 'total_pct_surplus_loss_e1', 'total_surp_loss_e2', 'total_pct_surplus_loss_e2'])
    print(result_df)


def simple_OD_calc(OD, comparisonOD,pos_trip_no):
    compare_thresh = np.greater(OD,comparisonOD)
    over_thresh_no = np.sum(compare_thresh) / 2
    return over_thresh_no / pos_trip_no
    
def SummariseOD(OD, fail_value, demand, baseline, GDP_per_capita, frac_counter):

    ans = []

    ### Function returns the % of total trips between origins and destinations that exceed fail value

    ## calculate trips destroyed
    masked_OD = np.ma.masked_greater(OD, value = (fail_value - 1))
    masked_baseline = np.ma.masked_greater(baseline, value = (fail_value - 1))
    adj_time = np.ma.masked_array(OD,masked_baseline.mask)
    masked_demand = np.ma.masked_array(demand, masked_baseline.mask)

    total_trips = masked_demand.sum()

    isolated_trips = np.ma.masked_array(masked_demand,~masked_OD.mask)
    isolated_trips_sum = isolated_trips.sum()

    potentially_disrupted_trips = np.ma.masked_array(masked_demand,masked_OD.mask)
    potentially_disrupted_trips_sum = potentially_disrupted_trips.sum()

    try:
        pct_isolated = (isolated_trips_sum / total_trips)
    except:
        pct_isolated  = 0

    ## Set up for ratio-based calculations

    fail_ratio = 50.0 #((fail_value-1) / baseline.max()) # headlimit above which trip destoryed

    potentially_disrupted_trips_original_time = np.ma.masked_array(masked_baseline, masked_OD.mask)
    delta_time_OD = (masked_OD - potentially_disrupted_trips_original_time)
    average_time_disruption = (delta_time_OD * potentially_disrupted_trips).sum() / potentially_disrupted_trips.sum()

    frac_OD = masked_OD / potentially_disrupted_trips_original_time

    def PctDisrupt(x, frac_OD, demand):
        masked_frac_OD = np.ma.masked_inside(frac_OD, 1, (1+x))
        m_demand = np.ma.masked_array(demand, masked_frac_OD.mask)
        return ((m_demand.sum()) / (demand.sum()))

    pct_thirty_plus = PctDisrupt(0.3, frac_OD, potentially_disrupted_trips)
    pct_twice_plus = PctDisrupt(2, frac_OD, potentially_disrupted_trips)
    pct_thrice_plus = PctDisrupt(3, frac_OD, potentially_disrupted_trips)

    # Flexing demand with trip cost
    def surplus_loss(e, C2, C1, D1):

        Y_intercept_max_cost = C1 - (e * D1)

        C2 = np.minimum(C2, Y_intercept_max_cost)

        delta_cost = C2 - C1

        delta_demand = (delta_cost / e)

        D2 = (D1 + delta_demand)

        surplus_loss_ans = ((delta_cost * D2) + ((delta_cost * -delta_demand) / 2))

        triangle = (D1 * (Y_intercept_max_cost - C1) ) / 2

        total_surp_loss = surplus_loss_ans.sum()

        total_pct_surplus_loss = total_surp_loss / triangle.sum()

        return total_surp_loss, total_pct_surplus_loss

    adj_cost = (adj_time * GDP_per_capita) / (365 * 8 * 3600)
    baseline_cost = (masked_baseline * GDP_per_capita) / (365 * 8 * 3600)

    total_surp_loss_e1, total_pct_surplus_loss_e1 = surplus_loss(-0.15, adj_cost, baseline_cost, masked_demand)
    total_surp_loss_e2, total_pct_surplus_loss_e2 = surplus_loss(-0.36, adj_cost, baseline_cost, masked_demand)

    return frac_counter, pct_isolated, average_time_disruption, pct_thirty_plus, pct_twice_plus, pct_thrice_plus,total_surp_loss_e1, total_pct_surplus_loss_e1, total_surp_loss_e2, total_pct_surplus_loss_e2


#Resets the ids of the nodes and edges, editing the refereces in edge table 
#using dict masking
def reset_ids(edges, nodes):
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

#Del frac is now a fraction of total edge length rather than of number of edges
def percolation_by_length(graph, OD_nodes, del_frac,isolated_threshold=2.5):
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
    


'''
    for frac in intervals:
        no_edge_del = math.floor(frac * edge_no)
        edges_del = random.sample(range(edge_no),no_edge_del)
        g.delete_edges(edges_del)
        for e in g.es:
            print(e)
        print(edges_del)
        showMore(g)
        print(frac)
        print(no_edge_del)
'''
def alt(edges, isolated_threshold=3, OD_no = 100, del_frac=0.02):
    g = ig.Graph(directed=False)
    max_node_id = max(max(edges.from_id),max(edges.to_id))
    g.add_vertices(max_node_id+1)
    zippy = zip(edges.from_id,edges.to_id)
    g.add_edges(zippy)
    g.es['distance'] = edges.distance
    randomOD = random.sample(range(g.vcount()-1),OD_no)
    return percolation(g,randomOD,del_frac, isolated_threshold)

def edge_to_perc(edges, isolated_threshold = 3, OD_no=100, del_frac=0.02):
    g = ig.Graph(directed=False)
    max_node_id = max(max(edges.from_id),max(edges.to_id))
    g.add_vertices(max_node_id+1)
    zippy = zip(edges.from_id,edges.to_id)
    g.add_edges(zippy)
    g.es['distance'] = edges.distance
    randomOD = random.sample(range(g.vcount()-1),OD_no)
    return percolation_by_length(g,randomOD,del_frac, isolated_threshold)


def convert_nx(graph):
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


def percolation(graph, OD_nodes, del_frac, isolated_threshold=2):
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
'''
    plt.plot(x_ax,isolated_trip_results)
    plt.ylabel("Isolated Trips")
    plt.title("Simple percolation")
    plt.xlabel("Percentage of edges destroyed")
    plt.show()
    print(isolated_trip_results)
'''
    
#https://plotly.com/python/v3/igraph-networkx-comparison/
def showMore(graph):
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



if __name__ == '__main__':     
    largest_component_df()
