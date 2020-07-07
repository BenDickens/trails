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


def graph_load(gdfNet):
    return ig.Graph.TupleList(gdfNet.edges[['from_id','to_id','distance']].itertuples(index=False),edge_attrs=['distance'])

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


def alt(edges, isolated_threshold=2.5, OD_no = 100, del_frac=0.02):
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

        compare_thresh = np.greater(perc_matrix,OD_thresh)
        over_thresh_no = np.sum(compare_thresh) / 2 
        isolated_trip_results.append(over_thresh_no/pos_trip_no)
        if over_thresh_no >= pos_trip_no: break
 
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
            compare_thresh = np.greater(perc_matrix,OD_thresh)
            over_thresh_no = np.sum(compare_thresh) / 2 
            isolated_trip_results.append(over_thresh_no/pos_trip_no)
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

if __name__ == '__main__':     
    largest_component_df()
