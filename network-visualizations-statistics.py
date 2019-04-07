import networkx as nx
import matplotlib.pyplot as plt
import pickle,json
from math import log

import networkx as nx
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from math import log



def save_visualization(g, file_name, title):
    plt.figure(figsize=(18, 12))
    degrees = dict(nx.degree(g))

    # Draw networkx graph -- scale node size by log(degree+1)
    nx.draw_spring(g, with_labels=False,
                   linewidths=2.0,
                   nodelist=degrees.keys(),
                   node_size=[log(degree_val + 1) * 100 for degree_val in degrees.values()], \
                   node_color='r')

    # Create black border around node shapes
    ax = plt.gca()
    ax.collections[0].set_edgecolor("#000000")

    #     plt.title(title)
    plt.savefig(file_name)
    plt.clf()


def get_network_statistics(g):
    num_connected_components = nx.number_connected_components(g)
    num_nodes = nx.number_of_nodes(g)
    num_edges = nx.number_of_edges(g)
    density = nx.density(g)
    avg_clustering_coef = nx.average_clustering(g)
    avg_degree = sum(dict(g.degree()).values()) / float(num_nodes)
    transitivity = nx.transitivity(g)

    if num_connected_components == 1:
        diameter = nx.diameter(g)
    else:
        diameter = None  # infinite path length between connected components

    network_statistics = {
        'num_connected_components': num_connected_components,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': density,
        'diameter': diameter,
        'avg_clustering_coef': avg_clustering_coef,
        'avg_degree': avg_degree,
        'transitivity': transitivity
    }

    return network_statistics

def save_network_statistics(g, file_name):
    network_statistics = get_network_statistics(g)
    with open(file_name, 'wb') as f:
        pickle.dump(network_statistics, f)

def save_network_statistics_json(g, file_name):
    network_statistics = get_network_statistics(g)
    with open(file_name, 'w') as f:
        json.dump(network_statistics, f, indent=4)

def facebook_networks():
    FB_EGO_USERS = [0, 107, 1684, 1912, 3437, 348, 3980, 414, 686, 698]

    # Individual ego networks
    for user in FB_EGO_USERS:
        network_dir = './data/fb-processed/{0}-adj-feat.pkl'.format(user)
        with open(network_dir, 'rb') as f:
            adj, features = pickle.load(f)

        G = nx.Graph(adj)

        visualization_file_name = './visualizations/fb-ego-{0}-visualization.png'.format(user)
        statistics_file_name_pkl = './network-statistics/fb-ego-{0}-statistics.pkl'.format(user)
        statistics_file_name_json = './network-statistics/fb-ego-{0}-statistics.json'.format(user)
        title = 'Facebook Ego Network: ' + str(user)

        save_visualization(G, visualization_file_name, title)
        save_network_statistics(G, statistics_file_name_pkl)
        save_network_statistics_json(G, statistics_file_name_json)

    # Combined FB network
    combined_dir = './data/fb-processed/combined-adj-sparsefeat.pkl'
    with open(combined_dir, 'rb') as f:
        adj, features = pickle.load(f)
        G = nx.Graph(adj)

        visualization_file_name = './visualizations/fb-combined-visualization.png'
        statistics_file_name_pkl = './network-statistics/fb-combined-statistics.pkl'
        statistics_file_name_json = './network-statistics/fb-combined-statistics.json'
        title = 'Facebook Ego Networks: Combined'

        save_visualization(G, visualization_file_name, title)
        save_network_statistics(G, statistics_file_name_pkl)
        save_network_statistics_json(G, statistics_file_name_json)


#other social networks
def other_social_networks():
    NetWorks = ['twitter', 'gplus', 'hamster', 'advogato']
    for network in NetWorks:
        # Read edge-list
        print('')
        print('Reading {} edgelist'.format(network))
        network_edges_dir = './data/{}/{}.txt'.format(network, network)

        # Parse edgelist into undirected graph
        if network in ['hamster', 'jazz', 'karate']:
            with open(network_edges_dir, 'rb')as edges_f:
                network_g = nx.read_edgelist(edges_f, nodetype=int, create_using=nx.Graph(), encoding='latin1',
                                             data=(('weight', float),))

            # print('Generating network visualization')
            visualization_file_name = './visualizations/{0}-visualization.png'.format(network)
            save_visualization(network_g, visualization_file_name, '{} Network'.format(network))

        else:
            with open(network_edges_dir, 'rb')as edges_f:
                network_g = nx.read_edgelist(edges_f, nodetype=int, create_using=nx.DiGraph(), encoding='latin1',
                                             data=(('weight', float),))
            print('Num. weakly connected components: ', nx.number_weakly_connected_components(network_g))

            # print('Generating network visualization and statistics')
            visualization_file_name = './visualizations/{0}-visualization.png'.format(network)
            statistics_file_name_pkl = './network-statistics/{0}-statistics.pkl'.format(network)
            statistics_file_name_json = './network-statistics/{0}-statistics.json'.format(network)
            save_visualization(network_g, visualization_file_name, '{} Network'.format(network))
            save_network_statistics(network_g, statistics_file_name_pkl)
            save_network_statistics_json(network_g, statistics_file_name_json)


#nx networks
def random_networks():
    RANDOM_SEED = 0

    # Dictionary to store all nx graphs
    nx_graphs = {}

    # Small graphs
    N_SMALL = 200
    nx_graphs['er-small'] = nx.erdos_renyi_graph(n=N_SMALL, p=.03, seed=RANDOM_SEED)  # Erdos-Renyi
    nx_graphs['ws-small'] = nx.watts_strogatz_graph(n=N_SMALL, k=11, p=.1, seed=RANDOM_SEED)  # Watts-Strogatz
    nx_graphs['ba-small'] = nx.barabasi_albert_graph(n=N_SMALL, m=6, seed=RANDOM_SEED)  # Barabasi-Albert
    nx_graphs['pc-small'] = nx.powerlaw_cluster_graph(n=N_SMALL, m=6, p=.02, seed=RANDOM_SEED)  # Powerlaw Cluster
    nx_graphs['sbm-small'] = nx.random_partition_graph(sizes=[N_SMALL // 10] * 10, p_in=.1, p_out=.01,
                                                       seed=RANDOM_SEED)  # Stochastic Block Model

    # Larger graphs
    N_LARGE = 1000
    nx_graphs['er-large'] = nx.erdos_renyi_graph(n=N_LARGE, p=.03, seed=RANDOM_SEED)  # Erdos-Renyi
    nx_graphs['ws-large'] = nx.watts_strogatz_graph(n=N_LARGE, k=11, p=.1, seed=RANDOM_SEED)  # Watts-Strogatz
    nx_graphs['ba-large'] = nx.barabasi_albert_graph(n=N_LARGE, m=6, seed=RANDOM_SEED)  # Barabasi-Albert
    nx_graphs['pc-large'] = nx.powerlaw_cluster_graph(n=N_LARGE, m=6, p=.02, seed=RANDOM_SEED)  # Powerlaw Cluster
    nx_graphs['sbm-large'] = nx.random_partition_graph(sizes=[N_LARGE // 10] * 10, p_in=.05, p_out=.005,
                                                       seed=RANDOM_SEED)  # Stochastic Block Model

    # Remove isolates from random graphs
    for g_name, nx_g in nx_graphs.items():
        isolates = nx.isolates(nx_g)
        if len(list(isolates)) > 0:
            for isolate_node in isolates:
                nx_graphs[g_name].remove_node(isolate_node)

    for name, g in nx_graphs.items():
        if nx.number_connected_components(g) > 1:
            print('Unconnected graph: ', name)

        visualization_file_name = './visualizations/{0}-visualization.png'.format(name)
        statistics_file_name_pkl = './network-statistics/{0}-statistics.pkl'.format(name)
        statistics_file_name_json = './network-statistics/{0}-statistics.json'.format(name)
        title = "Random NetworkX Graph: " + name

        save_visualization(g, visualization_file_name, title)
        save_network_statistics(g, statistics_file_name_pkl)
        save_network_statistics_json(g, statistics_file_name_json)

#facebook_networks()
#other_social_networks()
#random_networks()

 # Combined FB network
    combined_dir = './data/fb-processed/combined-adj-sparsefeat.pkl'
    with open(combined_dir, 'rb') as f:
        adj, features = pickle.load(f)
        G = nx.Graph(adj)

        visualization_file_name = './visualizations/fb-combined-visualization.png'
        statistics_file_name_pkl = './network-statistics/fb-combined-statistics.pkl'
        statistics_file_name_json = './network-statistics/fb-combined-statistics.json'
        title = 'Facebook Ego Networks: Combined'

        save_visualization(G, visualization_file_name, title)
        save_network_statistics(G, statistics_file_name_pkl)
        save_network_statistics_json(G, statistics_file_name_json)