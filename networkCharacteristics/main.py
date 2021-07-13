import networkx as nx

from utils.buildNetwork import build_whole_network, build_multilayer_network
from utils.readNodeData import read_node_label
from cycleCoff import Ck


def compute_network_characteristics(file_network, file_label, remove_roles):
    """Compute the main characteristics of datasets.

    Parameters
    ----------
    file_network : String
        The path of the .csv file, with the format "{node1_name}, {node2_name}, {edge_weight}".
    file_label : String
        The path of the .csv file, with the format "{node_name}, {node_role}".
    remove_roles : list
        Nodes having role in this list will all be removed.

    Returns
    -------
    The network characteristics.
    """
    G3_n, _, name2id, id2name = build_whole_network(file_network=file_network)
    name2role = read_node_label(G3_n, file_label=file_label)
    G2_n = build_multilayer_network(G3_n, remove_roles=remove_roles, name2role=name2role)

    D = nx.degree(G3_n)
    num_nodes = nx.number_of_nodes(G2_n)
    nodes = list(G2_n.nodes)
    num_inter_layer_edges = 0
    num_intra_layer_edges = 0
    sum_degree = 0
    for node in nodes:
        sum_degree += D[node]
        for neighbor in nx.neighbors(G3_n, node):
            if neighbor in nodes:
                num_intra_layer_edges += 1
            else:
                num_inter_layer_edges += 1
    ave_degree = sum_degree / num_nodes
    ave_intra_degree = num_intra_layer_edges / num_nodes
    num_intra_layer_edges = num_intra_layer_edges / 2
    k = 10
    ave_cc = Ck(G3_n, nodes, k-2)
    return num_nodes, num_inter_layer_edges, num_intra_layer_edges, ave_degree, ave_intra_degree, ave_cc


if __name__ == "__main__":
    # ---Net1---
    # num_nodes, num_inter_layer_edges, num_intra_layer_edges, ave_degree, ave_intra_degree, ave_cc = \
    #     compute_network_characteristics('../datasets/Net1.csv', "../datasets/Net1_label.csv",
    #                                     ['CR','AR'])

    # ---Net2---
    # num_nodes, num_inter_layer_edges, num_intra_layer_edges, ave_degree, ave_intra_degree, ave_cc = \
    #     compute_network_characteristics('../datasets/Net2.csv', "../datasets/Net2_label.csv",
    #                                     ['CR','AR','RR'])

    # ---Net3---
    num_nodes, num_inter_layer_edges, num_intra_layer_edges, ave_degree, ave_intra_degree, ave_cc = \
        compute_network_characteristics('../datasets/Net3.csv', "../datasets/Net3_label.csv",
                                        ['IGW','P','POP','RR'])

    ave_cc_k3, ave_cc_k4, ave_cc_k5 = ave_cc[1], ave_cc[2], ave_cc[3]










