import networkx as nx


def build_whole_network(file_network):
    """Read network data from datasets, and build the whole network.

    Parameters
    ----------
    file_network : String
        The path of the .csv file, with the format "{node1_name}, {node2_name}, {edge_weight}".

    Returns
    -------
    G3_n : Graph (a networkx data structure)
        A whole network with all kinds of role, saving node name and edges.
    G3_i : Graph (a networkx data structure)
        A whole network with all kinds of role, saving node id and edges.
    name2id : dict
        Mapping node name to node id. (Why? Because id is easy to store as the column index of np.array)
    id2name : dict
        Mapping node id to node name.
    """
    G3_n = nx.Graph()
    with open(file_network, 'r') as fr:
        lines = fr.readlines()
    for line in lines:
        n = line.strip().split(',')
        e = (n[0], n[1])
        if e in G3_n.edges():
            weiValue = G3_n.edges._adjdict[e[0]][e[1]]['weight']
            G3_n.add_edge(e[0], e[1], weight=weiValue + float(n[2]))
        else:
            G3_n.add_edge(e[0], e[1], weight=float(n[2]))

    name2id = dict.fromkeys(G3_n.nodes())
    i = 0
    for name in name2id.keys():
        name2id[name] = i
        i = i + 1
    id2name = {v: k for k, v in name2id.items()}

    G3_i = nx.Graph()
    for line in lines:
        n = line.strip().split(',')
        e = (name2id[n[0]], name2id[n[1]])
        if e in G3_i.edges():
            weiValue = G3_i.edges._adjdict[e[0]][e[1]]['weight']
            G3_i.add_edge(e[0], e[1], weight=weiValue + float(n[2]))
        else:
            G3_i.add_edge(e[0], e[1], weight=float(n[2]))

    return G3_n, G3_i, name2id, id2name


def build_multilayer_network(G3_n, remove_roles, name2role):
    """Remove nodes with given roles, and build the multilayer network. Notes, the rest layers must be continuous.

    Parameters
    ----------
    G3_n : Graph (a networkx data structure)
        A whole network with all kinds of role, saving node name and edges.
    remove_roles : list
        Nodes having role in this list will all be removed.
    name2role : dict
        e.g., {node_name: "ASG"}

    Returns
    -------
    G2_n : Graph (a networkx data structure)
        A multilayer network with partial kinds of role, saving node name and edges.
    """
    G2_n = G3_n.copy()
    for role in remove_roles:
        remove_nodes = filter(lambda k: name2role[k] == role, name2role)
        G2_n.remove_nodes_from(remove_nodes)
    return G2_n


