from sklearn.preprocessing import OneHotEncoder
from itertools import islice
import re
import numpy as np

from utils.buildNetwork import build_whole_network


def read_node_label(G3_n, file_label):
    """Read node role(label) information of network.

    Parameters
    ----------
    G3_n : Graph (a networkx data structure)
        A whole network with all kinds of role, saving node name and edges.
    file_label : String
        The path of the .csv file, with the format "{node_name}, {node_role}".

    Returns
    -------
    name2role : dict
        Mapping node name to node role.
    """
    data = dict()
    with open(file_label, 'r') as fr:
        lines = fr.readlines()
    for line in islice(lines, 1, None):
        n = line.strip().split(',')
        data[n[0]] = n[1]
    name2role = dict()
    for node in G3_n.nodes():
        name2role[node] = data[node]
    return name2role


def read_node_label_matrix(G3_i, id2name, num_role_kinds, file_label):
    """Read node role(label) information and transform to matrix.

    Parameters
    ----------
    G3_i : Graph (a networkx data structure)
        A whole network with all kinds of role, saving node id and edges.
    id2name : dict
        Mapping node id to node name.
    num_role_kinds : int
        The number of role kinds in the network.
    file_label : String
        The path of the .csv file, with the format "{node_name}, {node_role}".

    Returns
    -------
    y : np.array
        dimension - (1, num_nodes)
    y_trans : np.array
        dimension - (num_nodes, num_role_kinds)
    """
    data = dict()
    with open(file_label, 'r') as fr:
        lines = fr.readlines()
    for line in islice(lines, 1, None):
        n = line.strip().split(',')
        data[n[0]] = n[1]
    id2role = dict()
    y_trans = np.zeros((len(G3_i.nodes), num_role_kinds))
    for i in G3_i.nodes:
        label = data[id2name[i]]
        id2role[i] = label
        if label == 'CR':
            y_trans[i, :] = [1, 0, 0]
        if label == 'BR':
            y_trans[i, :] = [0, 1, 0]
        if label == 'AR':
            y_trans[i,:] = [0, 0, 1]
        if label == 'IGW':
            y_trans[i, :] = [1, 0, 0, 0]
        if label == 'P':
            y_trans[i, :] = [0, 1, 0, 0]
        if label == 'PE':
            y_trans[i, :] = [0, 0, 1, 0]
        if label == 'POP':
            y_trans[i, :] = [0, 0, 0, 1]
    y = np.array(list(id2role.values()))
    return y, y_trans


def read_node_port(G3_i, name2id, file_port):
    """Read node port information from network.

    Parameters
    ----------
    G3_i : Graph (a networkx data structure)
        A whole network with all kinds of role, saving node id and edges.
    name2id : dict
        Mapping node name to node id.
    file_port : String
        The path of the .csv file, with the format "{node_name}, {node_port}".

    Returns
    -------
    P : dict
        Mapping node port to modified one-hot encoding.
    """
    if file_port is None:
        return None
    data = np.loadtxt(file_port, delimiter=',', skiprows=1, dtype='<U20')
    n2 = data[:, 1].astype('<U3')
    # 取前3位看一共几大类
    kinds = list(set(n2))
    onehot = {}
    for k, v in enumerate(kinds):
        onehot[v] = k
    P = np.zeros((len(G3_i.nodes), len(kinds)))
    for i, kind in enumerate(n2):
        if data[i, 0] not in name2id.keys():  # 该点虽有label但连接虚拟节点，被删除，故不在网中
            continue
        j = onehot[kind]  # 第几列
        if kind == 'Gig':
            number = int(re.match(r'GigabitEthernet(.*?)/', data[i, 1]).group(1))
            if number > P[name2id[data[i, 0]], j]:
                P[name2id[data[i, 0]], j] = number
        if kind == 'Pos':
            number = int(re.match(r'Pos(.*?)/', data[i, 1]).group(1))
            if number > P[name2id[data[i, 0]], j]:
                P[name2id[data[i, 0]], j] = number
        if kind == '100':
            number = int(re.match(r'100GE(.*?)/', data[i, 1]).group(1))
            if number > P[name2id[data[i, 0]], j]:
                P[name2id[data[i, 0]], j] = number
        if kind == 'Ten':
            number = int(re.match(r'TenGigE0/(.*?)/', data[i, 1]).group(1))
            if number > P[name2id[data[i, 0]], j]:
                P[name2id[data[i, 0]], j] = number
        if kind == 'Hun':
            number = int(re.match(r'HundredGigE0/(.*?)/', data[i, 1]).group(1))
            if number > P[name2id[data[i, 0]], j]:
                P[name2id[data[i, 0]], j] = number
    return P


def read_node_model(G3_i, id2name, file_model):
    """Read node model information.

    Parameters
    ----------
    G3_i : Graph (a networkx data structure)
        A whole network with all kinds of role, saving node id and edges.
    id2name : dict
        Mapping node id to node name.
    file_model : String
        The path of the .csv file, with the format "{node_name}, {node_model}".

    Returns
    -------
    M : dict
        Mapping node name to node model.
    """
    if file_model is None:
        return None
    data = np.loadtxt(file_model, delimiter=',', skiprows=1, dtype='<U16')
    n1, n2 = data[:, 0].reshape(-1, 1), data[:, 1].reshape(-1, 1)
    enc = OneHotEncoder().fit_transform(n2).toarray()
    j = enc.shape[1]
    M = np.zeros((len(G3_i.nodes), j))
    for i in range(len(M)):
        index = np.where(n1 == id2name[i])[0]
        M[i] = enc[index, :]
    return M


if __name__ == "__main__":
    G3_n, G3_i, name2id, id2name = build_whole_network(file_network="../datasets/Net3.csv")
    # name2role = read_node_label(G3_n, "../datasets/Net2_label.csv")
    # P = read_node_port(G3_n, name2id, "../datasets/Net2_port.csv")
    M = read_node_model(G3_i, id2name, "../datasets/Net3_model.csv")