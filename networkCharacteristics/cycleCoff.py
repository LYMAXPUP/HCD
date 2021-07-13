"""
The detail information of CycleCoefficient computation is available on
https://github.com/kbuci/CycleCoefficientPython.
"""

import queue
import matplotlib.pyplot as plt
import networkx as nx

from utils.buildNetwork import build_whole_network


def update_ck(graph, src, dests, ck_vals, k):
    frontier = queue.Queue()
    frontier.put(src)
    frontier.put(None)
    explored = set()
    explored.add(src)
    path_length = 0
    while not frontier.empty() and path_length <= k:
        cur = frontier.get()
        if cur == None:
            path_length += 1
            frontier.put(None)
            continue
        elif cur in dests:
            for i in range(path_length,k+1):  
                ck_vals[i] += 1
        for neighbor in graph[cur]:
            if neighbor not in explored:
                explored.add(neighbor)
                frontier.put(neighbor)


def get_ck_node(graph, node, k):
    ck_vals = [0 for i in range(k+1)]
    
    neighbors = [n for n in graph[node]]
    graph.remove_node(node)
   
    dests = set(neighbors)
    for src in neighbors:

        dests.remove(src)
        update_ck(graph, src, dests, ck_vals, k)
    n = len(neighbors)
    n = max(((n - 1) * n) / 2.0, 1)
   
    for i,_ in enumerate(ck_vals):
        ck_vals[i] /= n
    graph.add_edges_from((node, neighbor) for neighbor in neighbors)
    graph.add_edges_from((neighbor, node) for neighbor in neighbors)
    return ck_vals


def Ck(Gn, rest, k):
    ck_vals = [0 for i in range(k+1)]
    for node in rest:
        ck_node = get_ck_node(Gn, node, k)
        for i, ck_val in enumerate(ck_node):
            ck_vals[i] += ck_val
    for i, _ in enumerate(ck_vals):
        ck_vals[i] /= len(rest)
    return ck_vals


if __name__ == '__main__':
    # File 123.csv is a simple test network for this demo.
    k = 10
    Gn, _, name2id, id2name = build_whole_network(file_network='123.csv')
    gis = nx.spring_layout(Gn, seed=0)
    # C_kVals_arr = []
    # C_kVals = Ck(Gn, Gn.nodes, k-2)
    # print(C_kVals)
    # C_kVals_arr.append(C_kVals)
    #
    # cyclePlots = []
    # for c in C_kVals_arr:
    #     cyclePlot, = plt.plot([i for i in range(1, k-2 + 1)], [c[i] for i in range(1,k-2 +1)])
    #     cyclePlots.append(cyclePlot)
    # plt.xticks(range(1,k-2+1), range(3,k+1))
    # plt.xlabel('K values (Note that paths for a given k are from 1 to k-2)')
    # plt.ylabel('Cycle coefficients')
    # plt.show()
    # print(nx.average_clustering(Gn))
