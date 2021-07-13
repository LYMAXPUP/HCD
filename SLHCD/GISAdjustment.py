import numpy as np


def get_euclidean_distance(vecA, vecB):
    """Compute the distance between two vectors.

    Parameters
    ----------
    vecA : np.array
    vecB : np.array

    Returns
    -------
        int
    """
    return np.sqrt(np.sum(np.power((vecA - vecB), 2)))


def geography_adjust_communities(gis, C, coms):
    """Adjust communities according to geographical locations.

    Parameters
    ----------
    gis : dict
        The hypothetical location of network. Format:
        { node_name : np.array(location_x, location_y) }
    C : dict
        Mapping the node to the communities it belongs. Each communities is allocated a different id.
        Format: { node_name : [comA_id, comB_id...] }.
        e.g., {'JUFW':[1], 'YYID':[1,2,3], 'XUKK':[2] }
    coms : list
        The list of all communities.
        Format: [[node_name1, node_name2], [node_name3, node_name4, node_name5], ...]

    Returns
    -------
    C : dict
        The updated C, in which each node only belongs to one community.
    """
    # The average location of each community.
    gis_C = dict()
    for i, com in enumerate(coms):
        x, y = [], []
        for node in com:
            x.append(gis[node][0])
            y.append(gis[node][1])
        gis_C[i] = np.array([np.mean(x), np.mean(y)])

    over_ids_set = set()
    for C_ids in C.values():
        if len(C_ids) > 1:
            over_ids_set.add(frozenset(C_ids))

    # Adjust the overlapped part to the nearest community.
    for over_ids in over_ids_set:
        part = [k for k,v in C.items() if frozenset(v) == over_ids]
        x, y = [], []
        for node in part:
            x.append(gis[node][0])
            y.append(gis[node][1])
        gis_over = np.array([np.mean(x), np.mean(y)])
        best_Cid = -1
        dis = dict()
        for id in over_ids:
            dis[id] = get_euclidean_distance(gis_over, gis_C[id])
        best_Cid = min(dis, key=dis.get)
        for node in part:
            C[node] = [best_Cid]
    return C


