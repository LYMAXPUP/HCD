def get_the_largest_set(coms):
    """使得到的coms只含所有最大社团（去除已被大社团包含的小社团集合）
    return:
    communities : [[name1, name2], [name3, name4]]
    """
    coms_new = sorted(coms, reverse=True, key=lambda i: len(i))
    communities = []
    for c1 in coms_new:  # 新来集合
        flag = 0
        for c2 in communities:  # 已有集合
            if set(c1).issubset(set(c2)):
                flag = 1
                break
        if flag == 0:
            communities.append(list(c1))
    return communities


def get_higher_nodes_connections(G2_n, G_high, G_low, coms_low):
    """Get the relationship of nodes in G_high connecting communities in coms_low.

    Parameters
    ----------
    G2_n : Graph
        A multilayer network with partial kinds of role, saving node name and edges.
        e.g., contains roles POP+PE+P
    G_high : Graph
        A network only contains the highest role in G2_n.
        e.g., contains role P.
    G_low : Graph
        A network contains all the lower roles in G2_n.
        e.g., contains roles POP+PE.
    coms_low : list
        The communities of nodes in all the lower roles in G_low. [[name1, name2], [name3], ...]

    Returns
    -------
    connections : dict
        { higher_node1 : {com_low1, com_low2, com_low3}, higher_node2 : {com_low2, com_low3}, ...}
    """
    # M : { com_low1: {higher_node1, higher_node2}, com_low1 : {higher_node2, higher_node3} }
    # N : { {higher_node1, higher_node2} : com_low1, {higher_node1, higher_node3, higher_node4} : com_low2}
    M, N = dict(), dict()
    connections = dict()
    for node in G_high.nodes():
        connections[node] = set()

    for com_low in coms_low:
        for node_i in com_low:
            nbr_high = set(G2_n.neighbors(node_i)) - set(G_low.nodes())
            if nbr_high:
                for node_j in nbr_high:
                    tmp = M.get(frozenset(com_low), set())
                    tmp.add(node_j)
                    M[frozenset(com_low)] = tmp
    for com_low, higher_nodes in M.items():
        tmp = N.get(frozenset(higher_nodes), set()) | set(com_low)
        N[frozenset(higher_nodes)] = tmp
    # 合并有共同higher nodes的低层社团，视作一个part
    for higher_nodes, com_low in N.items():
        for node in higher_nodes:
            tmp = connections.get(node, set())
            tmp.add(frozenset(com_low))
            connections[node] = tmp
    return connections


def coms2C(G2_n, coms):
    """ Transform coms to C.

    Parameters
    ----------
    G2_n : Graph
        A multilayer network with partial kinds of role, saving node name and edges.
    coms : list
        [[name1, name2], [name3], ...]

    Returns
    -------
    C : dict
        { name1 : [com_id1, com_id2], name2 : [com_id1], name3 : [com_id1, com_id3], ...}
    """
    C = dict.fromkeys(G2_n.nodes, [])
    for i,com in enumerate(coms):
        for node in com:
            C_ids = C.get(node, []).copy()
            C_ids.append(i)
            C[node] = C_ids
    return C


def C2coms(C):
    """Transform C to coms.

    Parameters
    ----------
    C : dict
        { name1 : [com_id1, com_id2], name2 : [com_id1], name3 : [com_id1, com_id3], ...}

    Returns
    -------
    coms_new : list
        [[name1, name2], [name3], ...]
    """
    num = max(max(C.values()))
    coms = [[] for i in range(num+1)]
    for key, value in C.items():
        for id in value:
            coms[id].append(key)
    coms_new = [com for com in coms if com]
    return coms_new