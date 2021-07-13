"""
Set node attributes, so as to export to Gephi and show.
"""


def set_community_in_gephi(G3_n, C, com):
    """Set the `community` that each node belongs to. In Gephi, nodes belonging to different
    communities show different colors.

    Parameters
    ----------
    G3_n : Graph (a networkx data structure)
        A whole network with all kinds of role, saving node name and edges.
    C : dict
        Mapping the node to the communities it belongs. Each communities is allocated a different id.
        Format: { node_name : [comA_id, comB_id...] }.
        e.g., {'JUFW':[1], 'YYID':[1,2,3], 'XUKK':[2] }
    coms : list
        The list of all communities.
        Format: [[node_name1, node_name2], [node_name3, node_name4, node_name5], ...]

    """
    for i in C.keys():
        G3_n.nodes[i][com] = C[i][0]


def set_role_in_gephi(G3_n, name2role):
    """Set the `role` that each node is in. In Gephi, nodes in different roles show
    different size.

    Parameters
    ----------
    G3_n : Graph (a networkx data structure)
        A whole network with all kinds of role, saving node name and edges.
    name2role : dict
        Mapping node name to node role.

    """
    for node in G3_n.nodes():
        role = name2role[node]
        # Net1And2 ----------------------------------------
        if role == 'AR':
            G3_n.nodes[node]['role'] = 0
        if role == 'BR':
            G3_n.nodes[node]['role'] = 5
        if role == 'CR':
            G3_n.nodes[node]['role'] = 25
        # Net3 -------------------------------------
        if role == 'POP':
            G3_n.nodes[node]['role'] = 0
        if role == 'PE':
            G3_n.nodes[node]['role'] = 20
        if role == 'P':
            G3_n.nodes[node]['role'] = 50
        if role == 'IGW':
            G3_n.nodes[node]['role'] = 100
