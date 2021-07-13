import numpy as np
from functools import reduce

from GISAdjustment import geography_adjust_communities
from changeComsFormat import get_higher_nodes_connections, get_the_largest_set, coms2C


class ExpandInit:
    """This model is the classic algorithm of `local community detection`, which is applied
    on the plane network. Here, we take it only on the lowest role layer as the initial computation.

    """
    def __init__(self, G, gis, alpha):
        self.G = G
        self.gis = gis
        self.alpha = alpha

    def compute_L(self, i, ind, outd, D, B, S):
        # i is a candidate node in S
        ind_i, outd_i = 0, 0
        i_in, i_out = set(), set()
        D.add(i)
        for j in self.G.neighbors(i):
            if j in D:
                ind_i += self.G[i][j]['weight']
                i_in.add(j)
            else:
                outd_i += self.G[i][j]['weight']
                i_out.add(j)
        S.remove(i)
        if i_out:
            B.add(i)
            S = S | i_out
        for j in i_in:
            if S & set(self.G.neighbors(j)) == set():
                B.remove(j)
        L_in = (ind + 2 * ind_i) / (len(D) + 1)
        # C is a component
        if len(B) == 0:
            return L_in, float("inf"), 0, D, B, S, ind_i, outd_i
        else:
            L_out = (outd - ind_i + outd_i) / len(B)
            L = L_in/np.power(L_out, self.alpha)
            return L_in, L_out, L, D, B, S, ind_i, outd_i

    def expand_seed(self, seed):
        D, B, S = {seed}, {seed}, set(self.G.neighbors(seed))
        ind, outd = 0, self.G.degree(seed, weight='weight')
        L, L_in, L_out = -1, -1, -1
        Li_in, Li_out, Li = -1, -1, -1
        ind_i, outd_i = 0, 0
        D_i, B_i, S_i = set(), set(), set()
        while 1:
            maxL = -10
            for node in S:
                Dn = D.copy()
                Bn = B.copy()
                Sn = S.copy()
                L_in_next, L_out_next, L_next, D_next, B_next, S_next, ind_next, outd_next \
                    = self.compute_L(node, ind, outd, Dn, Bn, Sn)
                if L_out_next == float("inf"):
                    D.add(node)
                    return D
                if maxL < L_next:
                    maxL = L_next
                    i = node
                    Li_in, Li_out, Li = L_in_next, L_out_next, L_next
                    D_i, B_i, S_i = D_next, B_next, S_next
                    ind_i, outd_i = ind_next, outd_next
            if Li < L:
                return D
            # The seed node successfully join the community.
            if Li > L and Li_in > L_in:
                L = Li
                L_in = Li_in
                L_out = Li_out
                D = D_i
                B = B_i
                S = S_i
                ind, outd = ind + 2 * ind_i, outd - ind_i + outd_i
            else:
                return D

    def apply(self):
        coms = []
        # The nodes which are not allocated to a community.
        R = dict()
        D = {n: d for n, d in self.G.degree(weight='weight')}
        for node in self.G.nodes:
            R[node] = D[node]
        while R:
            # Or randomly select seed
            seed = max(R, key=R.get)
            D = self.expand_seed(seed)
            coms.append(D)
            remove = set(R.keys()) & D
            for node in remove:
                del R[node]
        coms_over = get_the_largest_set(coms)
        C_over = coms2C(self.G, coms_over)

        # Geographical location adjustment.
        C_unover = geography_adjust_communities(self.gis, C_over.copy(), coms_over.copy())

        return C_unover


class ExpandCommunity:
    def __init__(self, G, G_high, G_low, coms_low, gis, alpha):
        self.G = G
        self.Gup = G_high
        self.coms_low = coms_low
        self.connections = get_higher_nodes_connections(G, G_high, G_low, coms_low)
        self.gis = gis
        self.alpha = alpha

    def add_lower_nodes(self, seed):
        # When adding nodes on the same role layer, must extra add the lower communities it connects.
        ALL = set()
        if len(self.connections[seed]) == 0:
            ALL = {seed}
        elif len(self.connections[seed]) == 1:
            for com in self.connections[seed]:
                ALL = {seed} | com
        elif len(self.connections[seed]) > 1:
            ALL = {seed} | reduce(lambda x, y: set(x) | set(y), self.connections[seed])
        return ALL

    def init_state(self, seed):
        """
        D : set
            A community which expand from the seed node.
        B : set
            The boundary set of community.
        S : set
            The shell set, which contains the neighbor nodes of D.
        ind : int
            The internal degree of nodes in D.
        outd : int
            The external degree of nodes in B.
        """
        D = self.add_lower_nodes(seed)
        B = set()
        S = set()
        ind = 0  # 社团内部连边
        outd = 0  # 社团外部连边
        for i in D:
            for j in self.G.neighbors(i):
                if j in D:
                    ind += self.G[i][j]['weight']
                else:
                    outd += self.G[i][j]['weight']
                    B.add(i)
        for k in self.Gup.neighbors(seed):
            if k not in D:
                S.add(k)
        if B == set():
            return D
        L_in = ind / len(D)
        L_out = outd / len(B)
        L = L_in / np.power(L_out, self.alpha)
        return (L_in, L_out, L, D, S)

    def compute_L(self, D):
        B = set()
        S = set()
        ind = 0
        outd = 0
        for i in D:
            for j in self.G.neighbors(i):
                if j in D:
                    ind += self.G[i][j]['weight']
                else:
                    outd += self.G[i][j]['weight']
                    B.add(i)
            if i in self.Gup.nodes:
                for k in self.G.neighbors(i):
                    if k not in D and k in self.Gup.nodes:
                        S.add(k)
        if B == set():
            return 0, 0, float("inf"), D, S
        L_in = ind / len(D)
        L_out = outd / len(B)
        L = L_in / np.power(L_out, self.alpha)
        return L_in, L_out, L, D, S

    def expand_seed(self, state):
        if type(state) is set:
            return state
        elif len(state) == 5:
            (L_in, L_out, L, D, S) = state
            Li_in, Li_out, Li = -1, -1, -1
            D_i, S_i = set(), set()
            i = str()
        while True:
            maxL = -10
            for node in S:
                Dn = D.copy()
                Dn = Dn | self.add_lower_nodes(node)
                L_in_next, L_out_next, L_next, D_next, S_next = self.compute_L(Dn)
                if L_next == float("inf"):
                    return D_next
                if maxL < L_next:
                    maxL = L_next
                    i = node
                    Li_in, Li_out, Li = L_in_next, L_out_next, L_next
                    D_i, S_i = D_next, S_next
            if Li < L:
                return D
            # This node successfully join in the community.
            if Li > L and Li_in > L_in:
                L_in, L_out, L = Li_in, Li_out, Li
                D, S = D_i, S_i
            else:
                return D

    def apply(self):
        result = self.coms_low
        D_high = dict()
        for node in self.Gup.nodes():
            D_high[node] = self.G.degree(node, weight="weight")
        while D_high:
            seed = max(D_high, key=D_high.get)
            state = self.init_state(seed)
            com = self.expand_seed(state)
            result.append(com)
            remove = set(D_high.keys()) & com
            for r in remove:
                del D_high[r]
        coms_over = get_the_largest_set(result)
        C_over = coms2C(self.G, coms_over)

        C_unover = geography_adjust_communities(self.gis, C_over, coms_over)

        return C_unover










