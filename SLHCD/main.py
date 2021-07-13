import networkx as nx

from expandCommunity import ExpandInit, ExpandCommunity
from utils.buildNetwork import build_whole_network, build_multilayer_network
from utils.readNodeData import read_node_label
from changeComsFormat import C2coms
from drawNetwork import set_community_in_gephi, set_role_in_gephi

_NET1 = {0: "AR", 1: "BR", 2: "CR"}
_NET2 = {0: "AR", 1: "BR", 2: "CR"}
_NET3 = {0: "POP", 1: "PE", 2: "P", 3: "IGW"}


class SLHCD:
    def __init__(self, file_network, file_label, id2role, alpha=1):
        self.G3_n, self.G3_i, self.name2id, self.id2name = build_whole_network(file_network=file_network)
        self.alpha = alpha
        self.gis = nx.spring_layout(self.G3_n, seed=0)
        self.id2role = id2role
        self.name2role = read_node_label(self.G3_n, file_label=file_label)

    def get_current_G(self, currRole):
        roles = list(self.id2role.values())
        for role in range(currRole+1):
            roles.remove(self.id2role[role])
        currGn = build_multilayer_network(self.G3_n, remove_roles=roles, name2role=self.name2role)
        return currGn

    def get_the_lowest_coms(self):
        G = self.get_current_G(currRole=0)
        C_unover = ExpandInit(G, self.gis, self.alpha).apply()
        return G, C_unover

    def get_higher_coms(self, currRole, Glow, C_low):
        G = self.get_current_G(currRole=currRole)  # 1+2层网
        Gup = G.copy()
        Gup.remove_nodes_from(Glow.nodes)
        coms_low = C2coms(C_low)
        C_unover = ExpandCommunity(G, Gup, Glow, coms_low, self.gis, self.alpha).apply()
        return G, C_unover

    def apply(self):
        hierarchy = dict()
        G, C = self.get_the_lowest_coms()
        hierarchy[0] = C
        for currRole in range(1, len(self.id2role)):
            G, C = self.get_higher_coms(currRole, G, C)
            hierarchy[currRole] = C

        set_role_in_gephi(self.G3_n, self.name2role)
        for i in range(len(self.id2role)):
            set_community_in_gephi(self.G3_n, hierarchy[i], "SLHCD_{:d}".format(i))

        # return modularity(self.Gn, coms3)
        # return C2coms(C3)
        nx.write_gexf(self.G3_n, path="network_communities.gexf")
        return hierarchy


if __name__ == "__main__":
    # ----Net1----
    # model = SLHCD(file_network="../datasets/Net1.csv", file_label="../datasets/Net1_label.csv",
    #               id2role=_NET1, alpha=1)

    # ----Net2----
    # model = SLHCD(file_network="../datasets/Net2.csv", file_label="../datasets/Net2_label.csv",
    #               id2role=_NET2, alpha=1)

    # ----Net3----
    model = SLHCD(file_network="../datasets/Net3.csv", file_label="../datasets/Net3_label.csv",
                  id2role=_NET3, alpha=1)

    hierarchy = model.apply()















