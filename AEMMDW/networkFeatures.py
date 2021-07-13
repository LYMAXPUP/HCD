import numpy as np
import networkx as nx
from utils.readNodeData import read_node_model, read_node_port


class CentralityFeatures(object):
    def __init__(self, Gi):
        self.G = Gi.copy()

    def degree(self):
        return dict(nx.degree(self.G))

    def ave_weight(self):
        for i in self.G.nodes():
            edge_weight = []
            for node1, node2, datas in self.G.edges(i, data=True):
                edge_weight.append(datas.get('weight'))
            self.G.nodes[i]['meanWei'] = np.mean(edge_weight)
        return nx.get_node_attributes(self.G, 'meanWei')

    def betweenness(self):
        return nx.betweenness_centrality(self.G)

    def pagerank_weight(self):
        return nx.pagerank(self.G, weight='weight')

    def ave_neighbor_degree(self):
        return nx.average_neighbor_degree(self.G, weight='weight')

    def clustering_weight(self):
        return nx.clustering(self.G, weight='weight')

    def eigenvector_weight(self):
        return nx.eigenvector_centrality(self.G, tol=1.0e-3, weight='weight')

    def closeness_centrality(self):
        return nx.closeness_centrality(self.G)

    def get_centrality_features(self):
        # Select the appropriate parameters according to the actual situation.
        centrality_features = (self.degree(), self.ave_weight(), self.betweenness(), self.betweenness(),
                               self.ave_neighbor_degree(), self.clustering_weight(), self.eigenvector_weight(),
                               self.closeness_centrality())
        X = np.zeros((len(self.G.nodes), len(centrality_features)))
        i = 0
        for parameter in centrality_features:
            for key, value in parameter.items():
                X[key, i] = value
            i += 1
        return X


class AttributeFeatures(object):
    def __init__(self, G3_i, name2id, id2name, file_model, file_port):
        self.G3_i = G3_i
        self.name2id = name2id
        self.id2name = id2name
        self.file_model = file_model
        self.file_port = file_port

    def get_attribute_features(self):
        X_model = read_node_model(self.G3_i, self.id2name, self.file_model)
        X_port = read_node_port(self.G3_i, self.name2id, self.file_port)
        if X_model is not None and X_port is not None:
            return np.hstack((X_model, X_port))
        elif X_model is not None:
            return X_model
        elif X_port is not None:
            return X_port
        else:
            return None


