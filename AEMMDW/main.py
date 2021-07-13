import numpy as np

from utils.buildNetwork import build_whole_network
from utils.readNodeData import read_node_label_matrix
from networkFeatures import CentralityFeatures, AttributeFeatures
from splitTrainAndTest import Dataset
from learnBestX import LearnBestX
from evaluateSVM import EvaluateSVM


def node_role_classification(file_network, file_label, file_model, file_port,
                             num_role_kinds, low_dim=100, ratio=0.5):
    """The algorithm of Attribute-associated Enhanced Max-Margin DeepWalk (A-EMMDW). Given a ratio
    of labeled nodes, predict the role of the rest nodes.

    Parameters
    ----------
    file_network : String
        The path of the .csv file, with the format "{node1_name}, {node2_name}, {edge_weight}".
    file_label : String
        The path of the .csv file, with the format "{node_name}, {node_role}".
    file_model : String
        The path of the .csv file, with the format "{node_name}, {node_model}".
    file_port : String
        The path of the .csv file, with the format "{node_name}, {node_port}".
    num_role_kinds : int
        The number of role kinds in the network.
    low_dim : int
        The dimension that you want to embed the network in.
    ratio : double, in (0, 1)
        The ratio of the training set.

    """
    G3_n, G3_i, name2id, id2name = build_whole_network(file_network=file_network)
    H = AttributeFeatures(G3_i, name2id, id2name, file_model=file_model, file_port=file_port)\
        .get_attribute_features()
    S = CentralityFeatures(G3_i).get_centrality_features()
    S = S.T  # 8*439
    y, y_trans = read_node_label_matrix(G3_i, id2name, num_role_kinds=num_role_kinds, file_label=file_label)
    data = Dataset(y, y_trans, ratio)
    X1 = LearnBestX(G3_i, low_dim, data, S).run()
    X = np.hstack((X1, H))

    micro, macro = [], []
    print("\nratio=", ratio)
    for n in range(10):
        result = EvaluateSVM(data, X, y, y_trans).predict()
        f1_micro, f1_macro, accuracy = result[3], result[4], result[5]
        y_pred = result[6]
        print("The final result of the ", n+1, "th round：\nf1_micro=", f1_micro)
        print("f1_macro=", f1_macro)
        print("accuracy=", accuracy, "\n")
        micro.append(f1_micro)
        macro.append(f1_macro)
    print("Finish")


if __name__ == "__main__":
    # ----Net2----
    node_role_classification(file_network="../datasets/Net2.csv", file_label="../datasets/Net2_label.csv",
                             file_model=None, file_port="../datasets/Net2_port.csv",
                             num_role_kinds=3, ratio=0.5)

    # ----Net3----
    # node_role_classification(file_network="../datasets/Net3.csv", file_label="../datasets/Net3_label.csv",
    #                          file_model="../datasets/Net3_model.csv", file_port="../datasets/Net3_port.csv",
    #                          num_role_kinds=4, ratio=0.5)
