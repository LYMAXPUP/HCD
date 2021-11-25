# HCD
HCD is the **H**ierarchical **C**ommunity **D**iscovery method which can be implemented in large-scale IP bearer network to discover communities. This method firstly predicts the node role of the whole network to get the hierarchy and then detects the hierarchical community through semi-local expansion. The initial communities are overlapped, but you can optionally adjust the overlapped parts according to the geographical locations. The ultimate community possess the following characteristics:
* densely connecting in topology;
* having hierarchical structure inside; 
* preserving the integrity of the tree/ring structures.

The details of this method can be found in [paper](https://www.researchgate.net/publication/352728745_Hierarchical_community_discovery_for_multi-stage_IP_bearer_network_upgradation).
```
@article{liu2021hierarchical,
  title={Hierarchical community discovery for multi-stage IP bearer network upgradation},
  author={Liu, Yuan and Gu, Rentao and Yang, Zeyuan and Ji, Yuefeng},
  journal={Journal of Network and Computer Applications},
  pages={103151},
  year={2021},
  publisher={Elsevier}
}
```
## Requirements
The example is run in python3 (3.7.3):
* scikit-learn: 0.23.1
* numpy: 1.18.5
* networkx: 2.5
* pandas: 0.24.2

## Dataset Information
In the folder `/datasets`, all the datasets are partial IP bearer networks of China offered by local operator. 
* Net1(`418 nodes`) and Net2(`627 nodes`) are two metropolitan area network segments;
* Net3(`247 nodes`) is a backbone network segment.
* **!!!Note that all datasets are set as undirected networks.** </br>

![image](https://user-images.githubusercontent.com/53416615/126338085-0626ef3f-3b6c-4fa6-8244-d7d840a78605.png)

## Usage
The algorithm is divided into three independent parts. And you can run each part in `../main.py`:
1. `/networkCharacteristics`: Compute the main network characteristics of each dataset.
2. `/AEMMDW`: Node role classification. This is a model to predict the node role of the whole IP bear network.
3. `/SLHCD`: Community detection. This is a model to detect the appropreate hierarchical communities in IP bear network.

## (optional) Visualization
The method supports to output the community results in the `.gexf` file (open in gephi). [Gephi](https://gephi.org/) is a visualization and exploration software for all kinds of networks, which is much clear than **matplotlib** to show the network details especially when the network is large-scale.
![image](https://user-images.githubusercontent.com/53416615/126345841-611f0794-c703-4bc4-8c79-48b05248979c.png)
![image](https://user-images.githubusercontent.com/53416615/126345910-8cb20fe0-d1da-4b26-a6f3-b3155eb8cf43.png)
