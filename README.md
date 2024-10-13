# ADMP-GNN: ADAPTIVE DEPTH MESSAGE PASSING GNN

## Overview

This repository contains python codes and datasets necessary to run the proposed ADMP-GNN approach, a novel framework that dynamically adjusts the number of message-passing layers for each node, leading to enhanced performance.


## Requirements

Code is written in Python 3.6 and requires:
- PyTorch
- Torch Geometric
- NetworkX

To run a GCN using ST settings, and run the learning the centrality based policy

E.g: Degree Centrality, and number of cluster is 5 
```bash
cd ./ADMPGNN/
python /home/yassine/Projects/ADMPGNN/main.py --dataset "Cora" 
python /home/yassine/Projects/ADMPGNN/inference_centrality.py --dataset "Cora" --centrality "DEGREE" --num_clusters 5 
```

E.g: Kcore Centrality, and number of cluster is 5 
```bash
cd ./ADMPGNN/
python /home/yassine/Projects/ADMPGNN/main.py --dataset "Cora" 
python /home/yassine/Projects/ADMPGNN/inference_centrality.py --dataset "Cora" --centrality "KCORE" --num_clusters 5 
```


E.g: PageRank Centrality, and number of cluster is 5 
```bash
cd ./ADMPGNN/
python /home/yassine/Projects/ADMPGNN/main.py --dataset "Cora" 
python /home/yassine/Projects/ADMPGNN/inference_centrality.py --dataset "Cora" --centrality "PAGERANK" --num_clusters 5 
```

E.g: Path Cout. Centrality, and number of cluster is 5 
```bash
cd ./ADMPGNN/
python /home/yassine/Projects/ADMPGNN/main.py --dataset "Cora" 
python /home/yassine/Projects/ADMPGNN/inference_centrality.py --dataset "Cora" --centrality "DEPTH" --num_clusters 5 
```