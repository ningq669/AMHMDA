# AMHMDA

AMHMDA is a novel attention aware multi-view similarity networks and hyper-graph learning for miRNA-disease associations identification. AMHMDA consists of three steps to realization of miRNA-disease associations identification. First, multiple similarity networks are constructed to obtain similarity information. Then, some hypernodes are introduced to construct heterogeneous hyper-graph. Finally, a layer-level attention is used to fuse node representations. 

# Requirements
  * Python 3.7 or higher
  * PyTorch 1.8.0 or higher
  * torch-geometric 2.0.4
  * GPU (default)

# Data
  * Download associated data and similarity data.
  * Multiple similarity calculations are detailed in the supplementary material.

# Running  the Code
  * Execute ```python main.py``` to run the code.
  * Parameter state='valid'. Start the 5-fold cross validation training model.
  * Parameter state='test'. Start the independent testing.

# Note
```
 1.Torch-geometric has a strong dependency, so it is recommended to install a matching version.
 2.The trained model are stored in folder named cross valid . You can import directly to implement valid and test.
```
