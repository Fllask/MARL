# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:38:57 2022

@author: valla
"""

import torch_geometric as pyg
from torch_geometric.data import HeteroData
class Replay_buffer(pyg.HeteroData):
    def __init__(self,length):
        super.__init__()
        self.n_graphs
class Sparse_graph(pyg.HeteroData):
    def __init__(self):
        super.__init__()
    
    