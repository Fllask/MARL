# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:38:57 2022

@author: valla
"""
import os
import networkx as nx
import torch_geometric as pyg
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch_geometric.data import HeteroData,Batch
from torch_geometric.utils import to_undirected,add_self_loops
import torch_geometric.transforms as T
import numpy as np
import copy
import wandb
class ReplayBufferSingleAgent():
    def __init__(self,length,action_list,side_sup,side_b,fully_connected = False,device='cpu',use_mask = False):
        self.states =  [None]*length
        self.nstates =  [None]*length
        if use_mask:
            self.masks =[None]*length
            self.nmasks = [None]*length
            self.use_mask = True
        else:
            self.use_mask = False
        self.actions=torch.zeros(length,device=device,dtype=torch.long)
        self.rewards = torch.zeros(length,device=device)
        self.is_terminal = torch.zeros(length,device = device, dtype=bool)
        self.counter = 0
        self.length = length
        self.full = False
        self.device = device
        self.fully_connected=fully_connected
        
        self.agent_action_list =action_list
        self.agent_sides_sup=side_sup
        self.agent_sides_b=side_b
        
    def push(self,rid,state,action,nstate,reward,terminal = False,mask=None,nmask=None,last_only=True):
        #create an HeteroData object from a graph defined by an adjacency matrix
        if self.fully_connected:
            assert False, "Not implemented"
            #self.states[self.counter]=DenseGraph(state)
            #self.nstates[self.counter]=DenseGraph(nstate)
        else:
            self.states[self.counter]=create_sparse_graph(state,rid,self.agent_action_list,self.device,self.agent_sides_sup,self.agent_sides_b,last_only=last_only)
            self.states[self.counter].validate()
            if self.use_mask:
                self.masks[self.counter]=torch.tensor(mask,device=self.device,dtype=bool)
                if terminal:
                    self.nmasks[self.counter]=torch.zeros(0,device=self.device,dtype=bool)
                else:
                    self.nmasks[self.counter]=torch.tensor(nmask,device=self.device,dtype=bool)
            self.actions[self.counter]=action
            self.rewards[self.counter]=reward
            if not terminal:
                self.nstates[self.counter]=create_sparse_graph(nstate,rid,self.agent_action_list,self.device,self.agent_sides_sup,self.agent_sides_b,last_only=last_only)
                self.nstates[self.counter].validate()
            else:
                self.nstates[self.counter]=create_sparse_graph(None,None,self.agent_action_list,self.device,state.n_side_oriented_sup,state.n_side_oriented,empty=True,last_only=last_only)
                self.nstates[self.counter].validate()
                self.is_terminal[self.counter]=True
            
        self.counter = self.counter+1
        if  self.counter==self.length:
            self.counter=0
            self.full = True
            
    def sample(self,batchsize):
        #extract a batch from the replay buffer
        
        if self.full:
            last_idx = self.length
            perm = torch.randperm(self.length,device = self.device)
            idx = perm[:batchsize]
        else:
            assert self.counter >= batchsize, "Not enough samples"
            last_idx = self.counter
            perm = torch.randperm(self.counter,device = self.device)
            idx = perm[:batchsize]
            
        states_minibatch = Batch.from_data_list([self.states[i] for i in idx])
        nstates_minibatch = Batch.from_data_list([self.nstates[i] for i in idx])
        # if self.fully_connected:
        #     states_minibatch = states_batch.index_select(idx)
        #     nstates_minibatch = nstates_batch.index_select(idx)
        # else:
            
        #     states_minibatch = states_batch.index_select(idx)
        #     nstates_minibatch = nstates_batch.index_select(idx)
        actions_minibatch = self.actions[idx]
        rewards_minibatch = self.rewards[idx]
        terminal_minibatch = self.is_terminal[idx]
        if self.use_mask:
            mask_minibatch = [self.masks[i] for i in idx]
            nmask_minibatch = [self.nmasks[i] for i in idx]
            return (states_minibatch,actions_minibatch,nstates_minibatch,rewards_minibatch,terminal_minibatch,mask_minibatch,nmask_minibatch)
        return (states_minibatch,actions_minibatch,nstates_minibatch,rewards_minibatch,terminal_minibatch)
def create_dense_graph(sim,robot_to_act,action_list,device,oriented_sides_sup,oriented_sides_b,last_only=True,empty=False):
    graph = HeteroData()
    if empty:
        graph['ground'].x = torch.zeros((0,3),device=device,dtype=torch.float)
        graph['block'].x = torch.zeros((0,3),device=device,dtype=torch.float)
        graph['robot'].x =  torch.zeros(0,6,device=device,dtype=torch.float)
        graph['side_sup'].x = torch.zeros(0,2,device=device,dtype=torch.float)
        graph['new_block'].x = torch.zeros(0,2,device=device,dtype=torch.float)
        graph['ground','edge','ground'].edge_index =  torch.zeros((2,0),device=device,dtype=torch.long)
        graph['ground','edge','block'].edge_index =  torch.zeros((2,0),device=device,dtype=torch.long)
        graph['ground','edge','robot'].edge_index =  torch.zeros((2,0),device=device,dtype=torch.long)
        graph['ground','edge','side_sup'].edge_index =  torch.zeros((2,0),device=device,dtype=torch.long)
        graph['block','edge','block'].edge_index =  torch.zeros((2,0),device=device,dtype=torch.long)
        graph['block','edge','robot'].edge_index =  torch.zeros((2,0),device=device,dtype=torch.long)
        graph['block','edge','side_sup'].edge_index =  torch.zeros((2,0),device=device,dtype=torch.long)
        graph['side_sup','edge','new_block'].edge_index =  torch.zeros((2,0),device=device,dtype=torch.long)
        graph['new_block','edge','robot'].edge_index =  torch.zeros((2,0),device=device,dtype=torch.long)
        graph['robot','holds','block'].edge_index =  torch.zeros((2,0),device=device,dtype=torch.long)
        graph = T.AddSelfLoops()(graph)
        graph = T.ToUndirected()(graph)
        return graph
    else:
        graph['ground'].x = torch.tensor(np.hstack([np.expand_dims(-1-sim.type_id[(sim.type_id > sim.empty_id) & (sim.type_id <0)],1),
                                         sim.graph.grounds[sim.graph.active_grounds,2:]]),device=device,dtype=torch.float) #features:[groundtype,coords]
        graph['block'].x = torch.tensor(sim.graph.blocks[sim.graph.active_blocks,1:],device=device,dtype=torch.float) # features:[btype,coords]
        if sim.ph_mod.last_res is None or sim.ph_mod.last_res.x is None:
            graph['robot'].x = torch.zeros((sim.ph_mod.nr,6),device=device,dtype=torch.float) #features: force applied
        else:
            graph['robot'].x = torch.tensor(sim.ph_mod.last_res.x[:sim.ph_mod.nr*6].reshape(sim.ph_mod.nr,6),device=device,dtype=torch.float) #features: force applied
        ng = graph['ground'].x.shape[0]
        nb = graph['block'].x.shape[0]
        
        
        
        if 'Ph' in action_list:
            #define the action tree:
                
                #6 nodes are created connected to each ground or block
            if last_only:
                #
                
                placed_block_typeid = sim.type_id[sim.type_id > sim.empty_id]
                if placed_block_typeid[-1]<0:
                    type_sup = -placed_block_typeid[-1]-1
                else:
                    type_sup=placed_block_typeid[-1]-sim.empty_id-1
                side_sup = np.zeros((np.sum(oriented_sides_sup[type_sup]),2))
                node_id = 0
                for side_ori, n_sides in enumerate(oriented_sides_sup[type_sup]):
                    side_sup[node_id:node_id+n_sides]=np.array([[side_ori]*n_sides,np.arange(n_sides)]).T
                    node_id +=n_sides
                graph['side_sup'].x = torch.tensor(side_sup,device = device,dtype=torch.float)
                
                if not np.any(sim.graph.active_blocks):
                    graph['ground','edge','side_sup'].edge_index=torch.vstack([(sim.graph.n_ground-1)*torch.ones(graph['side_sup'].x.shape[0],dtype=torch.long,device=device),
                                                                                      torch.arange(graph['side_sup'].x.shape[0],device=device)])
                    graph['block','edge','side_sup'].edge_index=torch.zeros((2,0),dtype=torch.long,device=device)
                else:
                    graph['block','edge','side_sup'].edge_index=torch.vstack([(sim.graph.n_blocks-1)*torch.ones(graph['side_sup'].x.shape[0],dtype=torch.long,device=device),
                                                                                     torch.arange(graph['side_sup'].x.shape[0],device=device)])
                    graph['ground','edge','side_sup'].edge_index=torch.zeros((2,0),dtype=torch.long,device=device)
            else:
                node_id = 0
                placed_block_typeid = sim.type_id[sim.type_id > sim.empty_id]
                graph['ground','edge','side_sup'].edge_index=torch.zeros((2,0),dtype=torch.long,device=device)
                graph['block','edge','side_sup'].edge_index=torch.zeros((2,0),dtype=torch.long,device=device)
                graph['side_sup'].x = torch.zeros((0,2),device = device,dtype=torch.float)
                ng=0
                for sup_id, support in enumerate(placed_block_typeid):
                    type_sup = support-sim.empty_id-1
                    side_sup = np.zeros((np.sum(oriented_sides_sup[type_sup]),2))
                    side_id = 0
                    for side_ori, n_sides in enumerate(oriented_sides_sup[type_sup]):
                        side_sup[side_id:side_id+n_sides]=np.array([[side_ori]*n_sides,np.arange(n_sides)]).T
                        side_id +=n_sides
                    
                    side_sup = torch.tensor(side_sup,device = device,dtype=torch.float)
                    
                    if support<0:
                        graph['ground','edge','side_sup'].edge_index=torch.hstack([graph['ground','edge','side_sup'].edge_index,
                                                                                          torch.vstack([sup_id*torch.ones(side_sup.shape[0],dtype=torch.long,device=device),
                                                                                                        torch.arange(graph['side_sup'].x.shape[0],graph['side_sup'].x.shape[0]+side_sup.shape[0],device=device)])])
                        ng+=1
                    else:
                        graph['block','edge','side_sup'].edge_index=torch.hstack([graph['block','edge','side_sup'].edge_index,
                                                                                          torch.vstack([(sup_id-ng)*torch.ones(side_sup.shape[0],dtype=torch.long,device=device),
                                                                                                        torch.arange(graph['side_sup'].x.shape[0],graph['side_sup'].x.shape[0]+side_sup.shape[0],device=device)])])
                    graph['side_sup'].x = torch.vstack([graph['side_sup'].x,side_sup])
                    node_id+=side_sup.shape[0]
            new_block = []
            put_against = []
            n_actions = 0
            for blocktype in range(oriented_sides_b.shape[0]):
                for node_id, side_ori in enumerate(graph['side_sup'].x[:,0]):
                    new_block.append(np.hstack([blocktype*np.ones((oriented_sides_b[blocktype,int(side_ori)],1)),
                                                np.arange(oriented_sides_b[blocktype,int(side_ori)]).reshape(-1,1)]))
                    put_against.append(np.vstack([node_id*np.ones(oriented_sides_b[blocktype,int(side_ori)]),
                                                  np.arange(n_actions,n_actions+oriented_sides_b[blocktype,int(side_ori)])]).reshape(2,-1))
                    n_actions+=oriented_sides_b[blocktype,int(side_ori)]
                                            
            new_block = np.vstack(new_block)
            graph['new_block'].x = torch.tensor(new_block,device = device,dtype=torch.float)#features: [blocktype, sideid]
            
            graph['side_sup','edge','new_block'].edge_index = torch.tensor(np.hstack(put_against),device = device,dtype=torch.long)
            
            graph['robot','edge','new_block'].edge_index = torch.vstack([robot_to_act*torch.ones(graph['new_block'].x.shape[0],device=device,dtype=torch.long),
                                                                           torch.arange(graph['new_block'].x.shape[0],device=device)
                                                                          ])
            
            bid_held, = np.nonzero(sim.graph.blocks[sim.graph.active_blocks][:,0]>-1)
            graph['robot','holds','block'].edge_index = torch.vstack([
                                                            torch.tensor(sim.graph.blocks[sim.graph.active_blocks][bid_held,0],device=device,dtype=torch.long),
                                                            torch.tensor(bid_held,device=device)])
            
            graph['block','edge','block'].edge_index = torch.vstack([torch.arange(graph['block'].x.shape[0],device=device,dtype=torch.long).repeat(torch.arange(graph['block'].x.shape[0])),
                                                                     torch.arange(graph['block'].x.shape[0],device=device,dtype=torch.long).repeat_interleave(torch.arange(graph['block'].x.shape[0]))
                                                                     ])
            
            graph['ground','edge','block'].edge_index = torch.vstack([torch.arange(graph['ground'].x.shape[0],device=device,dtype=torch.long).repeat(torch.arange(graph['block'].x.shape[0])),
                                                                     torch.arange(graph['block'].x.shape[0],device=device,dtype=torch.long).repeat_interleave(torch.arange(graph['ground'].x.shape[0]))
                                                                     ])
            graph['ground','edge','robot'].edge_index = torch.vstack([torch.arange(graph['ground'].x.shape[0],device=device,dtype=torch.long).repeat(torch.arange(graph['robot'].x.shape[0])),
                                                                     torch.arange(graph['robot'].x.shape[0],device=device,dtype=torch.long).repeat_interleave(torch.arange(graph['ground'].x.shape[0]))
                                                                     ])
            graph['block','edge','robot'].edge_index = torch.vstack([torch.arange(graph['block'].x.shape[0],device=device,dtype=torch.long).repeat(torch.arange(graph['robot'].x.shape[0])),
                                                                     torch.arange(graph['robot'].x.shape[0],device=device,dtype=torch.long).repeat_interleave(torch.arange(graph['block'].x.shape[0]))
                                                                     ])
            graph = T.AddSelfLoops()(graph)
            graph = T.ToUndirected()(graph)
            return graph
            
def create_sparse_graph(sim,robot_to_act,action_list,device,oriented_sides_sup,oriented_sides_b,last_only=True,empty=False,one_hot=False):
        graph = HeteroData()
        if empty:
            graph['ground'].x = torch.zeros((0,oriented_sides_sup.shape[0]-oriented_sides_b.shape[0]+2),device=device,dtype=torch.float)
            graph['block'].x = torch.zeros((0,oriented_sides_b.shape[0]+2),device=device,dtype=torch.float)
            graph['robot'].x =  torch.zeros(0,6,device=device,dtype=torch.float)
            graph['side_sup'].x = torch.zeros(0,7,device=device,dtype=torch.float)
            graph['new_block'].x = torch.zeros(0,oriented_sides_b.shape[0]+1+int('S' in action_list),device=device,dtype=torch.float)
            graph['block','touches','ground'].edge_index = torch.zeros((2,0),device=device,dtype=torch.long)
            graph['block','touches','block'].edge_index = torch.zeros((2,0),device=device,dtype=torch.long)
            graph['ground','touches','block'].edge_index= torch.zeros((2,0),device=device,dtype=torch.long)
            graph['side_sup','put_against','new_block'].edge_index= torch.zeros((2,0),device=device,dtype=torch.long)
            graph['ground','action_desc','side_sup'].edge_index = torch.zeros((2,0),device=device,dtype=torch.long)
            graph['block','action_desc','side_sup'].edge_index = torch.zeros((2,0),device=device,dtype=torch.long)
            graph['robot','choses','new_block'].edge_index = torch.zeros((2,0),device=device,dtype=torch.long)
            graph['robot','holds','block'].edge_index = torch.zeros((2,0),device=device,dtype=torch.long)
            graph['robot', 'reaches', 'block'].edge_index = torch.zeros((2,0),device=device,dtype=torch.long)
            graph['robot', 'reaches', 'ground'].edge_index = torch.zeros((2,0),device=device,dtype=torch.long)
            graph['robot', 'communicate', 'robot'].edge_index = torch.zeros((2,0),device=device,dtype=torch.long)
            graph = T.AddSelfLoops()(graph)
            graph = T.ToUndirected(merge=False)(graph)
            return graph
        
        type_grounds = -1-sim.type_id[(sim.type_id > sim.empty_id) & (sim.type_id <0)]
        pos_grounds = sim.graph.grounds[sim.graph.active_grounds,2:]
        graph['ground'].x = F.one_hot(torch.tensor(type_grounds,device=device,dtype=torch.long),oriented_sides_sup.shape[0]-oriented_sides_b.shape[0]+2).to(torch.float)
        graph['ground'].x[:,-2:]=torch.tensor(pos_grounds,device=device)
        
        type_blocks = sim.graph.blocks[sim.graph.active_blocks,1]
        pos_blocks = sim.graph.blocks[sim.graph.active_blocks,-2:]
        graph['block'].x = F.one_hot(torch.tensor(type_blocks,device=device,dtype=torch.long),oriented_sides_b.shape[0]+2).to(torch.float)
        graph['block'].x[:,-2:]=torch.tensor(pos_blocks,device=device)
        
        if sim.ph_mod.last_res is None or sim.ph_mod.last_res.x is None:
            graph['robot'].x = torch.zeros((sim.ph_mod.nr,6),device=device,dtype=torch.float) #features: force applied
        else:
            graph['robot'].x = torch.tensor(sim.ph_mod.last_res.x[:sim.ph_mod.nr*6].reshape(sim.ph_mod.nr,6),device=device,dtype=torch.float) #features: force applied
        ng = graph['ground'].x.shape[0]
        nb = graph['block'].x.shape[0]
        
        
        
        if ('Ph' in action_list) or ('P' in action_list):
            #define the action tree:
                
                #6 nodes are created connected to each ground or block
            if last_only:
                #
                placed_block_typeid = sim.type_id[sim.type_id > sim.empty_id]
                if placed_block_typeid[-1]<0:
                    type_sup = -placed_block_typeid[-1]-1
                else:
                    type_sup=placed_block_typeid[-1]-sim.empty_id-1
                side_sup = np.zeros((np.sum(oriented_sides_sup[type_sup]),7))
                node_id = 0
                for side_ori, n_sides in enumerate(oriented_sides_sup[type_sup]):
                    side_sup[node_id:node_id+n_sides,side_ori]=1
                    side_sup[node_id:node_id+n_sides,-1]=np.arange(n_sides)
                    node_id +=n_sides
                graph['side_sup'].x = torch.tensor(side_sup,device = device,dtype=torch.float)
                
                if not np.any(sim.graph.active_blocks):
                    graph['ground','action_desc','side_sup'].edge_index=torch.vstack([(sim.graph.n_ground-1)*torch.ones(graph['side_sup'].x.shape[0],dtype=torch.long,device=device),
                                                                                      torch.arange(graph['side_sup'].x.shape[0],device=device)])
                    graph['block','action_desc','side_sup'].edge_index=torch.zeros((2,0),dtype=torch.long,device=device)
                else:
                    graph['block','action_desc','side_sup'].edge_index=torch.vstack([(sim.graph.n_blocks-1)*torch.ones(graph['side_sup'].x.shape[0],dtype=torch.long,device=device),
                                                                                     torch.arange(graph['side_sup'].x.shape[0],device=device)])
                    graph['ground','action_desc','side_sup'].edge_index=torch.zeros((2,0),dtype=torch.long,device=device)
            else:
                node_id = 0
                placed_block_typeid = sim.type_id[sim.type_id > sim.empty_id]
                graph['ground','action_desc','side_sup'].edge_index=torch.zeros((2,0),dtype=torch.long,device=device)
                graph['block','action_desc','side_sup'].edge_index=torch.zeros((2,0),dtype=torch.long,device=device)
                graph['side_sup'].x = torch.zeros((0,7),device = device,dtype=torch.float)
                ng=0
                for sup_id, support in enumerate(placed_block_typeid):
                    type_sup = support-sim.empty_id-1
                    side_sup = np.zeros((np.sum(oriented_sides_sup[type_sup]),7))
                    side_id = 0
                    for side_ori, n_sides in enumerate(oriented_sides_sup[type_sup]):
                        side_sup[side_id:side_id+n_sides,side_ori]=1
                        side_sup[side_id:side_id+n_sides,-1]=np.arange(n_sides)
                        side_id +=n_sides
                    
                    side_sup = torch.tensor(side_sup,device = device,dtype=torch.float)
                    
                    if support<0:
                        graph['ground','action_desc','side_sup'].edge_index=torch.hstack([graph['ground','action_desc','side_sup'].edge_index,
                                                                                          torch.vstack([sup_id*torch.ones(side_sup.shape[0],dtype=torch.long,device=device),
                                                                                                        torch.arange(graph['side_sup'].x.shape[0],graph['side_sup'].x.shape[0]+side_sup.shape[0],device=device)])])
                        ng+=1
                    else:
                        graph['block','action_desc','side_sup'].edge_index=torch.hstack([graph['block','action_desc','side_sup'].edge_index,
                                                                                          torch.vstack([(sup_id-ng)*torch.ones(side_sup.shape[0],dtype=torch.long,device=device),
                                                                                                        torch.arange(graph['side_sup'].x.shape[0],graph['side_sup'].x.shape[0]+side_sup.shape[0],device=device)])])
                    graph['side_sup'].x = torch.vstack([graph['side_sup'].x,side_sup])
                    node_id+=side_sup.shape[0]
            new_block = []
            put_against = []
            n_actions = 0
            for blocktype in range(oriented_sides_b.shape[0]):
                for node_id, side_ori in enumerate(torch.nonzero(graph['side_sup'].x[:,:-1])[:,1]):
                    blocktype_oh = np.zeros((oriented_sides_b[blocktype,int(side_ori)],oriented_sides_b.shape[0]+1+int('S' in action_list)))
                    blocktype_oh[:,blocktype]=1
                    blocktype_oh[:,-1]=np.arange(oriented_sides_b[blocktype,int(side_ori)])
                    new_block.append(blocktype_oh)
                    put_against.append(np.vstack([node_id*np.ones(oriented_sides_b[blocktype,int(side_ori)]),
                                                  np.arange(n_actions,n_actions+oriented_sides_b[blocktype,int(side_ori)])]).reshape(2,-1))
                    n_actions+=oriented_sides_b[blocktype,int(side_ori)]
                                            
            new_block = np.vstack(new_block)
            graph['new_block'].x = torch.tensor(new_block,device = device,dtype=torch.float)#features: [blocktype, sideid]
            
            graph['side_sup','put_against','new_block'].edge_index = torch.tensor(np.hstack(put_against),device = device,dtype=torch.long)
            
            
        if 'S' in action_list:
            #model S as a block only connected to the robot
            stay_block = torch.zeros((1,oriented_sides_b.shape[0]+1+int('S' in action_list)),dtype=torch.float,device=device)
            stay_block[0,-2]=1
            graph['new_block'].x = torch.vstack([graph['new_block'].x,stay_block])
        graph['robot','choses','new_block'].edge_index = torch.vstack([robot_to_act*torch.ones(graph['new_block'].x.shape[0],device=device,dtype=torch.long),
                                                                       torch.arange(graph['new_block'].x.shape[0],device=device)
                                                                      ])
        if 'L' in action_list:
            assert False, 'not implemented'
            graph['leave'].x = torch.zeros(1,0)#features: []
            graph['robot','choses','leave'].edge_index = torch.tensor([robot_to_act,0],device=device).unsqueeze(1) #features: []
            
        bid_held, = np.nonzero(sim.graph.blocks[sim.graph.active_blocks][:,0]>-1)
        graph['robot','holds','block'].edge_index = torch.vstack([
                                                        torch.tensor(sim.graph.blocks[sim.graph.active_blocks][bid_held,0],device=device,dtype=torch.long),
                                                        torch.tensor(bid_held,device=device)]) #features: []
        graph['block','touches','block'].edge_index = torch.tensor(sim.graph.edges_index_bb[:,sim.graph.active_edges_bb],device=device,dtype=torch.long)
        graph['ground','touches','block'].edge_index = torch.tensor(sim.graph.edges_index_gb[:,sim.graph.active_edges_gb],device=device,dtype=torch.long)
        graph['block','touches','ground'].edge_index = torch.tensor(sim.graph.edges_index_bg[:,sim.graph.active_edges_bg],device=device,dtype=torch.long)
        
        graph['robot','reaches', 'block'].edge_index = torch.vstack([torch.arange(graph['robot'].x.shape[0],device=device).repeat(graph['block'].x.shape[0]),
                                                          torch.repeat_interleave(torch.arange(graph['block'].x.shape[0],device=device),graph['robot'].x.shape[0])]) #features: []
        graph['robot','reaches','ground'].edge_index =torch.vstack([torch.arange(graph['robot'].x.shape[0],device=device).repeat(ng),
                                                          torch.repeat_interleave(torch.arange(ng,device=device),graph['robot'].x.shape[0])])#features: []
        graph['robot','communicate','robot'].edge_index=torch.vstack([torch.arange(graph['robot'].x.shape[0],device=device).repeat(graph['robot'].x.shape[0]),
                                                          torch.repeat_interleave(torch.arange(graph['robot'].x.shape[0],device=device),graph['robot'].x.shape[0])]) #features: []
        graph = T.AddSelfLoops()(graph)
        graph = T.ToUndirected()(graph)
        return graph
def build_hetero_GNN(config,simulator,rid,action_list,sides_sup,sides_b, empty=False):
    sample_data = create_sparse_graph(simulator,rid,action_list,config['torch_device'],sides_sup,sides_b,last_only=False)
    if config['GNN_arch']=='ResNet':
        model =  pyg.nn.to_hetero_with_bases(GATskip(config),sample_data.metadata(),5,
                                             in_channels = {'x':16},debug=False).to(config['torch_device'])
       
    elif config['GNN_arch']=='GAT':
        model = pyg.nn.to_hetero_with_bases(pyg.nn.GAT(-1, config['GNN_hidden_dim'], config['GNN_n_layers'], 1,v2 =True),sample_data.metadata(),5,
                                            in_channels = {'x':16}).to(config['torch_device'])
        
    else:
        assert False, "Not implemented"
    if sample_data is not None:
        with torch.no_grad():  # Initialize lazy modules.
            out = model(sample_data.x_dict, sample_data.edge_index_dict)
    return model
class GATskip(torch.nn.Module):
    def __init__(self,
                 config,
                 use_wandb=False):
        super().__init__()
        #unpck the config
        n_layers = config['GNN_n_layers']
        internal_dims = config['GNN_hidden_dim']
        n_heads = config['GNN_att_head']
        device = config['torch_device']
        self.convs = torch.nn.ModuleList([pyg.nn.ResGatedGraphConv((-1, -1), internal_dims, edge_dim=-1,add_self_loops=False,heads = n_heads) for i in range(n_layers-1)])
        self.conv_out = pyg.nn.ResGatedGraphConv((-1, -1), 1,edge_dim=-1, add_self_loops=False)
        self.use_wandb=use_wandb
        
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x=x.relu()
        x = self.conv_out(x,edge_index)
        
        return x
# class GAT(torch.nn.Module):
#     def __init__(self,
#                  config,
#                  use_wandb=False):
#         super().__init__()
#         #unpck the config
#         n_layers = config['GNN_n_layers']
#         internal_dims = config['GNN_hidden_dim']
#         n_heads = config['GNN_att_head']
#         device = config['torch_device']
#         self.convs = torch.nn.ModuleList([pyg.nn.ResGatedGraphConv((-1, -1), internal_dims, edge_dim=-1,add_self_loops=False,heads = n_heads) for i in range(n_layers-1)])
#         self.conv_out = pyg.nn.GATv2Conv((-1, -1), 1,edge_dim=-1, add_self_loops=False)
#         self.use_wandb=use_wandb
        
#     def forward(self, x, edge_index):
#         for conv in self.convs:
#             x = conv(x, edge_index)
#             x=x.relu()
#         x = self.conv_out(x,edge_index)
        
#         return x
class SACOptimizerGeometricNN():
    def __init__(self,simulator,rid,action_list,sides_sup,sides_b,policy,config,use_wandb=False,log_freq =None,name=""):
        self.use_wandb = use_wandb
        self.pol = policy
        self.log_freq = log_freq
        self.Qs = [build_hetero_GNN(config,simulator,rid,action_list,sides_sup,sides_b)
                  for i in range(2)]
        
        self.target_Qs = copy.deepcopy(self.Qs)
        lr = config['opt_lr']
        wd = config['opt_weight_decay']
        self.max_norm = config['opt_max_norm']
        self.pol_over_val = config['opt_pol_over_val']
        self.tau = config['opt_tau']
        self.exploration_factor=config['opt_exploration_factor']
        self.Qval_reduction = config['opt_Q_reduction']
        self.lbound = config['opt_lower_bound_Vt']
        self.value_clip = config['opt_value_clip']
        self.opt_pol = torch.optim.NAdam(self.pol.parameters(),lr=lr*self.pol_over_val,weight_decay=wd)
        self.opt_Q = [torch.optim.NAdam(Q.parameters(),lr=lr,weight_decay=wd) for Q in self.Qs]
        self.target_entropy = config['opt_target_entropy']
        
        self.alpha = torch.tensor(config.get('opt_init_alpha') or 1.,device = config['torch_device'],requires_grad=True)
        self.opt_alpha = torch.optim.NAdam([self.alpha],lr=config.get('opt_lr_alpha') or 1e-3)
        self.beta = 0.5 #same as in paper
        self.clip_r = 0.5 #same as in paper
        self.step = 0
        self.name = name
        self.device = config['torch_device']
    def optimize(self,state,actionid,rewards,nstate,terminal,gamma,masks = None,nmasks=None):
        state = state.to(self.device)
        nstate = nstate.to(self.device)
        actionid = actionid.to(self.device)
        rewards = rewards.to(self.device)
        terminal = terminal.to(self.device)
        batches,n_actions = torch.unique(state['new_block'].batch,return_counts=True)
        batch_size = actionid.shape[0]
        
        pol_graph = self.pol(state.x_dict,state.edge_index_dict)
        if masks is None:
            prob = pyg.utils.softmax(pol_graph['new_block'],index = state['new_block'].batch)
        else:
            mask = torch.cat(masks)
            prob = pyg.utils.softmax(pol_graph['new_block'][mask],index = state['new_block'].batch[mask])
        Qvals = [self.Qs[i](state.x_dict,state.edge_index_dict)['new_block'] for i in range(2)]
            
        entropy = (-prob*torch.log(prob+1e-30)).sum()/batch_size
        
        tV = F.relu(self.alpha)*self.target_entropy*torch.ones(batch_size,device = self.device,dtype=torch.float)
        with torch.no_grad():
            if nstate['new_block'].x.shape[0] >0:
                nonterminal, inv =  torch.unique(nstate['new_block'].batch,return_inverse=True)
                npol_graph = self.pol(nstate.x_dict,nstate.edge_index_dict)
                if nmasks is None:
                    nprob = pyg.utils.softmax(npol_graph['new_block'],index = nstate['new_block'].batch)
                else:
                    nmask = torch.cat(nmasks)
                    nprob = pyg.utils.softmax(npol_graph['new_block'][nmask],index = nstate['new_block'].batch[nmask])
                nentropy = (-nprob*torch.log(nprob+1e-30)).sum()/torch.unique(nstate['new_block'].batch).shape[0]
                tQvals = [self.target_Qs[i](nstate.x_dict,nstate.edge_index_dict)['new_block'] for i in range(2)]
                tV_double=[]
                for tQval in tQvals:
                    tVi = torch.zeros(nonterminal.shape,device=self.device)
                    if nmasks is None:
                        tVi.scatter_(0,inv, (nprob*tQval).squeeze(),reduce='add')
                    else:
                        tVi.scatter_(0,inv[nmask], (nprob*tQval[nmask]).squeeze(),reduce='add')
                    tV_double.append(tVi+F.relu(self.alpha)*nentropy)
                tV_double = torch.stack(tV_double,dim=1)
            
                if self.Qval_reduction == 'min':
                    tV[nonterminal],_=torch.min(tV_double,dim=1)
                elif self.Qval_reduction == 'mean':
                    assert False, 'not implemented'
                    tV = torch.mean(tV_double,dim=1)          
        #the entropy bonus is kept in the terminal state value, as its value has no upper bound
        
        
        #clamp the target value 
        tV = torch.clamp(tV,min=self.lbound)
        #update the critics
        losses = torch.zeros(2,device = self.device)
        mean_sampled_Qs = torch.zeros(2,device = self.device)
        for i in range(2):
            self.opt_Q[i].zero_grad()
            sampled_Q = Qvals[i][state['new_block'].ptr[:-1]+actionid].squeeze()
            mean_sampled_Qs[i] = sampled_Q.mean()
            loss = F.huber_loss(sampled_Q,(rewards+gamma*tV).detach())
            if self.value_clip:
                target_sampled_Q = tQvals[i][torch.arange(Qvals[i].shape[0]),actionid]
                loss_clipped = F.mse_loss(target_sampled_Q+torch.clamp(sampled_Q-target_sampled_Q,-self.clip_r,self.clip_r),
                                         (rewards+gamma*tV).detach())
                loss = torch.max(loss,loss_clipped)
            loss.backward()
            self.opt_Q[i].step()
            losses[i]=loss
      
        if self.Qval_reduction=='min':
            minQvals,_ = torch.min(torch.stack(Qvals,dim=1),dim=1)
        else:
            minQvals = torch.mean(torch.stack(Qvals,dim=1),dim=1)
        _,argmax= torch.max(minQvals,dim=1)
        if masks is None:
            l_p = (-F.relu(self.alpha)*entropy-(minQvals.detach()*prob).sum(dim=1)).mean()
        else:
            l_p = (-F.relu(self.alpha)*entropy-(minQvals.detach()[mask]*prob).sum(dim=1)).mean()
        self.opt_pol.zero_grad()
        l_p.backward()
        if self.max_norm is not None:
            norm_p = torch.nn.utils.clip_grad_norm_(self.pol.parameters(),self.max_norm)
        self.opt_pol.step()
        #update alpha
        l_alpha = (entropy.detach().mean()-self.target_entropy)*F.elu(self.alpha)
        
        self.opt_alpha.zero_grad()
        l_alpha.backward()
        self.opt_alpha.step()
        #update the target
        for i in range(2):
            sd_target = self.target_Qs[i].state_dict()
            sd = self.Qs[i].state_dict()
            for key in sd_target:
                sd_target[key]= (1-self.tau)*sd_target[key]+self.tau*sd[key]
            self.target_Qs[i].load_state_dict(sd_target)
        #print(f"total: {t11-t00}")
        if self.use_wandb and self.step % self.log_freq == 0:
            
            wandb.log({self.name+"l_p": l_p.detach().cpu().numpy(),
                       self.name+"rewards":rewards.detach().mean().cpu().numpy(),
                       self.name+"best_action": argmax.cpu().numpy(),
                       self.name+"Qval_0":mean_sampled_Qs[0].detach().cpu().numpy(),
                       self.name+"Qval_1":mean_sampled_Qs[1].detach().cpu().numpy(),
                       self.name+'policy_grad_norm':norm_p.detach().cpu().numpy(),
                       self.name+"V_target":tV.detach().mean().cpu().numpy(),
                       self.name+'policy_entropy':entropy.detach().mean().cpu().numpy(),
                       self.name+"Qloss_0":losses[0].detach().cpu().numpy(),
                       self.name+"Qloss_1":losses[1].detach().cpu().numpy(),
                       self.name+"alpha": self.alpha.detach().cpu().numpy(),
                       self.name+"l_alpha": l_alpha.detach().cpu().numpy(),
                      },step=self.step)
            #wandb.watch(self.model)
        self.step+=1
        return l_p.detach().cpu().numpy()
    def save(self,log_dir,name):
        torch.save(self.pol.state_dict(),os.path.join(log_dir,f'{name}_pol.h5'))
        [torch.save(self.Qs[i].state_dict(),os.path.join(log_dir,f'{name}_Q_{i}.h5')) for i in range(2)]
        [torch.save(self.target_Qs[i].state_dict(),os.path.join(log_dir,f'{name}_targetQ_{i}.h5')) for i in range(2)]
        torch.save(self.alpha,os.path.join(log_dir,f'{name}_alpha.h5'))
        torch.save(self.opt_alpha.state_dict(),os.path.join(log_dir,f'{name}_opt_alpha.h5'))
        torch.save(self.opt_pol.state_dict(),os.path.join(log_dir,f'{name}_opt_pol.h5'))
        [torch.save(self.opt_Q[i].state_dict(),os.path.join(log_dir,f'{name}_opt_Q_{i}.h5')) for i in range(2)]
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = pyg.nn.GATv2Conv((-1, -1), hidden_channels, edge_dim=-1,add_self_loops=False)
        self.lin1 = pyg.nn.Linear(-1, hidden_channels)
        self.conv2 = pyg.nn.GATv2Conv((-1, -1), out_channels,edge_dim=-1, add_self_loops=False)
        self.lin2 = pyg.nn.Linear(-1, out_channels)
    def forward(self, x, edge_index,edge_attr):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        
        return x





        
    