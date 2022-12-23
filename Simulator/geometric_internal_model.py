# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:38:57 2022

@author: valla
"""
import networkx as nx
import torch_geometric as pyg
import torch
from torch_geometric.data import HeteroData,Batch
from torch_geometric.utils import to_undirected,add_self_loops
import torch_geometric.transforms as T
import numpy as np
import copy
import wandb
class ReplayBufferSingleAgent():
    def __init__(self,length,agent_params,fully_connected = False,device='cpu'):
        self.states =  [None]*length
        self.nstates =  [None]*length
        self.actions=torch.zeros(length,2,device=device,dtype=torch.long)
        self.rewards = torch.zeros(length,device=device)
        self.is_terminal = torch.zeros(length,device = device, dtype=bool)
        self.counter = 0
        self.length = length
        self.full = False
        self.device = device
        self.fully_connected=fully_connected
        
        self.agent_action_list =agent_params['action_list']
        self.agent_sides_sup=agent_params['sides_sup']
        self.agent_sides_b=agent_params['sides_b']
        
    def push(self,rid,state,action,nstate,reward,terminal = False):
        #create an HeteroData object from a graph defined by an adjacency matrix
        if self.fully_connected:
            assert False, "Not implemented"
            #self.states[self.counter]=DenseGraph(state)
            #self.nstates[self.counter]=DenseGraph(nstate)
        else:
            self.states[self.counter]=create_sparse_graph(state,rid,self.agent_action_list,self.device,self.agent_sides_sup,self.agent_sides_b)
            self.states[self.counter].validate()
            if not terminal:
                self.nstates[self.counter]=create_sparse_graph(nstate,rid,self.agent_action_list,self.device,self.agent_sides_sup,self.agent_sides_b)
                self.nstates[self.counter].validate()
            else:
                self.nstates[self.counter]=HeteroData()
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
        nstates_minibatch = Batch.from_data_list([self.states[i] for i in idx])
        # if self.fully_connected:
        #     states_minibatch = states_batch.index_select(idx)
        #     nstates_minibatch = nstates_batch.index_select(idx)
        # else:
            
        #     states_minibatch = states_batch.index_select(idx)
        #     nstates_minibatch = nstates_batch.index_select(idx)
        actions_minibatch = self.actions[idx]
        rewards_minibatch = self.rewards[idx]
        terminal_minibatch = self.is_terminal[idx]
        return (states_minibatch,actions_minibatch,nstates_minibatch,rewards_minibatch,terminal_minibatch)

def create_sparse_graph(sim,robot_to_act,action_list,device,sides_sup,sides_b,last_only=True):
        graph = HeteroData()
        graph['ground'].x = torch.tensor(sim.graph.grounds[sim.graph.active_grounds,2:],device=device,dtype=torch.float) #features:[coords]
        graph['block'].x = torch.tensor(sim.graph.blocks[sim.graph.active_blocks,1:],device=device,dtype=torch.float) # features:[btype,coords]
        graph['robot'].x = torch.tensor(sim.ph_mod.last_res.x[:sim.ph_mod.nr*6].reshape(sim.ph_mod.nr,6),device=device,dtype=torch.float) #features: force applied
        ng = graph['ground'].x.shape[0]
        nb = graph['block'].x.shape[0]
        
        
        
        if 'Ph' in action_list:
            #define the action tree:
                
                #6 nodes are created connected to each ground or block
            if last_only:
                graph['side_ori'].x = torch.arange(6,device=device,dtype=torch.float).view(-1,1)
                if not np.any(sim.graph.active_blocks):
                    graph['ground','action_desc','side_ori'].edge_index=torch.vstack([(sim.graph.n_ground-1)*torch.ones(6,dtype=torch.long,device=device),
                                                                                      torch.arange(6,device=device)])
                else:
                    graph['block','action_desc','side_ori'].edge_index=torch.vstack([(sim.graph.n_blocks-1)*torch.ones(6,dtype=torch.long,device=device),
                                                                                     torch.arange(6,device=device)])
            else:
                graph['side_ori'].x = torch.arange(6,device=device,dtype=torch.float).repeat(ng+nb).view(-1,1)
                graph['ground','action_desc','side_ori'].edge_index=torch.vstack([torch.repeat_interleave(torch.arange(ng,device=device),6),
                                                                                  torch.arange(6*ng,device=device)])
                graph['block','action_desc','side_ori'].edge_index=torch.vstack([torch.repeat_interleave(torch.arange(nb,device=device),6),
                                                                                 torch.arange(6*ng,6*(ng+nb),device=device)])
            
            graph['side_supid'].x = torch.zeros((0,1),device=device)
            graph['new_block'].x = torch.zeros((0,2),device=device)#features: [blocktype, sideid]
            graph['side_ori','action_desc','side_supid'].edge_index = torch.zeros((2,0),device=device,dtype=torch.long)
            graph['side_supid','action_desc','new_block'].edge_index = torch.zeros((2,0),device=device,dtype=torch.long)
            
            
            for i in range(graph['side_ori'].x.shape[0]):
                ori = int(graph['side_ori'].x[i,0])
                if i < 6*ng:
                
                    new_nodes = torch.arange(sides_sup[0,int(graph['side_ori'].x[i])]).unsqueeze(1)
                    
                else:
                    connected_block = graph['block','action_desc','side_ori'].edge_index[0,i-6*ng]
                    block_type = graph['block'].x[connected_block,0]
                    new_nodes = torch.arange(sides_sup[int(block_type+1),ori]).unsqueeze(1)
                    
                new_edges = torch.vstack([i*torch.ones((1,new_nodes.shape[0]),device=device,dtype = torch.long),
                                          graph['side_supid'].x.shape[0]+new_nodes.T])
                graph['side_supid'].x = torch.vstack([graph['side_supid'].x,new_nodes])
                graph['side_ori','action_desc','side_supid'].edge_index = torch.hstack([graph['side_ori','action_desc','side_supid'].edge_index,
                                                                                        new_edges])
                
                for sidsup_node_idx in new_edges[1,:]:#ids of each new nodes
                    for btype in range(sides_b.shape[0]):
                        new_nodes = torch.hstack([btype*torch.ones((sides_b[btype,int(graph['side_ori'].x[i,0])],1),device=device),
                                                  torch.arange(sides_b[btype,int(graph['side_ori'].x[i,0])]).view(-1,1)])
                        new_edges = torch.vstack([sidsup_node_idx*torch.ones(new_nodes.shape[0],device=device,dtype = torch.long),
                                                  graph['new_block'].x.shape[0]+torch.arange(new_nodes.shape[0])])
                        
                        graph['new_block'].x = torch.vstack([graph['new_block'].x,new_nodes])
                        graph['side_supid','action_desc','new_block'].edge_index = torch.hstack([graph['side_supid','action_desc','new_block'].edge_index,
                                                                                                 new_edges])
                
                
                
                
                
                
            #graph['possible_new_block'].x  = torch.zeros((n_blocktype*(graph['blocks'].x.shape[0]+graph['grounds'].x.shape[0])*,0),device=device) #n_nodes: ntype_b, features: []
            
            
            graph['robot','choses','new_block'].edge_index = torch.vstack([robot_to_act*torch.ones(graph['new_block'].x.shape[0],device=device,dtype=torch.long),
                                                                           torch.arange(graph['new_block'].x.shape[0],device=device)
                                                                          ])
        if 'L' in action_list:
            graph['leave'].x = torch.zeros(1,0)#features: []
            graph['robot','choses','leave'].edge_index = torch.tensor([robot_to_act,0],device=device).unsqueeze(1) #features: []
            
        bid_held, = np.nonzero(sim.graph.blocks[sim.graph.active_blocks][:,0]>-1)
        graph['robot','holds','block'].edge_index = torch.vstack([
                                                        torch.tensor(sim.graph.blocks[sim.graph.active_blocks][bid_held,0],device=device),
                                                        torch.tensor(bid_held,device=device)]) #features: []
        graph['block','touches','block'].edge_index = torch.tensor(sim.graph.edges_index_bb[:,sim.graph.active_edges_bb],device=device,dtype=torch.long)
        graph['ground','touches','block'].edge_index = torch.tensor(sim.graph.edges_index_gb[:,sim.graph.active_edges_gb],device=device,dtype=torch.long)
        graph['block','touches','ground'].edge_index = torch.tensor(sim.graph.edges_index_bg[:,sim.graph.active_edges_bg],device=device,dtype=torch.long)
        # graph['block','touches','block'].edge_attr = torch.tensor(sim.graph.i_a_bb[sim.graph.active_edges_bb],device = device,dtype=torch.float)
        # graph['block','touches','ground'].edge_attr = torch.tensor(sim.graph.i_a_bg[sim.graph.active_edges_bg],device = device,dtype=torch.float)
        # graph['ground','touches','block'].edge_attr = torch.tensor(sim.graph.i_a_gb[sim.graph.active_edges_gb],device = device,dtype=torch.float)
        
        # graph['possible_new_block','put_on','block'].edge_index = torch.vstack(
        #                                                 [torch.arange(n_blocktype,device=device).repeat(graph['block'].x.shape[0]),
        #                                                  torch.repeat_interleave(torch.arange(graph['block'].x.shape[0],device=device),n_blocktype)])
        # graph['possible_new_block','put_on','ground'].edge_index =  torch.vstack(
        #                                                 [torch.arange(n_blocktype,device=device).repeat(ng),
        #                                                  torch.repeat_interleave(torch.arange(ng,device=device),n_blocktype)])#features: [side_ori, sup_side_id, side_id]
        # #features: [side_ori, sup_side_id, side_id]
        
        graph['robot','reaches', 'block'].edge_index = torch.vstack([torch.arange(graph['robot'].x.shape[0],device=device).repeat(graph['block'].x.shape[0]),
                                                          torch.repeat_interleave(torch.arange(graph['block'].x.shape[0],device=device),graph['robot'].x.shape[0])]) #features: []
        graph['robot','reaches','ground'].edge_index =torch.vstack([torch.arange(graph['robot'].x.shape[0],device=device).repeat(ng),
                                                          torch.repeat_interleave(torch.arange(ng,device=device),graph['robot'].x.shape[0])])#features: []
        graph['robot','communicate','robot'].edge_index=torch.vstack([torch.arange(graph['robot'].x.shape[0],device=device).repeat(graph['robot'].x.shape[0]),
                                                          torch.repeat_interleave(torch.arange(graph['robot'].x.shape[0],device=device),graph['robot'].x.shape[0])]) #features: []
        graph = T.AddSelfLoops()(graph)
        graph = T.ToUndirected(merge=False)(graph)
        return graph
def build_hetero_GNN(config,sample_data=None):
    if config['GNN_arch']=='GATskip':
        model =  pyg.nn.to_hetero(GeometricNN(config),sample_data.metadata())
        if sample_data is not None:
            with torch.no_grad():  # Initialize lazy modules.
                out = model(sample_data.x_dict, sample_data.edge_index_dict)
    else:
        assert False, "Not implemented"
    
    return model.to(device=config['torch_device'])
class GeometricNN(torch.nn.Module):
    def __init__(self,
                 config,
                 use_wandb=False):
        super().__init__()
        #unpck the config
        n_layers = config['GNN_n_layers']
        internal_dims = config['GNN_hidden_dim']
        n_heads = config['GNN_att_head']
        device = config['torch_device']
        self.convs = torch.nn.ModuleList([pyg.nn.GATv2Conv((-1, -1), internal_dims, edge_dim=-1,add_self_loops=False,heads = n_heads) for i in range(n_layers-1)])
        self.lins =  torch.nn.ModuleList([pyg.nn.Linear(-1, internal_dims) for i in range(n_layers-1)])
        self.conv_out = pyg.nn.GATv2Conv((-1, -1), 1,edge_dim=-1, add_self_loops=False)
        self.lin_out = pyg.nn.Linear(-1, 1)
        self.use_wandb=use_wandb
        
    def forward(self, x, edge_index):
        for conv,lin in zip(self.convs,self.lins):
            x = conv(x, edge_index)+lin(x)
            x=x.relu()
        x = self.conv_out(x,edge_index)+self.lin_out(x)
        
        return x
    
class SACOptimizerGeometricNN():
    def __init__(self,policy,config,use_wandb=False):
        self.use_wandb = use_wandb
        self.pol = policy
        self.log_freq = self.pol.log_freq
        self.Qs = [build_hetero_GNN(config)
                  for i in range(2)]
        self.target_Qs = copy.deepcopy(self.Qs)
        self.target_Qs.eval()
        
        lr = config['opt_lr']
        wd = config['opt_weight_decay']
        self.max_norm = config['opt_max_norm']
        self.pol_over_val = config['opt_pol_over_val']
        self.tau = config['opt_tau']
        self.exploration_factor=config['opt_exploration_factor']
        self.entropy_penalty = config['opt_entropy_penalty']
        self.Qval_reduction = config['opt_Q_reduction']
        self.value_clip = config['opt_value_clip']
        self.last_only = config['agent_last_only']
        self.opt_pol = torch.optim.NAdam(self.pol.parameters(),lr=lr*self.pol_over_val,weight_decay=wd)
        self.opt_Q = [torch.optim.NAdam(Q.parameters(),lr=lr,weight_decay=wd) for Q in self.Qs]
        self.target_entropy = config['opt_target_entropy']
        self.alpha = torch.tensor([1.],device = policy.device,requires_grad=True)
        self.opt_alpha = torch.optim.NAdam([self.alpha],lr=1e-3)
        self.beta = 0.5 #same as in paper
        self.clip_r = 0.5 #same as in paper
        self.step = 0
        if self.pol.name is not None and self.pol.name !="":
            self.name = self.pol.name+"_"
        else:
            self.name = ""
    def optimize(self,state,actionid,rewards,nstates,gamma,mask=None,nmask=None,old_entropy =None):
        rewards = torch.tensor(rewards,device=self.pol.device).squeeze()
        feats_pol = self.pol(state.x_dict(),state.edge_index_dict())
        #logits = 
        Qvals = [self.Qs[i](state.x_dict(),state.edge_index_dict()) for i in range(2)]
        with torch.inference_mode():
            tQvals = [self.target_Qs[i](nstates) for i in range(2)]
        
        nfeats = self.pol(nstates,mask=nmask,inference=True)
        entropy = dist.entropy()
        nentropy = ndist.entropy()
        tV = torch.stack([(ndist.probs*tQval).sum(dim=1)+F.relu(self.alpha)*nentropy for tQval in tQvals],dim=1)
        if self.Qval_reduction == 'min':
            tV, _ = torch.min(tV,dim=1)
        elif self.Qval_reduction == 'mean':
            tV = torch.mean(tV,dim=1)          
        tV[~torch.any(torch.tensor(nmask,device=self.pol.device),dim=1)]=0
        #update the critics
        losses = torch.zeros(2,device = self.pol.device)
        for i in range(2):
            self.opt_Q[i].zero_grad()
            
            sampled_Q = Qvals[i][torch.arange(Qvals[i].shape[0]),actionid]
            loss = F.mse_loss(sampled_Q,(rewards+gamma*tV).detach())
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
        l_p = (-F.relu(self.alpha)*entropy-(minQvals.detach()*dist.probs).sum(dim=1)).mean()
        if self.entropy_penalty:
            l_p += self.beta*F.mse_loss(entropy,old_entropy.squeeze())
        self.opt_pol.zero_grad()
        l_p.backward()
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
        if self.use_wandb and self.step % self.log_freq == 0:
            
            wandb.log({self.name+"l_p": l_p.detach().cpu().numpy(),
                       self.name+"rewards":rewards.detach().mean().cpu().numpy(),
                       self.name+"best_action": argmax,
                       self.name+"Qval_0":Qvals[0][torch.arange(Qvals[0].shape[0]),actionid].detach().mean().cpu().numpy(),
                       self.name+"Qval_1":Qvals[1][torch.arange(Qvals[1].shape[0]),actionid].detach().mean().cpu().numpy(),
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





        
    