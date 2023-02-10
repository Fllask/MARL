# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:33:58 2022

@author: valla
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import time
import os
import wandb
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import Dataset

class ReplayBufferGraph(Dataset):
    def __init__(self,
                 length,
                 nreg,
                 maxblocks,
                 maxinterface,
                 n_attributeV=5,
                 n_attributeE=1,
                 n_actions = 450,
                 device='cpu'):
        #store each component separatly to speedup the loading
        self.counter = 0
        self.max_l = length
        self.device = device
        self.Va = torch.zeros((length,n_attributeV,nreg+maxblocks),device = device,dtype=torch.long)
        self.Es = torch.zeros((length,nreg+maxblocks,maxinterface),device = device,dtype=torch.float)
        self.Er = torch.zeros((length,nreg+maxblocks,maxinterface),device = device,dtype=torch.float)
        self.Ea = torch.zeros((length,n_attributeE,maxinterface),device = device,dtype=torch.long)
        self.mask_e = torch.zeros((length,maxinterface),device = device,dtype=bool)
        self.mask_v = torch.zeros((length,nreg+maxblocks),device = device,dtype=bool)
        
        self.mask_out = torch.zeros((length,n_actions),device = device,dtype=bool)
        
        self.action = torch.zeros((length,1),device = device, dtype = torch.long)
        
        self.reward = torch.zeros((length,1),device = device, dtype = torch.float)
        
        self.nVa = torch.zeros((length,n_attributeV,nreg+maxblocks),device = device,dtype=torch.long)
        self.nEs = torch.zeros((length,nreg+maxblocks,maxinterface),device = device,dtype=torch.float)
        self.nEr = torch.zeros((length,nreg+maxblocks,maxinterface),device = device,dtype=torch.float)
        self.nEa = torch.zeros((length,n_attributeE,maxinterface),device = device,dtype=torch.long)
        self.nmask_e = torch.zeros((length,maxinterface),device = device,dtype=bool)
        self.nmask_v = torch.zeros((length,nreg+maxblocks),device = device,dtype=bool)
        
        self.nmask_out = torch.zeros((length,n_actions),device = device,dtype=bool)
        
        self.entropy = torch.zeros((length,1),device = device,dtype = torch.float)
        
        
        self.full = False
    def push(self,graph,mask,action,new_graph,new_mask,reward,entropy=None):
        if entropy is not None:
            self.entropy[self.counter] = entropy 
        self.Va[self.counter] = torch.tensor(np.concatenate([graph.grounds,graph.blocks],1))
        self.Es[self.counter] = torch.tensor(graph.i_s)
        self.Er[self.counter] = torch.tensor(graph.i_r)
        self.Ea[self.counter] = torch.tensor(graph.i_a)
        self.mask_v[self.counter] = torch.tensor(graph.active_nodes)
        self.mask_e[self.counter] = torch.tensor(graph.active_edges)
        
        self.mask_out[self.counter] = torch.tensor(mask)
        self.action[self.counter] = torch.tensor(action)
        
        self.reward[self.counter] = reward
        
        self.nVa[self.counter] = torch.tensor(np.concatenate([new_graph.grounds,new_graph.blocks],1))
        self.nEs[self.counter] = torch.tensor(new_graph.i_s)
        self.nEr[self.counter] = torch.tensor(new_graph.i_r)
        self.nEa[self.counter] = torch.tensor(new_graph.i_a)
        self.nmask_v[self.counter] = torch.tensor(new_graph.active_nodes)
        self.nmask_e[self.counter] = torch.tensor(new_graph.active_edges)
        
        self.nmask_out[self.counter] = torch.tensor(new_mask)
        
        self.counter+=1
        if self.counter == self.max_l:
            self.counter = 0
            self.full = True
        
    def __getitem__(self,i):
        return ((self.Va[i],
                self.Es[i],
                self.Er[i] ,
                self.Ea[i],
                self.mask_e[i],
                self.mask_v[i]),

                self.mask_out[i],

                self.action[i],

                self.reward[i],

                (self.nVa[i], 
                self.nEs[i], 
                self.nEr[i],
                self.nEa[i],
                self.nmask_e[i],
                self.nmask_v[i]),

                self.nmask_out[i],
                self.entropy[i])
    def __len__(self):
        return self.counter
    def sample(self,batch_size):
        if self.full:
            idxs = torch.randperm(self.max_l)
            idxs = idxs[:batch_size]
        else:
            idxs = torch.randperm(self.counter)
            idxs = idxs[:batch_size]
        return self[idxs]
    
class WolpertingerOpt():
    def __init__(self,act_net,QT,tar_act_net,tar_QT):
        
        self.optimizerpol = torch.optim.SGD(act_net.parameters(),lr=0.0001,#,momentum = 0.01,
                                            #nesterov = True,
                                            maximize=True)
        self.optimizerQT = torch.optim.SGD(QT.parameters(),lr=0.0001)#,momentum = 0.01,
                                           #nesterov = True)
        self.QT = QT
        self.act_net = act_net
        self.tar_QT = tar_QT
        self.tar_act_net = tar_act_net
    def optimize_pol(self,states):
        if self.act_net.state_dict()['out.weight'][0,0] != self.act_net.state_dict()['out.weight'][0,0]:
            pass
        protoaction = self.act_net.forward(states)
        with torch.no_grad():
            v = self.QT(states,torch.unsqueeze(protoaction,1)).to(self.act_net.device)
        sum_v = torch.sum(v)
        self.optimizerpol.zero_grad()
        sum_v.backward()
        self.optimizerpol.step()
        if self.act_net.state_dict()['out.weight'][0,0] != self.act_net.state_dict()['out.weight'][0,0]:
            pass
    def update_target(self,tau):
        sd_targetQT = self.tar_QT.state_dict()
        sd_targetactnet = self.tar_act_net.state_dict()
        sd_QT = self.QT.state_dict()
        sd_actnet = self.act_net.state_dict()
        for key in sd_QT:
            sd_targetQT[key]= (1-tau)*sd_targetQT[key]+tau*sd_QT[key]
        self.tar_QT.load_state_dict(sd_targetQT)
        for key in sd_actnet:
            sd_targetactnet[key]= (1-tau)*sd_targetactnet[key]+tau*sd_actnet[key]
        self.tar_act_net.load_state_dict(sd_targetactnet)
                
    def optimize_QT(self,state,action,rewards,nstates,nactions,gamma):
        next_v = self.tar_QT(nstates,nactions,inference = True)
        y = torch.tensor(rewards).to(next_v.device) + gamma*next_v
        if torch.any(torch.isnan(self.QT.state_dict()['out.weight'])) or torch.any(torch.isinf(self.QT.state_dict()['out.weight'])):
            pass
        v = self.QT.forward(state, action)
        if torch.any(torch.isnan(v)) or torch.any(torch.isinf(v)):
            self.QT.forward(state, action)
        l = self.QT.loss(v,y)

        self.optimizerQT.zero_grad()
        l.backward()
        self.optimizerQT.step()
        if torch.any(torch.isnan(self.QT.state_dict()['out.weight'])) or torch.any(torch.isinf(self.QT.state_dict()['out.weight'])):
            pass
        return l.detach().cpu().numpy()
    
class WolpertingerActionFinderNet(nn.Module):
    def __init__(self,
                 maxs_grid,
                 max_blocks,
                 n_robots,
                 n_regions,
                 n_fc_layer =2,
                 n_neurons = 100,
                 device='cpu',
                 actions_dim=25,
                 **encoder_args,
                 ):
        super().__init__()
        self.state_encoder = StateEncoder(maxs_grid,
                                          max_blocks,
                                          n_robots,
                                          n_regions,
                                          device=device,
                                          **encoder_args)
        
        self.FC = [nn.Linear(np.prod(self.state_encoder.out_dims),n_neurons,device=device)]
        self.FC+=[nn.Linear(n_neurons,n_neurons,device=device) for i in range(n_fc_layer-1)]
        self.out = nn.Linear(n_neurons,actions_dim,device=device)
        self.device=device
    def forward(self,grids,inference = False,noise_amp=0):
        with torch.inference_mode(inference):
            
            
            rep = torch.flatten(self.state_encoder(grids),1)
            for layer in self.FC:
                rep = F.relu(layer(rep))
            protoaction = self.out(rep)
            #protoaction+=torch.normal(mean=torch.zeros_like(protoaction),std = noise_amp*torch.ones_like(protoaction))
            return protoaction
class DeepQT(nn.Module):
    def __init__(self,
                 maxs_grid,
                 max_blocks,
                 n_robots,
                 n_regions,
                 n_actions,
                 n_fc_layer =2,
                 n_neurons = 100,
                 device='cpu',
                 **encoder_args):
        super().__init__()
        self.state_encoder = StateEncoder(maxs_grid,
                                          max_blocks,
                                          n_robots,
                                          n_regions,
                                          device=device,
                                          **encoder_args)
        self.FC = [nn.Linear(np.prod(self.state_encoder.out_dims),n_neurons,device=device)]
        self.FC+=[nn.Linear(n_neurons,n_neurons,device=device) for i in range(n_fc_layer-1)]
        self.out = nn.Linear(n_neurons,n_actions,device=device)
        self.loss = nn.MSELoss()
        self.device=device
    def forward(self,grids,inference = False,prob=False):
        with torch.inference_mode(inference):
            rep = torch.flatten(self.state_encoder(grids),1)
            for layer in self.FC:
                rep = F.relu(layer(rep))
            action_values = self.out(rep)
            if prob:
                action_values = F.softmax(action_values,dim=1)
            return action_values
class DeepQTOptimizer():
    def __init__(self,QT):
        
        self.optimizer = torch.optim.SGD(QT.parameters(),lr=0.000005,momentum = 0.01,
                                         nesterov = True)
        self.QT = QT
    def optimize(self,state,action,rewards,nstates,gamma):
        next_v,_ = torch.max(self.QT(nstates,inference = True),dim=1)
        y = torch.tensor(rewards).to(next_v.device) + gamma*next_v
        v = self.QT(state)
        va = v[np.arange(len(action)),action]
        l = self.QT.loss(va,y)

        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()
        return l.detach().cpu().numpy()
class PolNetDense(nn.Module):
    def __init__(self,
                 maxs_grid,
                 max_blocks,
                 max_interfaces,
                 n_type_block,
                 n_sides,
                 n_robots,
                 n_regions,
                 n_actions,
                 config,
                 use_wandb=False,
                 log_freq = None,
                 **GNN_args):
        super().__init__()
        self.use_wandb=use_wandb
        
        self.maxgrid = maxs_grid
        self.bt = n_type_block+1 #-1 is used to discribe the ground
        self.nr = n_robots+1 # -1 is used when the block is not held
        self.maxsides = max(n_sides)
        self.sumsides = sum(n_sides)
        self.log_freq = log_freq
        #unwrap the config file:
        self.last_only = config['agent_last_only']
        n_GNblocks = config['A2C_nblocks']
        inter_Va = config['graph_inter_Va']
        inter_Ea = config['graph_inter_Ea']
        inter_u = config['graph_inter_u']
        device = config['torch_device']
        if config['graph_nonlinearity'] == 'gelu':
            self.non_lin = F.gelu
        elif config['graph_nonlinearity'] == 'relu':
            self.non_lin = F.relu
        self.pol_enc = config['graph_pol_enc']
        GNN_args = {'n_neurons':config['graph_neurons_chanel'],
                    'n_hidden_layers' :config['graph_n_hidden_layers'],
                    'non_lin': self.non_lin,
                    'device': config['torch_device']
            }
        self.GN_pol = nn.ModuleList([GNBlock(0, self.bt+self.nr+8, self.maxsides, inter_u, inter_Va, inter_Ea,**GNN_args)]+
                                   [GNBlock(inter_u, inter_Va, inter_Ea, inter_u, inter_Va, inter_Ea,**GNN_args) for i in range(n_GNblocks-2)]+
                                   [GNBlock(inter_u, inter_Va, inter_Ea, inter_u, inter_Va, 0,**GNN_args)]
                                   )
        if self.pol_enc == 'one_edge':
            self.GN_out_pol = GNBlock(inter_u, inter_Va, 0, 0, 0, n_actions,**GNN_args)
        if self.pol_enc =='edge_action':
            self.GN_out_pol = GNBlock(inter_u, inter_Va, 0, 0, 0, 1,**GNN_args)
        self.device = device
    def forward(self,graph=None,graph_desc=None,inference = False,mask =None,eps=None):
        with torch.inference_mode(inference):
            if inference:
                self.eval()
            else:
                self.train()
            #extract the matrices from the graph
            if graph is not None:
                blocks = torch.tensor(graph.blocks,device=self.device,dtype=torch.long).unsqueeze(0)
                grounds = torch.tensor(graph.grounds,device=self.device,dtype=torch.long).unsqueeze(0)
                Va = torch.cat([grounds,blocks],2)#all the attibutes of the vertices shape: (B,5,n_reg+n_blocks)
                Es = torch.tensor(graph.i_s,device=self.device,dtype=torch.float).unsqueeze(0)#shape:(B,n_reg+n_blocks,maxinterface)
                Er = torch.tensor(graph.i_r,device=self.device,dtype=torch.float).unsqueeze(0)#shape:(B,n_reg+n_blocks,maxinterface)
                Ea = torch.tensor(graph.i_a,dtype=torch.long,device=self.device).unsqueeze(0)#shape:(B,1,maxinterface)
                mask_v = torch.tensor(graph.active_nodes,device=self.device,dtype=bool).unsqueeze(0)
                mask_e = torch.tensor(graph.active_edges,device=self.device,dtype=bool).unsqueeze(0)
                mask=torch.tensor(mask,device=self.device,dtype=bool).unsqueeze(0)
            else:
                Va = graph_desc[0]
                Es = graph_desc[1]
                Er = graph_desc[2]
                Ea = graph_desc[3]
                mask_e = graph_desc[4]
                mask_v = graph_desc[5]
   
            Va_enc = torch.cat([F.one_hot(Va[:,0,:]+1,num_classes=self.bt).permute(0,2,1),
                                F.one_hot(Va[:,1,:]+1,num_classes=self.nr).permute(0,2,1),
                                2*torch.unsqueeze(Va[:,2,:],1).to(torch.float)/self.maxgrid[0]-1,
                                2*torch.unsqueeze(Va[:,3,:],1).to(torch.float)/self.maxgrid[1]-1,
                                F.one_hot(Va[:,4,:],num_classes=6).permute(0,2,1)
                                          ],1)
            
            Ea_enc = F.one_hot(Ea[:,0,:],num_classes=self.maxsides).permute(0,2,1).to(torch.float)
            
            u = torch.zeros((Ea.shape[0],0,1),device=self.device)
            #compute the policy
            pol_v = Va_enc
            pol_e = Ea_enc
            pol_u = u
            
            for block in self.GN_pol:
                pol_e,pol_v,pol_u = block(pol_e,Es,Er,pol_v,pol_u,mask_e = mask_e,mask_v=mask_v)
            
            #add the action node:
            if self.last_only:
       
                if self.pol_enc == 'one_edge':
                    #this method does not work when 'R'is in the action choice
                    last_node = torch.sum(mask_v,1).to(int)-1
                    #get the maximum number of nodes in the batch
                    last_node_max = last_node.max()
                    Es_outpol = F.one_hot(last_node,last_node_max+2).to(torch.float).unsqueeze(2)
                    Er_outpol= torch.zeros(Es_outpol.shape,device=self.device)
                    Er_outpol[:,-1]=1
                    pol_v = torch.cat([pol_v[:,:,:last_node_max+1],torch.ones(pol_v.shape[0],pol_v.shape[1],1,device=self.device)],2)
                    pol_e = torch.zeros(pol_e.shape[0],0,1,device=self.device)
                else:
                    assert False,"Not implemented"
                

                pol_e,pol_v,pol_u = self.GN_out_pol(pol_e,Es_outpol,Er_outpol,pol_v,pol_u)
            
            
                
            if self.pol_enc == 'one_edge':
                pol_e = pol_e.squeeze(2)
                if mask is not None:
                    
                    #mask_eout = mask-1
                    pol_e[~mask]-= 1e10
            dist = Categorical(logits = pol_e)
            pol = pol_e
            if torch.any(torch.isnan(pol)):
                assert False, 'nans in the policy'
            return dist,pol
class ValNetDense(nn.Module):
    def __init__(self,
                 maxs_grid,
                 max_blocks,
                 max_interfaces,
                 n_type_block,
                 n_sides,
                 n_robots,
                 n_output,
                 config,
                 use_wandb=False):
        super().__init__()
        self.use_wandb=use_wandb
        
        self.maxgrid = maxs_grid
        self.bt = n_type_block+1 #-1 is used to discribe the ground
        self.nr = n_robots+1 # -1 is used when the block is not held
        self.maxsides = max(n_sides)
        self.sumsides = sum(n_sides)
        
        #unwrap the config file:
        self.last_only = config['agent_last_only']
        n_GNblocks = config['A2C_nblocks']
        inter_Va = config['graph_inter_Va']
        inter_Ea = config['graph_inter_Ea']
        inter_u = config['graph_inter_u']
        device = config['torch_device']
        self.batch_norm = config['graph_batch_norm']
        if n_output > 1:
            self.dueling = config['graph_duel']
        else: 
            self.dueling = False
        if config['graph_nonlinearity'] == 'gelu':
            self.non_lin = F.gelu
        elif config['graph_nonlinearity'] == 'relu':
            self.non_lin = F.relu
        self.pol_enc = config['graph_pol_enc']
        GNN_args = {'n_neurons':config['graph_neurons_chanel'],
                    'n_hidden_layers' :config['graph_n_hidden_layers'],
                    'non_lin': self.non_lin,
                    'device':config['torch_device']
            }
        
        if self.dueling:
            self.GN_common = nn.ModuleList([GNBlock(0, self.bt+self.nr+8, self.maxsides, inter_u, inter_Va, inter_Ea,**GNN_args)]+
                                       [GNBlock(inter_u, inter_Va, inter_Ea, inter_u, inter_Va, inter_Ea,**GNN_args) for i in range(n_GNblocks-2)]
                                       )
            self.GN_V = GNBlock(inter_u, inter_Va, inter_Ea, inter_u, inter_Va, inter_Ea,**GNN_args)
            self.GN_A = GNBlock(inter_u, inter_Va, inter_Ea, inter_u, inter_Va, inter_Ea,**GNN_args)
            self.Vout = nn.Linear(inter_u,1,device=device)
            self.Aout = nn.Linear(inter_u,n_output,device=device)
            if config['V_optimistic']:
                self.Aout.bias.data.fill_(config['V_optimistic'])
        else:
            self.GN = nn.ModuleList([GNBlock(0, self.bt+self.nr+8, self.maxsides, inter_u, inter_Va, inter_Ea,**GNN_args)]+
                                       [GNBlock(inter_u, inter_Va, inter_Ea, inter_u, inter_Va, inter_Ea,**GNN_args) for i in range(n_GNblocks-2)]+
                                       [GNBlock(inter_u, inter_Va, inter_Ea, inter_u, inter_Va, inter_Ea,**GNN_args)]
                                       )

            self.out = nn.Linear(inter_u,n_output,device=device)
            if config['V_optimistic']:
                self.out.bias.data.fill_(1.)
        self.device = device
    def forward(self,graph=None,graph_desc=None,inference = False,mask =None,eps=None):
        with torch.inference_mode(inference):
            if inference:
                self.eval()
            else:
                self.train()
            #extract the matrices from the graph
            if graph is not None:
                blocks = torch.tensor(graph.blocks,device=self.device,dtype=torch.long).unsqueeze(0)
                grounds = torch.tensor(graph.grounds,device=self.device,dtype=torch.long).unsqueeze(0)
                Va = torch.cat([grounds,blocks],2)#all the attibutes of the vertices shape: (B,5,n_reg+n_blocks)
                Es = torch.tensor(graph.i_s,device=self.device,dtype=torch.float).unsqueeze(0)#shape:(B,n_reg+n_blocks,maxinterface)
                Er = torch.tensor(graph.i_r,device=self.device,dtype=torch.float).unsqueeze(0)#shape:(B,n_reg+n_blocks,maxinterface)
                Ea = torch.tensor(graph.i_a,dtype=torch.long,device=self.device).unsqueeze(0)#shape:(B,1,maxinterface)
                mask_v = torch.tensor(graph.active_nodes,device=self.device,dtype=bool).unsqueeze(0)
                mask_e = torch.tensor(graph.active_edges,device=self.device,dtype=bool).unsqueeze(0)
                mask=torch.tensor(mask,device=self.device,dtype=bool).unsqueeze(0)
            else:
                Va = graph_desc[0]
                Es = graph_desc[1]
                Er = graph_desc[2]
                Ea = graph_desc[3]
                mask_e = graph_desc[4]
                mask_v = graph_desc[5]
   
            Va_enc = torch.cat([F.one_hot(Va[:,0,:]+1,num_classes=self.bt).permute(0,2,1),
                                F.one_hot(Va[:,1,:]+1,num_classes=self.nr).permute(0,2,1),
                                2*torch.unsqueeze(Va[:,2,:],1).to(torch.float)/self.maxgrid[0]-1,
                                2*torch.unsqueeze(Va[:,3,:],1).to(torch.float)/self.maxgrid[1]-1,
                                F.one_hot(Va[:,4,:],num_classes=6).permute(0,2,1)
                                          ],1)
            
            Ea_enc = F.one_hot(Ea[:,0,:],num_classes=self.maxsides).permute(0,2,1).to(torch.float)
            
            u = torch.zeros((Ea.shape[0],0,1),device=self.device)
            #compute the policy
            val_v = Va_enc
            val_e = Ea_enc
            val_u = u
            if self.dueling:
                for block in self.GN_common:
                    val_e,val_v,val_u = block(val_e,Es,Er,val_v,val_u,mask_e = mask_e,mask_v=mask_v)
                V_e, V_v, V_u = self.GN_V(val_e,Es,Er,val_v,val_u,mask_e = mask_e,mask_v=mask_v)
                A_e, A_v, A_u = self.GN_A(val_e,Es,Er,val_v,val_u,mask_e = mask_e,mask_v=mask_v)
                t0 = time.perf_counter()
                V = self.Vout(V_u.squeeze(2))
                A = self.Aout(A_u.squeeze(2))
                A = A - A.mean()
                val = V+A
                t1 = time.perf_counter()
                #print(f'   lin_val: {t1-t0}')
            else:
                for block in self.GN:
                    val_e,val_v,val_u = block(val_e,Es,Er,val_v,val_u,mask_e = mask_e,mask_v=mask_v)
                val = self.out(val_u.squeeze(2))
                
            if torch.any(torch.isnan(val)):
                assert False, 'nans value'
            return val
class SACDenseOptimizer():
    def __init__(self,maxs_grid,max_blocks,max_interfaces,n_type_block,n_sides,n_robots,n_actions,policy,config):
        self.use_wandb = policy.use_wandb
        self.pol = policy
        self.log_freq = self.pol.log_freq
        self.Qs = [ValNetDense(maxs_grid,max_blocks,max_interfaces,n_type_block,n_sides,n_robots,n_actions,config,self.use_wandb)
                  for i in range(2)]
        self.target_Qs = copy.deepcopy(self.Qs)
        lr = config['opt_lr']
        wd = config['opt_weight_decay']
        self.max_norm = config['opt_max_norm']
        self.pol_over_val = config['opt_pol_over_val']
        self.tau = config['opt_tau']
        self.exploration_factor=config['opt_exploration_factor']
        self.entropy_penalty = config['opt_entropy_penalty']
        self.Qval_reduction = config['opt_Q_reduction']
        self.value_clip = config['opt_value_clip']
        self.opt_pol = torch.optim.NAdam(self.pol.parameters(),lr=lr*self.pol_over_val,weight_decay=wd)
        self.opt_Q = [torch.optim.NAdam(Q.parameters(),lr=lr,weight_decay=wd) for Q in self.Qs]
        self.target_entropy = config['opt_target_entropy']
        
        self.alpha = torch.tensor([config.get('opt_init_alpha') or 1.],device = config['torch_device'],requires_grad=True)
        self.opt_alpha = torch.optim.NAdam([self.alpha],lr=config.get('opt_lr_alpha') or 1e-3)
        
        self.beta = 0.5 #same as in paper
        self.clip_r = 0.5 #same as in paper
        self.step = 0
    def optimize(self,state,actionid,rewards,nstates,gamma,mask=None,nmask=None,old_entropy =None):
        t00 = time.perf_counter()
        t0 = time.perf_counter()
        dist,pol = self.pol(graph_desc = state,mask=mask)
        t1 = time.perf_counter()
        #print(f"policy with backprop: {t1-t0}")
        t0 = time.perf_counter()
        Qvals = [self.Qs[i](graph_desc =state) for i in range(2)]
        t1 = time.perf_counter()
        #print(f"values with backprop: {t1-t0}")
        tQvals = [self.target_Qs[i](graph_desc =nstates,inference=True) for i in range(2)]
        
        ndist,npol = self.pol(graph_desc = nstates,mask=nmask,inference=True)
        entropy = dist.entropy()
        nentropy = ndist.entropy()
        tV = torch.stack([(ndist.probs*tQval).sum(dim=1)+F.relu(self.alpha)*nentropy for tQval in tQvals],dim=1)
        if self.Qval_reduction == 'min':
            tV, _ = torch.min(tV,dim=1)
        elif self.Qval_reduction == 'mean':
            tV = torch.mean(tV,dim=1)
                                     
                                     
        tV[~torch.any(nmask,dim=1)]=0

        
        
        #update the critics
        losses = torch.zeros(2,device = self.pol.device)
        for i in range(2):
            self.opt_Q[i].zero_grad()
            
            sampled_Q = Qvals[i][torch.arange(Qvals[i].shape[0]),actionid.detach()]
            loss = F.mse_loss(sampled_Q,(rewards+gamma*tV).detach())
            if self.value_clip:
                target_sampled_Q = tQvals[i][torch.arange(Qvals[i].shape[0]),actionid.detach()]
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
            assert False, "Not implemented"
        self.opt_pol.zero_grad()
        l_p.backward()
        if self.max_norm is not None:
            norm_p = nn.utils.clip_grad_norm_(self.pol.parameters(),self.max_norm)
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
        t11  = time.perf_counter()
        #print(f"total: {t11-t00}")
        if self.use_wandb and self.step % self.log_freq == 0:
            wandb.log({"l_p": l_p.detach().cpu().numpy(),
                       "rewards":rewards.detach().mean().cpu().numpy(),
                       "best_action": argmax,
                       "Qval_0":Qvals[0][torch.arange(Qvals[0].shape[0]),actionid].detach().mean().cpu().numpy(),
                       "Qval_1":Qvals[1][torch.arange(Qvals[1].shape[0]),actionid].detach().mean().cpu().numpy(),
                       'policy_grad_norm':norm_p.detach().cpu().numpy(),
                       "V_target":tV.detach().mean().cpu().numpy(),
                       'policy_entropy':entropy.detach().mean().cpu().numpy(),
                       "Qloss_0":losses[0].detach().cpu().numpy(),
                       "Qloss_1":losses[1].detach().cpu().numpy(),
                       "alpha": self.alpha.detach().cpu().numpy(),
                       "l_alpha": l_alpha.detach().cpu().numpy(),
                      },step=self.step)
            #wandb.watch(self.model)
        self.step+=1
        return l_p.detach().cpu().numpy()


class PolNetSparse(nn.Module):
    def __init__(self,
                 maxs_grid,
                 max_blocks,
                 n_robots,
                 n_regions,
                 n_actions,
                 config,
                 use_wandb=False,
                 log_freq = None,
                 name = ""):
        super().__init__()
        #unpack the config file
        
        n_fc_layer = config['SAC_n_fc_layer']
        n_neurons = config['SAC_n_neurons']
        batch_norm = config['SAC_batch_norm']
        device = config['torch_device']
        encoder_args = {'n_channels':config['SEnc_n_channels'],
                        'n_internal_layer':config['SEnc_n_internal_layer'],
                        'stride':config['SEnc_stride']}
        self.name = name
        self.use_wandb=use_wandb
        self.log_freq = log_freq
        if config['SEnc_order_insensitive']:
            self.state_encoder = StateEncoderOE(maxs_grid,
                                          n_robots,
                                          n_regions,
                                          config['agent_last_only'],
                                          device=device,
                                          **encoder_args)
        else:
            self.state_encoder = StateEncoder(maxs_grid,
                                          max_blocks,
                                          n_robots,
                                          n_regions,
                                          device=device,
                                          **encoder_args)
        self.input_norm = nn.BatchNorm2d(self.state_encoder.out_dims[0],device=device)
        self.FC = nn.ModuleList([nn.Linear(np.prod(self.state_encoder.out_dims),n_neurons,device=device)])
        self.FC+=nn.ModuleList([nn.Linear(n_neurons,n_neurons,device=device) for i in range(n_fc_layer-1)])
        self.out_pol = nn.Linear(n_neurons,n_actions,device=device)
        self.device=device
        self.batch_norm = batch_norm
    def forward(self,grids,inference = False,mask =None):
        with torch.inference_mode(inference):
            if inference:
                self.eval()
            else:
                self.train()
            if self.batch_norm:
                normed_rep = self.input_norm(self.state_encoder(grids))
                rep = torch.flatten(normed_rep,1)
            else:
                rep = torch.flatten(self.state_encoder(grids),1)
            for layer in self.FC:
                rep = F.relu(layer(rep))
            if mask is not None:
                mask = torch.tensor(mask,device=self.device,dtype=bool).reshape(len(grids),-1)
                pol = self.out_pol(rep)
                pol[~mask] = -1e10
            else:
                pol = self.out_pol(rep)
            dist = Categorical(logits = pol)
            if torch.any(torch.isnan(pol)):
                assert False, 'nans in the policy'
            return dist,pol
class ValNetSparse(nn.Module):
    def __init__(self,
                 maxs_grid,
                 max_blocks,
                 n_robots,
                 n_regions,
                 n_actions,
                 config,
                 use_wandb=False):
        super().__init__()
        #unpack the config file
        n_fc_layer = config['SAC_n_fc_layer']
        n_neurons = config['SAC_n_neurons']
        batch_norm = config['SAC_batch_norm']
        if n_actions == 1:
            self.dueling = False
        else:
            self.dueling = config['Q_duel']
        device = config['torch_device']
        encoder_args = {'n_channels':config['SEnc_n_channels'],
                        'n_internal_layer':config['SEnc_n_internal_layer'],
                        'stride':config['SEnc_stride']}
        
        
        self.state_encoder = StateEncoder(maxs_grid,
                                          max_blocks,
                                          n_robots,
                                          n_regions,
                                          device=device,
                                          **encoder_args)
        self.input_norm = nn.BatchNorm2d(self.state_encoder.out_dims[0],device=device)
        if self.dueling:
            self.FC_V = nn.ModuleList([nn.Linear(np.prod(self.state_encoder.out_dims),n_neurons,device=device)])
            self.FC_V+=nn.ModuleList([nn.Linear(n_neurons,n_neurons,device=device) for i in range(n_fc_layer-1)])
            self.out_V = nn.Linear(n_neurons,1,device=device)
            self.FC_A = nn.ModuleList([nn.Linear(np.prod(self.state_encoder.out_dims),n_neurons,device=device)])
            self.FC_A+=nn.ModuleList([nn.Linear(n_neurons,n_neurons,device=device) for i in range(n_fc_layer-1)])
            self.out_A = nn.Linear(n_neurons,n_actions,device=device)
            self.out_norm_V = nn.BatchNorm1d(n_neurons,device=device)
            self.out_norm_A = nn.BatchNorm1d(n_neurons,device=device)
        else:
            self.FC_Q = nn.ModuleList([nn.Linear(np.prod(self.state_encoder.out_dims),n_neurons,device=device)])
            self.FC_Q+=nn.ModuleList([nn.Linear(n_neurons,n_neurons,device=device) for i in range(n_fc_layer-1)])
            self.out_Q = nn.Linear(n_neurons,n_actions,device=device)
            self.out_norm_Q = nn.BatchNorm1d(n_neurons,device=device)
        self.device=device
        self.batch_norm = batch_norm
        self.use_wandb=use_wandb
        
    def forward(self,grids,inference = False,mask =None):
        with torch.inference_mode(inference):
            if inference:
                self.eval()
            else:
                self.train()
                
            if self.batch_norm:
                normed_rep = self.input_norm(self.state_encoder(grids))
                rep = torch.flatten(normed_rep,1)
            else:
                rep = torch.flatten(self.state_encoder(grids),1)
            if self.dueling:
                rep_V = rep
                rep_A = rep
                for layer in self.FC_A:
                    rep_A = F.relu(layer(rep_A))
                for layer in self.FC_V:
                    rep_V = F.relu(layer(rep_V))
                if self.batch_norm:
                    rep_A = self.out_norm_A(rep_A)
                    rep_V = self.out_norm_V(rep_V)
                A = self.out_A(rep_A)
                V = self.out_V(rep_V)
                Q = V+A-A.mean(dim=1,keepdim=True)
            else:
                for layer in self.FC_Q:
                    rep = F.relu(layer(rep))
                if self.batch_norm:
                    rep = self.out_norm_Q(rep)
                Q = self.out_Q(rep)
            if mask is not None:
                mask = torch.tensor(~mask,device=self.device,dtype=bool).reshape(len(grids),-1)
                Q[~mask] = -1e10
            if torch.any(torch.isnan(Q)):
                assert False, 'nans in the Q values'
            return Q
class SACSparseOptimizer():
    def __init__(self,maxs_grid,max_blocks,n_robots,n_regions,n_actions,policy,config):
        self.use_wandb = policy.use_wandb
        self.pol = policy
        self.log_freq = self.pol.log_freq
        self.Qs = [ValNetSparse(maxs_grid,max_blocks,n_robots,n_regions,n_actions,config,self.use_wandb)
                  for i in range(2)]
        self.max_entropy = np.log(n_actions)
        self.target_Qs = copy.deepcopy(self.Qs)
        lr = config['opt_lr']
        wd = config['opt_weight_decay']
        self.max_norm = config['opt_max_norm']
        self.pol_over_val = config['opt_pol_over_val']
        self.tau = config['opt_tau']
        self.exploration_factor=config['opt_exploration_factor']
        self.entropy_penalty = config['opt_entropy_penalty']
        self.Qval_reduction = config['opt_Q_reduction']
        self.lbound = config['opt_lower_bound_Vt']
        self.value_clip = config['opt_value_clip']
        self.opt_pol = torch.optim.NAdam(self.pol.parameters(),lr=lr*self.pol_over_val,weight_decay=wd)
        self.opt_Q = [torch.optim.NAdam(Q.parameters(),lr=lr,weight_decay=wd) for Q in self.Qs]
        self.target_entropy = config['opt_target_entropy']
        self.alpha = torch.tensor([1.],device = policy.device,requires_grad=True)
        self.opt_alpha = torch.optim.NAdam([self.alpha],lr=1e-3)
        self.beta = 0.5 #same as in paper
        self.clip_r = 0.5 #same as in paper
        self.step = 0
        if self.pol.name !="":
            self.name = self.pol.name+"/"
        else:
            self.name = ""
    def optimize(self,state,actionid,rewards,nstates,gamma,mask=None,nmask=None,old_entropy =None):
        t00 = time.perf_counter()
        t0 = time.perf_counter()
        rewards = torch.tensor(rewards,device=self.pol.device).squeeze()
        dist,pol = self.pol(state,mask=mask)
        t1 = time.perf_counter()
        #print(f"policy with backprop: {t1-t0}")
        t0 = time.perf_counter()
        Qvals = [self.Qs[i](state) for i in range(2)]
        t1 = time.perf_counter()
        #print(f"values with backprop: {t1-t0}")
        tQvals = [self.target_Qs[i](nstates,inference=True) for i in range(2)]
        
        ndist,npol = self.pol(nstates,mask=nmask,inference=True)
        entropy = dist.entropy()
        nentropy = ndist.entropy()
        tV = torch.stack([(ndist.probs*tQval).sum(dim=1)+F.relu(self.alpha)*nentropy for tQval in tQvals],dim=1)
        if self.Qval_reduction == 'min':
            tV, _ = torch.min(tV,dim=1)
        elif self.Qval_reduction == 'mean':
            tV = torch.mean(tV,dim=1)          
        #the entropy bonus is kept in the terminal state value, as its value has no upper bound
        tV[~torch.any(torch.tensor(nmask,device=self.pol.device),dim=1)]=F.relu(self.alpha)*self.target_entropy
        
        #clamp the target value 
        tV = torch.clamp(tV,min=self.lbound)
        #update the critics
        losses = torch.zeros(2,device = self.pol.device)
        for i in range(2):
            self.opt_Q[i].zero_grad()
            
            sampled_Q = Qvals[i][torch.arange(Qvals[i].shape[0]),actionid]
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
        l_p = (-F.relu(self.alpha)*entropy-(minQvals.detach()*dist.probs).sum(dim=1)).mean()
        if self.entropy_penalty:
            l_p += self.beta*F.mse_loss(entropy,old_entropy.squeeze())
        self.opt_pol.zero_grad()
        l_p.backward()
        if self.max_norm is not None:
            norm_p = nn.utils.clip_grad_norm_(self.pol.parameters(),self.max_norm)
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
        t11  = time.perf_counter()
        #print(f"total: {t11-t00}")
        if self.use_wandb and self.step % self.log_freq == 0:
            
            wandb.log({self.name+"l_p": l_p.detach().cpu().numpy(),
                       self.name+"rewards":rewards.detach().mean().cpu().numpy(),
                       self.name+"best_action": argmax,
                       self.name+"Qval_0":Qvals[0][torch.arange(Qvals[0].shape[0]),actionid].detach().mean().cpu().numpy(),
                       self.name+"Qval_1":Qvals[1][torch.arange(Qvals[1].shape[0]),actionid].detach().mean().cpu().numpy(),
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
class A2CDense(nn.Module):
    def __init__(self,
                 maxs_grid,
                 max_blocks,
                 max_interfaces,
                 n_type_block,
                 n_sides,
                 n_robots,
                 n_regions,
                 n_actions,
                 config,
                 use_wandb=False,
                 **GNN_args):
        super().__init__()
        self.use_wandb=use_wandb
        
        self.maxgrid = maxs_grid
        self.bt = n_type_block+1 #-1 is used to discribe the ground
        self.nr = n_robots+1 # -1 is used when the block is not held
        self.maxsides = max(n_sides)
        self.sumsides = sum(n_sides)
        
        #unwrap the config file:
        self.last_only = config['agent_last_only']
        n_GNblocks = config['A2C_nblocks']
        inter_Va = config['graph_inter_Va']
        inter_Ea = config['graph_inter_Ea']
        inter_u = config['graph_inter_u']
        device = config['torch_device']
        self.pol_enc = config['graph_pol_enc']
        GNN_args = {'n_neurons':config['graph_neurons_chanel'],
                    'n_hidden_layers' :config['graph_n_hidden_layers'],
                    'device':config['torch_device']
            }
        self.GN_pol = nn.ModuleList([GNBlock(0, self.bt+self.nr+8, self.maxsides, inter_u, inter_Va, inter_Ea,device=device,**GNN_args)]+
                                   [GNBlock(inter_u, inter_Va, inter_Ea, inter_u, inter_Va, inter_Ea,device=device,**GNN_args) for i in range(n_GNblocks-2)]+
                                   [GNBlock(inter_u, inter_Va, inter_Ea, inter_u, inter_Va, 0,device=device,**GNN_args)]
                                   )
        if self.pol_enc == 'one_edge':
            self.GN_out_pol = GNBlock(inter_u, inter_Va, 0, 0, 0, (2*self.maxsides*self.sumsides+1)*n_robots,device=device,**GNN_args)
        if self.pol_enc =='edge_action':
            self.GN_out_pol = GNBlock(inter_u, inter_Va, 0, 0, 0, 1,device=device,**GNN_args)
        
        self.GN_val = nn.ModuleList([GNBlock(0, self.bt+self.nr+8, self.maxsides, inter_u, inter_Va, inter_Ea,device=device,**GNN_args)]+
                                   [GNBlock(inter_u, inter_Va, inter_Ea, inter_u, inter_Va, inter_Ea,device=device,**GNN_args) for i in range(n_GNblocks-1)])
        self.lin_val = nn.Linear(inter_u,1,device=device)
        
        self.loss_val = nn.MSELoss(reduction="mean")
        self.loss_act = None
        self.device=device
        self.pol_net = nn.ModuleList([self.GN_pol,self.GN_out_pol])
        self.val_net = nn.ModuleList([self.GN_val,self.lin_val])
    def forward(self,graph=None,graph_desc=None,inference = False,mask =None,eps=None):
        with torch.inference_mode(inference):
            if inference:
                self.eval()
            else:
                self.train()
            #extract the matrices from the graph
            if graph is not None:
                blocks = torch.tensor(graph.blocks,device=self.device,dtype=torch.long).unsqueeze(0)
                grounds = torch.tensor(graph.grounds,device=self.device,dtype=torch.long).unsqueeze(0)
                Va = torch.cat([grounds,blocks],2)#all the attibutes of the vertices shape: (B,5,n_reg+n_blocks)
                Es = torch.tensor(graph.i_s,device=self.device,dtype=torch.float).unsqueeze(0)#shape:(B,n_reg+n_blocks,maxinterface)
                Er = torch.tensor(graph.i_r,device=self.device,dtype=torch.float).unsqueeze(0)#shape:(B,n_reg+n_blocks,maxinterface)
                Ea = torch.tensor(graph.i_a,dtype=torch.long,device=self.device).unsqueeze(0)#shape:(B,1,maxinterface)
                mask_v = torch.tensor(graph.active_nodes,device=self.device,dtype=bool).unsqueeze(0)
                mask_e = torch.tensor(graph.active_edges,device=self.device,dtype=bool).unsqueeze(0)
                mask=torch.tensor(mask,device=self.device,dtype=bool).unsqueeze(0)
            else:
                Va = graph_desc[0]
                Es = graph_desc[1]
                Er = graph_desc[2]
                Ea = graph_desc[3]
                mask_e = graph_desc[4]
                mask_v = graph_desc[5]
   
            Va_enc = torch.cat([F.one_hot(Va[:,0,:]+1,num_classes=self.bt).permute(0,2,1),
                                F.one_hot(Va[:,1,:]+1,num_classes=self.nr).permute(0,2,1),
                                2*torch.unsqueeze(Va[:,2,:],1).to(torch.float)/self.maxgrid[0]-1,
                                2*torch.unsqueeze(Va[:,3,:],1).to(torch.float)/self.maxgrid[1]-1,
                                F.one_hot(Va[:,4,:],num_classes=6).permute(0,2,1)
                                          ],1)
            
            Ea_enc = F.one_hot(Ea[:,0,:],num_classes=self.maxsides).permute(0,2,1).to(torch.float)
            
            u = torch.zeros((Ea.shape[0],0,1),device=self.device)
 
  
            #comput the value function

            val_v = Va_enc
            val_e = Ea_enc
            val_u = u
            for block in self.GN_val:
                val_e,val_v,val_u = block(val_e,Es,Er,val_v,val_u,mask_e = mask_e,mask_v=mask_v)
            val = self.lin_val(val_u.squeeze(2))

        
            
            #compute the policy
            pol_v = Va_enc
            pol_e = Ea_enc
            pol_u = u
            
            for block in self.GN_pol:
                pol_e,pol_v,pol_u = block(pol_e,Es,Er,pol_v,pol_u,mask_e = mask_e,mask_v=mask_v)
            
            #add the action node:
            if self.last_only:
       
                if self.pol_enc == 'one_edge':
                    #this method does not work when 'R'is in the action choice
                    last_node = torch.sum(mask_v,1).to(int)-1
                    #get the maximum number of nodes in the batch
                    last_node_max = last_node.max()
                    Es_outpol = F.one_hot(last_node,last_node_max+2).to(torch.float).unsqueeze(2)
                    Er_outpol= torch.zeros(Es_outpol.shape,device=self.device)
                    Er_outpol[:,-1]=1
                    pol_v = torch.cat([pol_v[:,:,:last_node_max+1],torch.ones(pol_v.shape[0],pol_v.shape[1],1,device=self.device)],2)
                    pol_e = torch.zeros(pol_e.shape[0],0,1,device=self.device)
                else:
                    assert False,"Not implemented"
                

                pol_e,pol_v,pol_u = self.GN_out_pol(pol_e,Es_outpol,Er_outpol,pol_v,pol_u)
            
            
                
            if self.pol_enc == 'one_edge':
                pol_e = pol_e.squeeze(2)
                if mask is not None:
                    
                    #mask_eout = mask-1
                    pol_e[~mask]-= 1e10
            pol = F.softmax(pol_e,dim=1)
           
            if torch.any(torch.isnan(pol)):
                assert False, 'nans in the policy'
            
            return val,pol
class A2CDenseOptimizer():
    def __init__(self,model,config):
        lr = config['opt_lr']
        wd = config['opt_weight_decay']
        self.max_norm = config['opt_max_norm']
        self.pol_over_val = config['opt_pol_over_val']
        self.tau = config['opt_tau']
        self.exploration_factor=config['opt_exploration_factor']
        
        self.target_model = copy.deepcopy(model)
        
        self.optimizer_pol = torch.optim.NAdam(model.pol_net.parameters(),lr=lr*self.pol_over_val,weight_decay=wd)
        self.optimizer_val = torch.optim.NAdam(model.val_net.parameters(),lr=lr,weight_decay=wd)
        self.model = model
        self.step = 0
    def optimize(self,state,actionid,rewards,nstates,gamma,mask=None,nmask=None):
        v,pol = self.model(graph_desc = state,mask=mask)
        entropy = torch.sum(-pol*torch.log(pol+1e-10),dim=1)
        entropy = entropy.mean()
        next_v,_  = self.target_model(graph_desc=nstates,mask=nmask,inference = True)
        #put the value of the terminal state at 0
        with torch.inference_mode(): 
            next_v[~torch.any(nmask,dim=1)]=0
        Qval = rewards + gamma*next_v
        if self.model.use_wandb:
            wandb.log({'value_est': v.mean().detach().cpu().numpy(), 'target_value_est':Qval.mean().detach().cpu().numpy(), 'rewards':rewards.mean().detach().cpu().numpy()},step=self.step)
        #compute the critique loss
        l_v = self.model.loss_val(v,Qval)
        advantage = Qval-v.detach()
        nll = -torch.log(1e-10+pol[torch.arange(pol.shape[0]),actionid])
        l_p = torch.mean(torch.unsqueeze(nll,1)*advantage)
        if self.model.use_wandb:
            wandb.log({'policy_entropy':entropy.detach().cpu().numpy()},step=self.step)
        l_p -= self.exploration_factor*entropy
       
            
        self.optimizer_val.zero_grad()
        l_v.backward()
        if self.max_norm is not None:
            norm_v = nn.utils.clip_grad_norm_(self.model.val_net.parameters(),self.max_norm)
            wandb.log({'value_grad_norm':norm_v.detach().cpu().numpy()},step=self.step)
        self.optimizer_val.step()
        
        self.optimizer_pol.zero_grad()
        l_p.backward()
        if self.max_norm is not None:
            norm_p = nn.utils.clip_grad_norm_(self.model.pol_net.parameters(),self.max_norm)
            wandb.log({'policy_grad_norm':norm_p.detach().cpu().numpy()},step=self.step)
        self.optimizer_pol.step()
    
        #update the target
        sd_target = self.target_model.state_dict()
        sd = self.model.state_dict()
        for key in sd_target:
            sd_target[key]= (1-self.tau)*sd_target[key]+self.tau*sd[key]
        self.target_model.load_state_dict(sd_target)
        if self.model.use_wandb:
            wandb.log({"l_v": l_v.detach().cpu().numpy(),
                       "l_p": l_p.detach().cpu().numpy()},step=self.step)
        #wandb.watch(self.model)
        self.step+=1
        return l_v.detach().cpu().numpy(), l_p.detach().cpu().numpy()

    
class A2CShared(nn.Module):
    def __init__(self,
                 maxs_grid,
                 max_blocks,
                 n_robots,
                 n_regions,
                 n_actions,
                 config,
                 use_wandb=False):
        super().__init__()
        #unpack the config file
        n_fc_layer = config['A2C_n_fc_layer']
        n_neurons = config['A2C_n_neurons']
        batch_norm = config['A2C_batch_norm']
        device = config['torch_device']
        shared = config['A2C_shared']
        encoder_args = {'n_channels':config['SEnc_n_channels'],
                        'n_internal_layer':config['SEnc_n_internal_layer'],
                        'stride':config['SEnc_stride']}
        
        self.use_wandb=use_wandb
        self.shared = shared
        if shared:
            self.state_encoder = StateEncoder(maxs_grid,
                                                  max_blocks,
                                                  n_robots,
                                                  n_regions,
                                                  device=device,
                                                  **encoder_args)
            self.input_norm = nn.BatchNorm2d(self.state_encoder.out_dims[0],device=device)
            self.FC_val = nn.ModuleList([nn.Linear(np.prod(self.state_encoder.out_dims),n_neurons,device=device)])
            self.FC_pol = nn.ModuleList([nn.Linear(np.prod(self.state_encoder.out_dims),n_neurons,device=device)])
        else:
            self.state_encoder_val = StateEncoder(maxs_grid,
                                                  max_blocks,
                                                  n_robots,
                                                  n_regions,
                                                  device=device,
                                                  **encoder_args)
            self.state_encoder_pol = StateEncoder(maxs_grid,
                                                  max_blocks,
                                                  n_robots,
                                                  n_regions,
                                                  device=device,
                                                  **encoder_args)
            self.input_norm_pol = nn.BatchNorm2d(self.state_encoder_pol.out_dims[0],device=device)
            self.input_norm_val = nn.BatchNorm2d(self.state_encoder_val.out_dims[0],device=device)
            self.FC_pol = nn.ModuleList([nn.Linear(np.prod(self.state_encoder_pol.out_dims),n_neurons,device=device)])
            self.FC_val = nn.ModuleList([nn.Linear(np.prod(self.state_encoder_val.out_dims),n_neurons,device=device)])
            
        self.FC_val+=nn.ModuleList([nn.Linear(n_neurons,n_neurons,device=device) for i in range(n_fc_layer-1)])
        self.val_norm = nn.BatchNorm1d(n_neurons,device=device)
        self.out_val =nn.Linear(n_neurons,1,device=device)
        
        self.FC_pol+=nn.ModuleList([nn.Linear(n_neurons,n_neurons,device=device) for i in range(n_fc_layer-1)])
        self.out_pol = nn.Linear(n_neurons,n_actions,device=device)
        
        self.loss_val = nn.MSELoss(reduction="mean")
        self.loss_act = None
        self.device=device
        self.batch_norm = batch_norm
        if shared:
            self.pol_net = nn.ModuleList([self.state_encoder,self.input_norm,self.FC_pol,self.out_pol])
            self.val_net = nn.ModuleList([self.state_encoder,self.input_norm,self.FC_val,self.val_norm,self.out_val])
        else:
            self.pol_net = nn.ModuleList([self.state_encoder_pol,self.input_norm_pol,self.FC_pol,self.out_pol])
            self.val_net = nn.ModuleList([self.state_encoder_val,self.input_norm_val,self.FC_val,self.val_norm,self.out_val])
    def forward(self,grids,inference = False,mask =None):
        with torch.inference_mode(inference):
            if inference:
                self.eval()
            else:
                self.train()
            if self.shared:
                if self.batch_norm:
                    normed_rep = self.input_norm(self.state_encoder(grids))
                    rep = torch.flatten(normed_rep,1)
                else:
                    rep = torch.flatten(self.state_encoder(grids),1)
                rep_val = rep
                rep_pol = rep
            else:
                if self.batch_norm:
                    normed_rep_pol = self.input_norm_pol(self.state_encoder_pol(grids))
                    rep_pol = torch.flatten(normed_rep_pol,1)
                    normed_rep_val = self.input_norm_val(self.state_encoder_val(grids))
                    rep_val = torch.flatten(normed_rep_val,1)
                else:
                    rep_val = torch.flatten(self.state_encoder_val(grids),1)
                    rep_pol = torch.flatten(self.state_encoder_pol(grids),1)
            for layer in self.FC_val:
                rep_val = F.relu(layer(rep_val))
            # if self.batch_norm:
            #     if rep_val.shape[0] > 1:
            #         val = self.out_val(self.val_norm(rep_val))
            #     else:
            #         val = self.out_val(0*rep_val)
            # else:
            val = self.out_val(rep_val)
            for layer in self.FC_pol:
                rep_pol = F.relu(layer(rep_pol))
            if mask is not None:
                mask = torch.tensor(~mask,device=self.device,dtype=torch.float).reshape(len(grids),-1)
                rep_pol = self.out_pol(rep_pol)
                rep_pol = rep_pol-mask*1e10
                dist = Categorical(logits = rep_pol)
            else:
                dist = Categorical(logits = rep_pol)
            if torch.any(torch.isnan(rep_pol)):
                assert False, 'nans in the policy'
            
            return val,dist
class A2CSharedOptimizer():
    def __init__(self,model,config):
        lr = config['opt_lr']
        wd = config['opt_weight_decay']
        self.pol_over_val = config['opt_pol_over_val']
        self.tau = config['opt_tau']
        self.exploration_factor=config['opt_exploration_factor']
        
        self.target_model = copy.deepcopy(model)
        if model.shared:
            self.optimizer = torch.optim.NAdam(model.parameters(),lr=lr,weight_decay=wd)
        else:
            self.optimizer_pol = torch.optim.NAdam(model.pol_net.parameters(),lr=lr*self.pol_over_val,weight_decay=wd)
            self.optimizer_val = torch.optim.NAdam(model.val_net.parameters(),lr=lr,weight_decay=wd)
        #torch.optim.Adam(model.out_val.parameters(),lr=lr_val)#,momentum = 0.01,
                                             #nesterov = True)
        #self.optimizer_pol = torch.optim.NAdam(model.parameters(),lr=lr_pol,weight_decay=0.001)
        #torch.optim.Adam(model.parameters(),lr=lr_pol)#,momentum = 0.01,
                                            #nesterov = True)
        self.model = model
        self.step=0
    def optimize(self,state,actionid,rewards,nstates,gamma,mask=None,nmask=None):
        
        v,pol = self.model(state,mask=mask)
        #entropy = torch.sum(-pol*torch.log(pol+1e-10),dim=1)
        entropy = pol.entropy()
        entropy = entropy.mean()
        next_v,_  = self.target_model(nstates,mask=nmask,inference = True)
        #put the value of the terminal state at 0
        with torch.inference_mode(): 
            next_v[~np.any(nmask,axis=1)]=0
        Qval = torch.tensor(rewards).to(next_v.device) + gamma*next_v
        if self.model.use_wandb:
            wandb.log({'V': v.mean(), 'tV':next_v.mean(), 'rewards':rewards.mean()},step=self.step)
        #compute the critique loss
        l_v = self.model.loss_val(v,Qval)
        advantage = Qval-v.detach()
        nll = -torch.log(1e-10+pol.probs[torch.arange(pol.probs.shape[0]),actionid])
        l_p = torch.mean(torch.unsqueeze(nll,1)*advantage)
        l_p -= self.exploration_factor*entropy
        if self.model.shared:
            self.optimizer.zero_grad()
            l_tot = l_v+l_p*self.pol_over_val
            #self.optimizer_pol.zero_grad()
            #l_pol.backward(retain_graph=True)
            
            
            l_tot.backward()
            self.optimizer.step()
            #self.optimizer_pol.step()
        else:
            
            self.optimizer_val.zero_grad()
            l_v.backward()
            self.optimizer_val.step()
            
            self.optimizer_pol.zero_grad()
            l_p.backward()
            self.optimizer_pol.step()
        
        #update the target
        sd_target = self.target_model.state_dict()
        sd = self.model.state_dict()
        for key in sd_target:
            sd_target[key]= (1-self.tau)*sd_target[key]+self.tau*sd[key]
        self.target_model.load_state_dict(sd_target)
        if self.model.use_wandb:
            wandb.log({"l_v": l_v.detach().cpu().numpy(),
                       "l_p": l_p.detach().cpu().numpy(),
                       'policy_entropy':entropy},step=self.step)
        #wandb.watch(self.model)
        self.step+=1
        return l_v.detach().cpu().numpy(), l_p.detach().cpu().numpy()
class A2CSharedEncDec(nn.Module):
    def __init__(self,
                 maxs_grid,
                 max_blocks,
                 n_robots,
                 n_regions,
                 n_block_type,
                 encoder_args={},
                 decoder_args={},
                 n_fc_layer_val =2,
                 n_fc_layer_pol =2,
                 n_neurons = 100,
                 batch_norm= True,
                 device='cpu',
                 use_wandb=False
                 ):
        super().__init__()
        self.use_wandb=use_wandb
        self.state_encoder = StateEncoder(maxs_grid,
                                              max_blocks,
                                              n_robots,
                                              n_regions,
                                              device=device,
                                              **encoder_args)
        self.action_decoder = ActionDecoder(maxs_grid,n_block_type, n_robots,
                                            device=device,
                                            **decoder_args
                                            )
        self.input_norm = nn.BatchNorm2d(self.state_encoder.out_dims[0],device=device)
        self.FC_val = nn.ModuleList([nn.Linear(np.prod(self.state_encoder.out_dims),n_neurons,device=device)])
        self.FC_val+=nn.ModuleList([nn.Linear(n_neurons,n_neurons,device=device) for i in range(n_fc_layer_val-1)])
        self.val_norm = nn.BatchNorm1d(n_neurons,device=device)
        self.out_val =nn.Linear(n_neurons,1,device=device)
        
        self.FC_pol = nn.ModuleList([nn.Linear(np.prod(self.state_encoder.out_dims),n_neurons,device=device)])
        self.FC_pol+=nn.ModuleList([nn.Linear(n_neurons,n_neurons,device=device) for i in range(n_fc_layer_pol-2)])
        self.FC_pol += nn.ModuleList([nn.Linear(n_neurons,np.prod(self.action_decoder.in_dims),device=device)])
        
        self.loss_val = nn.MSELoss(reduction="mean")
        self.loss_act = None
        self.device=device
        self.batch_norm = batch_norm
    def forward(self,grids,inference = False,mask =None):
        with torch.inference_mode(inference):
            if inference:
                self.eval()
            else:
                self.train()
            if self.batch_norm:
                normed_rep = self.input_norm(self.state_encoder(grids))
                rep = torch.flatten(normed_rep,1)
            else:
                rep = torch.flatten(self.state_encoder(grids),1)
            rep_val = rep
            for layer in self.FC_val:
                rep_val = F.relu(layer(rep_val))
            if self.batch_norm:
                if rep_val.shape[0] > 1:
                    val = self.out_val(self.val_norm(rep_val))
                else:
                    val = self.out_val(0*rep_val)
            else:
                val = self.out_val(rep_val)
            rep_pol = rep
            for layer in self.FC_pol:
                rep_pol = F.relu(layer(rep_pol))
            pol = self.action_decoder(rep_pol.reshape((rep_pol.shape[0],
                                                       self.action_decoder.in_dims[0],
                                                       self.action_decoder.in_dims[1],
                                                       self.action_decoder.in_dims[2])),mask=mask)
            if torch.any(torch.isnan(pol)):
                assert False, 'nans in the policy'
            
            return val,pol
class A2CSharedEncDecOptimizer():
    def __init__(self,model,lr=1e-3,pol_over_val = 2e-5,tau=1e-5):
        self.pol_over_val = pol_over_val
        #lr_pol = 5e-6
        self.tau = tau
        self.target_model = copy.deepcopy(model)
        self.optimizer = torch.optim.NAdam(model.parameters(),lr=lr,weight_decay=0.001)
        #torch.optim.Adam(model.out_val.parameters(),lr=lr_val)#,momentum = 0.01,
                                             #nesterov = True)
        #self.optimizer_pol = torch.optim.NAdam(model.parameters(),lr=lr_pol,weight_decay=0.001)
        #torch.optim.Adam(model.parameters(),lr=lr_pol)#,momentum = 0.01,
                                            #nesterov = True)
        self.model = model
    def optimize(self,state,actioncoords,rewards,nstates,gamma,mask=None,nmask=None,exploration_factor=0.1):
        v,pol = self.model(state,mask=mask)
        entropy = torch.sum((-pol*torch.log(pol+1e-10)).flatten(start_dim=1),dim=1)
        entropy = entropy.mean()
        next_v,_  = self.target_model(nstates,mask=nmask,inference = True)
        Qval = torch.tensor(rewards).to(next_v.device) + gamma*next_v
        if self.model.use_wandb:
            wandb.log({'value_est': v.mean(), 'target_value_est':Qval.mean(), 'rewards':rewards.mean()})
        #compute the critique loss
        l_v = self.model.loss_val(v,Qval)
        
        advantage = Qval-v.detach()
        nll = -torch.log(1e-10+pol[torch.arange(pol.shape[0]),actioncoords[:,0],actioncoords[:,1],actioncoords[:,2]])
        l_p = torch.mean(torch.unsqueeze(nll,1)*advantage)
        if self.model.use_wandb:
            wandb.log({'policy_entropy':entropy})
        l_p -= exploration_factor*entropy
         
        self.optimizer.zero_grad()
        l_tot = l_v+l_p*self.pol_over_val
        #self.optimizer_pol.zero_grad()
        #l_pol.backward(retain_graph=True)
        
        
        l_tot.backward()
        self.optimizer.step()
        #self.optimizer_pol.step()
        
        
        #update the target
        sd_target = self.target_model.state_dict()
        sd = self.model.state_dict()
        for key in sd_target:
            sd_target[key]= (1-self.tau)*sd_target[key]+self.tau*sd[key]
        self.target_model.load_state_dict(sd_target)
        if self.model.use_wandb:
            wandb.log({"l_v": l_v.detach().cpu().numpy(),
                       "l_p": l_p.detach().cpu().numpy()})
        return l_v.detach().cpu().numpy(), l_p.detach().cpu().numpy()
  
    
class GNBlock(nn.Module):
    def __init__(self,
                 in_u,
                 in_Va,
                 in_Ea,
                 out_u,
                 out_Va,
                 out_Ea,
                 n_neurons=32,
                 n_hidden_layers = 3,
                 device='cpu',
                 non_lin=F.relu):
        super().__init__()
        self.fe = nn.ModuleList([nn.Linear(in_Ea+2*in_Va+in_u,n_neurons,device=device)]+
                                [nn.Linear(n_neurons,n_neurons,device=device) 
                                     for i in range(n_hidden_layers)]+
                                [nn.Linear(n_neurons,out_Ea,device=device)])
        self.fv = nn.ModuleList([nn.Linear(in_Va+out_Ea+in_u,n_neurons,device=device)]+
                                [nn.Linear(n_neurons,n_neurons,device=device) 
                                     for i in range(n_hidden_layers)]+
                                [nn.Linear(n_neurons,out_Va,device=device)])
        self.fu = nn.ModuleList([nn.Linear(in_u+out_Va+out_Ea,n_neurons,device=device)]+
                                [nn.Linear(n_neurons,n_neurons,device=device) 
                                     for i in range(n_hidden_layers)]+
                                [nn.Linear(n_neurons,out_u,device=device)])
        self.device=device
        self.in_u=in_u
        self.in_Va=in_Va
        self.in_Ea=in_Ea
        self.out_u=out_u
        self.out_Va=out_Va
        self.out_Ea=out_Ea
        self.non_lin = non_lin
    def forward(self,E_a,E_s,E_r,V_a,u,mask_v=None,mask_e=None,mask_u = None):
        t_shape = 0
        t_lin = 0
        t_shape-= time.perf_counter()
        B = torch.cat([E_a,V_a@E_s,V_a@E_r,u.expand(-1,-1,E_a.shape[-1])],1)
        new_e = B.permute(0,2,1)[mask_e]
        out_E = torch.zeros((E_a.shape[0],
                             E_a.shape[2],
                             self.out_Ea,),device=self.device)
        t_shape+= time.perf_counter()
        t_lin-= time.perf_counter()
        for layer in self.fe[:-1]:
            new_e = self.non_lin(layer(new_e))
        new_e = self.fe[-1](new_e)
        t_lin+= time.perf_counter()
        t_shape-= time.perf_counter()
        out_E[mask_e] = new_e
        out_E = out_E.permute(0,2,1)
        # print(f"edges: {t1-t0}")
        C = torch.cat([V_a, out_E@E_r.mT,u.expand(-1,-1,V_a.shape[-1])],1)
        new_v = C.permute(0,2,1)[mask_v]
        out_V = torch.zeros((V_a.shape[0],V_a.shape[2],self.out_Va),device=self.device)
        t_shape+= time.perf_counter()
        t_lin-= time.perf_counter()
        for layer in self.fv[:-1]:
            new_v = self.non_lin(layer(new_v))
        new_v = self.fv[-1](new_v)
        t_lin+= time.perf_counter()
        t_shape-= time.perf_counter()
        out_V[mask_v]=new_v
        out_V = out_V.permute(0,2,1)
        # if mask_v is not None:
        #     out_V = out_V*mask_v.unsqueeze(1).expand(-1,out_V.shape[1],-1)
        # print(f"vertices: {t1-t0}")
        P = torch.cat([u.squeeze(2),out_V.sum(dim=2),out_E.sum(dim=2)],1)
        out_u = P
        t_shape+= time.perf_counter()
        t_lin-= time.perf_counter()
        for layer in self.fu[:-1]:
            out_u = self.non_lin(layer(out_u))
        out_u = self.fu[-1](out_u)
        t_lin+= time.perf_counter()
        #print(f"     {t_shape=}")
        #print(f"     {t_lin=}")
        return out_E,out_V,out_u.unsqueeze(2)

class StateEncoder(nn.Module):
    def __init__(self,
                 maxs_grid,
                 max_blocks,
                 n_robots,
                 n_regions,
                 n_channels = 32,
                 n_internal_layer = 2,
                 stride=1,
                 device='cpu'):
        super().__init__()
        self.mbl=max_blocks
        self.nrob = n_robots
        self.nreg = n_regions
        self.convin = nn.Conv2d(6,n_channels,3,stride=1,device=device)
        self.convinternal = nn.ModuleList([nn.Conv2d(n_channels,n_channels,3,stride=stride,device=device) for i in range(n_internal_layer)])
        self.device=device
        dimx = maxs_grid[0]-1
        dimy = maxs_grid[1]-1
        for i in range(n_internal_layer):
            dimx = (dimx-3)//stride+1
            dimy = (dimy-3)//stride+1
        assert dimx > 0 and dimy>0, "out dims are negative"
        self.out_dims = [n_channels,dimx,dimy]
    def forward(self,grids):
        # occ = self.occ_emb(torch.tensor(np.array([grid.occ for grid in grids]),device=self.device)+1)
        # hold = self.hold_emb(torch.tensor(np.array([grid.hold for grid in grids]),device=self.device)+1)
        # con = self.con_emb(torch.tensor(np.array([grid.connection for grid in grids]),device=self.device)+1)
        occ = torch.tensor(np.array([grid.occ for grid in grids]),device=self.device)/self.mbl
        hold = torch.tensor(np.array([grid.hold for grid in grids]),device=self.device)/self.nrob
        con = torch.tensor(np.array([grid.connection for grid in grids]),device=self.device)/self.nreg
        inputs = torch.cat([occ,hold,con],3)
        if torch.any(torch.isnan(inputs)):
            assert False, 'nans in the input'
        inputs= inputs.permute(0,3,1,2)
        #inputs= inputs.permute(0,4,1,2,3)
        # rep = torch.cat([F.relu(self.convinup(inputs[...,0])),
        #                  F.relu(self.convindown(inputs[...,1]))],1)
        rep = F.relu(self.convin(inputs))
        for conv in self.convinternal:
            rep = F.relu(conv(rep))
        return rep
class StateEncoderOE(nn.Module):
    def __init__(self,
                 maxs_grid,
                 n_robots,
                 n_regions,
                 last_only,
                 n_channels = 32,
                 n_internal_layer = 2,
                 stride=1,
                 device='cpu'):
        super().__init__()
        if not last_only:
            print("Warning, this encoder cannot differentiate between blocks, so last-only option is highly recommended")
        self.nrob = n_robots
        self.nreg = n_regions
        self.convin = nn.Conv2d(14,n_channels,3,stride=1,device=device)
        self.convinternal = nn.ModuleList([nn.Conv2d(n_channels,n_channels,3,stride=stride,device=device) for i in range(n_internal_layer)])
        self.device=device
        dimx = maxs_grid[0]-1
        dimy = maxs_grid[1]-1
        for i in range(n_internal_layer):
            dimx = (dimx-3)//stride+1
            dimy = (dimy-3)//stride+1
        assert dimx > 0 and dimy>0, "out dims are negative"
        self.out_dims = [n_channels,dimx,dimy]
    def forward(self,grids):
        sides = torch.tensor(np.array([grid.neighbours[:,:,:,:,0]>-1 for grid in grids]),device=self.device,dtype=torch.float)
        last_block = torch.tensor(np.array([grid.occ== np.max(grid.occ) for grid in grids]),device=self.device,dtype=torch.float)
        ground = torch.tensor(np.array([grid.occ== 0 for grid in grids]),device=self.device,dtype=torch.float)
        hold = torch.tensor(np.array([grid.hold for grid in grids]),device=self.device)/self.nrob
        con = torch.tensor(np.array([grid.connection for grid in grids]),device=self.device)/self.nreg
        
        inputs = torch.cat([sides.flatten(-2),last_block,ground,hold,con],3)
        if torch.any(torch.isnan(inputs)):
            assert False, 'nans in the input'
        inputs= inputs.permute(0,3,1,2)
        #inputs= inputs.permute(0,4,1,2,3)
        # rep = torch.cat([F.relu(self.convinup(inputs[...,0])),
        #                  F.relu(self.convindown(inputs[...,1]))],1)
        rep = F.relu(self.convin(inputs))
        for conv in self.convinternal:
            rep = F.relu(conv(rep))
        return rep
class ActionDecoder(nn.Module):
    def __init__(self,
                 maxs_grid,
                 n_block_type,
                 n_robots,
                 n_channels = 32,
                 n_internal_layer = 1,
                 stride=1,
                 device='cpu'):
        super().__init__()
        self.actions_per_loc = n_robots*(n_block_type*12+5)#6rotations *2 + 2orientations*2+len([L])
        self.convout = nn.ConvTranspose2d(n_channels,self.actions_per_loc,3,stride=1,device=device)
        self.convinternal = [nn.ConvTranspose2d(n_channels,n_channels,3,stride=stride,device=device) for i in range(n_internal_layer)]
        self.device=device
        dimx = maxs_grid[0]-2
        dimy = maxs_grid[1]-2
        for i in range(n_internal_layer):
            dimx = (dimx-3)//stride+1
            dimy = (dimy-3)//stride+1
        assert dimx > 0 and dimy>0, "indims are negative"
        self.in_dims = [n_channels,dimx,dimy]
    def forward(self,dense_rep,mask=None):
        if torch.any(torch.isnan(dense_rep)):
            assert False, 'nans in the dense representation'
        rep=dense_rep
        for conv in self.convinternal:
            rep = F.relu(conv(rep))
        rep = self.convout(rep)
        if mask is not None:
            mask = torch.tensor(~mask,device=self.device,dtype=torch.float).reshape(rep.shape)
            rep = rep-mask*1e10
            out = F.softmax(rep.flatten(start_dim=1),dim=1).reshape(mask.shape)
        else:
            out = F.softmax(rep.flatten(start_dim=1),dim=1).reshape(rep.shape)
            
        return out
class WolpertingerQTable(nn.Module):
    #initialise a neural network that maps action and state to discounted total reward
    def __init__(self,
                 maxs_grid,
                 max_blocks,
                 n_robots,
                 n_regions,
                 n_fc_layer =2,
                 n_neurons = 100,
                 device='cpu',
                 **encoder_args):
        super().__init__()
        
        self.state_encoder = StateEncoder(maxs_grid,
                                          max_blocks,
                                          n_robots,
                                          n_regions,
                                          device=device,
                                          **encoder_args)
        self.ain = nn.Linear(25,n_neurons,device=device)
        
        self.FC = [nn.Linear(np.prod(self.state_encoder.out_dims)+n_neurons,n_neurons,device=device)]
        self.FC += [nn.Linear(n_neurons,n_neurons,device=device) for i in range(n_fc_layer-1)]
        self.out = nn.Linear(n_neurons,1,device=device)
        self.loss = nn.MSELoss(reduction="sum")
        self.device=device
    def forward(self,grids,actions_vec,inference = False,explore=False):
        with torch.inference_mode(inference):
            
            if not isinstance(actions_vec,torch.Tensor):
                actions_vec = torch.tensor(actions_vec,dtype=torch.float32,device=self.device)
            else:
                actions_vec = actions_vec.to(self.device)
            
            shape = actions_vec.shape[:-1]
            
            
            actions_vec = torch.flatten(actions_vec,0,len(shape)-1)
            
            reps = self.state_encoder(grids,inference=inference)
            reps = torch.flatten(reps,1)
            reps_aug = torch.repeat_interleave(reps, shape[1], dim=0)
            repa = F.relu(self.ain(actions_vec))
            rep = torch.cat([reps_aug,repa],1)
            for layer in self.FC:
                rep = F.relu(layer(rep))
            
            val_est = torch.reshape(self.out(rep),shape)
            if explore:
                val_est = torch.softmax(val_est,dim=1)
            return val_est
            