# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:33:58 2022

@author: valla
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiscreteStateEmbedding(nn.Module):
    def __init__(self,max_blocks,n_robots,n_regions,dim_b=4,dim_r=2,dim_c=2):
        super().__init()
        self.occ_emb = nn.Embedding(max_blocks,dim_b)
        self.hold_emb = nn.Embedding(n_robots, dim_r)
        self.con_emb = nn.Embedding(n_regions,dim_c)
    def forward(self,grid):
        occ = self.occ_emb(grid.occ)
        hold = self.hold_emb(grid.hold)
        con = self.con_emb(grid.connection)
        return 
class transition_estimator():
    def __init__(maxs, action_shape):
        pass
def UCB_approx(policy,
               transition,
               reward_f,
               state_0,
               n_sample=100,
               n_steps_max = 1000):
    cum_r = 0
    for step in range(n_steps_max):
        for hallu in range(n_sample):
            action = 0
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
            v = self.QT(states,protoaction).to(self.act_net.device)
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
        
        self.optimizer = torch.optim.SGD(QT.parameters(),lr=0.0005)#,momentum = 0.01,
                                           #nesterov = True)
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
class StateEncoder(nn.Module):
    def __init__(self,
                 maxs_grid,
                 max_blocks,
                 n_robots,
                 n_regions,
                 dim_b=4,
                 dim_r=2,
                 dim_c=2,
                 n_channels = 128,
                 n_internal_layer = 0,
                 device='cpu'):
        super().__init__()
        self.occ_emb = nn.Embedding(max_blocks+1,dim_b,device=device)
        self.hold_emb = nn.Embedding(n_robots+1, dim_r,device=device)
        self.con_emb = nn.Embedding(n_regions+1,dim_c,device=device)
        self.convinup = nn.Conv2d(dim_b+dim_r+dim_c, n_channels//2, 3,stride=1,device=device)
        self.convindown = nn.Conv2d(dim_b+dim_r+dim_c, n_channels//2, 3,stride=1,device=device)
        self.convinternal = [nn.Conv2d(n_channels,n_channels,3,stride=2,device=device) for i in range(n_internal_layer)]
        self.device=device
        dimx = maxs_grid[0]-1
        dimy = maxs_grid[1]-1
        for i in range(n_internal_layer):
            dimx = (dimx-3)//2+1
            dimy = (dimy-3)//2+1
        self.out_dims = [n_channels,dimx,dimy]
    def forward(self,grids,inference = False):
        with torch.inference_mode(inference):
            occ = self.occ_emb(torch.tensor(np.array([grid.occ for grid in grids]),device=self.device)+1)
            hold = self.hold_emb(torch.tensor(np.array([grid.hold for grid in grids]),device=self.device)+1)
            con = self.con_emb(torch.tensor(np.array([grid.connection for grid in grids]),device=self.device)+1)
            
            inputs = torch.cat([occ,hold,con],4)
            inputs= inputs.permute(0,4,1,2,3)
            rep = torch.cat([F.relu(self.convinup(inputs[...,0])),
                          F.relu(self.convindown(inputs[...,1]))],1)
            
            for conv in self.convinternal:
                rep = F.relu(conv(rep))
            return rep
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
    def forward(self,grids,actions_vec,inference = False,noise_amp=0):
        with torch.inference_mode(inference):
            
            if not isinstance(actions_vec,torch.Tensor):
                actions_vec = torch.tensor(actions_vec,dtype=torch.float32,device=self.device)
            else:
                actions_vec = actions_vec.to(self.device)
            if len(grids)==1:
                grids = actions_vec.shape[0]*grids
            
            reps = self.state_encoder(grids,inference=inference)
            repa = F.relu(self.ain(actions_vec))
            rep = torch.cat([torch.flatten(reps,1),repa],1)
            for layer in self.FC:
                rep = F.relu(layer(rep))
            
            val_est = self.out(rep)
            val_est += torch.normal(mean=torch.zeros_like(val_est),std = noise_amp*torch.ones_like(val_est))
            return val_est
            