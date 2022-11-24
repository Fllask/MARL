# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:33:58 2022

@author: valla
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
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
                 last_only = True,
                 n_GNblocks = 3,
                 embed_dims_V = 4,
                 embed_dims_E = 4,
                 inter_Va = 32,
                 inter_Ea = 32,
                 inter_u = 32,
                 device='cpu',
                 use_wandb=False,
                 **GNN_args):
        super().__init__()
        self.use_wandb=use_wandb
        
        self.maxgrid = maxs_grid
        self.bt = n_type_block+1 #-1 is used to discribe the ground
        self.nr = n_robots+1 # -1 is used when the block is not held
        self.maxsides = max(n_sides)
        self.sumsides = sum(n_sides)
        self.last_only = last_only
        self.GN_pol = torch.ModuleList([GNBlock(0, embed_dims_V, embed_dims_E, inter_u, inter_Va, inter_Ea,device=device,**GNN_args)]+
                                   [GNBlock(inter_u, inter_Va, inter_Ea, inter_u, inter_Va, inter_Ea,device=device,**GNN_args) for i in range(n_GNblocks-2)]
                                   )
        
        self.GN_out_pol = torch.ModuleList([GNBlock(inter_u, inter_Va, inter_Ea, 0, 0, 2*self.maxside*self.sumsides+1,device=device,**GNN_args) for i in range(2)])
        
        self.GN_val = torch.ModuleList([GNBlock(0, embed_dims_V, embed_dims_E, inter_u, inter_Va, inter_Ea,device=device,**GNN_args)]+
                                   [GNBlock(inter_u, inter_Va, inter_Ea, inter_u, inter_Va, inter_Ea,device=device,**GNN_args) for i in range(n_GNblocks-1)])
        self.lin_val = nn.Linear(inter_u,1)
        
        self.loss_val = nn.MSELoss(reduction="mean")
        self.loss_act = None
        self.device=device
    def forward(self,graph_ar,inference = False,mask =None):
        with torch.inference_mode(inference):
            if inference:
                self.eval()
            else:
                self.train()
            #extract the matrices from the graph
            blocks = torch.tensor([graph.blocks for graph in graph_ar])
            grounds = torch.tensor([graph.grounds for graph in graph_ar])
            Va = torch.cat([grounds,blocks],1)#all the attibutes of the vertices shape: (B,5,n_reg+n_blocks)
            Es = torch.tensor([graph.i_s for graph in graph_ar])#shape:(B,n_reg+n_blocks,maxinterface)
            Er = torch.tensor([graph.i_r for graph in graph_ar])#shape:(B,n_reg+n_blocks,maxinterface)
            Ea = torch.tensor([graph.i_a for graph in graph_ar])#shape:(B,1,maxinterface)
            if mask is not None:
                mask_e = torch.tensor(mask['e'])#shape: (B,maxinterface)
                mask_v = torch.tensor(mask['v'])#shape: (B,n_reg+n_blocks)
            else:
                mask_e = None
                mask_v = None
            Va_enc = torch.cat([F.one_hot(Va[:,0,:],num_classes=self.bt),
                                F.one_hot(Va[:,1,:],num_classes=self.nr),
                                2*Va[:,2,:]/self.maxgrid[0]-1,
                                2*Va[:,3,:]/self.maxgrid[1]-1,
                                F.one_hot(Va[:,4,:],num_classes=6)
                                          ],1)
            
            Ea_enc = F.one_hot(Ea,num_classes=self.max_sides)
            
            u = torch.zeros((Ea.shape[0],0),device=self.device)
            
            #comput the value function
            val_v = Va_enc
            val_e = Ea_enc
            val_u = u
            for block in self.GN_val:
                val_e,val_v,val_u = block(val_e,Es,Er,val_v,val_u,mask_e = mask_e,mask_v=mask_v)
            val = self.lin_val(val_u)
            
            
            
            #compute the policy
            pol_v = Va_enc
            pol_e = Ea_enc
            pol_u = u
            
            for block in self.GN_pol:
                pol_e,pol_v,pol_u = block(pol_e,Es,Er,pol_v,pol_u,mask_e = mask_e,mask_v=mask_v)
            
            #add the action node:
            if self.last_only:
                last_node = torch.unsqueeze(torch.tensor([graph.n_blocks+graph.n_reg-1 for graph in graph_ar],device=self.device),1)
                
                Es_outpol = torch.cat([Es,torch.zeros(Es.shape[0],1,Es.shape[2],device=self.device)],1)
                Es_outpol = torch.cat([Es_outpol,F.one_hot(last_node,Es_outpol.shape[1])],2)
                
                Er_outpol = torch.cat([Er,torch.zeros(Er.shape[0],1,Er.shape[2],device=self.device)],1)
                Er_outpol = torch.cat([Er_outpol,torch.zeros(Er_outpol.shape[0],Er_outpol.shape[1],1,device=self.device)],2)
                Er_outpol[:,-1,-1]=1
                pol_v = torch.cat([pol_v,torch.ones(pol_v.shape[0],1,pol_v.shape[2],device=self.device)],1)
                pol_e = torch.cat([pol_e,torch.ones(pol_e.shape[0],1,pol_e.shape[2],device=self.device)],1)
                if mask is not None:
                    mask_e = torch.cat([mask_e,torch.ones(mask_e.shape[0],1,device=self.device)],1)
                    mask_v = torch.cat([mask_v,torch.ones(mask_v.shape[0],1,device=self.device)],1)
            else:
                assert False,"Not implemented"
                
            
            for outblock in self.GN_out_pol:
                pol_e,pol_v,pol_u = outblock(pol_e,Es,Er,pol_v,pol_u,mask_e = mask_e,mask_v=mask_v)
            
            #only take the action edges
            if self.last_only:
                pol_e = pol_e[:,-1,:]
            if mask is not None:
                mask_eout = torch.tensor(~mask['e_out'],device=self.device,dtype=torch.float).reshape(len(graph_ar),-1)
                
                pol_e = pol_e-mask_eout*1e10
                
                
            
            pol = F.softmax(pol_e,dim=1)
            
            
            
            
            
            
            if torch.any(torch.isnan(pol)):
                assert False, 'nans in the policy'
            
            return val,pol
class A2CDenseOptimizer():
    def __init__(self,model,lr=1e-4,pol_over_val = 2e-4,tau=1e-5,exploration_factor=0.):
        self.pol_over_val = pol_over_val
        self.tau = tau
        self.exploration_factor=exploration_factor
        self.target_model = copy.deepcopy(model)
        self.optimizer = torch.optim.NAdam(model.parameters(),lr=lr,weight_decay=0.001)
        #torch.optim.Adam(model.out_val.parameters(),lr=lr_val)#,momentum = 0.01,
                                             #nesterov = True)
        #self.optimizer_pol = torch.optim.NAdam(model.parameters(),lr=lr_pol,weight_decay=0.001)
        #torch.optim.Adam(model.parameters(),lr=lr_pol)#,momentum = 0.01,
                                            #nesterov = True)
        self.model = model
    def optimize(self,state,actionid,rewards,nstates,gamma,mask=None,nmask=None):
        v,pol = self.model(state,mask=mask)
        entropy = torch.sum(-pol*torch.log(pol+1e-10),dim=1)
        entropy = entropy.mean()
        next_v,_  = self.target_model(nstates,mask=nmask,inference = True)
        #put the value of the terminal state at 0
        with torch.inference_mode(): 
            next_v[nmask['terminal']]=0
        Qval = torch.tensor(rewards).to(next_v.device) + gamma*next_v
        if self.model.use_wandb:
            wandb.log({'value_est': v.mean(), 'target_value_est':Qval.mean(), 'rewards':rewards.mean()})
        #compute the critique loss
        l_v = self.model.loss_val(v,Qval)
        
        advantage = Qval-v.detach()
        nll = -torch.log(1e-10+pol[torch.arange(pol.shape[0]),actionid])
        l_p = torch.mean(torch.unsqueeze(nll,1)*advantage)
        if self.model.use_wandb:
            wandb.log({'policy_entropy':entropy})
        l_p -= self.exploration_factor*entropy
         
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
        #wandb.watch(self.model)
        return l_v.detach().cpu().numpy(), l_p.detach().cpu().numpy()
class A2CShared(nn.Module):
    def __init__(self,
                 maxs_grid,
                 max_blocks,
                 n_robots,
                 n_regions,
                 n_actions,
                 n_fc_layer =2,
                 n_neurons = 100,
                 batch_norm= True,
                 device='cpu',
                 use_wandb=False,
                 shared = True,
                 **encoder_args):
        super().__init__()
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
            if self.batch_norm:
                if rep_val.shape[0] > 1:
                    val = self.out_val(self.val_norm(rep_val))
                else:
                    val = self.out_val(0*rep_val)
            else:
                val = self.out_val(rep_val)
            for layer in self.FC_pol:
                rep_pol = F.relu(layer(rep_pol))
            if mask is not None:
                mask = torch.tensor(~mask,device=self.device,dtype=torch.float).reshape(len(grids),-1)
                rep_pol = self.out_pol(rep_pol)
                rep_pol = rep_pol-mask*1e10
                pol = F.softmax(rep_pol,dim=1)
            else:
                pol = F.softmax(self.out_pol(rep_pol),dim=1)
            if torch.any(torch.isnan(pol)):
                assert False, 'nans in the policy'
            
            return val,pol
class A2CSharedOptimizer():
    def __init__(self,model,lr=1e-4,pol_over_val = 2e-4,tau=1e-5,exploration_factor=0.):
        self.pol_over_val = pol_over_val
        self.tau = tau
        self.exploration_factor=exploration_factor
        self.target_model = copy.deepcopy(model)
        self.optimizer = torch.optim.NAdam(model.parameters(),lr=lr,weight_decay=0.001)
        #torch.optim.Adam(model.out_val.parameters(),lr=lr_val)#,momentum = 0.01,
                                             #nesterov = True)
        #self.optimizer_pol = torch.optim.NAdam(model.parameters(),lr=lr_pol,weight_decay=0.001)
        #torch.optim.Adam(model.parameters(),lr=lr_pol)#,momentum = 0.01,
                                            #nesterov = True)
        self.model = model
    def optimize(self,state,actionid,rewards,nstates,gamma,mask=None,nmask=None):
        v,pol = self.model(state,mask=mask)
        entropy = torch.sum(-pol*torch.log(pol+1e-10),dim=1)
        entropy = entropy.mean()
        next_v,_  = self.target_model(nstates,mask=nmask,inference = True)
        #put the value of the terminal state at 0
        with torch.inference_mode(): 
            next_v[~np.any(nmask,axis=1)]=0
        Qval = torch.tensor(rewards).to(next_v.device) + gamma*next_v
        if self.model.use_wandb:
            wandb.log({'value_est': v.mean(), 'target_value_est':Qval.mean(), 'rewards':rewards.mean()})
        #compute the critique loss
        l_v = self.model.loss_val(v,Qval)
        
        advantage = Qval-v.detach()
        nll = -torch.log(1e-10+pol[torch.arange(pol.shape[0]),actionid])
        l_p = torch.mean(torch.unsqueeze(nll,1)*advantage)
        if self.model.use_wandb:
            wandb.log({'policy_entropy':entropy})
        l_p -= self.exploration_factor*entropy
         
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
        #wandb.watch(self.model)
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
                 device='cpu'):
        super().__init__()
        self.fe = nn.ModuleList([nn.Linear(in_Ea+2*in_Va+in_u,n_neurons,device=device)]+
                                [nn.Linear(n_neurons,n_neurons,device=device) 
                                     for i in range(n_hidden_layers-1)]+
                                [nn.Linear(n_neurons,out_Ea,device=device)])
        self.fv = nn.ModuleList([nn.Linear(in_Va+out_Ea+in_u,n_neurons,device=device)]+
                                [nn.Linear(n_neurons,n_neurons,device=device) 
                                     for i in range(n_hidden_layers-1)]+
                                [nn.Linear(n_neurons,out_Va,device=device)])
        self.fu = nn.ModuleList([nn.Linear(in_u+out_Va+out_Ea,n_neurons,device=device)]+
                                [nn.Linear(n_neurons,n_neurons,device=device) 
                                     for i in range(n_hidden_layers-1)]+
                                [nn.Linear(n_neurons,out_u,device=device)])
        self.device=device
        self.in_u=in_u
        self.in_Va=in_Va
        self.in_Ea=in_Ea
        self.out_u=out_u
        self.out_Va=out_Va
        self.out_Ea=out_Ea
    def forward(self,E_a,E_s,E_r,V_a,u,mask_v=None,mask_e=None,mask_u = None):
        B = torch.cat([E_a,V_a@E_s,V_a@E_r,u],1)
        out_E = torch.zeros((E_a.shape[0],
                             self.out_Ea,
                             E_a.shape[2]),device=self.device)
        for i in range(B.shape[2]):
            new_e = B[:,:,i]
            for layer in self.fe:
                new_e = F.gelu(layer(new_e))
            out_E[:,:,i] = new_e
        if mask_e is not None:
            out_E = out_E*mask_e.unsqueeze(1).expand(-1,out_E.shape[1],-1)
        C = torch.cat([V_a, out_E@E_r.T,u],0)
        out_V = torch.zeros((V_a.shape[0],self.out_Va,V_a.shape[2]),device=self.device)
        
        for i in range(C.shape[2]):
            new_v = C[:,:,i]
            for layer in self.fv:
                new_v = F.gelu(layer(new_v))
            out_V[:,:,i] = new_v
        if mask_v is not None:
            out_V = out_V*mask_v.unsqueeze(1).expand(-1,out_V.shape[1],-1)
        P = torch.cat([u,out_V.sum(dim=2),out_E.sum(dim=2)])
        out_u = P
        for layer in self.fu:
            out_u = F.gelu(layer(out_u))
        if mask_u is not None:
            out_u = out_u*mask_u
        return out_E,out_V,out_u

class StateEncoder(nn.Module):
    def __init__(self,
                 maxs_grid,
                 max_blocks,
                 n_robots,
                 n_regions,
                 # dim_b=4,
                 # dim_r=2,
                 # dim_c=2,
                 n_channels = 32,
                 n_internal_layer = 2,
                 stride=1,
                 device='cpu'):
        super().__init__()
        # self.occ_emb = nn.Embedding(max_blocks+1,dim_b,device=device)
        # self.hold_emb = nn.Embedding(n_robots+1, dim_r,device=device)
        # self.con_emb = nn.Embedding(n_regions+1,dim_c,device=device)
        self.mbl=max_blocks
        self.nrob = n_robots
        self.nreg = n_regions
        #self.convinup = nn.Conv2d(3, n_channels//2, 3,stride=1,device=device)
        #self.convindown = nn.Conv2d(3, n_channels//2, 3,stride=1,device=device)
        self.convin = nn.Conv2d(6,n_channels,3,stride=1,device=device)
        self.convinternal = [nn.Conv2d(n_channels,n_channels,3,stride=stride,device=device) for i in range(n_internal_layer)]
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
            