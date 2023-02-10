# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 14:23:31 2022

@author: valla
"""

import numpy as np
import geometric_internal_model as im
import wandb
from relative_single_agent import SupervisorRelative
from torch.distributions.categorical import Categorical

class SACSupervisorDense(SupervisorRelative):
    def __init__(self,
                 n_robots,
                 block_choices,
                 config,
                 setup_graph=None,
                 ground_blocks = None,
                 action_choice = ['Ph'],
                 grid_size = [10,10],
                 max_blocks=30,
                 max_interfaces = 120,
                 n_regions = 2,
                 discount_f = 0.1,
                 use_mask=True,
                 last_only=True,
                 use_wandb=False,
                 log_freq = None,
                 env='norot'
                 ):
        super().__init__(n_robots,block_choices,config,use_wandb=use_wandb,log_freq = log_freq,env=env)
  
        
        self.rep= 'graph'
        self.device = config['torch_device']
        self.max_interfaces =max_interfaces
        self.action_choice = action_choice
        self.grid_size = grid_size
        self.n_typeblock = len(block_choices)
        self.exploration_strat = config['agent_exp_strat']
        if self.exploration_strat == 'epsilon-greedy' or self.exploration_strat == 'epsilon-softmax':
            self.eps = config['agent_epsilon']
        self.max_blocks = max_blocks
        
        if self.env =='norot':
            self.n_side_oriented = np.array([[np.sum((block.neigh[:,2]==0) & (block.neigh[:,3]==0)),
                                              np.sum((block.neigh[:,2]==0) & (block.neigh[:,3]==1)),
                                              np.sum((block.neigh[:,2]==0) & (block.neigh[:,3]==2)),
                                              np.sum((block.neigh[:,2]==1) & (block.neigh[:,3]==0)),
                                              np.sum((block.neigh[:,2]==1) & (block.neigh[:,3]==1)),
                                              np.sum((block.neigh[:,2]==1) & (block.neigh[:,3]==2)),] for block in block_choices])
            
            self.n_side_oriented_sup = np.array([[np.sum((block.neigh[:,2]==1) & (block.neigh[:,3]==0)),
                                                  np.sum((block.neigh[:,2]==1) & (block.neigh[:,3]==1)),
                                                  np.sum((block.neigh[:,2]==1) & (block.neigh[:,3]==2)),
                                                  np.sum((block.neigh[:,2]==0) & (block.neigh[:,3]==0)),
                                                  np.sum((block.neigh[:,2]==0) & (block.neigh[:,3]==1)),
                                                  np.sum((block.neigh[:,2]==0) & (block.neigh[:,3]==2)),] for block in ground_blocks+ block_choices])
            #check the genererate_mask_norot function to understand why these parameters
            self.n_actions = n_robots*(1+len(block_choices))*(int('L' in action_choice) +len(block_choices))*np.max(self.n_side_oriented_sup)*np.max(self.n_side_oriented)*6
        else:
            assert False, "not implemented"
            
        self.action_per_robot =self.n_actions//n_robots
        
        
        
    def create_model(self,simulator,config):
        self.model = im.build_hetero_GNN(config,simulator,0,self.action_choice,self.n_side_oriented_sup,self.n_side_oriented)
        
        self.optimizer = im.SACOptimizerGeometricNN(simulator,0,self.action_choice,self.n_side_oriented_sup,self.n_side_oriented,
                                                    self.model,
                                                    config,
                                                    use_wandb=self.use_wandb,
                                                    log_freq = self.log_freq)
        
    def update_policy(self,buffer,buffer_count,batch_size,steps=1):
        for s in range(steps):
            if batch_size > buffer.counter:
                return buffer.counter
                
            (state,action,nstate,reward,terminal)=buffer.sample(batch_size)
            l_p = self.optimizer.optimize(state,action,reward,nstate,terminal,self.gamma)
            
    def choose_action(self,r_id,state,explore=True,mask=None):
        graph=im.create_sparse_graph(state,
                            r_id,
                            self.action_choice,
                            self.device,
                            self.n_side_oriented_sup,
                            self.n_side_oriented,
                            last_only=self.last_only)
        
        actions_graph = self.model(graph.x_dict,graph.edge_index_dict)
        actions_dist = Categorical(logits=actions_graph['new_block'].squeeze())
        if self.use_wandb and self.optimizer.step % self.log_freq == 0:
            wandb.log({'action_dist':actions_dist.probs[0,mask]},step=self.optimizer.step)
        if self.exploration_strat == 'softmax':
            actionid = int(actions_dist.sample().detach().cpu().numpy())
            if self.use_wandb and self.optimizer.step % self.log_freq == 0:
                wandb.log({'action_id':actionid},step=self.optimizer.step)
        elif self.exploration_strat == 'epsilon-greedy':
            if np.random.rand() > self.eps:
                actionid = np.argmax(actions_dist.probs.detach().cpu().numpy())
            else:
                ids, = np.nonzero(mask)
                actionid = np.random.choice(ids)
        elif self.exploration_strat == 'epsilon-softmax':
            if np.random.rand() > self.eps:
                actionid = actions_dist.sample().detach().cpu().numpy()[0]
            else:
                ids, = np.nonzero(mask)
                actionid = np.random.choice(ids)
        action,action_params = int2act(actionid,graph)
        return action,action_params,actionid,actions_dist.entropy()
        
    def generate_mask(self,state,rid):
        return None


def int2act(action_id,graph):
    graph = graph.cpu()
    action = 'Ph'
    new_block = graph['new_block'].x[action_id].cpu().numpy().astype(int)
    blocktype, = np.nonzero(new_block[:-1])
    blocktype = int(blocktype)
    
    side_node_id = graph['side_sup', 'put_against', 'new_block'].edge_index[0,action_id]
    
    side_sup_ori, = np.nonzero(graph['side_sup'].x[side_node_id,:-1].numpy())
    side_sup_ori = int(side_sup_ori)
    side_sup_id = int(graph['side_sup'].x[side_node_id,-1].numpy())
    rid = int(graph['robot','choses','new_block'].edge_index[0,action_id].numpy())
    
    if (graph['ground', 'action_desc', 'side_sup'].edge_index[1,:]==side_node_id).any():
        ground_node = graph['ground', 'action_desc', 'side_sup'].edge_index[0,graph['ground', 'action_desc', 'side_sup'].edge_index[1,:]==side_node_id]
        ground_id = ground_node.numpy().astype(int)
        action_params =  {'rid':rid,
                                'blocktypeid':blocktype,
                                'sideblock':new_block[-1],
                                'sidesup':side_sup_id,
                                'bid_sup':0,
                                'side_ori':side_sup_ori,
                                'idconsup': int(ground_id)
                                 }
    else:
        sup_bid = int(graph['block', 'action_desc', 'side_sup'].edge_index[0,graph['block', 'action_desc', 'side_sup'].edge_index[1,:]==side_node_id].numpy())
        action_params =  {'rid':rid,
                                'blocktypeid':blocktype,
                                'sideblock':new_block[-1],
                                'sidesup':side_sup_id,
                                'bid_sup':sup_bid+1,
                                'side_ori':side_sup_ori,
                                'idconsup': 1 #useless
                                 }
    return action,action_params




if __name__ == '__main__':
    print("Start test Agent")
    config = {'train_n_episodes':100000,
            'train_l_buffer':5000,
            'ep_batch_size':64,
            'ep_use_mask':True,
            'agent_discount_f':0.1,
            'agent_last_only':True,
            'torch_device':'cpu',
            'SEnc_n_channels':32,
            'SEnc_n_internal_layer':4,
            'SEnc_stride':1,
            'SAC_n_fc_layer':2,
            'SAC_n_neurons':64,
            'SAC_batch_norm':True,
            'Q_duel':True,
            'opt_lr':1e-4,
            'opt_pol_over_val': 1,
            'opt_tau': 1e-3,
            'opt_weight_decay':0.0001,
            'opt_exploration_factor':0.001,
            'agent_exp_strat':'softmax',
            'agent_epsilon':0.05,
            'opt_max_norm': 2,
            'opt_target_entropy':1.8,
            'opt_value_clip':False,
            'opt_entropy_penalty':False,
            'opt_Q_reduction': 'min',
            'V_optimistic':False,
            }
    from discrete_blocks_norot import discret_block_norot as Block
    hexagone = Block([[0,1,1],[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0]],muc=0.7)
    triangleUp = Block([[0,0,0]])
    triangleDown = Block([[0,0,1]])
    agent = SACSupervisorDense(2,
                        [hexagone],
                        config,
                        ground_block = [triangle.triangle],
                        action_choice = ['Pl','Ph','L'],
                        grid_size = [10,10],
                        max_blocks=30,
                        max_interfaces = 120,
                        n_regions = 2,
                        discount_f = 0.1,
                        use_mask=True,
                        last_only=True,
                        use_wandb=False,
                        log_freq = None,
                        env='norot'
                     )
    print("\nEnd test Agent")