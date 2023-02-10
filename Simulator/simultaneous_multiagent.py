# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 18:06:51 2022

@author: valla
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:43:04 2022

@author: valla
"""

import numpy as np
import internal_models as im
import geometric_internal_model as geo_im
import torch
from torch.distributions.categorical import Categorical
import abc
import wandb

class AgentSimultanous(metaclass=abc.ABCMeta):
    def __init__(self, n_robots,rid, block_choices,config,use_wandb=False,log_freq = None,env='norot'):
        super().__init__()
        self.rid = rid
        self.n_robots = n_robots
        self.block_choices = block_choices
        self.use_wandb=use_wandb
        self.log_freq = log_freq
        self.env = env
        self.gamma = 1-config['agent_discount_f']
    def prepare_action(self,simulator,action):
        if action in  {'P','L'}:
            simulator.leave(self.rid)
            
    def act(self,simulator,action,
            sideblock=None,
            sidesup = None,
            bid_sup = None,
            idconsup = None,
            blocktypeid = None,
            side_ori = None,
            draw= False,
            ):
        valid,closer = None,None
        if blocktypeid is not None:
            blocktype= self.block_choices[blocktypeid]
        else:
            blocktype = None
        if action == 'P':
            oldbid = simulator.leave(self.rid)
            if oldbid is not None:
                stable = simulator.check()
                if not stable:
                    return False,None,blocktype,None
            valid,closer,interfaces = simulator.put_rel(blocktype,sideblock,sidesup,bid_sup,side_ori,blocktypeid=blocktypeid,idconsup = idconsup)
                
            if valid:
                simulator.hold(self.rid,simulator.nbid-1)
        if action in {'L','S'}:
            return True,None,None,None
        return valid,closer,blocktype,interfaces
    @abc.abstractmethod
    def generate_mask(self,state):
        pass
    @abc.abstractmethod
    def update_policy(self,**kwargs):
        pass
    @abc.abstractmethod
    def choose_action(self,state):
        pass
class SACDense(AgentSimultanous):
    def __init__(self,
                 n_robots,
                 rid,
                 block_choices,
                 config,
                 ground_blocks = None,
                 action_choice = ['P','L','S'],
                 grid_size = [10,10],
                 max_blocks=30,
                 max_interfaces = 120,
                 n_regions = 2,
                 discount_f = 0.1,
                 use_mask=True,
                 last_only=False,
                 use_wandb=False,
                 log_freq = None,
                 env='norot'
                 ):
        super().__init__(n_robots,rid,block_choices,config,use_wandb=use_wandb,log_freq = log_freq,env=env)
        self.rep= 'graph'
        self.last_only=False
        self.action_choice = action_choice
        self.grid_size = grid_size
        self.n_ground_type = len(ground_blocks)
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
            
           
            self.n_actions = (max_blocks+n_regions)*(len(ground_blocks)+len(block_choices))*(('L'in self.action_choice) + ('S' in self.action_choice)+len(block_choices))*np.max(self.n_side_oriented_sup)*np.max(self.n_side_oriented)*6
            self.gamma = 1-discount_f
        else:
            assert False, 'Not implemented'

    def create_model(self,simulator,config):
        self.model = geo_im.build_hetero_GNN(config,simulator,0,self.action_choice,self.n_side_oriented_sup,self.n_side_oriented)
        
        self.optimizer = geo_im.SACOptimizerGeometricNN(simulator,0,self.action_choice,self.n_side_oriented_sup,self.n_side_oriented,
                                                    self.model,
                                                    config,
                                                    use_wandb=self.use_wandb,
                                                    log_freq = self.log_freq,
                                                    name = f"Robot_{self.rid}/")
        
    def update_policy(self,buffer,buffer_count,batch_size,steps=1):
        for s in range(steps):
            if batch_size > buffer.counter:
                return buffer.counter
                
            (state,action,nstate,reward,terminal,masks,nmasks)=buffer.sample(batch_size)
            l_p = self.optimizer.optimize(state,action,reward,nstate,terminal,self.gamma,masks=masks,nmasks=nmasks)
    def choose_action(self,r_id,state,explore=True,mask=None):
        graph=geo_im.create_sparse_graph(state,
                            r_id,
                            self.action_choice,
                            self.optimizer.device,
                            self.n_side_oriented_sup,
                            self.n_side_oriented,
                            last_only=self.last_only)
        
        actions_graph = self.model(graph.x_dict,graph.edge_index_dict)
        if mask is not None:
            actions_dist = Categorical(logits=actions_graph['new_block'].squeeze()-1e10*torch.tensor(~mask,dtype=torch.float,device = actions_graph['new_block'].device))
        else:
            actions_dist = Categorical(logits=actions_graph['new_block'].squeeze())
        if self.use_wandb and self.optimizer.step % self.log_freq == 0:
            wandb.log({"Robot_"+str(self.rid)+'/action_dist':actions_dist.probs},step=self.optimizer.step)
        if self.exploration_strat == 'softmax':
            actionid = int(actions_dist.sample().detach().cpu().numpy())
            if self.use_wandb and self.optimizer.step % self.log_freq == 0:
                wandb.log({"Robot_"+str(self.rid)+'/action_id':actionid},step=self.optimizer.step)
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
        action,action_params = int2actsim_graph(graph,actionid)
        return action,action_params,actionid#,actions_dist.entropy()  
    
        
    def generate_mask(self,state):
        graph = geo_im.create_sparse_graph(state, self.rid, self.action_choice, self.optimizer.device, self.n_side_oriented_sup, self.n_side_oriented,last_only=self.last_only)
        if self.env == 'norot':
            mask = generate_mask_graph(state.grid,graph,self.block_choices)
            return mask
        else:
            assert False, "Not implemented"
class SACSparse(AgentSimultanous):
    def __init__(self,
                 n_robots,
                 rid,
                 block_choices,
                 config,
                 ground_blocks = None,
                 action_choice = ['P','L','S'],
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
        super().__init__(n_robots,rid,block_choices,config,use_wandb=use_wandb,log_freq = log_freq,env=env)
        self.rep= 'grid'
        self.action_choice = action_choice
        self.grid_size = grid_size
        self.n_ground_type = len(ground_blocks)
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
            
           
            self.n_actions = (max_blocks+n_regions)*(len(ground_blocks)+len(block_choices))*(('L'in self.action_choice) + ('S' in self.action_choice)+len(block_choices))*np.max(self.n_side_oriented_sup)*np.max(self.n_side_oriented)*6
            self.gamma = 1-discount_f
        else:
            assert False, 'Not implemented'

        self.model = im.PolNetSparse(grid_size,
                                max_blocks,
                                1,
                                n_regions,
                                self.n_actions,
                                config,
                                use_wandb=self.use_wandb,
                                log_freq = self.log_freq,
                                name=f"Robot_{rid}")
        
        self.optimizer = im.SACSparseOptimizer(grid_size,
                                              max_blocks,
                                              1,
                                              n_regions,
                                              self.n_actions,
                                              self.model,
                                              config)
    def update_policy(self,buffer,buffer_count,batch_size,steps=1):
         for s in range(steps):
            if batch_size > buffer_count:
                return buffer_count
        
            if buffer_count <buffer.shape[0]:
                batch = np.random.choice(buffer[:buffer_count],batch_size,replace=False)
            else:
                batch = np.random.choice(buffer,batch_size,replace=False)
                #compute the state value using the target Q table
        
            states = [trans.state['grid'] for trans in batch]
            nstates = [trans.new_state['grid'] for trans in batch]
            mask = np.array([trans.state['mask'] for trans in batch])
            nmask = np.array([trans.new_state['mask'] for trans in batch])
       
            actions = np.array([trans.a[0] for trans in batch])
            rewards = np.array([[trans.r] for trans in batch],dtype=np.float32)

            l_p = self.optimizer.optimize(states,actions,rewards,nstates,self.gamma,mask,nmask=nmask)
            return l_p
    def choose_action(self,r_id,state,explore=True,mask=None):
        if mask is None:
            mask = np.zeros(self.n_actions,dtype=bool)
            mask[self.action_per_robot*r_id:self.action_per_robot*(1+r_id)]=True
        actions_dist,logits = self.model([state.grid],inference = True,mask=mask)
        if self.exploration_strat == 'softmax':
            actionid = actions_dist.sample().detach().cpu().numpy()[0]
            if self.use_wandb and self.optimizer.step % self.log_freq == 0:
                wandb.log({"Robot_"+str(self.rid)+'_action_id':actionid},step=self.optimizer.step)
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
        if self.env == 'norot':
            action,action_params = int2actsim_norot(actionid,
                                                    state.graph.n_blocks,
                                                    self.n_side_oriented,
                                                    self.n_side_oriented_sup,
                                                    self.max_blocks,
                                                    state.graph.n_ground)
        else:
            assert False, "Not implemented"
        return action,action_params,(actionid,actions_dist.entropy())
        
    def generate_mask(self,state):
        if self.env == 'norot':
            mask =  generate_mask_norot(self.rid,
                                       state.grid,
                                       self.n_side_oriented,
                                       self.n_side_oriented_sup,
                                       self.max_blocks,
                                       state.graph.n_reg,
                                       self.action_choice,
                                       state.type_id,
                                       empty_id=-1-self.n_ground_type)
            mask = complete_mask_norot(state.grid,
                                       self.block_choices,
                                       mask,
                                       state.graph.n_blocks,
                                       self.n_side_oriented,
                                       self.n_side_oriented_sup,
                                       self.max_blocks,
                                       state.graph.n_ground,
                                       )
            return mask
        else:
            assert False, "Not implemented"

def generate_mask_norot(rid,grid,n_side_b,n_side_sup,maxblocks,n_reg,action_choice,typeblocks,reverse=False,empty_id=-2):
    mask = np.zeros((maxblocks+n_reg,
                     n_side_sup.shape[0],
                     n_side_b.shape[0]+int('L' in action_choice)+int('S' in action_choice),
                     np.max(n_side_sup),
                     np.max(n_side_b),
                     6
                     ),dtype=bool)
    
    #only keep the blocks that are present
    if reverse:
        typeblocks = typeblocks[::-1]
    if 'P' in action_choice:
        for i,n_side in enumerate(n_side_b):
            for j,typesup in enumerate(typeblocks):
                if typesup ==empty_id:
                    continue
                if typesup > 0:
                    typesup-=empty_id+1
                else:
                    typesup = -typesup-1
                for k in range(6):
                    mask[j,typesup,i,:n_side_sup[typesup,k],:n_side[k],k]=True
    #both these actions are presented 6 times to allow a little entropy increase in the final steps of the episode
    #mask for the leave action
    if 'L' in action_choice:
        mask[0,0,-2,0,0,0]=rid in grid.hold
    #mask for the stay action
    if 'S' in action_choice:
        mask[0,0,-1,0,0,0]=True

   
    return mask.flatten()
def generate_mask_graph(grid,graph,block_choices):
    mask = np.zeros(graph['new_block'].x.shape[0],dtype=bool)
    for actionid in range(graph['new_block'].x.shape[0]):
        action,action_params = int2actsim_graph(graph,actionid)
        if action == 'P':
            mask[actionid],*_=grid.connect(block_choices[action_params['blocktypeid']],
                                             bid=-2,
                                             sideid=action_params['sideblock'],
                                             support_sideid=action_params['sidesup'],
                                             support_bid=action_params['bid_sup'],
                                             side_ori=action_params['side_ori'],
                                             idcon=action_params['idconsup'],
                                             test=True)
        else:
            mask[actionid]=True
    return mask
def int2actsim_graph(graph,action_id):
    graph = graph.cpu()
    new_block = graph['new_block'].x[action_id].cpu().numpy().astype(int)
    if new_block[-2]==1:
        action = 'S'
        action_params = {}
    else:
        action = 'P'
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
            action_params =  {
                                    'blocktypeid':blocktype,
                                    'sideblock':new_block[-1],
                                    'sidesup':side_sup_id,
                                    'bid_sup':0,
                                    'side_ori':side_sup_ori,
                                    'idconsup': int(ground_id)
                                     }
        else:
            sup_bid = int(graph['block', 'action_desc', 'side_sup'].edge_index[0,graph['block', 'action_desc', 'side_sup'].edge_index[1,:]==side_node_id].numpy())
            action_params =  {
                                    'blocktypeid':blocktype,
                                    'sideblock':new_block[-1],
                                    'sidesup':side_sup_id,
                                    'bid_sup':sup_bid+1,
                                    'side_ori':side_sup_ori,
                                    'idconsup': 1 #useless
                                     }
    return action,action_params
def complete_mask_norot(grid,block_choices,mask,n_blocks,n_side_b,n_side_sup,maxblocks,n_reg,reverse=False):
    #remove all actions that would end up in a collision
    possible_actions, = np.nonzero(mask)
    for actionid in possible_actions:
        action,action_params = int2actsim_norot(actionid,n_blocks,n_side_b,n_side_sup,maxblocks,n_reg,reverse=reverse)
        if action == 'P':
            mask[actionid],*_=grid.connect(block_choices[action_params['blocktypeid']],
                                             bid=-2,
                                             sideid=action_params['sideblock'],
                                             support_sideid=action_params['sidesup'],
                                             support_bid=action_params['bid_sup'],
                                             side_ori=action_params['side_ori'],
                                             idcon=action_params['idconsup'],
                                             test=True)
    return mask
def int2actsim_norot(actionid,n_blocks,n_side_b,n_side_sup,maxblocks,n_reg,reverse=False):
    blockid, btype_sup, btype, side_sup, side_b,side_ori =np.unravel_index(actionid,
                                                              (maxblocks+n_reg,
                                                               n_side_sup.shape[0],
                                                               n_side_b.shape[0]+1,
                                                               np.max(n_side_sup),
                                                               np.max(n_side_b),
                                                               6))
    if btype == n_side_b.shape[0]:
        action = 'S'
        action_params = {}
    else:
        action = 'P'
        if reverse:
            blockid = maxblocks+n_reg-blockid
        if blockid<n_reg:
            
            action_params = {'blocktypeid':btype,
                             'sideblock':side_b,
                             'sidesup':side_sup,
                             'bid_sup':0,
                             'side_ori':side_ori,
                             'idconsup':blockid,
                              }
        else:
            action_params = {'blocktypeid':btype,
                             'sideblock':side_b,
                             'sidesup':side_sup,
                             'bid_sup':blockid-n_reg+1,
                             'side_ori':side_ori,
                             'idconsup':None,
                              }
    return action,action_params
def reward_simultaneous1(action, valid, closer, success,failure):
    reward = 0
    if not valid:
        return -1
    if failure:
        reward-=0.5
    if success:
        reward+=1
    if action == 'P':
        if closer == 1:
            return reward+0.4
        else:
            return reward
    if action == 'L':
        return reward - 0.1
    if action == 'S':
        return reward -0.1
def modular_reward(action,valid,closer,success,fail,config=None,n_sides = None, **kwargs):
    if fail: 
        return config['reward_failure']
    reward =0
    reward+=config['reward_action'][action]
    if success:
        reward+= config['reward_success']
    if closer is not None and closer == 1:
        reward+=config['reward_closer']
    if n_sides is not None:
        reward += config['reward_nsides']*np.sum(n_sides)
        if not np.all(np.logical_xor(n_sides[:3],n_sides[3:])):
            reward += config['reward_opposite_sides']
    return reward
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    