# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:43:04 2022

@author: valla
"""

import numpy as np
import internal_models as im
import abc

class SupervisorSimultanous(metaclass=abc.ABCMeta):
    def __init__(self, n_robots, block_choices,use_wandb=False):
        super().__init__()
        self.n_robots = n_robots
        self.block_choices = block_choices
        self.use_wandb=use_wandb
    def act(self,simulator,robot_actions):
        if 'R' in robot_actions:
            simulator.save()
            hard_save = True
        elif not 'L' in robot_actions:
            return True
        
        old_bid = np.zeros(len(robot_actions))
        for rid, action in enumerate(robot_actions):
            if action == 'N':
                old_bid[rid] = None
            if action == 'L':
                old_bid[rid] = simulator.leave(rid)
            elif action == 'R':
                bid = simulator.grid.occ[simulator.grid.hold==rid][0]
                old_bid[rid] = simulator.remove(rid,bid,save=False)
        valid = simulator.check()
        if not valid:
            if hard_save:
                simulator.undo()
            else:
                for rid, bid in enumerate(old_bid):
                    simulator.hold(rid,bid)
        return valid
    def select(self,simulator,rid,pos,ori,blocktypeid):
        oldbid = simulator.leave(rid)
        if blocktypeid == -1:
            bid = simulator.grid.occ[pos[0],pos[1],ori%2]
        else:
            bid = simulator.nbid
        if oldbid is not None:
            valid = simulator.check()
        else:
            valid = True
        if valid:
            if blocktypeid!=-1:
                simulator.put(self.block_choices[blocktypeid],pos,ori)
                simulator.hold(rid,bid)
        return valid
    @abc.abstractmethod
    def generate_mask(self,state,rid):
        pass
    @abc.abstractmethod
    def update_policy(self,**kwargs):
        pass
    @abc.abstractmethod
    def choose_action(self,state):
        pass
    
class A2CSupervisor(SupervisorSimultanous):
    def __init__(self,
                 n_robots,
                 block_choices,
                 action_choice = ['L','H','R'],
                 grid_size = [10,10],
                 max_blocks=30,
                 n_regions = 2,
                 discount_f = 0.1,
                 device='cuda',
                 use_mask=True,
                 use_wandb=False,
                 ):
        super().__init__(n_robots,block_choices,use_wandb)
        self.grid_size = grid_size
        self.n_typeblock = len(block_choices)
        self.max_blocks = max_blocks
        self.action_list = generate_actions_supervisor(n_robots,len(block_choices),grid_size,max_blocks)
        self.action_per_robot = len(self.action_list)//n_robots
        
        #parameters of the internal model:
        n_fc_layer=4
        n_neurons=100
        n_internal_layer=2
        n_channels = 128
        
        self.model = im.A2CShared(grid_size,
                                  max_blocks,
                                  n_robots,
                                  n_regions,
                                  len(self.action_list),
                                  n_fc_layer =n_fc_layer,
                                  n_neurons = n_neurons,
                                  device=device,
                                  use_wandb=self.use_wandb,
                                  n_internal_layer=n_internal_layer,
                                  n_channels = n_channels)
        
        self.optimizer = im.A2CSharedOptimizer(self.model,lr=1e-4,pol_over_val=2e-1,tau=1e-4,exploration_factor=0.001)
        
        self.gamma = 1-discount_f
        self.use_mask = use_mask
    def update_policy(self,buffer,buffer_count,batch_size,steps=1):
        if buffer_count==0:
            return
        
        # _,_,nactions = self.choose_action(nstates,explore=False)
       
        #while loss > conv_tol:
        for s in range(steps):
            if buffer_count <buffer.shape[0]:
                batch_size = np.clip(batch_size,0,buffer_count)
                batch = np.random.choice(buffer[:buffer_count],batch_size,replace=False)
            else:
                batch = np.random.choice(np.delete(buffer,buffer_count%buffer.shape[0]),batch_size,replace=False)
            #compute the state value using the target Q table
            if self.use_mask:
                states = [trans.state[0] for trans in batch]
                nstates = [trans.new_state[0] for trans in batch]
                mask = np.array([trans.state[1] for trans in batch])
                nmask = None
                
            else:
                states = [trans.state for trans in batch]
                nstates = [trans.new_state for trans in batch]
                mask=None
                nmask=None
            actions = np.concatenate([trans.a[0] for trans in batch],axis=0)
            rewards = np.array([[trans.r] for trans in batch],dtype=np.float32)
            
            l_v,l_p = self.optimizer.optimize(states,actions,rewards,nstates,self.gamma,mask,nmask=nmask)
            
    def choose_action(self,r_id,state,explore=True,mask=None):
        if mask is None:
            mask = np.zeros(len(self.action_list),dtype=bool)
            mask[self.action_per_robot*r_id:self.action_per_robot*(1+r_id)]=True
        if not isinstance(state,list):
            state = [state]
        _,actions = self.model(state,inference = True,mask=mask)
        if explore:
            actionid = np.zeros(actions.shape[0],dtype=int)
            for i,p in enumerate(actions.cpu().detach().numpy()):
                actionid[i] = int(np.random.choice(len(self.action_list),p=p))
        else:
            actionid = np.argmax(actions.cpu().detach().numpy(),axis=1)
        if len(state)==1:
            if actionid < 30:
                pass
            action,action_params = vec2act_sup(self.action_list[actionid[0],:],r_id,self.n_typeblock,self.grid_size,self.max_blocks)
            return action,action_params,actionid
        else:
            return None,None,actionid
    def generate_mask(self,state,rid):
        return generate_mask(state, rid, self.block_choices, self.grid_size, self.max_blocks)

def generate_mask(state,rid,block_type,grid_size,n_robots=2):
    mask_select = np.zeros((6*len(block_type),grid_size[0],grid_size[1]),dtype=bool)
    mask_action = np.zeros((n_robots,3),dtype=bool)
    
    #get the ids of the feasible put actions (note that the are not all hidden)
    for idb, block in enumerate(block_type):
        pos,ori = state.touch_side(block)
        mask_select[idb*6+ori,pos[:,0],pos[:,1]]=True
    return (mask_select,mask_action)

def reward_link_fail(action,valid,closer,terminal):
    #reward specific to the case where the robots need to link two points, and where an
    #invalid action end the simulation

    #add a penalty for holding a block
    hold_penalty = 0.3
    #add a penalty if no block are put
    slow_penalty = 1
    #add a penatly for forbidden actions
    forbiden_penalty = 1
    #add the terminal reward
    terminal_reward = 1
    #add a cost for each block
    block_cost = -0.1
    #add a cost for the blocks going away from the target, 
    #or a reward if the block is going toward the target
    closer_reward = 0.2

    
    reward = 0

    if not valid:
        return -forbiden_penalty
    if action in {'H', 'L','R'}:
        reward=-slow_penalty
        return reward
    if action =='Ph':
        reward-=hold_penalty
    
    elif action in {'Ph', 'Pl'}:
        reward += closer_reward*closer-block_cost
    if terminal:
        reward += terminal_reward
    return reward