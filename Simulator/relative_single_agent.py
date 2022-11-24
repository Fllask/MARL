# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 15:48:45 2022

@author: valla
"""

import numpy as np
import internal_models as im
import abc     
    
class SupervisorRelative(metaclass=abc.ABCMeta):
    def __init__(self, n_robots, block_choices,use_wandb=False,last_only=False):
        super().__init__()
        self.n_robots = n_robots
        self.block_choices = block_choices
        self.use_wandb=use_wandb
        self.last_only = last_only
    def Act(self,simulator,action,
            rid=None,
            sideblock=None,
            sidesup = None,
            bid_sup = None,
            idconsup = None,
            blocktypeid = None,
            draw= False
            ):
        valid,closer = None,None
        if blocktypeid is not None:
            blocktype= self.block_choices[blocktypeid]
        else:
            blocktype = None
        if bid_sup is None:
            bid_sup = simulator.nbid-1
        if action in {'Ph','Pl'}:
            oldbid = simulator.leave(rid)
            if oldbid is not None:
                stable = simulator.check()
                if not stable:
                    if draw:
                        #if draw, place the block on the grid and remove it. this way the block is located at the right position
                        valid_pos,closer = simulator.put_rel(blocktype,sideblock,sidesup,bid_sup,idconsup = idconsup)
                        if valid_pos:
                            simulator.remove(simulator.nbid-1,save=False)
                        simulator.hold(rid,oldbid)
                        return False,None,blocktype
                    else:
                        #the robot cannot move from there
                        simulator.hold(rid,oldbid)
                        return False,None,blocktype
            valid,closer = simulator.put_rel(blocktype,sideblock,sidesup,bid_sup,blocktypeid=blocktypeid,idconsup = idconsup)
                
            if valid:
                if action == 'Ph':
                    simulator.hold(rid,simulator.nbid-1)
                if action == 'Pl':
                    stable = simulator.check()
                    if not stable:
                        simulator.remove(simulator.nbid-1,save=False)
                        valid = False
            if not valid:
                simulator.hold(rid,oldbid)
                            
        elif action == 'L':
            oldbid = simulator.leave(rid)
            if oldbid is not None:
                stable = simulator.check()
                valid = stable
                if not stable:
                    simulator.hold(rid,oldbid)
            else:
                valid = False
        else:
            assert False,'Unknown action'
        return valid,closer,blocktype
    @abc.abstractmethod
    def generate_mask(self,state,rid):
        pass
    @abc.abstractmethod
    def update_policy(self,**kwargs):
        pass
    @abc.abstractmethod
    def choose_action(self,state):
        pass
    
class A2CSupervisorDense(SupervisorRelative):
    def __init__(self,
                 n_robots,
                 block_choices,
                 action_choice = ['Pl','Ph','L'],
                 grid_size = [10,10],
                 max_blocks=30,
                 n_regions = 2,
                 discount_f = 0.1,
                 device='cuda',
                 use_mask=True,
                 last_only=True,
                 use_wandb=False,
                 ):
        super().__init__(n_robots,block_choices,use_wandb,last_only)
        self.action_choice = action_choice
        self.grid_size = grid_size
        self.n_typeblock = len(block_choices)
        self.max_blocks = max_blocks
        self.n_side = [block.neigh.shape[0] for block in block_choices]
        
        if last_only:
            self.n_actions = (2*max(self.n_side)*sum(self.n_side)+1)*n_robots
        else:
            self.n_actions = (2*max(self.n_side)*sum(self.n_side)+1)*n_robots
        self.action_per_robot =self.n_actions//n_robots
        
        #parameters of the internal model:
        n_fc_layer=4
        n_neurons=500
        n_internal_layer=4
        n_channels = 256
        batch_norm = True
        self.model = im.A2CShared(grid_size,
                                  max_blocks,
                                  n_robots,
                                  n_regions,
                                  self.n_actions,
                                  n_fc_layer =n_fc_layer,
                                  n_neurons = n_neurons,
                                  device=device,
                                  use_wandb=self.use_wandb,
                                  n_internal_layer=n_internal_layer,
                                  shared = False,
                                  n_channels = n_channels,
                                  batch_norm=batch_norm)
        
        self.optimizer = im.A2CSharedOptimizer(self.model,lr=1e-4,pol_over_val=2e-2,tau=1e-4,exploration_factor=0.001)
        
        self.gamma = 1-discount_f
        self.use_mask = use_mask
    def update_policy(self,buffer,buffer_count,batch_size,steps=1):
        if buffer_count==0 or (self.model.batch_norm and buffer_count < 2):
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
                nmask = np.array([trans.new_state[1] for trans in batch])
                
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
            mask = np.zeros(self.n_actions,dtype=bool)
            mask[self.action_per_robot*r_id:self.action_per_robot*(1+r_id)]=True
        if not isinstance(state,list):
            state = [state]
        _,actions = self.model(state,inference = True,mask=mask)
        if explore:
            actionid = np.zeros(actions.shape[0],dtype=int)
            for i,p in enumerate(actions.cpu().detach().numpy()):
                actionid[i] = int(np.random.choice(self.n_actions,p=p))
        else:
            actionid = np.argmax(actions.cpu().detach().numpy(),axis=1)
        if len(state)==1:
            if actionid < 30:
                pass
            action,action_params = int2act_sup(actionid[0],self.n_side,self.last_only,self.max_blocks)
            return action,action_params,actionid
        else:
            return None,None,actionid
    def generate_mask(self,state,rid):
        return generate_mask(state, rid, self.n_side,self.last_only,self.max_blocks,self.n_robots)


class A2CDenseSupervisor(SupervisorRelative):
    def __init__(self,
                 n_robots,
                 block_choices,
                 action_choice = ['Pl','Ph','L'],
                 grid_size = [10,10],
                 max_blocks=30,
                 n_regions = 2,
                 discount_f = 0.1,
                 device='cuda',
                 use_mask=True,
                 last_only=True,
                 use_wandb=False,
                 ):
        super().__init__(n_robots,block_choices,use_wandb,last_only)
        self.action_choice = action_choice
        self.grid_size = grid_size
        self.n_typeblock = len(block_choices)
        self.max_blocks = max_blocks
        self.n_side = [block.neigh.shape[0] for block in block_choices]
        
        if last_only:
            self.n_actions = (2*max(self.n_side)*sum(self.n_side)+1)*n_robots
        else:
            self.n_actions = (2*max(self.n_side)*sum(self.n_side)+1)*n_robots
        self.action_per_robot =self.n_actions//n_robots
        
        #parameters of the internal model:
        n_fc_layer=4
        n_neurons=500
        n_internal_layer=4
        n_channels = 256
        batch_norm = True
        self.model = im.A2CShared(grid_size,
                                  max_blocks,
                                  n_robots,
                                  n_regions,
                                  self.n_actions,
                                  n_fc_layer =n_fc_layer,
                                  n_neurons = n_neurons,
                                  device=device,
                                  use_wandb=self.use_wandb,
                                  n_internal_layer=n_internal_layer,
                                  shared = False,
                                  n_channels = n_channels,
                                  batch_norm=batch_norm)
        
        self.optimizer = im.A2CSharedOptimizer(self.model,lr=1e-4,pol_over_val=2e-2,tau=1e-4,exploration_factor=0.001)
        
        self.gamma = 1-discount_f
        self.use_mask = use_mask
    def update_policy(self,buffer,buffer_count,batch_size,steps=1):
        if buffer_count==0 or (self.model.batch_norm and buffer_count < 2):
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
                states = [trans.state[1] for trans in batch]
                nstates = [trans.new_state[1] for trans in batch]
                mask = np.array([trans.state[2] for trans in batch])
                nmask = np.array([trans.new_state[2] for trans in batch])
                
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
            mask = np.zeros(self.n_actions,dtype=bool)
            mask[self.action_per_robot*r_id:self.action_per_robot*(1+r_id)]=True
        if not isinstance(state,list):
            state = [state]
        _,actions = self.model(state,inference = True,mask=mask)
        if explore:
            actionid = np.zeros(actions.shape[0],dtype=int)
            for i,p in enumerate(actions.cpu().detach().numpy()):
                actionid[i] = int(np.random.choice(self.n_actions,p=p))
        else:
            actionid = np.argmax(actions.cpu().detach().numpy(),axis=1)
        if len(state)==1:
            if actionid < 30:
                pass
            action,action_params = int2act_sup(actionid[0],self.n_side,self.last_only,self.max_blocks)
            return action,action_params,actionid
        else:
            return None,None,actionid
    def generate_mask(self,state,rid):
        return generate_mask_dense(state, rid, self.n_side,self.last_only,self.max_blocks,self.n_robots)

class A2CSupervisor(SupervisorRelative):
    def __init__(self,
                 n_robots,
                 block_choices,
                 action_choice = ['Pl','Ph','L'],
                 grid_size = [10,10],
                 max_blocks=30,
                 n_regions = 2,
                 discount_f = 0.1,
                 device='cuda',
                 use_mask=True,
                 last_only=True,
                 use_wandb=False,
                 ):
        super().__init__(n_robots,block_choices,use_wandb,last_only)
        self.action_choice = action_choice
        self.grid_size = grid_size
        self.n_typeblock = len(block_choices)
        self.max_blocks = max_blocks
        self.n_side = [block.neigh.shape[0] for block in block_choices]
        
        if last_only:
            self.n_actions = (2*max(self.n_side)*sum(self.n_side)+1)*n_robots
        else:
            self.n_actions = (2*max(self.n_side)*sum(self.n_side)+1)*n_robots
        self.action_per_robot =self.n_actions//n_robots
        
        #parameters of the internal model:
        n_fc_layer=4
        n_neurons=500
        n_internal_layer=4
        n_channels = 256
        batch_norm = True
        self.model = im.A2CShared(grid_size,
                                  max_blocks,
                                  n_robots,
                                  n_regions,
                                  self.n_actions,
                                  n_fc_layer =n_fc_layer,
                                  n_neurons = n_neurons,
                                  device=device,
                                  use_wandb=self.use_wandb,
                                  n_internal_layer=n_internal_layer,
                                  shared = False,
                                  n_channels = n_channels,
                                  batch_norm=batch_norm)
        
        self.optimizer = im.A2CSharedOptimizer(self.model,lr=1e-4,pol_over_val=2e-2,tau=1e-4,exploration_factor=0.001)
        
        self.gamma = 1-discount_f
        self.use_mask = use_mask
    def update_policy(self,buffer,buffer_count,batch_size,steps=1):
        if buffer_count==0 or (self.model.batch_norm and buffer_count < 2):
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
                nmask = np.array([trans.new_state[1] for trans in batch])
                
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
            mask = np.zeros(self.n_actions,dtype=bool)
            mask[self.action_per_robot*r_id:self.action_per_robot*(1+r_id)]=True
        if not isinstance(state,list):
            state = [state]
        _,actions = self.model(state,inference = True,mask=mask)
        if explore:
            actionid = np.zeros(actions.shape[0],dtype=int)
            for i,p in enumerate(actions.cpu().detach().numpy()):
                actionid[i] = int(np.random.choice(self.n_actions,p=p))
        else:
            actionid = np.argmax(actions.cpu().detach().numpy(),axis=1)
        if len(state)==1:
            if actionid < 30:
                pass
            action,action_params = int2act_sup(actionid[0],self.n_side,self.last_only,self.max_blocks)
            return action,action_params,actionid
        else:
            return None,None,actionid
    def generate_mask(self,state,rid):
        return generate_mask(state, rid, self.n_side,self.last_only,self.max_blocks,self.n_robots)


def int2act_sup(action_id,n_side,last_only,max_blocks):
    maxs = max(n_side)
    sums = sum(n_side)
    cumsum = np.cumsum(n_side)
    if last_only:
        r_id = action_id//(2*maxs*sums+1)
        action_id = action_id%(2*maxs*sums+1)
        action_type = action_id//(maxs*sums)
        action = ['Ph','Pl','L'][action_type]
        if action != 'L':
            action_id = action_id%(maxs*sums)
            
            side_support = action_id//sums
            action_id = action_id%sums
            blocktypeid = np.searchsorted(cumsum,action_id,side='right')
            if blocktypeid > 0:
                action_id -= cumsum[blocktypeid-1]
            side_block = action_id
            action_params = {'rid':r_id,
                             'blocktypeid':blocktypeid,
                             'sideblock':side_block,
                             'sidesup':side_support,
                             'idconsup':0,
                              }
   
        else:
            action_params = {'rid':r_id,
                              }
       
    else:
        r_id = action_id//(2*maxs*sums*max_blocks+1)
        action_id = action_id%(2*maxs*sums*max_blocks+1)
        action_type = action_id//(maxs*sums*max_blocks)
        action = ['Ph','Pl','L'][action_type]
        if action != 'L':
            action_id = action_id%(maxs*sums*max_blocks)
            bid_sup = action_id//(maxs*sums)
            
            action_id = action_id%(maxs*sums)
            
            side_support = action_id//sums
            action_id = action_id%sums
            blocktypeid = np.searchsorted(cumsum,action_id)
            if blocktypeid > 0:
                action_id -= cumsum[blocktypeid-1]
            side_block = action_id
         
            action_params = {'rid':r_id,
                             'blocktypeid':blocktypeid,
                             'sideblock':side_block,
                             'sidesup':side_support,
                             'bid_sup':bid_sup,
                              }
        else:
            action_params = {'rid':r_id,
                              }
    return action,action_params
def generate_mask_dense(state,rid,n_side,last_only,max_blocks,n_robots):
    pass
def generate_mask(state,rid,n_side,last_only,max_blocks,n_robots):
    if last_only:
        n_actions = (2*max(n_side)*sum(n_side)+1)
    else:
        n_actions = (2*max(n_side)*sum(n_side)+1)*max_blocks
    #only ph,pl and l
    
    mask = np.zeros(n_actions*n_robots,dtype=bool)
        
    base_idx = rid*n_actions
    #get the ids of the feasible put actions (note that the are not all hidden)
    if last_only:
        mask[base_idx:base_idx+n_actions]=True
        idlast = np.max(state.neighbours)
        if idlast ==0:
            n_side_last = np.sum((state.neighbours[state.connection==0]==idlast))
        else:
            n_side_last = np.sum(state.neighbours==idlast)
        #hide out the remaining indices )if the last block had 2 sides less than the max, hide out these sides
        for i in range(n_side_last,max(n_side)):
            mask[base_idx+i*sum(n_side):base_idx+(i+1)*sum(n_side)]=False
            mask[base_idx+n_actions//2+i*sum(n_side):base_idx+n_actions//2+(i+1)*sum(n_side)]=False
    else:
        #only allows the ids that are already present:
        n_current= np.max(state.neigbours)
        mask[base_idx:base_idx+n_current*(n_actions-1)//max_blocks]=True
        for bid in range(n_current+1):
            n_side_bid = np.sum(state.neigbours==bid)
            for i in range(n_side_bid,max(n_side)):
                mask[base_idx+i*sum(n_side)+bid*sum(n_side)*max(n_side):base_idx+(i+1)*sum(n_side)+bid*sum(n_side)*max(n_side)]=False
                mask[base_idx+n_actions//2+i*sum(n_side)+bid*sum(n_side)*max(n_side):base_idx+n_actions//2+(i+1)*sum(n_side)+bid*sum(n_side)*max(n_side)]=False
    #leave
    mask[base_idx+n_actions-1]=rid in state.hold
        
    return mask
def args2idx(pos,ori,grid_size):
    idx =  (pos[:,0]*grid_size[1]*6+pos[:,1]*6+ori).astype(int)
    return idx
def reward_link2(action,valid,closer,terminal,fail):
    #reward specific to the case where the robots need to link two points

    #add a penalty for holding a block
    hold_penalty = 0.
    #add a penalty if no block are put
    slow_penalty = 0.1
    #add a penatly for forbidden actions
    forbiden_penalty = 0.9
    #add the terminal reward
    terminal_reward = 1
    #add a cost for each block
    block_cost = -0.2
    #add a cost for the blocks going away from the target, 
    #or a reward if the block is going toward the target
    closer_reward = 0.2
    
    #the cost of failing
    failing_cost = 1
    
    reward = 0
    if fail:
        return -failing_cost
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

if __name__ == '__main__':
    print("Start test Agent")
    
    print("\nEnd test Agent")