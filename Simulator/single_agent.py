# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 16:34:42 2022

@author: valla
"""

import numpy as np
import internal_models as im
import abc
from sklearn.neighbors import NearestNeighbors            
    
class Supervisor(metaclass=abc.ABCMeta):
    def __init__(self, n_robots, block_choices,use_wandb=False):
        super().__init__()
        self.n_robots = n_robots
        self.block_choices = block_choices
        self.use_wandb=use_wandb
    def Act(self,simulator,action,
            rid=None,
            #bid=None,
            blocktype=None,
            ori=None,
            pos=None,
            bid=None,
            blocktypeid = None,
            ):
        valid,closer = None,None
        if blocktypeid is not None:
            blocktype= self.block_choices[blocktypeid]
        if action in {'Ph','Pl'}:
            oldbid = simulator.leave(rid)
            if oldbid is not None:
                stable = simulator.check()
                if not stable:
                    #the robot cannot move from there
                    simulator.hold(rid,oldbid)
                    return False,None,blocktype
                
            valid,closer = simulator.put(blocktype,pos,ori)
            
            if valid:
                if action == 'Ph':
                    simulator.hold(rid,simulator.nbid-1)
                if action == 'Pl':
                    stable = simulator.check()
                    if not stable:
                        simulator.remove(simulator.nbid-1)
                        valid = False
            if not valid:
                simulator.hold(rid,oldbid)
                            
        elif action == 'H':
            oldbid = simulator.leave(rid)
            if bid is None:
                bid = simulator.grid.occ[pos[0],pos[1],ori%2]
            if oldbid is not None and oldbid != bid:
                stable = simulator.check()
                valid = stable
                if not stable:
                    #the robot cannot move from there
                    simulator.hold(rid,oldbid)
                else:
                    #chect if the block can be held
                    if not simulator.hold(rid,bid):
                        simulator.hold(rid,oldbid)
                        valid=False
            else:
                valid = simulator.hold(rid,bid)
        elif action == 'R':
            if bid is None:
                bid = simulator.grid.occ[pos[0],pos[1],ori%2]
                
            oldbid = simulator.leave(rid)
            
            if oldbid != bid and not simulator.check():
                    simulator.hold(rid,oldbid)
                    valid=False
            else:
            
                block_pres = simulator.remove(bid)
                if block_pres:
                    valid = simulator.check()
                    if not valid:
                        simulator.undo()
                        simulator.hold(rid,oldbid)
                else:
                    simulator.hold(rid,oldbid)
                    valid=False
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
    
class A2CSupervisor(Supervisor):
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
                 use_wandb=False,
                 ):
        super().__init__(n_robots,block_choices,use_wandb)
        self.action_choice = action_choice
        self.grid_size = grid_size
        self.n_typeblock = len(block_choices)
        self.max_blocks = max_blocks
        self.action_list = generate_actions_supervisor(n_robots,len(block_choices),grid_size,max_blocks,action_choice)
        self.action_per_robot = len(self.action_list)//n_robots
        
        #parameters of the internal model:
        n_fc_layer=4
        n_neurons=100
        n_internal_layer=5
        n_channels = 128
        batch_norm = False
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
                                  n_channels = n_channels,
                                  batch_norm = batch_norm,
                                  )
        
        self.optimizer = im.A2CSharedOptimizer(self.model,lr=1e-4,pol_over_val=2e-1,tau=1e-4,exploration_factor=0.1)
        
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
        return generate_mask_supervisor(state, rid, self.block_choices, self.grid_size, self.max_blocks,self.action_choice,self.n_robots)

class A2CSupervisorStruc(Supervisor):
    def __init__(self,
                 n_robots,
                 block_choices,
                 action_choice = ['Pl','Ph','H','L','R'],
                 grid_size = [10,10],
                 max_blocks=30,
                 n_regions = 2,
                 discount_f = 0.1,
                 device='cuda',
                 use_mask=True,
                 use_wandb=False
                 ):
        super().__init__(n_robots,block_choices,use_wandb)
        self.grid_size = grid_size
        self.n_typeblock = len(block_choices)
        self.max_blocks = max_blocks        
        #parameters of the internal model:
        n_fc_layer=2
        n_neurons=100
        
        self.model = im.A2CSharedEncDec(grid_size,
                                        max_blocks,
                                        n_robots,
                                        n_regions,
                                        len(block_choices),
                                        n_fc_layer_val =n_fc_layer,
                                        n_fc_layer_pol =n_fc_layer,
                                        n_neurons = n_neurons,
                                        device=device,
                                        use_wandb=self.use_wandb)
        
        self.optimizer = im.A2CSharedEncDecOptimizer(self.model,lr=1e-4,pol_over_val=2e-1,tau=1e-3)
        
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
            actionid = np.zeros((actions.shape[0],3),dtype=int)
            for i,p in enumerate(actions.cpu().detach().numpy()):
                actionid[i] = np.unravel_index(np.random.choice(p.size,p=p.flatten()),p.shape)
        else:
            assert False, "should only use exproratory policies"
        if len(state)==1:
            action,action_params = vec2struct_act_sup(actionid[0],n_robots=self.n_robots,n_block_type=self.n_typeblock)
            return action,action_params,actionid
        else:
            return None,None,actionid
    def generate_mask(self,state,rid):
        return generate_struct_mask_supervisor(state, rid, self.block_choices, self.grid_size, n_robots=self.n_robots)

def vec2act_sup(action_vec,r_id,nblocktypes,gridsize,max_block):
    action_vec = np.reshape(action_vec,(5,6))
    actionid = np.argmax(action_vec[:,1])
    if actionid == 0:
        action = 'Ph'
        action_params = {'rid':r_id,
                         'blocktypeid': int((action_vec[actionid,2]/2+0.5)*(nblocktypes-1)),
                         'pos': [int((action_vec[actionid,3]/2+0.5)*(gridsize[0]-1)),
                                 int((action_vec[actionid,4]/2+0.5)*(gridsize[1]-1))],
                         'ori': int((action_vec[actionid,5]/2+0.5)*5),
                         }
    if actionid == 1:
        action = 'Pl'
        action_params = {'rid':r_id,
                         'blocktypeid': int((action_vec[actionid,2]/2+0.5)*(nblocktypes-1)),
                         'pos': [int((action_vec[actionid,3]/2+0.5)*(gridsize[0]-1)),
                                 int((action_vec[actionid,4]/2+0.5)*(gridsize[1]-1))],
                         'ori': int((action_vec[actionid,5]/2+0.5)*5),
                         }
    if actionid == 2:
        action = 'H'
        action_params = {'rid':r_id,'bid':int((action_vec[actionid,2]/2+0.5)*max_block)}
    if actionid == 3:
        action = 'R'
        action_params = {'rid':r_id,'bid':int((action_vec[actionid,2]/2+0.5)*max_block)}
    if actionid == 4:
        action = 'L'
        action_params = {'rid':r_id,}
    return action,action_params
def vec2struct_act_sup(coords,n_robots=2,n_block_type=2):
    act_per_robot = n_block_type*12+5
    rid = coords[0]//act_per_robot
    actionid = coords[0]-rid*act_per_robot
    if actionid in range(6*n_block_type):
        action = 'Ph'
        blocktypeid = actionid//6
        ori = actionid % 6
        action_params = {'rid':rid,
                         'blocktypeid': blocktypeid,
                         'pos': [coords[1],coords[2]],
                         'ori': ori,
                         }
    if actionid in range(6*n_block_type,12*n_block_type):
        action = 'Pl'
        blocktypeid = (actionid-6*n_block_type)//6
        ori = actionid % 6
        action_params = {'rid':rid,
                         'blocktypeid': blocktypeid,
                         'pos': [coords[1],coords[2]],
                         'ori': ori,
                         }
    if actionid in range(12*n_block_type,12*n_block_type+2):
        action = 'H'
        ori = actionid-12*n_block_type
        action_params = {'rid':rid,
                         'pos': [coords[1],coords[2]],
                         'ori': ori,}
    if actionid in range(12*n_block_type+2,12*n_block_type+4):
        action = 'R'
        ori = actionid-12*n_block_type-2
        action_params = {'rid':rid,
                        'pos': [coords[1],coords[2]],
                        'ori': ori,}
    if actionid == 12*n_block_type+4:
        action = 'L'
        action_params = {'rid':rid}
    return action,action_params
def generate_actions_supervisor(n_robots,ntype_blocks,grid_size,max_blocks,action_list):
    if len(action_list)==5:
        #first generate all action for a given robot
        actions = np.zeros((grid_size[0]*grid_size[1]*6*ntype_blocks*2+2*max_blocks+1,5,5))
        
        #each hold action:
        actions[:max_blocks,2,0]=1
        actions[:max_blocks,2,1:]=np.linspace(-1,1,max_blocks)[...,None]
        #each remove action:
        actions[max_blocks:2*max_blocks,3,0]=1
        actions[max_blocks:2*max_blocks,3,1:]=np.linspace(-1,1,max_blocks)[...,None]
        #each place action:
        tv,xv, yv,rv = np.meshgrid(np.linspace(-1,1,ntype_blocks),
                                   np.linspace(-1,1,grid_size[0]),
                                   np.linspace(-1,1,grid_size[1]),
                                   np.linspace(-1,1,6),
                                   indexing='ij')
        
        range_Ph = (2*max_blocks, grid_size[0]*grid_size[1]*6*ntype_blocks+2*max_blocks)
        actions[range_Ph[0]:range_Ph[1],0,0]=1
        actions[range_Ph[0]:range_Ph[1],0,1]=tv.flatten()
        actions[range_Ph[0]:range_Ph[1],0,2]=xv.flatten()
        actions[range_Ph[0]:range_Ph[1],0,3]=yv.flatten()
        actions[range_Ph[0]:range_Ph[1],0,4]=rv.flatten()
        
        range_Pl = (grid_size[0]*grid_size[1]*6*ntype_blocks+2*max_blocks,
                    2*grid_size[0]*grid_size[1]*6*ntype_blocks+2*max_blocks)
        actions[range_Pl[0]:range_Pl[1],1,0]=1
        actions[range_Pl[0]:range_Pl[1],1,1]=tv.flatten()
        actions[range_Pl[0]:range_Pl[1],1,2]=xv.flatten()
        actions[range_Pl[0]:range_Pl[1],1,3]=yv.flatten()
        actions[range_Pl[0]:range_Pl[1],1,4]=rv.flatten()
        #leave action
        actions[-1,4,:]=1
        
        #add as many copies of actions as there are robots
        actions_sup = np.tile(actions,(n_robots,1,1))
        rids = np.tile(np.repeat(np.linspace(-1,1,n_robots),actions.shape[0])[...,None],(1,5))
        actions_sup = np.concatenate([rids[...,None],actions_sup],axis=2)
        return np.reshape(actions_sup,(-1,30))
    elif len(action_list)==3:
        #first generate all action for a given robot
        actions = np.zeros((grid_size[0]*grid_size[1]*6*ntype_blocks*2+1,5,5))
        
        #each place action:
        tv,xv, yv,rv = np.meshgrid(np.linspace(-1,1,ntype_blocks),
                                   np.linspace(-1,1,grid_size[0]),
                                   np.linspace(-1,1,grid_size[1]),
                                   np.linspace(-1,1,6),
                                   indexing='ij')
        
        range_Ph = (0, grid_size[0]*grid_size[1]*6*ntype_blocks)
        actions[range_Ph[0]:range_Ph[1],0,0]=1
        actions[range_Ph[0]:range_Ph[1],0,1]=tv.flatten()
        actions[range_Ph[0]:range_Ph[1],0,2]=xv.flatten()
        actions[range_Ph[0]:range_Ph[1],0,3]=yv.flatten()
        actions[range_Ph[0]:range_Ph[1],0,4]=rv.flatten()
        
        range_Pl = (grid_size[0]*grid_size[1]*6*ntype_blocks,
                    2*grid_size[0]*grid_size[1]*6*ntype_blocks)
        actions[range_Pl[0]:range_Pl[1],1,0]=1
        actions[range_Pl[0]:range_Pl[1],1,1]=tv.flatten()
        actions[range_Pl[0]:range_Pl[1],1,2]=xv.flatten()
        actions[range_Pl[0]:range_Pl[1],1,3]=yv.flatten()
        actions[range_Pl[0]:range_Pl[1],1,4]=rv.flatten()
        #leave action
        actions[-1,4,:]=1
        
        #add as many copies of actions as there are robots
        actions_sup = np.tile(actions,(n_robots,1,1))
        rids = np.tile(np.repeat(np.linspace(-1,1,n_robots),actions.shape[0])[...,None],(1,5))
        actions_sup = np.concatenate([rids[...,None],actions_sup],axis=2)
        return np.reshape(actions_sup,(-1,30))
def generate_struct_mask_supervisor(state,rid,block_type,grid_size,n_robots=2):
    mask = np.zeros((n_robots*(12*len(block_type)+5),grid_size[0],grid_size[1]),dtype=bool)
    base_idx = rid*(12*len(block_type)+5)
    #mask for the hold and remove:
    coords = np.nonzero(state.occ>0)
    #each feasible hold action of ids are placed at ids
    mask[base_idx+12*len(block_type)+coords[2],coords[0],coords[1]]=True
    #each feasible remove action is placed at maxblock+ids
    mask[base_idx+12*len(block_type)+2+coords[2],coords[0],coords[1]]=True
    
    
    #get the ids of the feasible put actions (note that the are not all hidden)
    for idb, block in enumerate(block_type):
        pos,ori = state.touch_side(block)
        #ph
        mask[base_idx+idb*6+ori,pos[:,0],pos[:,1]]=True
        #pl
        mask[base_idx+len(block_type)*6+idb*6+ori,pos[:,0],pos[:,1]]=True
    #leave
    if rid in state.hold:
        mask[base_idx+len(block_type)*6+4,:,:]=True
    return mask

def generate_mask_supervisor(state,rid,block_type,grid_size,max_blocks,action_list,n_robots=2,last_only=True):
    if len(action_list)==5:
        mask = np.zeros((n_robots*(grid_size[0]*grid_size[1]*6*len(block_type)*2+2*max_blocks+1)),dtype=bool)
        base_idx = rid*(grid_size[0]*grid_size[1]*6*len(block_type)*2+2*max_blocks+1)
        #mask the hold and remove:
        ids = np.unique(state.occ)
        #remove the -1 and 0
        ids = ids[2:]
        #if all the other robots are holding a block, the robot cannot hold as 
        holders = np.unique(state.hold)
        if not np.all(np.isin(np.delete(np.arange(n_robots),rid), holders)):
            #each feasible hold action of ids are placed at ids
            mask[base_idx+ids]=True
        #each feasible remove action is placed at maxblock+ids
        mask[base_idx+max_blocks+ids]=True
    elif len(action_list) ==3:
        #only ph,pl and l
        mask = np.zeros((n_robots*(grid_size[0]*grid_size[1]*6*len(block_type)*2+1)),dtype=bool)
        base_idx = rid*(grid_size[0]*grid_size[1]*6*len(block_type)*2+1)
        
        #get the ids of the feasible put actions (note that the are not all hidden)
        for idb, block in enumerate(block_type):
            pos,ori = state.touch_side(block,last=last_only)
            ids = args2idx(pos,ori,grid_size)
            if not np.all(ids<grid_size[0]*grid_size[1]*6):
                args2idx(pos,ori,grid_size)
            #ph
            mask[base_idx+idb*np.prod(grid_size)*6+ids]=True
            #pl
            mask[base_idx+(len(block_type)+idb)*np.prod(grid_size)*6+ids]=True
        #leave
        mask[base_idx+(grid_size[0]*grid_size[1]*6*len(block_type)*2)]=rid in state.hold
    return mask
def args2idx(pos,ori,grid_size):
    idx =  (pos[:,0]*grid_size[1]*6+pos[:,1]*6+ori).astype(int)
    return idx
def reward_link2(action,valid,closer,terminal,fail):
    #reward specific to the case where the robots need to link two points

    #add a penalty for holding a block
    hold_penalty = 0.
    #add a penalty if no block are put
    slow_penalty = 1
    #add a penatly for forbidden actions
    forbiden_penalty = 0.9
    #add the terminal reward
    terminal_reward = 1
    #add a cost for each block
    block_cost = 0.
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