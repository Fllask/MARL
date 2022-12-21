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
import abc
import wandb

class AgentSimultanous(metaclass=abc.ABCMeta):
    def __init__(self, n_robots,rid, block_choices,config,use_wandb=False,log_freq = None,env='norot'):
        super().__init__()
        self.rid = rid
        self.n_robots = n_robots
        self.block_choices = block_choices
        self.use_wandb=use_wandb
        self.log_freq = None
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
                    return False,None,blocktype
            valid,closer = simulator.put_rel(blocktype,sideblock,sidesup,bid_sup,side_ori,blocktypeid=blocktypeid,idconsup = idconsup)
                
            if valid:
                simulator.hold(self.rid,simulator.nbid-1)
        if action in {'L','S'}:
            return True,None,None
        return valid,closer,blocktype
    @abc.abstractmethod
    def generate_mask(self,state):
        pass
    @abc.abstractmethod
    def update_policy(self,**kwargs):
        pass
    @abc.abstractmethod
    def choose_action(self,state):
        pass
    
class SACSparse(AgentSimultanous):
    def __init__(self,
                 n_robots,
                 rid,
                 block_choices,
                 config,
                 ground_block = None,
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
                                                  np.sum((block.neigh[:,2]==0) & (block.neigh[:,3]==2)),] for block in [ground_block]+ block_choices])
            #check the genererate_mask_norot function to understand why these parameters
            
           
            self.n_actions = (max_blocks+n_regions)*(1+len(block_choices))*(2+len(block_choices))*np.max(self.n_side_oriented_sup)*np.max(self.n_side_oriented)*6
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
        if self.use_wandb and self.optimizer.step % self.log_freq == 0:
            wandb.log({"Robot_"+self.rid+'_action_dist':actions_dist.probs[0,mask]},step=self.optimizer.step)
        if self.exploration_strat == 'softmax':
            actionid = actions_dist.sample().detach().cpu().numpy()[0]
            if self.use_wandb and self.optimizer.step % self.log_freq == 0:
                wandb.log({"Robot_"+self.rid+'action_id':actionid},step=self.optimizer.step)
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
            return generate_mask_norot(self.rid,
                                       state.grid,
                                       self.n_side_oriented,
                                       self.n_side_oriented_sup,
                                       self.max_blocks,
                                       state.graph.n_reg,
                                       self.action_choice,
                                       state.type_id)
        else:
            assert False, "Not implemented"

def generate_mask_norot(rid,grid,n_side_b,n_side_sup,maxblocks,n_reg,action_choice,typeblocks,reverse=True):
    mask = np.zeros((maxblocks+n_reg,
                     n_side_sup.shape[0],
                     n_side_b.shape[0]+2,
                     np.max(n_side_sup),
                     np.max(n_side_b),
                     6
                     ),dtype=bool)
    #only keep the blocks that are present
    if reverse:
        typeblocks = reverse(typeblocks)
    if 'P' in action_choice:
        for i,n_side in enumerate(n_side_b):
            for j,typesup in enumerate(typeblocks):
                if typesup ==-2:
                    continue
                typesup+=1#set the ground block at 0
                for k in range(6):
                    mask[j,typesup,i,:n_side_sup[typesup,k],:n_side[k],k]=True
    #both these actions are presented 6 times to allow a little entropy increase in the final steps of the episode
    #mask for the leave action
    if 'L' in action_choice:
        mask[0,0,-2,0,0]=rid in grid.hold
    #mask for the stay action
    if 'S' in action_choice:
        mask[0,0,-1,0,0]=True

   
    return mask.flatten()
def int2actsim_norot(actionid,n_blocks,n_side_b,n_side_sup,maxblocks,n_reg,reverse=True):
    blockid, btype_sup, btype, side_sup, side_b,side_ori =np.unravel_index(actionid,
                                                              (maxblocks+n_reg,
                                                               n_side_sup.shape[0],
                                                               n_side_b.shape[0]+2,
                                                               np.max(n_side_sup),
                                                               np.max(n_side_b),
                                                               6))
    if btype == n_side_b.shape[0]:
        action = 'L'
        action_params = {}
    elif btype == n_side_b.shape[0]+1:
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
                             'idconsup':blockid,#always place the block on the second target
                              }
        else:
            action_params = {'blocktypeid':btype,
                             'sideblock':side_b,
                             'sidesup':side_sup,
                             'bid_sup':blockid-n_reg,
                             'side_ori':side_ori,
                             'idconsup':None,#always place the block on the second target
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    