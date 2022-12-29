# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 15:48:45 2022

@author: valla
"""

import numpy as np
import internal_models as im
import abc     
import wandb
import pickle
import os
class SupervisorRelative(metaclass=abc.ABCMeta):
    def __init__(self, n_robots, block_choices,config,log_freq=None,use_wandb=False,env='rot'):
        super().__init__()
        self.n_robots = n_robots
        self.block_choices = block_choices
        self.use_wandb=use_wandb
        self.log_freq = log_freq
        self.env = env
        self.last_only = config['agent_last_only']
        self.gamma = 1-config['agent_discount_f']
        self.use_mask = config['ep_use_mask']
    def Act(self,simulator,action,
            rid=None,
            sideblock=None,
            sidesup = None,
            bid_sup = None,
            idconsup = None,
            blocktypeid = None,
            side_ori = None,
            draw= False,
            ):
        valid,closer,interfaces = None,None,None
        if blocktypeid is not None:
            blocktype= self.block_choices[blocktypeid]
        else:
            blocktype = None
        if action in {'Ph','Pl'}:
            oldbid = simulator.leave(rid)
            if oldbid is not None:
                stable = simulator.check()
                if not stable:
                    #the robot cannot move from there
                    #simulator.hold(rid,oldbid)
                    return False,None,blocktype,None
            valid,closer,interfaces = simulator.put_rel(blocktype,sideblock,sidesup,bid_sup,side_ori,blocktypeid=blocktypeid,idconsup = idconsup)
                
            if valid:
                if action == 'Ph':
                    simulator.hold(rid,simulator.nbid-1)
                if action == 'Pl':
                    stable = simulator.check()
                    if not stable:
                        #simulator.remove(simulator.nbid-1,save=False)
                        valid = False
            # if not valid:
            #     simulator.hold(rid,oldbid)
                            
        elif action == 'L':
            oldbid = simulator.leave(rid)
            if oldbid is not None:
                stable = simulator.check()
                valid = stable
                # if not stable:
                #     simulator.hold(rid,oldbid)
            else:
                valid = False
        else:
            assert False,'Unknown action'
        return valid,closer,blocktype,interfaces
    def save(self,name,path):
        file = open(os.path.join(path,name+'.pickle'), 'wb')
        pickle.dump(self,file)
        file.close()
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
                 config,
                 action_choice = ['Pl','Ph','L'],
                 grid_size = [10,10],
                 max_blocks=30,
                 max_interfaces = 120,
                 n_regions = 2,
                 discount_f = 0.1,
                 use_mask=True,
                 last_only=True,
                 use_wandb=False,
                 ):
        super().__init__(n_robots,block_choices,config,use_wandb)
        self.rep= 'graph'
        self.action_choice = action_choice
        self.grid_size = grid_size
        self.n_typeblock = len(block_choices)
        self.exploration_strat = config['agent_exp_strat']
        if self.exploration_strat == 'epsilon-greedy' or self.exploration_strat == 'epsilon-softmax':
            self.eps = config['agent_epsilon']
        self.max_blocks = max_blocks
        self.n_side = [block.neigh.shape[0] for block in block_choices]
        
        if last_only:
            self.n_actions = (2*max(self.n_side)*sum(self.n_side)+1)*n_robots
        else:
            self.n_actions = (2*max(self.n_side)*sum(self.n_side)+1)*n_robots
        self.action_per_robot =self.n_actions//n_robots
        

        self.model = im.A2CDense(grid_size,
                                max_blocks,
                                max_interfaces,
                                self.n_typeblock,
                                self.n_side,
                                n_robots,
                                n_regions,
                                self.n_actions,
                                config,
                                use_wandb=self.use_wandb)
        
        self.optimizer = im.A2CDenseOptimizer(self.model,config)
        #self.reader = Dataloader()
    def update_policy(self,buffer,buffer_count,batch_size,steps=1):
        for s in range(steps):
            if batch_size > buffer.counter:
                return buffer.counter
                
            (state,mask,action,reward,nstate,nmask)=buffer.sample(batch_size)
            
            l_v,l_p = self.optimizer.optimize(state,action,reward,nstate,self.gamma,mask,nmask=nmask)
            
    def choose_action(self,r_id,state,explore=True,mask=None):
        if mask is None:
            mask = np.zeros(self.n_actions,dtype=bool)
            mask[self.action_per_robot*r_id:self.action_per_robot*(1+r_id)]=True
        _,actions = self.model(graph = state.graph,inference = True,mask=mask)
        actions_dist = actions.squeeze().cpu().detach().numpy()
        if self.exploration_strat == 'softmax':
            actionid = np.random.choice(self.n_actions,p=actions_dist)
        elif self.exploration_strat == 'epsilon-greedy':
            if np.random.rand() > self.eps:
                actionid = np.argmax(actions_dist)
            else:
                ids, = np.nonzero(mask)
                actionid = np.random.choice(ids)
        elif self.exploration_strat == 'epsilon-softmax':
            if np.random.rand() > self.eps:
                actionid = np.random.choice(self.n_actions,p=actions_dist)
            else:
                ids, = np.nonzero(mask)
                actionid = np.random.choice(ids)
        action,action_params = int2act_sup(actionid,self.n_side,self.last_only,state.graph.n_blocks,self.max_blocks)
        return action,action_params,actionid
        
    def generate_mask(self,state,rid):
        return generate_mask_dense(state, rid, self.n_side,self.last_only,self.max_blocks,self.n_robots)
class SACSupervisorDense(SupervisorRelative):
    def __init__(self,
                 n_robots,
                 block_choices,
                 config,
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
                 ):
        super().__init__(n_robots,block_choices,config,use_wandb=use_wandb,log_freq = log_freq)
        self.rep= 'graph'
        self.action_choice = action_choice
        self.grid_size = grid_size
        self.n_typeblock = len(block_choices)
        self.exploration_strat = config['agent_exp_strat']
        if self.exploration_strat == 'epsilon-greedy' or self.exploration_strat == 'epsilon-softmax':
            self.eps = config['agent_epsilon']
        self.max_blocks = max_blocks
        self.n_side = [block.neigh.shape[0] for block in block_choices]
        if 'Pl'in self.action_choice:
            if last_only:
                self.n_actions = (2*max(self.n_side)*sum(self.n_side)+1)*n_robots
            else:
                self.n_actions = (2*max(self.n_side)*sum(self.n_side)+1)*n_robots
        else:
            if last_only:
                self.n_actions = (max(self.n_side)*sum(self.n_side)+1)*n_robots
            else:
                self.n_actions = (max(self.n_side)*sum(self.n_side)+1)*n_robots
        self.action_per_robot =self.n_actions//n_robots
        

        self.model = im.PolNetDense(grid_size,
                                max_blocks,
                                max_interfaces,
                                self.n_typeblock,
                                self.n_side,
                                n_robots,
                                n_regions,
                                self.n_actions,
                                config,
                                use_wandb=self.use_wandb,
                                log_freq = self.log_freq)
        
        self.optimizer = im.SACDenseOptimizer(grid_size,
                                              max_blocks,
                                              max_interfaces,
                                              self.n_typeblock,
                                              self.n_side,
                                              n_robots,
                                              self.n_actions,
                                              self.model,
                                              config)
    def update_policy(self,buffer,buffer_count,batch_size,steps=1):
        for s in range(steps):
            if batch_size > buffer.counter:
                return buffer.counter
                
            (state,mask,action,reward,nstate,nmask,entropy)=buffer.sample(batch_size)
            
            l_p = self.optimizer.optimize(state,action,reward,nstate,self.gamma,mask,nmask=nmask,old_entropy = entropy)
            
    def choose_action(self,r_id,state,explore=True,mask=None):
        if mask is None:
            mask = np.zeros(self.n_actions,dtype=bool)
            mask[self.action_per_robot*r_id:self.action_per_robot*(1+r_id)]=True
        actions_dist,logits = self.model(graph = state.graph,inference = True,mask=mask)
        if self.use_wandb and self.optimizer.step % self.log_freq == 0:
            wandb.log({'action_dist':actions_dist.probs[0,mask]},step=self.optimizer.step)
        if self.exploration_strat == 'softmax':
            actionid = actions_dist.sample().detach().cpu().numpy()[0]
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
        action,action_params = int2act_sup(actionid,self.n_side,self.last_only,state.graph.n_blocks,self.max_blocks,self.action_choice)
        return action,action_params,actionid,actions_dist.entropy()
        
    def generate_mask(self,state,rid):
        if 'Pl' in self.action_choice:
            return generate_mask_dense(state, rid, self.n_side,self.last_only,self.max_blocks,self.n_robots,self.action_choice)
        else:
            return generate_mask_always_hold(state, rid, self.n_side,self.last_only,self.max_blocks,self.n_robots,self.action_choice)
class SACSupervisorSparse(SupervisorRelative):
    def __init__(self,
                 n_robots,
                 block_choices,
                 config,
                 ground_block = None,
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
                 env='rot'
                 ):
        super().__init__(n_robots,block_choices,config,use_wandb=use_wandb,log_freq = log_freq,env=env)
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
            self.n_actions = n_robots*(1+len(block_choices))*(1+len(block_choices))*np.max(self.n_side_oriented_sup)*np.max(self.n_side_oriented)*6
        
        else:
            self.n_sides = [len(block.neigh) for block in block_choices]
            if 'Pl'in self.action_choice:
                if last_only:
                    self.n_actions = (2*max(self.n_side)*sum(self.n_side)+1)*n_robots
                else:
                    self.n_actions = (2*max(self.n_side)*sum(self.n_side)+1)*n_robots
            else:
                if last_only:
                    self.n_actions = (max(self.n_side)*sum(self.n_side)+1)*n_robots
                else:
                    self.n_actions = (max(self.n_side)*sum(self.n_side)+1)*n_robots
        self.action_per_robot =self.n_actions//n_robots

        self.model = im.PolNetSparse(grid_size,
                                max_blocks,
                                n_robots,
                                n_regions,
                                self.n_actions,
                                config,
                                use_wandb=self.use_wandb,
                                log_freq = self.log_freq)
        
        self.optimizer = im.SACSparseOptimizer(grid_size,
                                              max_blocks,
                                              n_robots,
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
            if self.use_mask:
                states = [trans.state['grid'] for trans in batch]
                nstates = [trans.new_state['grid'] for trans in batch]
                mask = np.array([trans.state['mask'] for trans in batch])
                nmask = np.array([trans.new_state['mask'] for trans in batch])
            else:
                states = [trans.state['grid'] for trans in batch]
                nstates = [trans.new_state['grid'] for trans in batch]
                mask=None
                nmask=None
            actions = np.array([trans.a[0] for trans in batch])
            rewards = np.array([[trans.r] for trans in batch],dtype=np.float32)

            l_p = self.optimizer.optimize(states,actions,rewards,nstates,self.gamma,mask,nmask=nmask)
            
    def choose_action(self,r_id,state,explore=True,mask=None):
        if mask is None:
            mask = np.zeros(self.n_actions,dtype=bool)
            mask[self.action_per_robot*r_id:self.action_per_robot*(1+r_id)]=True
        actions_dist,logits = self.model([state.grid],inference = True,mask=mask)
        if self.use_wandb and self.optimizer.step % self.log_freq == 0:
            wandb.log({'action_dist':actions_dist.probs[0,mask]},step=self.optimizer.step)
        if self.exploration_strat == 'softmax':
            actionid = actions_dist.sample().detach().cpu().numpy()[0]
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
        if self.env == 'norot':
            action,action_params = int2act_norot(actionid,state.graph.n_blocks,self.n_robots,self.n_side_oriented,self.n_side_oriented_sup,self.last_only,self.max_blocks,self.action_choice)
        else:
            action,action_params = int2act_sup(actionid,self.n_side,self.last_only,state.graph.n_blocks,self.max_blocks,self.action_choice)
        return action,action_params,actionid,actions_dist.entropy()
        
    def generate_mask(self,state,rid):
        if self.env == 'norot':
            return generate_mask_no_rot(state.grid, rid, self.n_side_oriented,self.n_side_oriented_sup,self.last_only,self.max_blocks,self.n_robots,self.action_choice,state.type_id)
        if 'Pl' in self.action_choice:
            return generate_mask_dense(state, rid, self.n_side,self.last_only,self.max_blocks,self.n_robots,self.action_choice)
        else:
            return generate_mask_always_hold(state, rid, self.n_side,self.last_only,self.max_blocks,self.n_robots,self.action_choice)
class A2CSupervisor(SupervisorRelative):
    def __init__(self,
                 n_robots,
                 block_choices,
                 config,
                 action_choice = ['Pl','Ph','L'],
                 grid_size = [10,10],
                 max_blocks=30,
                 n_regions = 2,
                 use_wandb=False,
                 ):
        super().__init__(n_robots,block_choices,config,use_wandb)
        self.action_choice = action_choice
        self.grid_size = grid_size
        self.n_typeblock = len(block_choices)
        self.max_blocks = max_blocks
        self.n_side = [block.neigh.shape[0] for block in block_choices]
        
        if self.last_only:
            self.n_actions = (2*max(self.n_side)*sum(self.n_side)+1)*n_robots
        else:
            self.n_actions = (2*max(self.n_side)*sum(self.n_side)+1)*n_robots
        self.action_per_robot =self.n_actions//n_robots
        

        self.model = im.A2CShared(grid_size,
                                  max_blocks,
                                  n_robots,
                                  n_regions,
                                  self.n_actions,
                                  config,
                                  use_wandb=self.use_wandb,
                                  )
        
        self.optimizer = im.A2CSharedOptimizer(self.model,config)
        
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
                states = [trans.state['grid'] for trans in batch]
                nstates = [trans.new_state['grid'] for trans in batch]
                mask = np.array([trans.state['mask'] for trans in batch])
                nmask = np.array([trans.new_state['mask'] for trans in batch])
                
            else:
                states = [trans.state['grid'] for trans in batch]
                nstates = [trans.new_state['grid'] for trans in batch]
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
            state = [state.grid]
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
            action,action_params = int2act_sup(actionid[0],self.n_side,self.last_only,np.max(state[0].occ),self.max_blocks)
            return action,action_params,actionid
        else:
            return None,None,actionid
    def generate_mask(self,state,rid):
        return generate_mask(state, rid, self.n_side,self.last_only,self.max_blocks,self.n_robots)


def int2act_norot(action_id,n_block,n_robots, n_side_b,n_side_sup,last_only,max_blocks,action_choices):
    if last_only:
        rid, btype_sup, btype, side_sup, side_b,side_ori =np.unravel_index(action_id,
                                                                  (n_robots,
                                                                   n_side_sup.shape[0],
                                                                   n_side_b.shape[0]+1,
                                                                   np.max(n_side_sup),
                                                                   np.max(n_side_b),
                                                                   6))
        if btype == n_side_b.shape[0]:
            action = 'L'
            action_params = {'rid':rid,
                              }
        else:
            action = 'Ph'
            action_params =  {'rid':rid,
                             'blocktypeid':btype,
                             'sideblock':side_b,
                             'sidesup':side_sup,
                             'bid_sup':n_block,
                             'side_ori':side_ori,
                             'idconsup':1,#always place the block on the second target
                              }
        return action,action_params
    else:
        assert False, "not implemented"
def int2act_sup(action_id,n_side,last_only,n_block,max_blocks,action_choice):
    maxs = max(n_side)
    sums = sum(n_side)
    cumsum = np.cumsum(n_side)
    if 'Pl' in action_choice:
        if last_only:
            r_id = action_id//(2*maxs*sums+1)
            action_id = action_id%(2*maxs*sums+1)
            action_type = action_id//(maxs*sums)
            action = action_choice[action_type]
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
                                 'bid_sup':n_block,
                                 'idconsup':1,#always place the block on the second target
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
                                 'bid_sup':n_block-bid_sup,
                                  }
            else:
                action_params = {'rid':r_id,
                                  }
    else:
        if last_only:
            r_id = action_id//(maxs*sums+1)
            action_id = action_id%(maxs*sums+1)
            action_type = action_id//(maxs*sums)
            action = action_choice[action_type]
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
                                 'bid_sup':n_block,
                                 'idconsup':1,#always place the block on the second target
                                  }

            else:
                action_params = {'rid':r_id,
                                  }

        else:
            r_id = action_id//(maxs*sums*max_blocks+1)
            action_id = action_id%(maxs*sums*max_blocks+1)
            action_type = action_id//(maxs*sums*max_blocks)
            action = action_choice[action_type]
            if action != 'L':
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
                                 'bid_sup':n_block-bid_sup,
                                  }
            else:
                action_params = {'rid':r_id,
                                  }
    return action,action_params
def generate_mask_dense(state,rid,n_side,last_only,max_blocks,n_robots,action_choices):
    assert 'Pl' in action_choices, " Wrong mask / action choices combination"
        
    if last_only:
        n_actions = (max(n_side)*sum(n_side)+1)
    else:
        n_actions = (max(n_side)*sum(n_side))*max_blocks+1
    #only ph,pl and l
    
    mask = np.zeros(n_actions*n_robots,dtype=bool)
        
    base_idx = rid*n_actions
    #get the ids of the feasible put actions (note that the are not all hidden)
    if last_only:
        mask[base_idx:base_idx+n_actions]=True
        idlast = np.max(state.neighbours[:,:,:,:,0])
        if idlast ==0:
            n_side_last = np.sum(state.neighbours[state.connection==1][:,:,0]==0)
        else:
            n_side_last = np.sum(state.neighbours[:,:,:,:,0]==idlast)
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
def generate_mask_no_rot(state,rid,n_side_b, n_side_sup,last_only,max_blocks,n_robots,action_choices,placed_block_typeid):
    assert 'Pl' not in action_choices, "wrong action choices / mask combination"
    # for an interface to not require rotation, the 3 coordinate of each side needs to be different 
    #while the 4th needs to be equal:
        
    #get the ids of the feasible put actions (note that the are not all hidden)
    if last_only:
        mask = np.zeros((n_robots,
                         n_side_sup.shape[0],
                         n_side_b.shape[0]+1,
                         np.max(n_side_sup),
                         np.max(n_side_b),
                         6
                         ),dtype=bool)
        #only keep the blocks that are present
        placed_block_typeid = placed_block_typeid[placed_block_typeid > -2]
        type_sup = placed_block_typeid[-1]+1
        for i,n_side in enumerate(n_side_b):
            for k in range(6):
                mask[rid,type_sup,i,:n_side_sup[type_sup,k],:n_side[k],k]=True
        #mask for the leave action
        mask[rid,type_sup,-1,0,0]=rid in state.hold
    else:
        assert False, "not implemented"
        mask = np.zeros((n_robots,
                         max_blocks,
                         n_side_sup.shape[0],
                         n_side_b.shape[0],
                         np.max(n_side_sup),
                         np.max(n_side_b)
                         ),dtype=bool)
        #only allows the ids that are already present:
        n_current= np.max(state.neigbours[:,:,:,:,0])
        
    
    return mask.flatten()

def generate_mask_always_hold(state,rid,n_side,last_only,max_blocks,n_robots,action_choices):
    assert 'Pl' not in action_choices, "wrong action choices / mask combination"
    if last_only:
        n_actions = (max(n_side)*sum(n_side)+1)
    else:
        n_actions = (max(n_side)*sum(n_side))*max_blocks+1
    #only ph,pl and l
    
    mask = np.zeros(n_actions*n_robots,dtype=bool)
        
    base_idx = rid*n_actions
    #get the ids of the feasible put actions (note that the are not all hidden)
    if last_only:
        mask[base_idx:base_idx+n_actions]=True
        idlast = np.max(state.neighbours[:,:,:,:,0])
        if idlast ==0:
            side_last = state.neighbours[state.connection==1][:,:,0]
        else:
            side_last = np.sum(state.neighbours[:,:,:,:,0]==idlast)
        #hide out the remaining indices )if the last block had 2 sides less than the max, hide out these sides
        for i in range(side_last,max(n_side)):
            mask[base_idx+i*sum(n_side):base_idx+(i+1)*sum(n_side)]=False
    else:
        #only allows the ids that are already present:
        n_current= np.max(state.neigbours)
        mask[base_idx:base_idx+n_current*(n_actions-1)//max_blocks]=True
        for bid in range(n_current+1):
            n_side_bid = np.sum(state.neigbours==bid)
            for i in range(n_side_bid,max(n_side)):
                mask[base_idx+i*sum(n_side)+bid*sum(n_side)*max(n_side):base_idx+(i+1)*sum(n_side)+bid*sum(n_side)*max(n_side)]=False
    #leave
    mask[base_idx+n_actions-1]=rid in state.hold
        
    return mask
def generate_mask(state,rid,n_side,last_only,max_blocks,n_robots,action_choices):
    if last_only:
        n_actions = (2*max(n_side)*sum(n_side)+1)
    else:
        n_actions = (2*max(n_side)*sum(n_side))*max_blocks+1
    #only ph,pl and l
    
    mask = np.zeros(n_actions*n_robots,dtype=bool)
        
    base_idx = rid*n_actions
    #get the ids of the feasible put actions (note that the are not all hidden)
    if last_only:
        mask[base_idx:base_idx+n_actions]=True
        idlast = np.max(state.neighbours[:,:,:,:,0])
        if idlast ==0:
            side_last = state.neighbours[state.connection==1][:,:,0]
        else:
            side_last = np.sum(state.neighbours[:,:,:,:,0]==idlast)
        #hide out the remaining indices )if the last block had 2 sides less than the max, hide out these sides
        for i in range(side_last,max(n_side)):
            mask[base_idx+i*sum(n_side):base_idx+(i+1)*sum(n_side)]=False
            mask[base_idx+n_actions//2+i*sum(n_side):base_idx+n_actions//2+(i+1)*sum(n_side)]=False
    else:
        #only allows the ids that are already present:
        n_current= np.max(state.neigbours[:,:,:,:,0])
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

def generous_reward(action,valid,closer,terminal,fail,n_sides,**kwargs):
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
    block_cost = -0.
    #add a cost for the blocks going away from the target, 
    #or a reward if the block is going toward the target
    closer_reward = 0.4
    
    #the cost of failing
    failing_cost = 1
    
    reward = 0
    if fail:
        return -failing_cost
    if not valid:
        return -forbiden_penalty
    if action in {'H', 'L','R'}:
        reward=-slow_penalty
    if action =='Ph':
        reward-=hold_penalty
    if action in {'Ph', 'Pl'}:
        if closer == 1:
            reward += closer_reward
        reward-=block_cost
    if terminal:
        reward += terminal_reward
    return reward
def punitive_reward(action,valid,closer,terminal,fail,**kwargs):
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
    block_cost = -0.
    #add a cost for the blocks going away from the target, 
    #or a reward if the block is going toward the target
    closer_reward = 0.4
    
    #the cost of failing
    failing_cost = 1
    
    reward = 0
    if fail:
        return -failing_cost
    if not valid:
        return -forbiden_penalty
    if action in {'H', 'L','R'}:
        reward=-slow_penalty
    if action =='Ph':
        reward-=hold_penalty
    if action in {'Ph', 'Pl'}:
        if closer == 1:
            reward += closer_reward
        else:
            reward -= closer_reward
        reward-=block_cost
    if terminal:
        reward += terminal_reward
    return reward
def sparse_reward(action,valid,closer,terminal,fail,**kwargs):
    #reward specific to the case where the robots need to link two points

    terminal_reward = 1
    
    failing_cost = 1
    
    reward = 0
    if fail:
        return -failing_cost
    if terminal:
        reward += terminal_reward
    return reward
def n_side_reward(action,valid,closer,terminal,fail,**kwargs):
    assert False, 'not implemented'
def auto_scale_reward(action,valid,closer,terminal,fail,**kwargs):
    assert False, 'not implemented'
if __name__ == '__main__':
    print("Start test Agent")
    config_sparse_SAC = {'train_n_episodes':100000,
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
    agent = SACSupervisorSparse(2,
                        hexagone,
                        [hexagone],
                        config_sparse_SAC,
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