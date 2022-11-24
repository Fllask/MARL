# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 13:53:21 2022

@author: valla
"""
import numpy as np
import internal_models as im
import abc
from sklearn.neighbors import NearestNeighbors            
    
class Agent(metaclass=abc.ABCMeta):
    def __init__(self, rid,block_choices):
        super().__init__()
        self.rid = rid
        self.block_choices = block_choices
    def Act(self,simulator,action,
            bid=None,
            blocktype=None,
            ori=None,
            pos=None,
            blocktypeid = None,
            ):
        valid,closer = None,None
        if blocktypeid is not None:
            blocktype= self.block_choices[blocktypeid]
        if action in {'Ph','Pl'}:
            oldbid = simulator.leave(self.rid)
            if oldbid is not None:
                stable = simulator.check()
                if not stable:
                    #the robot cannot move from there
                    simulator.hold(self.rid,oldbid)
                    return False,None,blocktype
                
            valid,closer = simulator.put(blocktype,pos,ori)
            
            if valid:
                if action == 'Ph':
                    simulator.hold(self.rid,simulator.nbid-1)
                if action == 'Pl':
                    stable = simulator.check()
                    if not stable:
                        simulator.remove(simulator.nbid-1)
                        valid = False
            if not valid:
                simulator.hold(self.rid,oldbid)
                            
        elif action == 'H':
            oldbid = simulator.leave(self.rid)
            if oldbid is not None and oldbid != bid:
                stable = simulator.check()
                valid = stable
                if not stable:
                    #the robot cannot move from there
                    simulator.hold(self.rid,oldbid)
                else:
                    #chect if the block can be held
                    if not simulator.hold(self.rid,bid):
                        simulator.hold(self.rid,oldbid)
                        valid=False
            else:
                valid = simulator.hold(self.rid,bid)
        elif action == 'R':
            oldbid = simulator.leave(self.rid)
            if simulator.remove(bid):
                stable = simulator.check()
                valid = stable
                if not stable:
                    simulator.undo()
                    simulator.hold(self.rid,oldbid)
            else:
                simulator.hold(self.rid,oldbid)
                valid=False
        elif action == 'L':
            oldbid = simulator.leave(self.rid)
            if oldbid is not None:
                stable = simulator.check()
                valid = stable
                if not stable:
                    simulator.hold(self.rid,oldbid)
            else:
                valid = False
        else:
            assert False,'Unknown action'
        return valid,closer,blocktype
    @abc.abstractmethod
    def update_policy(self,**kwargs):
        pass
    @abc.abstractmethod
    def choose_action(self,state):
        pass
    

class NaiveRandom(Agent):
    def __init__(self,rid,placement_range,block_choices,action_choice = ['Pl','Ph','H','L','R'],**kwargs):
        super().__init__(rid,block_choices)
        self.pr = placement_range
        self.actions = action_choice
        self.policy = None
    def update_policy(self,reward,state,action):
        return True
    def choose_action(self,state):
        act = np.random.choice(self.actions)
        blocktype = np.random.choice(self.block_choices)
        posx = np.random.choice(np.nonzero(self.pr)[0])
        posy = np.random.choice(np.nonzero(self.pr[posx,:])[0])
        rot = np.random.randint(6)
        bid = np.random.randint(np.max(state.occ)+1)
        return act,{'blocktype': blocktype,'pos':[posx,posy],'ori':rot,'bid':bid}
class A2CLearner(Agent):
    def __init__(self,
                 rid,
                 placement_range,
                 block_choices,
                 action_choice = ['Pl','Ph','H','L','R'],
                 grid_size = [10,10],
                 max_blocks=30,
                 n_robots = 2,
                 n_regions = 2,
                 discount_f = 0.1,
                 device='cuda',
                 use_mask=True
                 ):
        super().__init__(rid,block_choices)
        self.pr = placement_range
        self.grid_size = grid_size
        self.n_typeblock = len(block_choices)
        self.max_blocks = max_blocks
        self.action_list = generate_actions(len(block_choices),grid_size,max_blocks)
        
        
        #parameters of the internal model:
        n_fc_layer=2
        n_neurons=100
        
        self.model = im.A2CShared(grid_size,
                                  max_blocks,
                                  n_robots,
                                  n_regions,
                                  len(self.action_list),
                                  n_fc_layer =n_fc_layer,
                                  n_neurons = n_neurons,
                                  device=device)
        
        self.optimizer = im.A2CSharedOptimizer(self.model)
        
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
            
            l_v,l_p = self.optimizer.optimize(states,actions,rewards,nstates,self.gamma,mask,nmask)
            
    def choose_action(self,state,explore=True,mask=None):
            
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
            action,action_params = vec2act(self.action_list[actionid[0],:],self.n_typeblock,self.grid_size,self.max_blocks)
            return action,action_params,actionid
        else:
            return None,None,actionid
class QTLearner(Agent):
    def __init__(self,
                 rid,
                 placement_range,
                 block_choices,
                 action_choice = ['Pl','Ph','H','L','R'],
                 grid_size = [10,10],
                 max_blocks=30,
                 n_robots = 2,
                 n_regions = 2,
                 discount_f = 0.1,
                 device='cuda'
                 ):
        super().__init__(rid,block_choices)
        self.pr = placement_range
        self.grid_size = grid_size
        self.n_typeblock = len(block_choices)
        self.max_blocks = max_blocks
        self.action_list = generate_actions(len(block_choices),grid_size,max_blocks)
        
        self.QT = im.DeepQT(grid_size,max_blocks,n_robots,n_regions,len(self.action_list),device=device)
        
        self.optimizer = im.DeepQTOptimizer(self.QT)
        
        
        
        self.gamma = 1-discount_f
        
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
            states = [trans.state for trans in batch]
            actions = np.concatenate([trans.a[0] for trans in batch],axis=0)
            rewards = np.array([trans.r for trans in batch])
            nstates = [trans.new_state for trans in batch]
            loss = self.optimizer.optimize(states,actions,rewards,nstates,self.gamma)

    def choose_action(self,state,explore=True):
        if not isinstance(state,list):
            state = [state]
        values = self.QT(state,inference = True,prob=True).cpu().detach().numpy()
        if explore:
            actionid = np.zeros(values.shape[0],dtype=int)
            for i,p in enumerate(values):
                actionid[i] = int(np.random.choice(len(self.action_list),p=p))
        else:
            actionid = np.argmax(values,axis=1)
        if len(state)==1:
            action,action_params = vec2act(self.action_list[actionid[0],:],self.n_typeblock,self.grid_size,self.max_blocks)
            return action,action_params,actionid
        else:
            return None,None,actionid
class WolpertingerLearner(Agent):
    def __init__(self,
                 rid,
                 placement_range,
                 block_choices,
                 action_choice = ['Pl','Ph','H','L','R'],
                 grid_size = [10,10],
                 max_blocks=30,
                 n_robots = 2,
                 n_regions = 2,
                 selected_actions = None,
                 discount_f = 0.1,
                 tau=0.1,
                 device='cpu'
                 ):
        super().__init__(rid,block_choices)
        self.pr = placement_range
        self.grid_size = grid_size
        self.n_typeblock = len(block_choices)
        self.max_blocks = max_blocks
        self.QT = im.WolpertingerQTable(grid_size,max_blocks,n_robots,n_regions,device=device)
        self.target_QT = im.WolpertingerQTable(grid_size,max_blocks,n_robots,n_regions,device=device)
        self.target_QT.load_state_dict(self.QT.state_dict())
        
        self.protopol = im.WolpertingerActionFinderNet(grid_size,max_blocks, n_robots, n_regions,device=device)
        self.target_protopol =  im.WolpertingerActionFinderNet(grid_size,max_blocks, n_robots, n_regions,device=device)
        self.target_protopol.load_state_dict(self.protopol.state_dict())
        
        self.optimizer = im.WolpertingerOpt(self.protopol,
                                            self.QT,
                                            self.target_protopol,
                                            self.target_QT)
        
        
        self.action_list = generate_actions(len(block_choices),grid_size,max_blocks)
        self.knn = NearestNeighbors().fit(self.action_list)   
        if selected_actions is None:
            self.selected_actions = int(0.05*self.action_list.shape[0])
        else:
            self.selected_actions = selected_actions
        self.gamma = 1-discount_f
        self.tau = tau
    def update_policy(self,buffer,buffer_count,batch_size,conv_tol = 1e-1,pol_steps=3):
        if buffer_count==0:
            return
        if buffer_count <buffer.shape[0]:
            batch_size = np.clip(batch_size,0,buffer_count)
            batch = np.random.choice(buffer[:buffer_count],batch_size,replace=False)
        else:
            batch = np.random.choice(np.delete(buffer,buffer_count%buffer.shape[0]),batch_size,replace=False)
        #compute the state value using the target Q table
        states = [trans.state for trans in batch]
        actions = np.concatenate([[trans.a[1]] for trans in batch],axis=0)
        rewards = np.array([[trans.r] for trans in batch])
        nstates = [trans.new_state for trans in batch]
        
        _,_,_,nactions = self.choose_action(nstates,target=True,explore=False)
       
        loss=1
        while loss > conv_tol:
        #for s in range(pol_steps):
            loss = self.optimizer.optimize_QT(states,
                                              actions,
                                              rewards,
                                              nstates,
                                              np.expand_dims(nactions,1),
                                              self.gamma)
        for s in range(pol_steps):
            self.optimizer.optimize_pol(states)
        
        self.optimizer.update_target(self.tau)
    def choose_action(self,state,target = False,grad=False,explore=True):
        if target:
            pol = self.target_protopol
            QT = self.target_QT
        else:
            pol = self.protopol
            QT = self.QT
        if not isinstance(state,list):
            state = [state]
        proto_action = pol(state,inference = not grad)
        
        if proto_action[0,0] != proto_action[0,0]:
            pass
        closest_actions = self.knn.kneighbors(proto_action.detach().cpu().numpy(),self.selected_actions,return_distance=False)
       
        values = QT(state,self.action_list[closest_actions,:],inference = True,explore=explore).cpu().detach().numpy()
        if explore:
            cactionid = np.zeros(values.shape[0],dtype=int)
            for i,p in enumerate(values):
                cactionid[i] = np.random.choice(closest_actions.shape[1],p=p)
        else:
            cactionid = np.argmax(values,axis=1)
        action = self.action_list[closest_actions[np.arange(closest_actions.shape[0]),cactionid].flatten()]
        if action.shape[0]==1:
            action_name,action_params = vec2act(action,self.n_typeblock,self.grid_size,self.max_blocks)
            return action_name,action_params,proto_action,action
        else:
            return None,None,proto_action,action

def vec2act(action_vec,nblocktypes,gridsize,max_block):
    action_vec = np.reshape(action_vec,(5,5))
    actionid = np.argmax(action_vec[:,0])
    if actionid == 0:
        action = 'Ph'
        action_params = {'blocktypeid': int((action_vec[actionid,1]/2+0.5)*(nblocktypes-1)),
                         'pos': [int((action_vec[actionid,2]/2+0.5)*(gridsize[0]-1)),
                                 int((action_vec[actionid,3]/2+0.5)*(gridsize[1]-1))],
                         'ori': int((action_vec[actionid,4]/2+0.5)*5),
                         }
    if actionid == 1:
        action = 'Pl'
        action_params = {'blocktypeid': int((action_vec[actionid,1]/2+0.5)*(nblocktypes-1)),
                         'pos': [int((action_vec[actionid,2]/2+0.5)*(gridsize[0]-1)),
                                 int((action_vec[actionid,3]/2+0.5)*(gridsize[1]-1))],
                         'ori': int((action_vec[actionid,4]/2+0.5)*5),
                         }
    if actionid == 2:
        action = 'H'
        action_params = {'bid':int((action_vec[actionid,1]/2+0.5)*max_block)}
    if actionid == 3:
        action = 'R'
        action_params = {'bid':int((action_vec[actionid,1]/2+0.5)*max_block)}
    if actionid == 4:
        action = 'L'
        action_params = {}
    return action,action_params
def generate_actions(ntype_blocks,grid_size,max_blocks):
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
    return np.reshape(actions,(-1,25))

def generate_mask(state,rid,block_type,grid_size,max_blocks):
    mask = np.zeros((grid_size[0]*grid_size[1]*6*len(block_type)*2+2*max_blocks+1),dtype=bool)
    #mask the hold and remove:
    ids = np.unique(state.occ)
    #remove the -1 and 0
    ids = ids[2:]
    
    #each feasible hold action of ids are placed at ids
    mask[ids]=True
    #each feasible remove action is placed at maxblock+ids
    mask[max_blocks+ids]=True
    
    
    #get the ids of the feasible put actions (note that the are not all hidden)
    for idb, block in enumerate(block_type):
        pos,ori = state.touch_side(block)
        ids = args2idx(pos,ori,grid_size)
        if not np.all(ids<grid_size[0]*grid_size[1]*6):
            args2idx(pos,ori,grid_size)
        #ph
        mask[idb*np.prod(grid_size)*6+max_blocks*2+ids]=True
        #pl
        mask[(len(block_type)+idb)*np.prod(grid_size)*6+max_blocks*2+ids]=True
    #leave
    mask[-1]=rid in state.hold
    return mask
def args2idx(pos,ori,grid_size):
    idx =  (pos[:,0]*grid_size[1]*6+pos[:,1]*6+ori).astype(int)
    return idx
def reward_link2(action,valid,closer,terminal):
    #reward specific to the case where the robots need to link two points

    #add a penalty for holding a block
    hold_penalty = 0.3
    #add a penalty if no block are put
    slow_penalty = 0.4
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
        reward-=slow_penalty
    if action in {'H','Ph'}:
        reward-=hold_penalty
    
    elif action in {'Ph', 'Pl'}:
        reward += closer_reward*closer-block_cost
    if terminal:
        reward += terminal_reward
    return reward

if __name__ == '__main__':
    print("Start test Agent")
    
    print("\nEnd test Agent")