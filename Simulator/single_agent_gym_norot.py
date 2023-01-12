# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 16:33:47 2022

@author: valla
"""

import copy
import time
import os
import wandb
import numpy as np
import pickle
from discrete_blocks_norot import discret_block_norot as Block
from geometric_internal_model import ReplayBufferSingleAgent
from relative_single_agent import SACSupervisorSparse,generous_reward,punitive_reward,modular_reward,A2CSupervisor
from discrete_simulator_norot import DiscreteSimulator as Sim,Transition as Trans
from pyg_single_agent import SACSupervisorDense

import discrete_graphics as gr
hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.5)
triangle = Block([[0,0,1]],muc=0.5)
link = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=0.5)
hextarget = Block([[1,0,1],[0,0,0],[2,0,0]])

class ReplayDiscreteGymSupervisor():
    def __init__(self,
                 config,
                 maxs = [10,10],
                 block_type = [hexagon,link],
                 random_targets = 'random',
                 targets = [triangle]*2,
                 targets_loc = [[3,0],[6,0]],
                 n_robots=2,
                 ranges = None,
                 agent_type = SACSupervisorSparse,
                 actions = ['Ph','Pl','L'],
                 max_blocks = 30,
                 max_interfaces = 100,
                 log_freq = 100,
                 reward_function = None,
                 use_wandb=False
            ):
        if use_wandb:
            self.use_wandb = True
            self.run = wandb.init(project="MARL", entity="flask",config=config)
            self.config = wandb.config
        else:
            self.use_wandb = False
            self.config = config
        if ranges is None:
            ranges = np.ones((n_robots,maxs[0],maxs[1],2),dtype = bool)
        self.log_freq = log_freq
        self.n_robots = n_robots
        
        self.gap_range = config.get('gap_range')
        if self.gap_range is None:
            self.gap_range = [1,self.sim.grid.shape[0]-2]
            
        if random_targets == 'random_gap':
            self.targets = [triangle]
            self.targets += [Block([[i,0,1] for i in range(maxs[0]-gap-1)],muc=0.5) for gap in range(self.gap_range[0],self.gap_range[1])]
        else:
            self.targets = targets
        
        self.sim = Sim(maxs,n_robots,block_type,len(targets_loc),max_blocks,max_interfaces,ground_blocks=self.targets)
        self.random_targets = random_targets
        
        
        if random_targets == 'fixed':
            for tar,loc in zip(targets,targets_loc):
                self.sim.add_ground(tar,loc)
        self.setup = copy.deepcopy(self.sim)
        
        self.agent = agent_type(n_robots,
                                block_type,
                                self.config,
                                ground_blocks = self.targets,
                                action_choice =actions,
                                grid_size=maxs,
                                use_wandb=use_wandb,
                                log_freq = self.log_freq,
                                env="norot")
        if self.agent.rep == 'graph':
            #create a dummy situation to initialize the graph
            
            self.sim.add_ground(triangle,[self.sim.grid.shape[0]-1,0])
            self.sim.add_ground(triangle,[1,0])
            self.sim.put_rel(hexagon, 0,0,0,0,idconsup = 0)
            self.sim.put_rel(hexagon, 0,0,1,0)
            self.sim.hold(0,2)
            self.agent.create_model(self.sim,config)
        if reward_function is None:
            if config['reward']=='punitive':
                self.rewardf = punitive_reward
            elif config['reward']=='generous':
                self.rewardf = generous_reward
            elif config['reward']=='modular':
                self.rewardf = modular_reward
        else:
            self.rewardf = reward_function
        
        
    def episode_restart(self,
                          max_steps,
                          draw=False,
                          buffer=None,
                          buffer_count=0,
                          auto_leave = True,
                          ):
        use_mask = self.config['ep_use_mask']
        batch_size = self.config['ep_batch_size']
        #if the action is not valid, stop the episode
        success = False
        failure = False
        rewards_ar = np.zeros((self.n_robots,max_steps))
        self.sim =copy.deepcopy(self.setup)
        gap=None
        if self.random_targets== 'random':
            validlocs = np.ones(self.sim.grid.shape,dtype=bool)
            #dont allow the target to be all the way to the extremity of the grid
            validlocs[:2,:]=False
            validlocs[-2:,:]=False
            validlocs[:,-2:]=False
            for tar in self.targets:
                valid = np.array(np.nonzero(validlocs)).T
                idx = np.random.randint(len(valid))
                self.sim.add_ground(tar,[valid[idx,0],valid[idx,1]])
                validlocs[max(valid[idx,0]-1,0):valid[idx,0]+2,max(valid[idx,1]-1,0):valid[idx,1]+2]=False
        if self.random_targets == 'random_flat':
            validlocs = np.ones(self.sim.grid.shape[0],dtype=bool)
            #dont allow the target to be all the way to the extremity of the grid
            validlocs[1]=False
            validlocs[-1]=False
            for tar in self.targets:
                valid = np.array(np.nonzero(validlocs)).flatten()
                idx = np.random.randint(len(valid))
                self.sim.add_ground(tar,[valid[idx],0])
                
                validlocs[max(valid[idx]-2,0):valid[idx]+3]=False
        if self.random_targets == 'random_gap':
            gap = np.random.randint(self.gap_range[0],self.gap_range[1])
            self.sim.add_ground(self.targets[gap-self.gap_range[0]+1],[0,0],ground_type=gap-self.gap_range[0]+1)
            self.sim.add_ground(self.targets[0],[self.sim.grid.shape[0]-1,0],ground_type=0)
            
        elif self.random_targets== 'half_fixed':
            assert False, "not implemented"
                    
        if draw:
            self.sim.setup_anim()
            self.sim.add_frame()
        if use_mask:
            mask = self.agent.generate_mask(self.sim,0)
        else:
            mask = None
        for step in range(max_steps):
            for idr in range(self.n_robots):
                
                if use_mask:
                    prev_state = {'grid':copy.deepcopy(self.sim.grid),
                                  'graph': copy.deepcopy(self.sim.graph),
                                  'mask':mask.copy(),
                                  'forces':copy.deepcopy(self.sim.ph_mod),
                                  }
                else:
                    prev_state = {'grid':copy.deepcopy(self.sim.grid),'graph': copy.deepcopy(self.sim.graph),'mask':None,'forces':copy.deepcopy(self.sim.ph_mod),'sim':copy.deepcopy(self.sim)}
                
                action,action_args,*action_enc = self.agent.choose_action(idr,self.sim,mask=mask)
                
                valid,closer,blocktype,interfaces = self.agent.Act(self.sim,action,**action_args,draw=draw)
                if use_mask:
                    mask = self.agent.generate_mask(self.sim,(idr+1)%self.n_robots)
                if valid:
                    if np.all(self.sim.grid.min_dist < 1e-5) and (auto_leave or np.all(self.sim.grid.hold==-1)):
                        if auto_leave:
                            bids = []
                            for r in range(self.n_robots):
                                bids.append(self.sim.leave(r))
                            if self.sim.check():
                                success = True
                                if use_mask:
                                    mask[:]=False
                            else:
                                for r,bid in enumerate(bids):
                                    self.sim.hold(r,bid)
                        else:
                            success = True
                            if use_mask:
                                mask[:]=False
                else:
                    failure = True
                    #mark the state as terminal
                    if use_mask:
                        mask[:]=False
                if step == max_steps-1 and idr == self.n_robots-1:
                    #the end of the episode is reached
                    if use_mask:
                        failure = True
                        mask[:]=False
                    
                    
                    
                if interfaces is not None:
                    sides_id,n_sides_ori = np.unique(interfaces[:,0],return_counts=True)
                    n_sides = np.zeros(6,dtype=int)
                    n_sides[sides_id.astype(int)]=n_sides_ori
                else:
                    n_sides = None
                
                reward =self.rewardf(action, valid, closer, success,failure,n_sides=n_sides,config=self.config)
                
                    
                rewards_ar[idr,step]=reward
                if self.agent.rep == 'graph':
                    buffer.push(idr,prev_state['sim'],action_enc[0],self.sim,reward,terminal=success or failure)
                else:
                    buffer[(buffer_count)%buffer.shape[0]] = Trans(prev_state,
                                                                    action_enc,
                                                                    reward,
                                                                   {'grid':copy.deepcopy(self.sim.grid),
                                                                    'graph': copy.deepcopy(self.sim.graph),
                                                                    'mask':mask,
                                                                    'forces':copy.deepcopy(self.sim.ph_mod)})
                    buffer_count +=1
                self.agent.update_policy(buffer,buffer_count,batch_size)
            
                if draw:
                    action_args.pop('rid')
                    
                    self.sim.draw_act(idr,action,blocktype,prev_state,**action_args)
                    self.sim.add_frame()
                            
                if success or failure:
                    break
            if success or failure:
                break
        if draw:
            anim = self.sim.animate()
        else:
            anim = None
        
        return rewards_ar,step, anim,buffer,buffer_count,success,gap
    
    def training(self,
                pfreq = 10,
                draw_freq=100,
                max_steps=100,
                save_freq = 1000,
                success_rate_decay = 0.01,
                log_dir=None):
        if self.random_targets == 'random_gap':
            success_rate = np.zeros(self.gap_range[1])
            success_rate[0]=1
            res_dict={}
        else:
            success_rate = 0
        if log_dir is None:
            if self.use_wandb:
                log_dir = self.run.dir
            else:
                log_dir = os.path.join('log','log'+str(np.random.randint(10000000)))
                os.mkdir(log_dir)
        if self.agent.rep == 'graph':
            buffer=ReplayBufferSingleAgent(self.config['train_l_buffer'],
                                           self.agent.action_choice,
                                           self.agent.n_side_oriented_sup,
                                           self.agent.n_side_oriented,
                                           fully_connected = False,device='cuda')
            buffer_count = 0
        else:
            buffer = np.empty(self.config['train_l_buffer'],dtype = object)
            buffer_count=0
        print("Training started")
        for episode in range(self.config['train_n_episodes']):
            (rewards_ep,n_steps_ep,
             anim,buffer,buffer_count, success,gap) = self.episode_restart(max_steps,
                                                              draw = episode % draw_freq == 0,#draw_freq-1,
                                                              buffer=buffer,
                                                              buffer_count=buffer_count,
                                                              )
            if self.random_targets == 'random_gap':
                if success:
                    success_rate[gap] = (1-success_rate_decay)*success_rate[gap] +success_rate_decay
                else:
                    success_rate[gap] = (1-success_rate_decay)*success_rate[gap]
            else:
                if success:
                    success_rate = (1-success_rate_decay)*success_rate +success_rate_decay
                else:
                    success_rate = (1-success_rate_decay)*success_rate
            
            if episode % pfreq==0:
                print(f'episode {episode}/{self.config["train_n_episodes"]} rewards: {np.sum(rewards_ep,axis=1)}')
            if episode % save_freq == 0 and self.agent.rep == 'grid':
                self.agent.save(f"ep_{episode}",log_dir)
            if self.use_wandb and episode % self.log_freq == 0:
                if self.random_targets == 'random_gap':
                    for i in np.arange(self.gap_range[0],self.gap_range[1]):
                        res_dict[f'success_rate_gap{i}']=success_rate[i]
                    wandb.log(res_dict)
                else:
                    wandb.log({'succes_rate':success_rate})
            if anim is not None:
                if self.use_wandb:
                    if success:
                        wandb.log({f'success_animation_gap_{gap}':wandb.Html(anim.to_jshtml())})
                        gr.save_anim(anim,os.path.join(log_dir, f"success_animation_gap_{i}_ep{episode}"),ext='gif')
                    else:
                        wandb.log({'animation':wandb.Html(anim.to_jshtml())})
                        
                else:
                    #gr.save_anim(anim,os.path.join(log_dir,'files','media', f"episode {episode}"),ext='gif')
                    gr.save_anim(anim,os.path.join(log_dir, f"episode {episode}"),ext='html')
                
        if self.use_wandb:
            self.run.finish()
        return anim
    def exploit(self,gap,alterations=None,max_steps=30,auto_leave=True,n_alter = 1,h=6,draw_robots=True):
        use_mask = self.config['ep_use_mask']
        rewards_ar = np.zeros((self.n_robots,max_steps))
        self.sim =copy.deepcopy(self.setup)
        gap=gap
                
        tar = Block([[i,0,1] for i in range(self.sim.grid.shape[0]-gap-1)],muc=0.7)
        self.sim.add_ground(tar,[0,0])
        self.sim.add_ground(triangle,[self.sim.grid.shape[0]-1,0])
        self.sim.setup_anim(h=h)
        self.sim.add_frame(draw_robots=draw_robots)
        
        if use_mask:
            mask = self.agent.generate_mask(self.sim,0)
        else:
            mask = None
        success = False
        failure = False
        for step in range(max_steps):
            for idr in range(self.n_robots):
                if step == 8:
                    pass
                prev_state = {'grid':copy.deepcopy(self.sim.grid),'graph': copy.deepcopy(self.sim.graph),'mask':mask.copy(),'forces':copy.deepcopy(self.sim.ph_mod)}
                action,action_args,*action_enc = self.agent.choose_action(idr,self.sim,mask=mask)
                if alterations is not None and step in alterations[:,0] and idr in alterations[:,1]:
                    for n_alter in range(n_alter):
                        mask[action_enc[0]]=False
                        action,action_args,*action_enc = self.agent.choose_action(idr,self.sim,mask=mask)
                    
                valid,closer,blocktype,interfaces = self.agent.Act(self.sim,action,**action_args,draw=True)
                if use_mask:
                    mask = self.agent.generate_mask(self.sim,(idr+1)%self.n_robots)
                if valid:
                    if np.all(self.sim.grid.min_dist < 1e-5) and (auto_leave or np.all(self.sim.grid.hold==-1)):
                        if auto_leave:
                            bids = []
                            for r in range(self.n_robots):
                                bids.append(self.sim.leave(r))
                            if self.sim.check():
                                success = True
                                if use_mask:
                                    mask[:]=False
                            else:
                                for r,bid in enumerate(bids):
                                    self.sim.hold(r,bid)
                        else:
                            success = True
                            if use_mask:
                                mask[:]=False
                else:
                    failure = True
                    #mark the state as terminal
                    mask[:]=False
                if step == max_steps-1 and idr == self.n_robots-1:
                    #the end of the episode is reached
                    mask[:]=False
                    
                    
                    
                if interfaces is not None:
                    sides_id,n_sides_ori = np.unique(interfaces[:,0],return_counts=True)
                    n_sides = np.zeros(6,dtype=int)
                    n_sides[sides_id.astype(int)]=n_sides_ori
                else:
                    n_sides = None
                
                reward =self.rewardf(action, valid, closer, success,failure,n_sides=n_sides,config=self.config)
                rewards_ar[idr,step]=reward
                
                action_args.pop('rid')
                    
                self.sim.draw_act(idr,action,blocktype,prev_state,draw_robots=draw_robots,**action_args)
                self.sim.add_frame(draw_robots=draw_robots)
                            
                if success or failure:
                    break
            if success or failure:
                break
        
        anim = self.sim.animate()
        
        return rewards_ar, anim
    def test(self,
             draw=True):
        from relative_single_agent import int2act_norot
        self.agent.Act(self.sim,'Ph',rid=0,
                        sideblock=0,
                        sidesup = 1,
                        bid_sup = 0,
                        idconsup = 1,
                        blocktypeid = 0,
                        side_ori = 0,
                        draw= False)
        self.agent.Act(self.sim,'Ph',rid=1,
                        sideblock=0,
                        sidesup = 0,
                        bid_sup = 1,
                        idconsup = 1,
                        blocktypeid = 1,
                        side_ori = 4,
                        draw= False)
        setup = copy.deepcopy(self.sim)
        mask = self.agent.generate_mask(self.sim,0)
        #self.sim.setup_anim()
        #self.sim.add_frame()
        while mask.any():
            
            actionids,= np.nonzero(mask)
            action,action_params = int2act_norot(actionids[0],self.sim.graph.n_blocks,
                                                 self.n_robots,
                                                 self.agent.n_side_oriented,
                                                 self.agent.n_side_oriented_sup,
                                                 self.agent.last_only,
                                                 self.agent.max_blocks,
                                                 self.agent.action_choice)
            self.agent.Act(self.sim,action,**action_params,draw=True)
            self.sim.draw_state_debug()
            mask[actionids[0]]=False
            self.sim =copy.deepcopy(setup)
            #self.sim.setup_anim()
            #self.sim.add_frame()
        #anim = self.sim.animate()
        return None


    def test_gap(self,gap=None):
        if gap is None:
            if self.gap_range is not None:
                gap = np.random.randint(self.gap_range[0],self.gap_range[1])
            else:
                gap = np.random.randint(1,self.sim.grid.shape[0]-3)
        tar = Block([[i,0,1] for i in range(self.sim.grid.shape[0]-gap-1)],muc=0.7)
        self.sim.add_ground(tar,[0,0])
        self.sim.add_ground(triangle,[self.sim.grid.shape[0]-1,0])
        
        from relative_single_agent import int2act_norot
        self.agent.Act(self.sim,'Ph',rid=0,
                        sideblock=0,
                        sidesup = 0,
                        bid_sup = 0,
                        idconsup = 1,
                        blocktypeid = 0,
                        side_ori = 0,
                        draw= False)
        self.agent.Act(self.sim,'Ph',rid=1,
                        sideblock=0,
                        sidesup = 0,
                        bid_sup = 1,
                        idconsup = 0,
                        blocktypeid =0,
                        side_ori = 5,
                        draw= False)
        self.agent.Act(self.sim,'Ph',rid=0,
                        sideblock=0,
                        sidesup = 0,
                        bid_sup = 2,
                        idconsup = 0,
                        blocktypeid =0,
                        side_ori = 1,
                        draw= False)
        self.agent.Act(self.sim,'L',rid=0,
                       )
        self.agent.Act(self.sim,'L',rid=1,
                        )
        setup = copy.deepcopy(self.sim)
        self.sim.draw_state_debug()
        
if __name__ == '__main__':
    print("Start test gym")
    # config={'train_n_episodes':60000,
    #         'train_l_buffer':4000,
    #         'ep_batch_size':256,
    #         'ep_use_mask':True,
    #         'agent_discount_f':0.1,
    #         'agent_last_only':True,
    #         'torch_device':'cuda',
    #         'A2C_n_fc_layer':4,
    #         'A2C_n_neurons':500,
    #         'A2C_shared':False,
    #         'A2C_batch_norm':True,
    #         'SEnc_n_channels':256,
    #         'SEnc_n_internal_layer':4,
    #         'SEnc_stride':1,
    #         'opt_lr':1e-4,
    #         'opt_pol_over_val': 1,
    #         'opt_tau': 1e-5,
    #         'opt_weight_decay':1e-3,
    #         'opt_exploration_factor':0.1}
    config = {'train_n_episodes':10000,
            'train_l_buffer':200,
            'ep_batch_size':32,
            'ep_use_mask':True,
            'agent_discount_f':0.1,
            'agent_last_only':True,
            'reward': 'modular',
            'torch_device':'cuda',
            'SEnc_n_channels':32,
            'SEnc_n_internal_layer':4,
            'SEnc_stride':1,
            'SEnc_order_insensitive':True,
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
            'opt_target_entropy':0.2,
            'opt_value_clip':False,
            'opt_entropy_penalty':False,
            'opt_Q_reduction': 'min',
            'V_optimistic':False,
            'reward_failure':-1,
            'reward_action':{'Ph': -0.2, 'L':-0.1},
            'reward_closer':0.4,
            'reward_nsides': 0.1,
            'reward_success':1,
            'reward_opposite_sides':0,
            'opt_lower_bound_Vt':-2,
            'gap_range':[0,1]
            }
    config_A2C = {'train_n_episodes':10000,
            'train_l_buffer':200,
            'ep_batch_size':32,
            'ep_use_mask':True,
            'agent_discount_f':0.1,
            'agent_last_only':True,
            'reward': 'modular',
            'torch_device':'cuda',
            'SEnc_n_channels':32,
            'SEnc_n_internal_layer':4,
            'SEnc_stride':1,
            'SEnc_order_insensitive':True,
            'A2C_n_fc_layer':2,
            'A2C_n_neurons':64,
            'A2C_batch_norm':True,
            'A2C_shared':False,
            'opt_lr':1e-4,
            'opt_pol_over_val': 1,
            'opt_tau': 1e-3,
            'opt_weight_decay':0.0001,
            'opt_exploration_factor':0,
            'agent_exp_strat':'epsilon-greedy',
            'agent_epsilon':0.05,
            'V_optimistic':False,
            'reward_failure':-2,
            'reward_action':{'Ph': -0.2, 'L':-0.1},
            'reward_closer':0.4,
            'reward_nsides': 0.05,
            'reward_success':5,
            'reward_opposite_sides':0,
            #'gap_range':[1,3]
            }
    config_dense = {'train_n_episodes':1000,
            'train_l_buffer':200,
            'ep_batch_size':32,
            'ep_use_mask':False,
            'agent_discount_f':0.1,
            'agent_last_only':True,
            'reward': 'modular',
            'torch_device':'cuda',
            'GNN_arch':'GAT',
            'GNN_n_layers':3,
            'GNN_hidden_dim':32,
            'GNN_att_head':1,
            'opt_lr':1e-4,
            'opt_pol_over_val': 1,
            'opt_tau': 1e-3,
            'opt_weight_decay':0.0001,
            'opt_exploration_factor':0.001,
            'agent_exp_strat':'softmax',
            'agent_epsilon':0.05,
            'opt_max_norm': 2,
            'opt_target_entropy':0.2,
            'opt_value_clip':False,
            'opt_entropy_penalty':False,
            'opt_Q_reduction': 'min',
            'V_optimistic':False,
            'reward_failure':-1,
            'reward_action':{'Ph': -0.2, 'L':-0.1},
            'reward_closer':0.4,
            'reward_nsides': 0.1,
            'reward_success':5,
            'reward_opposite_sides':0,
            'opt_lower_bound_Vt':-2,
            'gap_range':[1,20],
            
            }
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.5)
    linkr = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=0.5) 
    linkl = Block([[0,0,0],[0,1,1],[1,0,0],[0,1,0],[0,0,1],[-1,1,1]],muc=0.5) 
    linkh = Block([[0,0,0],[0,1,1],[1,0,0],[-1,2,1],[0,1,0],[0,2,1]],muc=0.5)
    #target = Block([[0,0,1],[1,0,1]])
    target = Block([[0,0,1]])
    gym = ReplayDiscreteGymSupervisor(config_dense,
              agent_type=SACSupervisorDense,
              use_wandb=False,
              actions= ['Ph'],
              block_type=[hexagon,link],
              random_targets='random_gap',
              targets_loc=[[2,0],[6,0]],
              n_robots=2,
              max_blocks = 20,
              targets=[target]*2,
              max_interfaces = 50,
              log_freq = 5,
              maxs = [30,15])
    t0 = time.perf_counter()
    anim = gym.training(max_steps = 20, draw_freq = 200,pfreq =10)
    #gym.test_gap()
    #gr.save_anim(anim,os.path.join(".", f"test_graph"),ext='html')
    t1 = time.perf_counter()
    print(f"time spent: {t1-t0}s")
    print("\nEnd test gym")