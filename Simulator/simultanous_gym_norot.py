# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:37:53 2022

@author: valla
"""

import copy
import time
import os
import wandb
import numpy as np
import pickle
from discrete_blocks_norot import discret_block_norot as Block
from internal_models import ReplayBufferGraph
from simultaneous_multiagent import reward_simultaneous1,modular_reward,SACSparse
from discrete_simulator_norot import DiscreteSimulator as Sim,Transition as Trans
import discrete_graphics as gr
hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.7)
triangle = Block([[0,0,1]],muc=0.7)
link = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=0.7)
hextarget = Block([[1,0,1],[0,0,0],[2,0,0]])

class ReplayDiscreteGym():
    def __init__(self,
                 config,
                 maxs = [10,10],
                 block_type = [hexagon,link],
                 random_targets = 'random',
                 targets = [triangle]*2,
                 targets_loc = [[3,0],[6,0]],
                 n_robots=2,
                 ranges = None,
                 agent_type = SACSparse,
                 actions = ['P','L','S'],
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
        self.sim = Sim(maxs,n_robots,block_type,len(targets_loc),max_blocks,max_interfaces)
        self.random_targets = random_targets
        self.targets = targets
        self.gap_range = config.get('gap_range')
        if self.gap_range is None:
            self.gap_range = [1,self.sim.grid.shape[0]-2]
        if random_targets == 'fixed':
            for tar,loc in zip(targets,targets_loc):
                self.sim.add_ground(tar,loc)
        self.agents = [agent_type(n_robots,
                                  rid,
                                  block_type,
                                  self.config,
                                  ground_blocks = targets,
                                  action_choice =actions,
                                  grid_size=maxs,
                                  use_wandb=use_wandb,
                                  log_freq = self.log_freq,
                                  env="norot") for rid in range(n_robots)]
        
        self.rewardf = reward_function
        self.setup = copy.deepcopy(self.sim)
        if reward_function is None:
            if config['reward']=='modular':
                self.rewardf = modular_reward
    def episode_restart(self,
                          max_steps,
                          draw=False,
                          buffer=None,
                          buffer_count=0,
                          ):
        batch_size = self.config['ep_batch_size']
        #if the action is not valid, stop the episode
        success = False
        failure = False
        rewards_ar = np.zeros((self.n_robots,max_steps))
        
        self.sim =copy.deepcopy(self.setup)
        
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
                
            #tar = Block([[i,0,1] for i in range(self.sim.grid.shape[0]-gap-1)],muc=0.7)
            [tar.move([0,0]) for tar in self.targets]
            width = [np.max(tar.parts[:,0]) for tar in self.targets]
            self.sim.add_ground(self.targets[0],[self.sim.grid.shape[0]-2-gap-width[0]-width[1],0])
            self.sim.add_ground(self.targets[1],[self.sim.grid.shape[0]-1-width[1],0])
            
        elif self.random_targets== 'half_fixed':
            assert False, "not implemented"
                    
        
        
        prev_state=[{}]*self.n_robots
        closer = np.zeros(self.n_robots)
        action_enc = np.zeros(self.n_robots,dtype=object)
        actions = np.zeros(self.n_robots,dtype=object)
        valids = np.zeros(self.n_robots,dtype=bool)
        interfaces = np.zeros((self.n_robots),dtype=object)
        action_args = np.zeros(self.n_robots,dtype=object)
        if draw:
            self.sim.setup_anim()
            self.sim.add_frame()
        
        masks= [self.agents[idr].generate_mask(self.sim) for idr in range(self.n_robots)]
        
        for step in range(max_steps):
            for idr in range(self.n_robots):
                prev_state[idr] = {'grid':copy.deepcopy(self.sim.grid),'graph': copy.deepcopy(self.sim.graph),'mask':masks[idr].copy()}
                action,action_args[idr],action_enc[idr] = self.agents[idr].choose_action(idr,self.sim,mask=masks[idr])
                actions[idr]=action
                self.agents[idr].prepare_action(self.sim,action)
            valid_prep = self.sim.check()
            if not valid_prep:
                #penalize all robot that chose something else than the stay action
                valids = np.array([action=='S' for action in actions])
                if draw:
                    self.sim.add_frame()
                    for idr in range(self.n_robots):
                        _,_,blocktype,_ = self.agents[idr].act(self.sim,actions[idr],**action_args[idr],draw=draw)
                        self.sim.draw_act(idr,actions[idr],blocktype,prev_state,redraw_state = False,**action_args[idr])
                
            
            if valid_prep:
                #setup the placement step drawing
                if draw:
                    self.sim.add_frame()
                for idr in range(self.n_robots):
                    valids[idr],closer[idr],blocktype,interfaces[idr] = self.agents[idr].act(self.sim,actions[idr],**action_args[idr],draw=draw)
                    if draw:
                        self.sim.draw_act(idr,actions[idr],blocktype,prev_state,redraw_state = False,**action_args[idr])
                 
                #if an action can interfere with the physical equilibrium, check the physics there
                #self.sim.check()
                if np.all(valids[idr]):
                    if np.all(self.sim.grid.min_dist < 1e-5) and np.all(self.sim.grid.hold==-1):
                        success = True
                        for idr in range(self.n_robots):
                            masks[idr][:]=False
                    else:
                        masks = [self.agents[idr].generate_mask(self.sim) for idr in range(self.n_robots)]
                    
                if draw:
                    self.sim.add_frame()
                    
            if not np.all(valids) or step == max_steps-1:
                failure = True
                #mark the state as terminal
                for idr in range(self.n_robots): 
                    masks[idr][:]=False
            #compute the common reward and the optmization step
            reward = 0
            for idr in range(self.n_robots):
                if interfaces[idr] is not None:
                    sides_id,n_sides_ori = np.unique(interfaces[idr][:,0],return_counts=True)
                    n_sides = np.zeros(6,dtype=int)
                    n_sides[sides_id.astype(int)]=n_sides_ori
                else:
                    n_sides = None
                
                reward_individual =self.rewardf(actions[idr], valids[idr], closer[idr], success,failure,n_sides=n_sides,config=self.config)
                
                #reward_individual =self.rewardf(actions[idr], valids[idr], closer[idr], success,failure)
                
                    
                rewards_ar[idr,step]=reward_individual
                reward += reward_individual/self.n_robots
            for idr in range(self.n_robots):
                if self.agents[idr].rep == 'graph':
                    buffer.push(prev_state['graph'],prev_state['mask'],action_enc[idr][0],self.sim.graph,masks[idr],reward,entropy = action_enc[idr][1])
                else:
                    if self.config['ep_common_reward']:
                        buffer[idr,(buffer_count)%buffer.shape[1]] = Trans(prev_state[idr],
                                                                        action_enc[idr],
                                                                        reward,
                                                                       {'grid':copy.deepcopy(self.sim.grid),
                                                                        'graph': copy.deepcopy(self.sim.graph),
                                                                        'mask':masks[idr]})
                    
                
                self.agents[idr].update_policy(buffer[idr],buffer_count+1,batch_size)
            buffer_count +=1
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
                log_dir=None):
        
        if self.random_targets == 'random_gap':
            success_rate = np.zeros(self.gap_range[1])
            success_rate[0]=1
            res_dict={}
        else:
            success_rate = 0
            
        if log_dir is None:
            log_dir = os.path.join('log','log'+str(np.random.randint(10000000)))
            os.mkdir(log_dir)
        if any([agent.rep == 'graph' for agent in self.agents]):
            buffer=ReplayBufferGraph(self.config["train_l_buffer"],
                                     len(self.targets),
                                     self.sim.graph.mblock,
                                     self.sim.graph.minter,
                                     n_actions=self.agent.n_actions,
                                     device=self.config['torch_device'])
            buffer_count = 0
        if any([agent.rep == 'grid' for agent in self.agents]):
            buffer = np.empty((self.n_robots,self.config['train_l_buffer']),dtype = object)
            buffer_count=0
        print("Training started")
        for episode in range(self.config['train_n_episodes']):
            (rewards_ep,n_steps_ep,
             anim,buffer,buffer_count,success,gap) = self.episode_restart(max_steps,
                                                              draw = episode % draw_freq == 0,#draw_freq-1,
                                                              buffer=buffer,
                                                              buffer_count=buffer_count,
                                                              )
            if episode % pfreq==0:
                print(f'episode {episode}/{self.config["train_n_episodes"]} rewards: {np.sum(rewards_ep,axis=1)}')
            if episode % save_freq == 0:
                file = open(os.path.join(log_dir,f'res{episode}.pickle'), 'wb')
                pickle.dump({"rewards":rewards_ep,"episode":episode,"n_steps":n_steps_ep},file)
                file.close()
            if anim is not None:
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
    
    def test(self,
             draw=True):
        anim = None
        return anim

if __name__ == '__main__':
    print("Start test gym")
    config = {'train_n_episodes':20000,
            'train_l_buffer':2000,
            'ep_batch_size':64,
            'agent_discount_f':0.1,
            'agent_last_only':True,
            'torch_device':'cuda',
            'SEnc_order_insensitive':False,
            'SEnc_n_channels':32,
            'SEnc_n_internal_layer':4,
            'SEnc_stride':1,
            'SAC_n_fc_layer':2,
            'SAC_n_neurons':64,
            'SAC_batch_norm':False,
            'Q_duel':True,
            'opt_lr':1e-4,
            'opt_pol_over_val': 1,
            'opt_tau': 1e-3,
            'opt_weight_decay':0.0001,
            'opt_exploration_factor':0.001,
            'agent_exp_strat':'softmax',
            'agent_epsilon':0.05,
            'opt_max_norm': 2,
            'opt_target_entropy':1.,
            'opt_value_clip':False,
            'opt_entropy_penalty':False,
            'opt_Q_reduction': 'min',
            'V_optimistic':False,
            'ep_common_reward':True,
            'opt_lower_bound_Vt':-2,
            'gap_range':[5,6],
            'reward':'modular',
            'reward_failure':-1,
            'reward_action':{'P': 0.2, 'L':-0.1, 'S':-0.1},
            'reward_closer':0.4,
            'reward_nsides': 0.05,
            'reward_success':5,
            'reward_opposite_sides':0,
            }
    
    gym = ReplayDiscreteGym(config,maxs=[10,10],use_wandb=True,agent_type = SACSparse,random_targets='random_gap',block_type=[hexagon],
                            targets=[Block([[i,0,1]for i in range(1)])]*2)
    t0 = time.perf_counter()
    anim = gym.training(max_steps = 6, draw_freq = 100,pfreq =10)
    #anim = gym.test()
    #gr.save_anim(anim,os.path.join(".", f"test_graph"),ext='html')
    t1 = time.perf_counter()
    print(f"time spent: {t1-t0}s")
    print("\nEnd test gym")