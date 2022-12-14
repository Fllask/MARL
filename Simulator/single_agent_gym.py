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
from discrete_blocks import discret_block as Block
from internal_models import ReplayBufferGraph
from relative_single_agent import reward_link2,A2CSupervisor,A2CSupervisorDense,SACSupervisorDense
#from single_agent import reward_link2,A2CSupervisor,A2CSupervisorStruc,generate_mask_supervisor,vec2act_sup
from discrete_simulator import DiscreteSimulator as Sim,Transition as Trans
import discrete_graphics as gr
hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.7)
triangle = Block([[0,0,1]],muc=0.7)
link = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=0.7)
hextarget = Block([[1,0,1],[0,0,0],[2,0,0]])

class ReplayDiscreteGymSupervisor():
    def __init__(self,
                 config,
                 maxs = [10,10],
                 block_type = [hexagon,link],
                 random_targets = 'random',
                 targets = [triangle]*2,
                 targets_loc = [[3,0],[6,0]],
                 targets_rot = [0,0],
                 n_robots=2,
                 ranges = None,
                 agent_type = A2CSupervisorDense,
                 actions = ['Ph','Pl','L'],
                 max_blocks = 30,
                 max_interfaces = 100,
                 log_freq = 100,
                 reward_function = reward_link2,
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
        if random_targets == 'fixed':
            for tar,loc,rot in zip(targets,targets_loc,targets_rot):
                self.sim.add_ground(tar,loc,rot)
        self.agent = agent_type(n_robots,block_type,self.config,action_choice =actions,grid_size=maxs,use_wandb=use_wandb,log_freq = self.log_freq)
        self.rewardf = reward_function
        self.setup = copy.deepcopy(self.sim)
        
    def episode_restart(self,
                          max_steps,
                          draw=False,
                          buffer=None,
                          buffer_count=0,
                          ):
        use_mask = self.config['ep_use_mask']
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
                self.sim.add_ground(tar,[valid[idx,0],valid[idx,1]],0)
                validlocs[max(valid[idx,0]-1,0):valid[idx,0]+2,max(valid[idx,1]-1,0):valid[idx,1]+2]=False
        elif self.random_targets== 'half_fixed':
            assert False, "not implemented"
                    
        if draw:
            self.sim.setup_anim()
            self.sim.add_frame()
        if use_mask:
            mask = self.agent.generate_mask(self.sim.grid,0)
        for step in range(max_steps):
            for idr in range(self.n_robots):
                if use_mask:
                    mask = self.agent.generate_mask(self.sim.grid,idr)
                    prev_state = {'grid':copy.deepcopy(self.sim.grid),'graph': copy.deepcopy(self.sim.graph),'mask':mask.copy()}
                else:
                    prev_state = {'grid':copy.deepcopy(self.sim.grid),'graph': copy.deepcopy(self.sim.graph),'mask':None}
                    mask = None
                
                action,action_args,*action_enc = self.agent.choose_action(idr,self.sim,mask=mask)
                
                valid,closer,blocktype = self.agent.Act(self.sim,action,**action_args,draw=draw)
                if use_mask:
                    mask = self.agent.generate_mask(self.sim.grid,(idr+1)%self.n_robots)
                if valid:
                    if np.all(self.sim.grid.min_dist < 1e-5) and np.all(self.sim.grid.hold==-1):
                        success = True
                        mask[:]=False
                else:
                    failure = True
                    #mark the state as terminal
                    mask[:]=False
                if step == max_steps-1 and idr == self.n_robots-1:
                    #the end of the episode is reached
                    mask[:]=False
                reward =self.rewardf(action, valid, closer, success,failure)
                
                    
                rewards_ar[idr,step]=reward
                if self.agent.rep == 'graph':
                    buffer.push(prev_state['graph'],prev_state['mask'],action_enc[0],self.sim.graph,mask,reward,entropy = action_enc[1])
                else:
                    buffer[(buffer_count)%buffer.shape[0]] = Trans(prev_state,
                                                                    action_enc,
                                                                    reward,
                                                                   {'grid':copy.deepcopy(self.sim.grid),
                                                                    'graph': copy.deepcopy(self.sim.graph),
                                                                    'mask':mask})
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
        
        return rewards_ar,step, anim,buffer,buffer_count
    def episode_multistep(self,
                          max_steps = 20,
                          draw=False,
                          buffer=np.zeros(0),
                          buffer_count=0,
                          batch_size=32,
                          use_mask=True):
        success = False
        failure = False
        rewards_ar = np.zeros((self.n_robots,max_steps))
        self.sim =copy.deepcopy(self.setup)
        if draw:
            self.sim.setup_anim()
            self.sim.add_frame()
        for step in range(max_steps):
            for idr in range(self.n_robots):
                valid = False
                if use_mask:
                    mask = self.agent.generate_mask(self.sim.grid,
                                                    idr)
                    prev_state = [copy.deepcopy(self.sim.grid),copy.deepcopy(self.sim.graph),mask.copy()]
                else:
                    prev_state = [copy.deepcopy(self.sim.grid),copy.deepcopy(self.sim.graph)]
                    mask = None
                n_tries = 0
                while not valid:
                    if np.sum(mask)==0:
                        #no actions are possible: end the episode
                        failure = True
                        valid=True
                    else:
                        n_tries +=1
                        action,action_args,*action_enc = self.agent.choose_action(idr,self.sim.grid,mask=mask)
                        valid,closer,blocktype = self.agent.Act(self.sim,action,**action_args)
                        
                    if valid:
                        if np.all(self.sim.grid.min_dist < 1e-5) and np.all(self.sim.grid.hold==-1):
                            success = True
                            mask['terminal']=True
                    else:
                        #remove the action from the mask so it cannot be chosen anymore
                        if use_mask:
                            mask['out_action'][action_enc[0]]=False
                    reward =self.rewardf(action, valid, closer, success,failure)
                    
                    rewards_ar[idr,step]=reward/n_tries+rewards_ar[idr,step]*(n_tries-1)/n_tries
                    buffer[(buffer_count)%buffer.shape[0]] = Trans(prev_state,
                                                                   action_enc,
                                                                   reward,
                                                                   [copy.deepcopy(self.sim.grid),mask])
                    buffer_count +=1
                    self.agent.update_policy(buffer,buffer_count,batch_size)
                
                if draw:
                    action_args.pop('rid')
                    self.sim.draw_act(idr,action,blocktype,**action_args)
                    self.sim.add_frame()
                        
                if success or failure:
                    break
            if success or failure:
                break
        if draw:
            anim = self.sim.animate()
        else:
            anim = None
        
        return rewards_ar,step, anim,buffer,buffer_count
    
    def training(self,
                pfreq = 10,
                draw_freq=100,
                max_steps=100,
                save_freq = 1000,
                log_dir=None):
        
        
        if log_dir is None:
            log_dir = os.path.join('log','log'+str(np.random.randint(10000000)))
            os.mkdir(log_dir)
        if self.agent.rep == 'graph':
            buffer=ReplayBufferGraph(self.config["train_l_buffer"],
                                     len(self.targets),
                                     self.sim.graph.mblock,
                                     self.sim.graph.minter,
                                     n_actions=self.agent.n_actions,
                                     device=self.config['torch_device'])
            buffer_count = 0
        else:
            buffer = np.empty(self.config['train_l_buffer'],dtype = object)
            buffer_count=0
        print("Training started")
        for episode in range(self.config['train_n_episodes']):
            (rewards_ep,n_steps_ep,
             anim,buffer,buffer_count) = self.episode_restart(max_steps,
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
                if self.use_wandb:
                    wandb.log({'animation':wandb.Html(anim.to_jshtml())})
                else:
                    gr.save_anim(anim,os.path.join(log_dir, f"episode {episode}"),ext='gif')
                    gr.save_anim(anim,os.path.join(log_dir, f"episode {episode}"),ext='html')
                
        if self.use_wandb:
            self.run.finish()
        return anim
    
    def test(self,
             draw=True):
        config = {'train_n_episodes':1,
                'train_l_buffer':3,
                'ep_batch_size':2,
                'ep_use_mask':True,
                'agent_discount_f':0.1,
                'agent_last_only':True,
                'torch_device':'cuda',
                'A2C_nblocks':3,
                'graph_inter_Va':10,
                'graph_inter_Ea':10,
                'graph_inter_u':10,
                'graph_neurons_chanel':10,
                'graph_n_hidden_layers':2,
                'graph_pol_enc':'one_edge',
                'opt_lr':1e-4,
                'opt_pol_over_val': 1,
                'opt_tau': 1e-5,
                'opt_weight_decay':1e-6,
                'opt_exploration_factor':0.1}
        
        buffer=ReplayBufferGraph(self.config["train_l_buffer"],
                                 len(self.targets),
                                 self.sim.graph.mblock,
                                 self.sim.graph.minter,
                                 n_actions=self.agent.n_actions,
                                 device=self.config['torch_device'])
        buffer_count = 0
        
        
        #put 2 grounds:
            
        self.sim.add_ground(self.targets[0],[1,0],0)
        self.sim.add_ground(self.targets[1],[4,0],0)
            
                    
        
        self.sim.setup_anim()
        self.sim.add_frame()
        
        idr = 0 
        step = 0
        success = False
        failure = False
        mask = self.agent.generate_mask(self.sim.grid,idr)
        prev_state = {'grid':copy.deepcopy(self.sim.grid),'graph': copy.deepcopy(self.sim.graph),'mask':mask.copy()}
        
        
        
        action,action_args,*action_enc = self.agent.choose_action(idr,self.sim,mask=mask)
        action = 'Pl'
        action_args = {'rid':0,
                         'blocktypeid':0,
                         'sideblock':0,
                         'sidesup':0,
                         'bid_sup':self.sim.graph.n_blocks,
                         'idconsup':0,
                          }
        valid,closer,blocktype = self.agent.Act(self.sim,action,**action_args,draw=draw)
                    
        if valid:
            if np.all(self.sim.grid.min_dist < 1e-5) and np.all(self.sim.grid.hold==-1):
                success = True
                mask[:]=False
        else:
            failure = True
            #mark the state as terminal
            mask[:]=False
        reward =self.rewardf(action, valid, closer, success,failure)
        
        print(reward)
        
        buffer.push(prev_state['graph'],prev_state['mask'],action_enc[0],self.sim.graph,mask,reward)
        print("pol update")
        self.agent.update_policy(buffer,buffer_count,2)
        print("pol updated")
        if draw:
            action_args.pop('rid')
            
            self.sim.draw_act(idr,action,blocktype,prev_state,**action_args)
            self.sim.add_frame()
                    
        idr = 1
        mask = self.agent.generate_mask(self.sim.grid,idr)
        prev_state = {'grid':copy.deepcopy(self.sim.grid),'graph': copy.deepcopy(self.sim.graph),'mask':mask.copy()}
        
        
        
        action,action_args,*action_enc = self.agent.choose_action(idr,self.sim,mask=mask)
        action = 'Ph'
        action_args = {'rid':1,
                         'blocktypeid':0,
                         'sideblock':0,
                         'sidesup':2,
                         'bid_sup':self.sim.graph.n_blocks,
                         'idconsup':0,
                          }
        valid,closer,blocktype = self.agent.Act(self.sim,action,**action_args,draw=draw)
                    
        if valid:
            if np.all(self.sim.grid.min_dist < 1e-5) and np.all(self.sim.grid.hold==-1):
                success = True
                mask[:]=False
        else:
            failure = True
            #mark the state as terminal
            mask[:]=False
        reward =self.rewardf(action, valid, closer, success,failure)
        
        print(reward)
        
        buffer.push(prev_state['graph'],prev_state['mask'],action_enc[0],self.sim.graph,mask,reward)
        print("pol update")
        self.agent.update_policy(buffer,buffer_count,2)
        print("pol updated")
        if draw:
            action_args.pop('rid')
            
            self.sim.draw_act(idr,action,blocktype,prev_state,**action_args)
            self.sim.add_frame()
        idr = 0
        mask = self.agent.generate_mask(self.sim.grid,idr)
        prev_state = {'grid':copy.deepcopy(self.sim.grid),'graph': copy.deepcopy(self.sim.graph),'mask':mask.copy()}
        
        
        
        action,action_args,*action_enc = self.agent.choose_action(idr,self.sim,mask=mask)
        action = 'Pl'
        action_args = {'rid':0,
                         'blocktypeid':0,
                         'sideblock':0,
                         'sidesup':2,
                         'bid_sup':self.sim.graph.n_blocks,
                         'idconsup':0,
                          }
        valid,closer,blocktype = self.agent.Act(self.sim,action,**action_args,draw=draw)
                    
        if valid:
            if np.all(self.sim.grid.min_dist < 1e-5) and np.all(self.sim.grid.hold==-1):
                success = True
                mask[:]=False
        else:
            failure = True
            #mark the state as terminal
            mask[:]=False
        reward =self.rewardf(action, valid, closer, success,failure)
        
        print(reward)
        
        buffer.push(prev_state['graph'],prev_state['mask'],action_enc[0],self.sim.graph,mask,reward)
        print("pol update")
        self.agent.update_policy(buffer,buffer_count,2)
        print("pol updated")
        if draw:
            action_args.pop('rid')
            
            self.sim.draw_act(idr,action,blocktype,prev_state,**action_args)
            self.sim.add_frame()
        idr = 1
        mask = self.agent.generate_mask(self.sim.grid,idr)
        prev_state = {'grid':copy.deepcopy(self.sim.grid),'graph': copy.deepcopy(self.sim.graph),'mask':mask.copy()}
        
        
        
        action,action_args,*action_enc = self.agent.choose_action(idr,self.sim,mask=mask)
        action = 'L'
        action_args = {'rid':1,
                          }
        valid,closer,blocktype = self.agent.Act(self.sim,action,**action_args,draw=draw)
                    
        if valid:
            if np.all(self.sim.grid.min_dist < 1e-5) and np.all(self.sim.grid.hold==-1):
                success = True
                mask[:]=False
        else:
            failure = True
            #mark the state as terminal
            mask[:]=False
        reward =self.rewardf(action, valid, closer, success,failure)
        
        print(reward)
        
        buffer.push(prev_state['graph'],prev_state['mask'],action_enc[0],self.sim.graph,mask,reward)
        print("pol update")
        self.agent.update_policy(buffer,buffer_count,2)
        print("pol updated")
        if draw:
            action_args.pop('rid')
            
            self.sim.draw_act(idr,action,blocktype,prev_state,**action_args)
            self.sim.add_frame()


        anim = self.sim.animate()
        
        return anim

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
    config={'train_n_episodes':60000,
            'train_l_buffer':4000,
            'ep_batch_size':128,
            'ep_use_mask':True,
            'agent_discount_f':0.1,
            'agent_last_only':True,
            'torch_device':'cuda',
            'A2C_nblocks':3,
            'graph_inter_Va':128,
            'graph_inter_Ea':64,
            'graph_inter_u':256,
            'graph_neurons_chanel':300,
            'graph_n_hidden_layers':3,
            'graph_pol_enc':'one_edge',
            'opt_lr':5e-4,
            'opt_pol_over_val': 1,
            'opt_tau': 1e-6,
            'opt_weight_decay':0,
            'opt_exploration_factor':0.001,
            'agent_exp_strat':'epsilon-softmax',
            'agent_epsilon':0.05,
            'opt_max_norm': 2,
            'opt_target_entropy':3,
            'opt_entropy_penalty':True,
            'opt_value_clip':True
            }
    
    gym = ReplayDiscreteGymSupervisor(config,use_wandb=False,agent_type = SACSupervisorDense)
    t0 = time.perf_counter()
    anim = gym.training(max_steps = 20, draw_freq = 100,pfreq =10)
    #anim = gym.test()
    #gr.save_anim(anim,os.path.join(".", f"test_graph"),ext='html')
    t1 = time.perf_counter()
    print(f"time spent: {t1-t0}s")
    print("\nEnd test gym")