# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:40:18 2022

@author: valla
"""

import copy
import time
import os
import wandb
import numpy as np
import pickle
from Blocks import discret_block as Block
from single_agent import reward_link2,A2CSupervisor,A2CSupervisorStruc,generate_mask_supervisor,vec2act_sup
from discrete_simulator import DiscreteSimulator as Sim,Transition as Trans
import discrete_graphics as gr
hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.7)
triangle = Block([[0,0,0]],muc=0.7)
link = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=0.7)
hextarget = Block([[1,0,1],[0,0,0],[2,0,0]])

class GymSupervisor():
    def __init__(self,
                 maxs = [9,6],
                 block_type = [hexagon,link],
                 targets = [triangle]*2,
                 targets_loc = [[1,0],[7,0]],
                 targets_rot = [1,1],
                 n_robots=2,
                 ranges = None,
                 agent_type = A2CSupervisor,
                 actions = ['H','L','R'],
                 reward_function = reward_link2,
                 use_wandb=True
            ):
        if ranges is None:
            ranges = np.ones((n_robots,maxs[0],maxs[1],2),dtype = bool)
        self.n_robots = n_robots
        self.sim = Sim(maxs,n_robots)
        for tar,loc,rot in zip(targets,targets_loc,targets_rot):
            self.sim.add_ground(tar,loc,rot)
        self.agent = agent_type(n_robots,block_type,actions,maxs,use_wandb=use_wandb)
        self.rewardf = reward_function
        self.setup = copy.deepcopy(self.sim)
        if use_wandb:
            wandb.init(project="MARL", entity="flask")
            wandb.config = {'agent_type': type(self.agent),
                            'n_robots': self.n_robots,
                            'targets_loc': targets_loc,
                            'targets_rot': targets_rot,
                        
            }
            wandb.config.update(self.agent.model.__dict__)
    
    def episode(self,
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
                    prev_state = [copy.deepcopy(self.sim.grid),mask.copy()]
                else:
                    prev_state = copy.deepcopy(self.sim.grid)
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
                            mask[:]=False
                    else:
                        #remove the action from the mask so it cannot be chosen anymore
                        if use_mask:
                            mask[action_enc[0]]=False
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
                n_episodes=10,
                pfreq = 10,
                draw_freq=100,
                max_steps=100,
                save_freq = 1000,
                l_buffer = 4000,
                batch_size = 256,
                log_dir=None):
        
        
        if log_dir is None:
            log_dir = os.path.join('log','log'+str(np.random.randint(10000000)))
            os.mkdir(log_dir)
        buffer = np.empty(l_buffer,dtype = object)
        buffer_count=0
        print("Training started")
        for episode in range(n_episodes):
            (rewards_ep,n_steps_ep,
             anim,buffer,buffer_count) = self.episode_multistep(max_steps,
                                                                draw = episode % draw_freq == draw_freq-1,
                                                                buffer=buffer,
                                                                buffer_count=buffer_count,
                                                                batch_size=batch_size)
            if episode % pfreq==0:
                print(f'episode {episode}/{n_episodes} rewards: {np.sum(rewards_ep,axis=1)}')
            if episode % save_freq == 0:
                file = open(os.path.join(log_dir,f'res{episode}.pickle'), 'wb')
                pickle.dump({"rewards":rewards_ep,"episode":episode,"n_steps":n_steps_ep},file)
                file.close()
            if anim is not None:
                gr.save_anim(anim,os.path.join(log_dir, f"episode {episode}"),ext='gif')
                gr.save_anim(anim,os.path.join(log_dir, f"episode {episode}"),ext='html')
        return anim
    
    def test(self,
             draw=True):
        pass

if __name__ == '__main__':
    print("Start test gym")
    gym = ReplayDiscreteGymSupervisor(agent_type=A2CSupervisor)
    t0 = time.perf_counter()
    anim = gym.training(n_episodes = 5000,max_steps = 10, draw_freq = 10,pfreq =1)
    #anim = gym.test()
    t1 = time.perf_counter()
    print(f"time spent: {t1-t0}s")
    print("\nEnd test gym")