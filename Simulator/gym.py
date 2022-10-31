# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:59:30 2022

@author: valla
"""
import copy
import time
import os
import numpy as np
import pickle
from Blocks import discret_block as Block
from Agents import NaiveRandom,reward_link2,WolpertingerLearner,QTLearner
from discrete_simulator import DiscreteSimulator as Sim,Transition as Trans
import discrete_graphics as gr
hexagon = Block([[0,1,1],[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0]],muc=0.7)
triangle = Block([[0,0,0]],muc=0.7)
link = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=0.7)
hextarget = Block([[1,0,1],[0,0,0],[2,0,0]])
class ReplayDiscreteGym():
    def __init__(self,
                 maxs = [9,6],
                 block_type = [hexagon,link],
                 targets = [hextarget]*2,
                 targets_loc = [[1,0],[7,0]],
                 targets_rot = [0,0],
                 n_robots=2,
                 ranges = None,
                 agent_type = NaiveRandom,
                 actions = ['Ph','Pl','H','L','R'],
                 reward_function = reward_link2
            ):
        if ranges is None:
            ranges = np.ones((n_robots,maxs[0],maxs[1],2),dtype = bool)
        self.n_robots = n_robots
        self.sim = Sim(maxs,n_robots)
        for tar,loc,rot in zip(targets,targets_loc,targets_rot):
            self.sim.add_ground(tar,loc,rot)
        self.agents = [agent_type(i,ranges[i],block_type,actions,maxs) for i in range(n_robots)]
        self.rewardf = reward_function
        self.setup = copy.deepcopy(self.sim)
    def episode_egoist(self,
                       max_steps = 1000,
                       draw=False,
                       buffer=np.array([]),
                       buffer_count=0,
                       batch_size=32):
        success = False
        rewards_ar = np.zeros((self.n_robots,max_steps))
        self.sim =copy.deepcopy(self.setup)
        if draw:
            self.sim.setup_anim()
            self.sim.add_frame()
        for step in range(max_steps):
            for idr in range(self.n_robots):
                prev_state = copy.deepcopy(self.sim.grid)
                # if buffer_count < buffer.shape[0]:
                #     explore = 100
                # elif buffer_count < buffer.shape[0]*2:
                #     explore = 50
                # else:
                #     explore = 10
                action,action_args,*action_enc = self.agents[idr].choose_action(self.sim.grid)
                
                valid,closer,blocktype = self.agents[idr].Act(self.sim,action,**action_args)
                if draw:
                    self.sim.draw_act(idr,action,blocktype,**action_args)
                    if valid:
                        self.sim.add_frame()
                if np.all(self.sim.grid.min_dist < 1e-5) and np.all(self.sim.grid.hold==-1):
                    success = True
                reward =self.rewardf(action, valid, closer, success)
                
                rewards_ar[idr,step]=reward
                buffer[buffer_count%buffer.shape[0]]=Trans(prev_state,
                                                           action_enc,
                                                           reward,
                                                           copy.deepcopy(self.sim.grid))
                buffer_count +=1
                self.agents[idr].update_policy(buffer,buffer_count,batch_size)
                if success:
                    break
            if success:
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
                save_freq = 5,
                l_buffer = 500,
                n_mini_batches = 20,
                log_dir=None):
        if log_dir is None:
            log_dir = os.path.join('log','log'+str(np.random.randint(10000000)))
            os.mkdir(log_dir)
        buffer = np.empty(l_buffer,dtype = object)
        buffer_count=0
        print("Training started")
        for episode in range(n_episodes):
            (rewards_ep,n_steps_ep,
             anim,buffer,buffer_count) = self.episode_egoist(max_steps,
                                                             draw = episode % draw_freq == draw_freq-1,
                                                             buffer=buffer,
                                                             buffer_count=buffer_count)
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
    
class DiscretGym():
    def __init__(self,
                 maxs = [9,6],
                 block_type = [hexagon,link],
                 targets = [hextarget]*2,
                 targets_loc = [[1,0],[7,0]],
                 targets_rot = [0,0],
                 n_robots=2,
                 ranges = None,
                 agent_type = NaiveRandom,
                 actions = ['Ph','Pl'],#,'H','L','R'],
                 reward_function = reward_link2
            ):
        if ranges is None:
            ranges = np.ones((n_robots,maxs[0],maxs[1],2),dtype = bool)
        self.n_robots = n_robots
        self.sim = Sim(maxs,n_robots)
        for tar,loc,rot in zip(targets,targets_loc,targets_rot):
            self.sim.add_ground(tar,loc,rot)
        self.agents = [agent_type(i,ranges[i],block_type,actions,maxs) for i in range(n_robots)]
        self.rewardf = reward_function
        self.setup = copy.deepcopy(self.sim)
    def episode_egoist(self,max_steps = 1000,draw=False):
        success = False
        rewards_ar = np.zeros((self.n_robots,max_steps))
        self.sim =copy.deepcopy(self.setup)
        if draw:
            self.sim.setup_anim()
            self.sim.add_frame()
        for step in range(max_steps):
            for idr in range(self.n_robots):
                prev_state = copy.deepcopy(self.sim.grid)
                action,action_args,*action_enc = self.agents[idr].choose_action(self.sim.grid)
                
                valid,dist,con = self.agents[idr].Act(self.sim,action,**action_args)
                if valid and draw:
                    self.sim.add_frame()
                reward =self.rewardf(action, valid, dist, con)
                
                rewards_ar[idr,step]=reward
               
                self.agents[idr].update_policy(reward, prev_state, action)
            if np.all(self.sim.grid.connection < 1) and np.all(self.sim.grid.hold==-1):
                success = True
                break
        if draw:
            anim = self.sim.animate()
        else:
            anim = None
        
        return rewards_ar,step, anim
    
    
   
    
    
    
    def training(self,n_episodes=10,pfreq = 10, draw_freq=100,max_steps=10000,save_freq = 10, log_dir=None):
        if log_dir is None:
            log_dir = os.path.join('log','log'+str(np.random.randint(10000000)))
            os.mkdir(log_dir)
        print("Training started")
        for episode in range(n_episodes):
            rewards_ep, n_steps_ep, anim = self.episode_egoist(max_steps,draw = episode % draw_freq == 0)
            if episode % pfreq==0:
                print(f'episode {episode}/{n_episodes} rewards: {np.sum(rewards_ep,axis=1)}')
            if episode % save_freq == 0:
                file = open(os.path.join(log_dir,f'res{episode}.pickle'), 'wb')
                pickle.dump({"rewards":rewards_ep,"episode":episode,"n_steps":n_steps_ep},file)
                file.close()
            if anim is not None:
                gr.save_anim(anim,os.path.join(log_dir, f"episode {episode}"),ext='gif')
        return anim
if __name__ == '__main__':
    print("Start test gym")
    gym = ReplayDiscreteGym(agent_type=QTLearner)
    t0 = time.perf_counter()
    anim = gym.training(n_episodes = 1000,max_steps = 100, draw_freq = 100,pfreq =1)
    t1 = time.perf_counter()
    print(f"time for 2000 steps: {t1-t0}s")
    print("\nEnd test gym")