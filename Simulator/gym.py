# -*- coding: utf-8 -*-

"""
Created on Mon Oct 24 16:59:30 2022

@author: valla
"""
import copy
import time
import os
import wandb
import numpy as np
import pickle
from Blocks import discret_block as Block
from Agents import NaiveRandom,reward_link2,WolpertingerLearner,QTLearner,A2CLearner,generate_mask,args2idx,vec2act
from discrete_simulator import DiscreteSimulator as Sim,Transition as Trans
import discrete_graphics as gr
hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.7)
triangle = Block([[0,0,0]],muc=0.7)
link = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=0.7)
hextarget = Block([[1,0,1],[0,0,0],[2,0,0]])

class ReplayDiscreteGym():
    def __init__(self,
                 maxs = [9,6],
                 block_type = [hexagon,link],
                 targets = [triangle]*2,
                 targets_loc = [[1,0],[7,0]],
                 targets_rot = [1,1],
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
        #wandb.init(project="MARL", entity="flask")
        wandb.config = {'agent_type': type(self.agents[0]),
                        'n_robots': self.n_robots,
                        'targets_loc': targets_loc,
                        'targets_rot': targets_rot,
                        
            }
        wandb.config.update(self.agents[0].model.__dict__)
    def episode_egoist(self,
                       max_steps = 1000,
                       draw=False,
                       buffer=np.zeros((2,0)),
                       buffer_count=0,
                       batch_size=32,
                       use_mask=True):
        success = False
        rewards_ar = np.zeros((self.n_robots,max_steps))
        self.sim =copy.deepcopy(self.setup)
        
        if draw:
            self.sim.setup_anim()
            self.sim.add_frame()
        
        for step in range(max_steps):
            
            for idr in range(self.n_robots):
                if use_mask:
                    mask = generate_mask(self.sim.grid,
                                         idr,
                                         self.agents[idr].block_choices, 
                                         self.agents[idr].grid_size,
                                         self.agents[idr].max_blocks)
                    prev_state = [copy.deepcopy(self.sim.grid),mask]
                else:
                    prev_state = copy.deepcopy(self.sim.grid)
                    mask = None
                if buffer_count >0:
                    buffer[idr,(buffer_count-1)%buffer.shape[1]].new_state = prev_state
                # if buffer_count < buffer.shape[0]:
                #     explore = 100
                # elif buffer_count < buffer.shape[0]*2:
                #     explore = 50
                # else:
                #     explore = 10
                
                action,action_args,*action_enc = self.agents[idr].choose_action(self.sim.grid,mask=mask)
                
                valid,closer,blocktype = self.agents[idr].Act(self.sim,action,**action_args)
                if draw:
                    self.sim.draw_act(idr,action,blocktype,**action_args)
                    if valid:
                        self.sim.add_frame()
                if valid:
                    if np.all(self.sim.grid.min_dist < 1e-5) and np.all(self.sim.grid.hold==-1):
                        success = True
                    
                reward =self.rewardf(action, valid, closer, success)
                
                rewards_ar[idr,step]=reward
                buffer[idr,buffer_count%buffer.shape[1]]=Trans(prev_state,
                                                               action_enc,
                                                               reward,
                                                               None)
                self.agents[idr].update_policy(buffer[idr,:],buffer_count,batch_size)
                if success:
                    break
            buffer_count +=1
            if success:
                break
        #complete the last state
        if use_mask:
            buffer[idr,(buffer_count-1)%buffer.shape[1]].new_state = [copy.deepcopy(self.sim),mask]
        else:
            buffer[idr,(buffer_count-1)%buffer.shape[1]].new_state = copy.deepcopy(self.sim)
        if draw:
            anim = self.sim.animate()
        else:
            anim = None
        
        return rewards_ar,step, anim,buffer,buffer_count
    def episode_multistep_collaborative(self,
                              max_steps = 20,
                              draw=False,
                              buffer=np.zeros((2,0)),
                              buffer_counts=np.zeros(2),
                              batch_size=32,
                              use_mask=True):
        success = False
        rewards_ar = np.zeros((self.n_robots,max_steps))
        self.sim =copy.deepcopy(self.setup)
        
        if draw:
            self.sim.setup_anim()
            self.sim.add_frame()
        for step in range(max_steps):
            
            for idr in range(self.n_robots):
                valid = False
                if use_mask:
                    mask = generate_mask(self.sim.grid,
                                         idr,
                                         self.agents[idr].block_choices, 
                                         self.agents[idr].grid_size,
                                         self.agents[idr].max_blocks)
                    prev_state = [copy.deepcopy(self.sim.grid),mask.copy()]
                else:
                    prev_state = copy.deepcopy(self.sim.grid)
                    mask = None
                if step >0:
                    buffer[idr,(buffer_counts[idr])%buffer.shape[1]].new_state = prev_state
                    #average the individual rewards of all agents
                    buffer[idr,(buffer_counts[idr])%buffer.shape[1]].r += np.sum(rewards_ar[:idr,step])+ np.sum(rewards_ar[idr+1:,step-1])
                    buffer[idr,(buffer_counts[idr])%buffer.shape[1]].r /= self.n_robots
                    buffer_counts[idr] +=1
                n_tries = 0
                while not valid:
                    
                    if n_tries >0:
                        buffer[idr,(buffer_counts[idr]-1)%buffer.shape[1]].new_state = prev_state
                        buffer_counts[idr] +=1
                    n_tries +=1
                    action,action_args,*action_enc = self.agents[idr].choose_action(self.sim.grid,mask=mask)
                    valid,closer,blocktype = self.agents[idr].Act(self.sim,action,**action_args)
                    
                    if valid:
                        if np.all(self.sim.grid.min_dist < 1e-5) and np.all(self.sim.grid.hold==-1):
                            success = True
                    else:
                        #remove the action from the mask so it cannot be chosen anymore
                        if use_mask:
                            mask[action_enc[0]]=False
                    reward =self.rewardf(action, valid, closer, success)
                    
                    rewards_ar[idr,step]=reward/n_tries+rewards_ar[idr,step]*(n_tries-1)/n_tries
                    # buffer[idr,buffer_counts[idr]%buffer.shape[1]]=Trans(prev_state,
                    #                                                action_enc,
                    #                                                reward,
                    #                                                None)
                    unfinished_trans[idr] = Trans(prev_state,
                                                  action_enc,
                                                  reward,
                                                  None)
                    
                    self.agents[idr].update_policy(buffer[idr,:],buffer_counts[idr]-1,batch_size)
                
                if draw:
                    self.sim.draw_act(idr,action,blocktype,**action_args)
                    self.sim.add_frame()
                        
                if success:
                    break
            if success:
                break
        if success:
            #make all robots end up in the terminal state:
            for idr_ter in range(idr+1):
                if use_mask:
                    buffer[idr_ter,(buffer_counts[idr_ter]-1)%buffer.shape[1]].new_state = [copy.deepcopy(self.sim),mask]
                else:
                    buffer[idr_ter,(buffer_counts[idr_ter]-1)%buffer.shape[1]].new_state = copy.deepcopy(self.sim)
        else:
            #only the last robot reaches the terminal state
            if use_mask:
                buffer[idr,(buffer_counts[idr]-1)%buffer.shape[1]].new_state = [copy.deepcopy(self.sim),mask]
            else:
                buffer[idr,(buffer_counts[idr]-1)%buffer.shape[1]].new_state = copy.deepcopy(self.sim)
            
        if draw:
            anim = self.sim.animate()
        else:
            anim = None
        
        return rewards_ar,step, anim,buffer,buffer_counts
    
    def training(self,
                n_episodes=10,
                pfreq = 10,
                draw_freq=100,
                max_steps=100,
                save_freq = 5,
                l_buffer = 2000,
                batch_size = 512,
                log_dir=None):
        
        
        if log_dir is None:
            log_dir = os.path.join('log','log'+str(np.random.randint(10000000)))
            os.mkdir(log_dir)
        buffer = np.empty((self.n_robots,l_buffer),dtype = object)
        buffer_count=np.zeros(self.n_robots,dtype=int)
        print("Training started")
        for episode in range(n_episodes):
            (rewards_ep,n_steps_ep,
             anim,buffer,buffer_count) = self.episode_multistep_collaborative(max_steps,
                                                                                draw = episode % draw_freq == draw_freq-1,
                                                                                buffer=buffer,
                                                                                buffer_counts=buffer_count,
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
    def episode_collaborative(self,
                              max_steps = 1000,
                              draw=False,
                              buffer=np.zeros((2,0)),
                              buffer_count=0,
                              batch_size=32,
                              use_mask=True):
        success = False
        rewards_ar = np.zeros((self.n_robots,max_steps))
        self.sim =copy.deepcopy(self.setup)
        
        if draw:
            self.sim.setup_anim()
            self.sim.add_frame()
        for step in range(max_steps):
            
            for idr in range(self.n_robots):
                if use_mask:
                    mask = generate_mask(self.sim.grid,
                                         idr,
                                         self.agents[idr].block_choices, 
                                         self.agents[idr].grid_size,
                                         self.agents[idr].max_blocks)
                    prev_state = [copy.deepcopy(self.sim.grid),mask]
                else:
                    prev_state = copy.deepcopy(self.sim.grid)
                    mask = None
                if buffer_count >0:
                    buffer[idr,(buffer_count-1)%buffer.shape[1]].new_state = prev_state
                    #average the individual rewards of all agents
                    buffer[idr,(buffer_count-1)%buffer.shape[1]].r += np.sum(rewards_ar[:idr,step])+ np.sum(rewards_ar[idr+1:,step-1])
                    buffer[idr,(buffer_count-1)%buffer.shape[1]].r /= self.n_robots
                # if buffer_count < buffer.shape[0]:
                #     explore = 100
                # elif buffer_count < buffer.shape[0]*2:
                #     explore = 50
                # else:
                #     explore = 10
                action,action_args,*action_enc = self.agents[idr].choose_action(self.sim.grid,mask=mask)
                
                valid,closer,blocktype = self.agents[idr].Act(self.sim,action,**action_args)
                if draw:
                    self.sim.draw_act(idr,action,blocktype,**action_args)
                    if valid:
                        self.sim.add_frame()
                if valid:
                    if np.all(self.sim.grid.min_dist < 1e-5) and np.all(self.sim.grid.hold==-1):
                        success = True
                    
                reward =self.rewardf(action, valid, closer, success)
                
                rewards_ar[idr,step]=reward
                buffer[idr,buffer_count%buffer.shape[1]]=Trans(prev_state,
                                                               action_enc,
                                                               reward,
                                                               None)
                self.agents[idr].update_policy(buffer[idr,:],buffer_count,batch_size)
                if success:
                    break
            buffer_count +=1
            if success:
                break
        if success:
            #make all robots end up in the terminal state:
            for idr_ter in range(idr+1):
                if use_mask:
                    buffer[idr_ter,(buffer_count-1)%buffer.shape[1]].new_state = [copy.deepcopy(self.sim),mask]
                else:
                    buffer[idr_ter,(buffer_count-1)%buffer.shape[1]].new_state = copy.deepcopy(self.sim)
            for idr_ter in range(idr+1,self.n_robots):
                if use_mask:
                    buffer[idr_ter,(buffer_count-2)%buffer.shape[1]].new_state = [copy.deepcopy(self.sim),mask]
                else:
                    buffer[idr_ter,(buffer_count-2)%buffer.shape[1]].new_state = copy.deepcopy(self.sim)
        else:
            #only the last robot reaches the terminal state
            if use_mask:
                buffer[idr,(buffer_count-1)%buffer.shape[1]].new_state = [copy.deepcopy(self.sim),mask]
            else:
                buffer[idr,(buffer_count-1)%buffer.shape[1]].new_state = copy.deepcopy(self.sim)
            
        if draw:
            anim = self.sim.animate()
        else:
            anim = None
        
        return rewards_ar,step, anim,buffer,buffer_count
    
    
    def test2(self,
             draw=True):
        success = False
        self.sim =copy.deepcopy(self.setup)
        if draw:
            self.sim.setup_anim()
            self.sim.add_frame()
        
                
        action = 'Pl'
        action_args = {'pos':[7,0],'ori':0,'blocktypeid':0}
        ag=0
        valid,closer,blocktype = self.agents[ag].Act(self.sim,action,**action_args)
        if draw:
            self.sim.draw_act(ag,action,blocktype,**action_args)
            if valid:
                self.sim.add_frame()
        action = 'Ph'
        action_args = {'pos':[6,2],'ori':0,'blocktypeid':0}
        ag=1
        valid,closer,blocktype = self.agents[ag].Act(self.sim,action,**action_args)
        if draw:
            self.sim.draw_act(ag,action,blocktype,**action_args)
            if valid:
                self.sim.add_frame()
                
        action = 'R'
        action_args = {'bid':1}
        ag=0
        valid,closer,blocktype = self.agents[ag].Act(self.sim,action,**action_args)
        if draw:
            self.sim.draw_act(ag,action,blocktype,**action_args)
            if valid:
                self.sim.add_frame()
                
        valid =False
        mask = generate_mask(self.sim.grid,
                             0,
                             self.agents[0].block_choices, 
                             self.agents[0].grid_size,
                             self.agents[0].max_blocks)
        actions_allowed = np.nonzero(mask)
        for i in actions_allowed[0]:
            action,action_args = vec2act(self.agents[0].action_list[i], 2, np.array([9,6]), 30)
            bid = action_args.get('blocktypeid')
            if bid is not None:
                blocktype = [hexagon,link][bid]
            else:
                blocktype=None
            self.sim.draw_act(0,action,blocktype,**action_args)
        self.sim.add_frame()
        if valid:
            if np.all(self.sim.grid.min_dist < 1e-5) and np.all(self.sim.grid.hold==-1):
                print("success")
        if draw:
            anim = self.sim.animate()
            gr.save_anim(anim,"test_agent",ext='html')
        else:
            anim = None
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
    gym = ReplayDiscreteGym(agent_type=A2CLearner)
    t0 = time.perf_counter()
    anim = gym.training(n_episodes = 5000,max_steps = 5, draw_freq = 8,pfreq =1)
    #anim = gym.test2()
    t1 = time.perf_counter()
    print(f"time spent: {t1-t0}s")
    print("\nEnd test gym")