# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:53:45 2023

@author: valla
"""
from single_agent_gym_norot import ReplayDiscreteGymSupervisor as Gym
from relative_single_agent import A2CSupervisor, A2CSupervisorDense,SACSupervisorDense,SACSupervisorSparse
from discrete_blocks_norot import discret_block_norot as Block
import pickle
import discrete_graphics as gr
import numpy as np
import matplotlib.pyplot as plt
def create_gym(config):
    
    #overwrite the action choice method:
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.5)
    linkr = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=0.5) 
    linkl = Block([[0,0,0],[0,1,1],[1,0,0],[0,1,0],[0,0,1],[-1,1,1]],muc=0.5) 
    linkh = Block([[0,0,0],[0,1,1],[1,0,0],[-1,2,1],[0,1,0],[0,2,1]],muc=0.5)
    target = Block([[0,0,1]])
    gym = Gym(config_sparse_SAC,
              agent_type=SACSupervisorSparse,
              use_wandb=False,
              actions= ['Ph'],
              block_type=[hexagon,linkr,linkl,linkh],
              random_targets='random_gap',
              n_robots=2,
              max_blocks = 100,
              targets=[target],
              targets_loc = [[5,0],[23,0]],
              max_interfaces = 300,
              log_freq = 50,
              maxs = [30,20])
    return gym
def load_agent(file,gym,explore=False):
    
    with open(file, "rb") as input_file:
        agent  = pickle.load(input_file)
    if not explore:
        agent.exploration_strat = 'epsilon-greedy'
        agent.eps = 0
    gym.agent = agent
    return gym

if __name__ == "__main__":
    config_sparse_SAC = {'train_n_episodes':60000,
            'train_l_buffer':30000,
            'reward':'modular',
            'ep_batch_size':256,
            'ep_use_mask':True,
            'agent_discount_f':0.05,
            'agent_last_only':True,
            'torch_device':'cuda',
            'SEnc_n_channels':64,
            'SEnc_n_internal_layer':6,
            'SEnc_stride':1,
            'SEnc_order_insensitive':True,
            'SAC_n_fc_layer':3,
            'SAC_n_neurons':128,
            'SAC_batch_norm':True,
            'Q_duel':True,
            'opt_lr':5e-4,
            'opt_pol_over_val': 1,
            'opt_tau': 2e-4,
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
            'reward_failure':-2,
            'reward_action':{'Ph': -0.2, 'L':-0.1},
            'reward_closer':0.4,
            'reward_nsides': 0.05,
            'reward_success':5,
            'reward_opposite_sides':0,
            'opt_lower_bound_Vt':2,
            'gap_range': [1,20],
            }
    gym = create_gym(config_sparse_SAC)
    gym = load_agent("1miosteps.pickle",gym,explore=False)
    alterations=None
    alterations=np.array([[5,0]])
    rewards, anim = gym.exploit(17,alterations= alterations,n_alter=1,h=12,draw_robots=False)
    gr.save_anim(anim,"exploit",ext='html')
    gr.save_anim(anim,"exploit",ext='gif')
    #name = 'struct8_a2'
    #plt.savefig(f'../graphics/results/experiment 3/{name}.pdf')
    #plt.savefig(f'../graphics/results/experiment 3/{name}.png')