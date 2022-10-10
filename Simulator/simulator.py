# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 11:28:57 2022

@author: valla
"""
import graphics as gr
import physics as ph
import matplotlib.pyplot as plt
from Blocks import Block,Slide
import numpy as np
import copy
import time
def scenario1(n_block = 5):
    
    block_list = [Block([[0,0],[3,0],[2,1.75],[1,1.75]]),
                  Block([[0,0],[2,0],[2,2],[0,2]]),
                  ]
    start_block = Block([[4,-5],[5,-4.6]],colors = {'facecolor': 'k'}).expand()
    i_list = []
    
    last_b = start_block
    block_placed = [start_block]
    last_face = 0
    for block_id in range(n_block):
        #chose a random block:
        b = copy.deepcopy(np.random.choice(block_list))
        #create an interface with the opposite face of the last block
        f = np.random.choice(4)
        for f_s in np.random.choice(np.delete(np.arange(4),last_face),3,replace=False):
            #i = Slide(b,last_b,f,(last_face+2)%4)
            i = Slide(b,last_b,f,f_s,safe=True)
            i.complete(block_placed)
            flag = i.find_valid()
            # if i.n_obst[-1] !=1:
            #     print('faillure point --------------------')
            #     print(f'faces: {f},{f_s}')
            #     print(i.blockm.corners)
            #     print(i.blocksf[0].corners)
            #     print([b.corners for b in block_placed])
            #     print(f"{i.min_list=}")
            #     print(f"{i.max_list=}")
            #     print(f"{i.smin_list=}")
            #     print(f"{i.smax_list=}")
            #     print('end faillure point ------------')
            if flag:
                break
        else:
            print('imposible to place the next block')
            break
        x = np.random.random()*0.5+0.5
        i.set_x(x)
        
        i_list.append(i)
        block_placed.append(b)
        last_b = b
        last_face = f
    return block_placed,i_list
if __name__ == '__main__':
    print("Start test simulator")
    fig,ax = plt.subplots(1,1,figsize=(6,6))
    ax.set_xlim([-20,20])
    ax.set_ylim([-20,20])
    time0 = time.perf_counter()
    bl,il = scenario1(n_block=100)
    time1 = time.perf_counter()
    print(f"time needed to put {len(bl)} blocks: {time1-time0} ")
    for block in bl:
        #support = ph.check_support(block,block_list)
        #block.support = block_list[support]
        _ = gr.draw_block(ax, block)
        #print(support)
    plt.show()
    print("End test simulator")
    