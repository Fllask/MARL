# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 20:11:43 2022

@author: valla
"""

import discrete_graphics as gr
import physics as ph
import matplotlib.pyplot as plt
from Blocks import discret_block as Block, Grid
import numpy as np
import time
def scenario1(maxs, n_block = 10,maxtry=100):
    #try to fill the grid with hexagones
    block_list = [Block([[0,1,1],[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0]])]
    
    
    grid = Grid(maxs)
    grid.put(block_list[0],[maxs[0]//2,maxs[1]//2],0,1,floating=True)
    bid = 2
    trys=0
    while bid < n_block+1 and trys < maxtry:
        block = np.random.choice(block_list)
        pos = np.random.randint(maxs)
        if grid.put(block,pos,0,bid):
            bid +=1
            trys=0
        else:
            trys+=1
    return grid,bid-1
def scenario2(maxs, n_block = 10,maxtry=100):
    #try to fill the grid with hexagones
    block_list = [Block([[0,1,1],[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0]]),
                  Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]])]
    
    
    grid = Grid(maxs)
    grid.put(block_list[0],[maxs[0]//2,maxs[1]//2],0,1,floating=True)
    bid = 2
    trys=0
    
    while bid < n_block+1 and trys < maxtry:
        block =block_list[bid%2]
        pos = np.random.randint(maxs)
        rot = np.random.randint(6)
        if grid.put(block,pos,rot,bid):
            bid +=1
            trys=0
        else:
            trys+=1
    return grid,bid-1
if __name__ == '__main__':
    print("Start test simulator")
    maxs = [50,50]
    
    fig,ax = gr.draw_grid(maxs,h=30,label_points=False)
    time0 = time.perf_counter()
    grid,bid = scenario1(maxs,n_block=500,maxtry=10000)
    time1 = time.perf_counter()
    print(f"time needed to put {bid} blocks: {time1-time0} ")
    gr.fill_grid(ax, grid)
    plt.show()
    print("End test simulator")
    