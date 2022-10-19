# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 20:11:43 2022

@author: valla
"""

import discrete_graphics as gr
import matplotlib.pyplot as plt
from Blocks import discret_block as Block, Grid
from physics_scipy import stability_solver_discrete as ph
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
def scenario3(maxs,n_block,maxtry = 100,mode='triangle',draw=False,physon=True):
    #pill up some shapes
    
    block_list = [Block([[0,1,1],[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0]],muc=0.7),
                  Block([[0,0,0]],muc=0.7)]
    ground = Block([[0,0,0],[maxs[0]-1,0,0]]+[[i,0,1] for i in range(0,maxs[0])],muc=0.7)
    grid = Grid(maxs)
    grid.put(ground, [0,0], 0, 1,floating=True)
    
    phys = ph(maxs,n_robots = 0)
    trys = 0
    bid = 2
    while bid < n_block+1 and trys < maxtry:
        if mode == 'triangle':
            block = block_list[1]
        elif mode == 'hex':
            block = block_list[0]
        else:
            block = np.random.choice(block_list)
        pos = np.random.randint(maxs)
        rot = np.random.randint(6)
        if grid.put(block,pos,rot,bid):
            if draw:
                fig,ax = gr.draw_grid(maxs,h=30,label_points=True)
                gr.fill_grid(ax,grid)
            if physon:    
                phys.add_block(grid, block, bid)
                res = phys.solve()
                if res.status==0:
                    bid+=1
                    trys=0
                    
                else:
                    trys+=1
                    grid.remove(bid)
                    if draw:
                        fig,ax = gr.draw_grid(maxs,h=30,label_points=True)
                        gr.fill_grid(ax,grid)
                    phys.remove_block(bid)
                    
                    
            else:
                bid+=1
                trys=0
                
        else:
            trys+=1
    return grid,bid-1
if __name__ == '__main__':
    print("Start test simulator")
    maxs = [30,20]
    
    fig,ax = gr.draw_grid(maxs,h=30,label_points=False,color='none')
    time0 = time.perf_counter()
    grid,bid = scenario3(maxs,n_block=300,maxtry=1000,mode='triangle',physon = False)
    time1 = time.perf_counter()
    print(f"time needed to put {bid} blocks: {time1-time0} ")
    gr.fill_grid(ax, grid)
    plt.show()
    print("End test simulator")
    