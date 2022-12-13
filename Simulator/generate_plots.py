# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:22:11 2022

@author: valla
"""

import discrete_graphics as gr
from discrete_blocks import discret_block as Block,Grid
import matplotlib.pyplot as plt
plt.tight_layout()
def gen_grid(h=10):
    gr.draw_grid([5,5],label_points=True,steps=1,color='k',h=h,linewidth=1,fontsize=13)
    plt.tight_layout()
    plt.savefig('../graphics/grid.pdf')  
def gen_blocks(h=20):
    fig,ax = gr.draw_grid([6,3],h=h,color='k',linewidth=0.8,label_points = False)
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]])
    link = Block([[3,0,0],[4,0,0],[3,1,1],[3,1,0],[2,2,1],[3,2,1]])
    ground = Block([[5,1,1]])
    gr.draw_block(ax, hexagon, color=plt.cm.Pastel1(0))
    gr.draw_block(ax, link, color=plt.cm.Pastel1(1))
    gr.draw_block(ax, ground, color=plt.cm.Pastel1(2))
    plt.tight_layout()
    plt.savefig('../graphics/blocks.pdf')
def gen_structures(h=20):
    maxs=[10,10]
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]])
    link = Block([[1,0,0],[2,0,0],[1,1,1],[1,1,0],[0,2,1],[1,2,1]])
    ground = Block([[0,0,1]])
    
    # #build a tower of hex
    # grid = Grid(maxs)
    # grid.put(ground, [6,1], 0, 0,floating=True)
    # grid.connect(hexagon, 1, 0, 0, 0,idcon=0)
    # grid.connect(hexagon, 2, 0, 1, 1)
    # grid.connect(hexagon, 3, 0, 1, 2)
    # grid.connect(hexagon, 4, 0, 1, 3)
    # fig,ax = gr.draw_grid(maxs,h=h,color='k',linewidth=0.7,label_points = False)
    # gr.fill_grid(ax, grid)
    # plt.tight_layout()
    # plt.savefig('../graphics/tower.pdf')  
    
    # #build a minimalist arc using hexagons
    # grid = Grid(maxs)
    # grid.put(ground, [4,2], 0, 0,floating=True)
    # grid.put(ground, [7,2], 0, 0,floating=True)
    # grid.connect(hexagon, 1, 0, 0, 0,idcon=0)
    # grid.connect(hexagon, 2, 0, 2, 1)
    # grid.connect(hexagon, 3, 0, 2, 2)
    # fig,ax = gr.draw_grid(maxs,h=h,color='k',linewidth=0.7,label_points = False)
    # gr.fill_grid(ax, grid)
    # plt.tight_layout()
    # plt.savefig('../graphics/small_arch.pdf')
    
    # #build a bigger arch using hexagons and link
    # grid = Grid(maxs)
    # grid.put(ground, [2,3], 0, 0,floating=True)
    # grid.put(ground, [8,3], 0, 0,floating=True)
    # grid.connect(hexagon, 1, 0, 0, 0,idcon=0)
    # grid.connect(link,2,4,1,1)
    # grid.connect(hexagon, 3, 0, 6, 2)
    # grid.connect(link,4,4,1,3)
    # grid.connect(hexagon, 5, 0, 6,4)
    # grid.connect(link,6,4,1,5)
    # grid.connect(hexagon, 7, 0, 6,6)
    # fig,ax = gr.draw_grid(maxs,h=h,color='k',linewidth=0.7,label_points = False)
    # gr.fill_grid(ax, grid)
    # plt.tight_layout()
    # plt.savefig('../graphics/big_arch.pdf')
    
    # #dodgy structure
    # grid = Grid(maxs)
    # grid.put(ground, [3,2], 0, 0,floating=True)
    # grid.put(ground, [8,2], 0, 0,floating=True)
    # grid.connect(hexagon, 1, 0, 0, 0,idcon=0)
    # grid.connect(link,2,4,1,1)
    # grid.connect(hexagon, 3, 0, 6, 2)
    # grid.connect(link,4,4,1,3)
    # grid.connect(link,5,6,2,4)
    # grid.connect(hexagon, 7, 0, 0, 0,idcon=1)
    # fig,ax = gr.draw_grid(maxs,h=h,color='k',linewidth=0.7,label_points = False)
    # gr.fill_grid(ax, grid)
    # plt.tight_layout()
    # plt.savefig('../graphics/disclocated.pdf')
    
    # #different heigths
    # grid = Grid(maxs)
    # grid.put(ground, [3,1], 0, 0,floating=True)
    # grid.put(ground, [6,7], 0, 0,floating=True)
    # grid.connect(hexagon, 1, 0, 0, 0,idcon=0)
    # grid.connect(link,2,4,1,1)
    # grid.connect(hexagon, 3, 0, 6, 2)
    # grid.connect(link,4,4,5,3)
    # grid.connect(hexagon, 5, 0, 6, 4)
    # grid.connect(link,6,6,2,5)
    # grid.connect(hexagon, 7, 0, 0, 0,idcon=1)
    # fig,ax = gr.draw_grid(maxs,h=h,color='k',linewidth=0.7,label_points = False)
    # gr.fill_grid(ax, grid)
    # plt.tight_layout()
    # plt.savefig('../graphics/asymetry.pdf')
    # plt.show()
    
    grid = Grid(maxs)
    grid.put(ground, [2,3], 0, 0,floating=True)
    grid.put(ground, [5,3], 0, 0,floating=True)
    grid.put(ground, [8,3], 0, 0,floating=True)
    grid.connect(hexagon, 1, 0, 0, 0,idcon=0)
    grid.connect(hexagon, 2, 0, 2, 1)
    grid.connect(hexagon, 3, 0, 2, 2)
    grid.connect(hexagon, 4, 0, 5, 3)
    grid.connect(hexagon, 5, 0, 2, 4)
    fig,ax = gr.draw_grid(maxs,h=h,color='k',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, grid)
    plt.tight_layout()
    plt.savefig('../graphics/multiple.pdf')
    plt.show()
    
if __name__ == "__main__":
    gen_structures(h=8)