# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:22:11 2022

@author: valla
"""

import discrete_graphics as gr
from discrete_blocks_norot import discret_block_norot as Block,Grid
from discrete_simulator_norot import DiscreteSimulator
from discrete_blocks import discret_block as BlockRot, Grid as GridRot
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,FancyArrowPatch
import numpy as np
from physics_scipy import get_cm
from matplotlib.colors import ListedColormap
from relative_single_agent import generate_mask_no_rot

def gen_grid(h=10):
    fig,ax=gr.draw_grid([5,5],label_points=True,steps=1,color='k',h=h,linewidth=1)
    fig.tight_layout()
    plt.savefig('../graphics/grid.png')  
def get_w(h,maxs):
    return h*(0.5+0.5*maxs[1]+maxs[0])/(np.sqrt(3)*maxs[1]/2+1)
def gen_blocks(h=20):
    fig,ax = gr.draw_grid([7,3],h=h,color='k',linewidth=0.8,label_points = False)
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]])
    link = Block([[3,0,0],[4,0,0],[3,1,1],[3,1,0],[2,2,1],[3,2,1]])
    ground = Block([[5,1,1],[6,1,1]])
    gr.draw_block(ax, hexagon, color=plt.cm.plasma(0))
    gr.draw_block(ax, link, color=plt.cm.plasma(0))
    gr.draw_block(ax, ground, color="darkslategrey")
    plt.tight_layout()
    plt.savefig('../graphics/blocks.png')
def gen_structures(h=20):
    maxs=[10,10]
    hexagon = BlockRot([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]])
    link = BlockRot([[1,0,0],[2,0,0],[1,1,1],[1,1,0],[0,2,1],[1,2,1]])
    ground = BlockRot([[0,0,1]])
    
    #build a tower of hex
    grid = GridRot(maxs)
    grid.put(ground, [6,1], 0, 0,floating=True)
    grid.connect(hexagon, 1, 0, 0, 0,idcon=0)
    grid.connect(hexagon, 2, 0, 1, 1)
    grid.connect(hexagon, 3, 0, 1, 2)
    grid.connect(hexagon, 4, 0, 1, 3)
    fig,ax = gr.draw_grid(maxs,h=h,color='k',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, grid,fixed_color=plt.cm.plasma(0),linewidth=5)
    plt.tight_layout()
    plt.savefig('../graphics/structure/tower.png')  
    
    #build a minimalist arc using hexagons
    grid = GridRot(maxs)
    grid.put(ground, [4,2], 0, 0,floating=True)
    grid.put(ground, [7,2], 0, 0,floating=True)
    grid.connect(hexagon, 1, 0, 0, 0,idcon=0)
    grid.connect(hexagon, 2, 0, 2, 1)
    grid.connect(hexagon, 3, 0, 2, 2)
    fig,ax = gr.draw_grid(maxs,h=h,color='k',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, grid,fixed_color=plt.cm.plasma(0),linewidth=5)
    plt.tight_layout()
    plt.savefig('../graphics/structure/small_arch.png')
    
    #build a bigger arch using hexagons and link
    grid = GridRot(maxs)
    grid.put(ground, [2,3], 0, 0,floating=True)
    grid.put(ground, [8,3], 0, 0,floating=True)
    grid.connect(hexagon, 1, 0, 0, 0,idcon=0)
    grid.connect(link,2,4,1,1)
    grid.connect(hexagon, 3, 0, 6, 2)
    grid.connect(link,4,4,1,3)
    grid.connect(hexagon, 5, 0, 6,4)
    grid.connect(link,6,4,1,5)
    grid.connect(hexagon, 7, 0, 6,6)
    fig,ax = gr.draw_grid(maxs,h=h,color='k',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, grid,fixed_color=plt.cm.plasma(0),linewidth=5)
    plt.tight_layout()
    plt.savefig('../graphics/structure/big_arch.png')
    
    #dodgy structure
    grid = GridRot(maxs)
    grid.put(ground, [3,2], 0, 0,floating=True)
    grid.put(ground, [8,2], 0, 0,floating=True)
    grid.connect(hexagon, 1, 0, 0, 0,idcon=0)
    grid.connect(link,2,4,1,1)
    grid.connect(hexagon, 3, 0, 6, 2)
    grid.connect(link,4,4,1,3)
    grid.connect(link,5,6,2,4)
    grid.connect(hexagon, 7, 0, 0, 0,idcon=1)
    fig,ax = gr.draw_grid(maxs,h=h,color='k',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, grid,fixed_color=plt.cm.plasma(0),linewidth=5)
    plt.tight_layout()
    plt.savefig('../graphics/structure/disclocated.png')
    
    #different heigths
    grid = GridRot(maxs)
    grid.put(ground, [3,1], 0, 0,floating=True)
    grid.put(ground, [6,7], 0, 0,floating=True)
    grid.connect(hexagon, 1, 0, 0, 0,idcon=0)
    grid.connect(link,2,4,1,1)
    grid.connect(hexagon, 3, 0, 6, 2)
    grid.connect(link,4,4,5,3)
    grid.connect(hexagon, 5, 0, 6, 4)
    grid.connect(link,6,6,2,5)
    grid.connect(hexagon, 7, 0, 0, 0,idcon=1)
    fig,ax = gr.draw_grid(maxs,h=h,color='k',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, grid,fixed_color=plt.cm.plasma(0),linewidth=5)
    plt.tight_layout()
    plt.savefig('../graphics/structure/asymetry.png')
    plt.show()
    
    grid = GridRot(maxs)
    grid.put(ground, [2,3], 0, 0,floating=True)
    grid.put(ground, [5,3], 0, 0,floating=True)
    grid.put(ground, [8,3], 0, 0,floating=True)
    grid.connect(hexagon, 1, 0, 0, 0,idcon=0)
    grid.connect(hexagon, 2, 0, 2, 1)
    grid.connect(hexagon, 3, 0, 2, 2)
    grid.connect(hexagon, 4, 0, 5, 3)
    grid.connect(hexagon, 5, 0, 2, 4)
    fig,ax = gr.draw_grid(maxs,h=h,color='k',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, grid,fixed_color=plt.cm.plasma(0),linewidth=5)
    plt.tight_layout()
    plt.savefig('../graphics/structure/multiple.png')
    plt.show()
def gen_forces_no_robot(h,muc,name):
    maxs=[10,10]
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=muc)
    linkr = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=muc) 
    linkl = Block([[0,0,0],[0,1,1],[1,0,0],[0,1,0],[0,0,1],[-1,1,1]],muc=muc) 
    linkh = Block([[1,0,0],[2,0,0],[1,1,1],[1,1,0],[0,2,1],[1,2,1]],muc=muc)
    
    ground = Block([[0,0,1]],muc=muc)
    sim = DiscreteSimulator(maxs, 1, [hexagon,linkr,linkl,linkh], 1, 30, 30,ground_blocks=[ground])
    sim.ph_mod.set_max_forces(0,Fx=[-50,50])
    sim.add_ground(ground, [6,1])
    sim.put_rel(hexagon, 0, 0, 0, 0,idconsup=0)
    sim.put_rel(linkr, 0, 0, 1, 0,idconsup=0)
    sim.hold(0,3)
    fig,ax = gr.draw_grid(maxs,h=h,color='none',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, sim.grid,forces_bag=sim.ph_mod,draw_hold=False)
    plt.tight_layout()
    plt.savefig(f'../graphics/forces/{name}.png')
    plt.show()
    
    ground = Block([[0,0,1]],muc=muc)
    sim = DiscreteSimulator(maxs, 1, [hexagon,linkr,linkl,linkh], 1, 30, 30,ground_blocks=[ground])
    sim.ph_mod.set_max_forces(0,Fx=[-50,50])
    sim.add_ground(ground, [6,1])
    sim.put_rel(hexagon, 0, 0, 0, 0,idconsup=0)
    sim.put_rel(linkh, 0, 0, 1, 4,idconsup=0)
    sim.put_rel(linkh, 1, 0, 2, 0,idconsup=0)
    sim.hold(0,3)
    fig,ax = gr.draw_grid(maxs,h=h,color='none',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, sim.grid,forces_bag=sim.ph_mod,draw_hold=False)
    plt.tight_layout()
    plt.savefig(f'../graphics/forces/{name}_interlock.png')
    plt.show()
    
def gen_forces_no_robot_kernel(h,kernel,name,unstable=False):
    maxs=[5,5]
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.7)
    linkr = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=0.7) 
    linkl = Block([[0,0,0],[0,1,1],[1,0,0],[0,1,0],[0,0,1],[-1,1,1]],muc=0.7) 
    linkh = Block([[1,0,0],[2,0,0],[1,1,1],[1,1,0],[0,2,1],[1,2,1]],muc=0.7)
    
    ground = Block([[0,0,1]],muc=0.7)
    sim = DiscreteSimulator(maxs, 1, [hexagon,linkr,linkl,linkh], 1, 30, 30)
    sim.ph_mod.safety_kernel=kernel
    sim.ph_mod.set_max_forces(0,Fx=[-50,50])
    sim.add_ground(ground, [3,0])
    sim.put_rel(hexagon, 0, 0, 0, 0,idconsup=0)
    if unstable:
        sim.put_rel(linkh, 0, 0, 1, 0,idconsup=0)
    else:
        sim.put_rel(linkr, 0, 0, 1, 0,idconsup=0)
    sim.hold(0,3)
    fig,ax = gr.draw_grid(maxs,h=h,color='none',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, sim.grid,forces_bag=sim.ph_mod,draw_hold=False,linewidth=3)
    plt.tight_layout()
    plt.savefig(f'../graphics/forces/{name}.png')
    plt.show()
def gen_forces_max(h,n_blocks,Fymax,name,maxs=[20,20]):
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.7)
    ground = Block([[0,0,1]],muc=0.7)
    sim = DiscreteSimulator(maxs, 1, [hexagon], 1, 30, 30)
    sim.ph_mod.set_max_forces(0,Fy=[-Fymax,Fymax],M=[-0,0])
    sim.add_ground(ground, [(maxs[0]+maxs[1]/2)//2-2,2])
    sim.put_rel(hexagon, 0, 0, 0, 2,idconsup=0)
    sim.hold(0,1)
    for i in range(n_blocks):
        sim.put_rel(hexagon, 0, 0, i+1, 0,idconsup=0)
    fig,ax = gr.draw_grid(maxs,h=h,color='none',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, sim.grid,forces_bag=sim.ph_mod,draw_hold=False,linewidth=1)
    plt.tight_layout()
    plt.savefig(f'../graphics/forces/{name}.png')
    plt.show()
def gen_forces_max_diag(h,n_blocks,Fxmax,name,maxs=[15,15]):
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.7)
    ground = Block([[0,0,0]],muc=0.7)
    sim = DiscreteSimulator(maxs, 1, [hexagon], 1, 30, 30)
    sim.ph_mod.set_max_forces(0,Fx=[-Fxmax,Fxmax],M=[-0,0])
    sim.add_ground(ground, [0,0])
    sim.put_rel(hexagon, 0, 0, 0, 4,idconsup=0)
    
    for i in range(n_blocks):
        sim.put_rel(hexagon, 0, 0, i+1, 4,idconsup=0)
    sim.hold(0,n_blocks+1)
    fig,ax = gr.draw_grid(maxs,h=h,color='none',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, sim.grid,forces_bag=sim.ph_mod,draw_hold=False,linewidth=1)
    plt.tight_layout()
    plt.savefig(f'../graphics/forces/{name}.png')
    plt.show()
def gen_forces_colors(h,name,n_blocks):
    if n_blocks % 2 ==1:
        maxs=[(n_blocks)//2+3*n_blocks-1,n_blocks+2]
    else:
        maxs=[(n_blocks)//2+3*n_blocks-1,n_blocks+3]
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.7)
    ground = Block([[0,0,1]],muc=0.7)
    sim = DiscreteSimulator(maxs, n_blocks-2, [hexagon], 2, 100, 100)
    
    sim.add_ground(ground, [maxs[0]-1,0])
    sim.add_ground(ground, [maxs[0]-3*n_blocks+2,0])
    for i in range(n_blocks):
        for j in range(n_blocks-i):
            sim.put(hexagon, [maxs[0]-3*n_blocks+2+3*i+j,j])    
    bid = n_blocks+1
    robot_order = [0,2,3,1,4,5,6]
    for r in range(n_blocks-2):
        sim.hold(robot_order[r],bid)
        bid+=n_blocks-r-1
        
    sim.ph_mod.set_max_forces(0,M=[-50,50],Fy = [-36*(n_blocks-1),36*(n_blocks-1)])
    fig,ax = gr.draw_grid(maxs,h=h,color='none',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, sim.grid,forces_bag=sim.ph_mod,draw_hold='dot',linewidth=2,draw_arrows=False)
    plt.tight_layout()
    plt.savefig(f'../graphics/colormaps/{name}_color_forces_strong_noforce.png')
    plt.show()
    
    
    sim.ph_mod.set_max_forces(0,M=[-50,50],Fy = [-6*(n_blocks-1),6*(n_blocks-1)])
    fig,ax = gr.draw_grid(maxs,h=h,color='none',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, sim.grid,forces_bag=sim.ph_mod,draw_hold='dot',linewidth=2,draw_arrows=False)
    plt.tight_layout()
    plt.savefig(f'../graphics/colormaps/{name}_color_forces_weak_noforce.png')
    plt.show()
    fig,ax = gr.draw_grid(maxs,h=h,color='none',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, sim.grid,draw_hold=True,linewidth=2)
    plt.tight_layout()
    plt.savefig(f'../graphics/colormaps/{name}_color_bid_noforce.png')
    plt.show()
    
    fig,ax = gr.draw_grid(maxs,h=h,color='none',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, sim.grid,draw_hold=True,linewidth=2,use_con=True)
    plt.tight_layout()
    plt.savefig(f'../graphics/colormaps/{name}_color_con_noforce.png')
    plt.show()
def colormap_robot(h,w, n_robots=2,vert=True,font_ratio =1/5,title_font=5,bar_aspect =1/20):
    # Create figure and adjust figure height to number of colormaps
    colorid = np.linspace(0,1,8)[:n_robots]
    colorid = np.vstack((colorid, colorid))
    if vert:
        colorid = np.repeat(np.arange(n_robots),100)
        colorid = np.tile(colorid, (int(n_robots*bar_aspect*100),1) )
        colorid = colorid.T
    fig, ax = plt.subplots(1, figsize=(w, h))
    #ax.set_title(f'Robots colormap', fontsize=title_font)
    ax.imshow(colorid, cmap=ListedColormap(gr.robot_colors),vmax=3)
    for i in range(n_robots):
        if vert:
            ax.set_yticks(np.arange(50,n_robots*100,100), labels=[f"Robot n°{i}" for i in range(n_robots)],fontsize=h/font_ratio,)
            ax.axes.get_xaxis().set_visible(False)
        else:
            ax.text((i+0.5)/n_robots,0.5, f"Robot n°{i}", va='center', ha='center', fontsize=h/font_ratio,
                    transform=ax.transAxes)
            ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(f'../graphics/colormaps/colormap_robots_{n_robots}.png')
    plt.show()
def colormap_region(h,w, n_region=4,vert=True,font_ratio =1/5,title_font=15,bar_aspect=1/20):
    # Create figure and adjust figure height to number of colormaps
    colorid = np.linspace(0,1,8)[:n_region]
    colorid = np.vstack((colorid, colorid))
    if vert:
        colorid = np.repeat(np.arange(n_region),100)
        colorid = np.tile(colorid, (int(n_region*bar_aspect*100),1) )
        colorid = colorid.T
    fig, ax = plt.subplots(1, figsize=(w, h))
    #ax.set_title(f'Region colormap', fontsize=title_font)
    ax.imshow(colorid, cmap=plt.cm.Pastel2,vmax=7)
    for i in range(n_region):
        if vert:
           ax.set_yticks(np.arange(50,n_region*100,100), labels=[f"Region n°{i}" for i in range(n_region)],fontsize=h/font_ratio,)
           ax.axes.get_xaxis().set_visible(False)
        else:
            ax.text((i+0.5)/n_region,0.5, f"Region n°{i}", va='center', ha='center', fontsize=h/font_ratio,
                    transform=ax.transAxes)
            ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(f'../graphics/colormaps/colormap_region_{n_region}.png')
    plt.show()
def colormap_forces(h,w,vert=False,font_ratio =1/5,bar_aspect =1/20):
    # Create figure and adjust figure height to number of colormaps
    
    if vert:
        colorid = np.linspace(1,0,256)
        colorid = np.tile(colorid, (int(256*bar_aspect),1) )
        colorid = colorid.T
    else:
        colorid = np.linspace(0,1,256)
        colorid = np.vstack((colorid, colorid))
    fig, ax = plt.subplots(1, figsize=(w, h))
    #ax.set_title(f'Maximum force colormap', fontsize=2*h/font_ratio)
    ax.imshow(colorid, cmap=plt.cm.plasma,vmax=1)
    ax.set_frame_on(False)
    if vert:
        ax.set_yticks([0,255], labels=["Max force","No force"],fontsize=h/font_ratio,)
        ax.axes.get_xaxis().set_visible(False)
    else:
        ax.set_xticks([0,255], labels=["No force", "Max force"],fontsize=h/font_ratio,)
        ax.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(f'../graphics/colormaps/colormap_Max_force.png')
    plt.show()
def colormap_bid(h,w,vert=False,font_ratio =1/5,bar_aspect =1/20):
    # Create figure and adjust figure height to number of colormaps
    if vert:
        colorid = np.linspace(1,0,256)
        colorid = np.tile(colorid, (int(256*bar_aspect),1) )
        colorid = colorid.T
    else:
        colorid = np.linspace(0,1,256)
        colorid = np.vstack((colorid, colorid))
    fig, ax = plt.subplots(1, figsize=(w, h))
    #ax.set_title(f'Maximum force colormap', fontsize=2*h/font_ratio)
    ax.imshow(colorid, cmap=plt.cm.turbo,vmax=1)
    ax.set_frame_on(False)
    if vert:
        ax.set_yticks([0,255], labels=["Last block","First block"],fontsize=h/font_ratio,)
        ax.axes.get_xaxis().set_visible(False)
    else:
        ax.set_xticks([0,255], labels=["First block", "Last block"],fontsize=h/font_ratio,)
        ax.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(f'../graphics/colormaps/colormap_bid.png')
    plt.show()
def colormap_arrows(h,w):
    # Create figure and adjust figure height to number of colormaps
    n_colors = 5
    colorid = np.arange(n_colors)
    colors = gr.force_colors
    fig, ax = plt.subplots(1, figsize=(w,h),subplot_kw={'projection': 'polar'})
    #ax.set_title(f'Forces arrow colormap', fontsize=2*h/font_ratio)
    ax.bar(x=np.arange(0,np.pi*2,np.pi*2/n_colors),
       bottom=0.9, height=0.5, width = np.pi*2/n_colors-0.05,
       color=colors, edgecolor='none', linewidth=1, align="edge")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(f'../graphics/colormaps/colormap_arrow_force.png')
    plt.show()
def color_arrows_example(h):
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.5)
    ground = Block([[0,0,0]],muc=0.5)
    sim = DiscreteSimulator([7,4], 1, [hexagon], 2, 30, 30)
    #sim.ph_mod.set_max_forces(0,Fx=[-Fxmax,Fxmax],M=[-0,0])
    sim.add_ground(ground, [1,0])
    sim.add_ground(ground, [6,0])
    sim.put_rel(hexagon, 0, 0, 0, 4,idconsup=0)
    sim.put_rel(hexagon, 0, 0, 1, 4,idconsup=0)
    sim.put_rel(hexagon, 0, 0, 2, 2,idconsup=0)
    
    fig,ax = gr.draw_grid([6,5],h=h,color='none',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, sim.grid,forces_bag=sim.ph_mod,draw_hold=False,linewidth=h)
    plt.tight_layout()
    plt.savefig(f'../graphics/forces/color_arrow_example.png')
    plt.show()
def build(sim,n_blocks=6):
    ground = Block([[0,0,1]],muc=0.7)
    hexagon = Block([[1,0,0],[0,1,1],[1,1,1],[1,1,0],[0,2,1],[0,1,0]])
    linkr = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]])
    linkl = Block([[1,0,1],[1,0,0],[0,1,1],[1,1,0],[1,1,1],[2,0,0]])
    linkh = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[1,2,1],[2,0,0]])
    #sim.ph_mod.set_max_forces(0,Fx=[-Fxmax,Fxmax],M=[-0,0])
    sim.add_ground(ground, [sim.grid.shape[0]-7,0])
    sim.add_ground(ground, [sim.grid.shape[0]-1,0])
    sim.put(hexagon, [sim.grid.shape[0]-7,0])
    sim.put(hexagon,[sim.grid.shape[0]-1,0])
    sim.put(linkr,[sim.grid.shape[0]-8,2])
    sim.put(linkl,[sim.grid.shape[0]-3,2])
    sim.put(hexagon,[sim.grid.shape[0]-7,3])
    if n_blocks>5:
        sim.put(linkh,[sim.grid.shape[0]-6,3])
    #sim.put(hexagon,[4,3])
    
    sim.hold(0,n_blocks-1)
    sim.hold(1,n_blocks)
def build_corner(sim,muc=0.5):
    ground = Block([[0,0,1]],muc=muc)
    hexagon = Block([[1,0,0],[0,1,1],[1,1,1],[1,1,0],[0,2,1],[0,1,0]],muc=muc)
    linkh = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[1,2,1],[2,0,0]],muc=muc)
    linkr = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=muc)
    linkl = Block([[1,0,1],[1,0,0],[0,1,1],[1,1,0],[1,1,1],[2,0,0]],muc=muc)
    sim.add_ground(ground, [sim.grid.shape[0]-1,0])
    sim.add_ground(ground, [sim.grid.shape[0]-5,0])
    sim.put_rel(linkl,0,0,0,0,blocktypeid=2,idconsup=0)
    sim.put_rel(linkh,1,0,1,0,blocktypeid=1)
    sim.put_rel(None,0,0,2,3,blocktypeid=3)
def build_torque(sim,n_blocks,muc=0.5):
    ground = Block([[0,0,1]],muc=muc)
    hexagon = Block([[1,0,0],[0,1,1],[1,1,1],[1,1,0],[0,2,1],[0,1,0]],muc=muc)
    linkh = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[1,2,1],[2,0,0]],muc=muc)
    sim.add_ground(ground, [sim.grid.shape[0]-1,0])
    sim.put_rel(hexagon,0,0,0,0,blocktypeid=0,idconsup=0)
    for i in range(1,n_blocks+1):
        if i%2==1:
            sim.put_rel(linkh,0,0,i,5,blocktypeid=1)
        else:
            sim.put_rel(hexagon,0,0,i,1,blocktypeid=0)
    sim.hold(0,n_blocks+1)
def build_line(sim,gap,start='hex',muc=0.5):
    #sim.ph_mod.safety_kernel=0
    ground = Block([[0,0,1]],muc=muc)
    hexagon = Block([[1,0,0],[0,1,1],[1,1,1],[1,1,0],[0,2,1],[0,1,0]],muc=muc)
    linkh = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[1,2,1],[2,0,0]],muc=muc)
    sim.add_ground(ground, [sim.grid.shape[0]-1-gap-1,0])
    sim.add_ground(ground, [sim.grid.shape[0]-1,0])
    
    if start=='hex':
        sim.put_rel(hexagon,0,0,0,0,blocktypeid=0,idconsup=1)
        i=1
        while np.any(sim.grid.min_dist > 1e-6):
            
            if i%2==1:
                sim.put_rel(linkh,0,0,i,5,blocktypeid=1)
            else:
                sim.put_rel(hexagon,0,0,i,1,blocktypeid=0)
            i+=1
        if gap % 3 == 0:
            sim.put_rel(linkh,0,0,i,5,blocktypeid=1)
        if gap%3 == 2:
            sim.put_rel(hexagon,0,0,i,1,blocktypeid=0)
def gen_counterintutive(h):
    muc=0.7
    ground = Block([[0,0,1]],muc=muc)
    hexagon = Block([[1,0,0],[0,1,1],[1,1,1],[1,1,0],[0,2,1],[0,1,0]],muc=muc)
    linkh = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[1,2,1],[2,0,0]],muc=muc)
    linkr = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=muc)
    linkl = Block([[1,0,1],[1,0,0],[0,1,1],[1,1,0],[1,1,1],[2,0,0]],muc=muc)
    sim = DiscreteSimulator([8,7], 1, [hexagon,linkh,linkl,linkr], 2, 30, 30)
    sim.add_ground(ground, [sim.grid.shape[0]-1,0])
    sim.put_rel(None, 0, 0, 0, 0, blocktypeid=0,idconsup=0)
    sim.put_rel(None, 0, 0, 1, 5, blocktypeid=0)
    sim.put_rel(None, 0, 0, 2, 5, blocktypeid=2)
    sim.put_rel(None, 1, 1, 3, 1, blocktypeid=2)
    sim.hold(0, 4)
    fig,ax = gr.draw_grid([8,5],h=h,color='none',linewidth=2,label_points = False)
    gr.fill_grid(ax, sim.grid,forces_bag=sim.ph_mod,linewidth=h)
    plt.tight_layout()
    plt.savefig(f'../graphics/validation/counter_intuitive_1.png')
def gen_visualisation_type(h,name,order_independant=False):
    hexagon = Block([[1,0,0],[0,1,1],[1,1,1],[1,1,0],[0,2,1],[0,1,0]])
    linkr = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]])
    linkl = Block([[1,0,1],[1,0,0],[0,1,1],[1,1,0],[1,1,1],[2,0,0]])
    linkh = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[1,2,1],[2,0,0]])
    sim = DiscreteSimulator([8,6], 1, [hexagon,linkr,linkl,linkh], 2, 30, 30)
    
    build(sim)
    
    # fig,ax = gr.draw_grid([8,5],h=h,color='none',linewidth=0.7,label_points = False)
    # gr.fill_grid(ax, sim.grid,forces_bag=sim.ph_mod,draw_hold=False,linewidth=1)
    if order_independant:
        gr.write_state_OI(sim.grid, h,scale=10,alpha=0.7)
        pass
    else:
        gr.write_state_OD(sim.grid, h,scale=10,alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'../graphics/visualisation/{name}.png')
    plt.show()
def gen_scenario(h,draw_forces=False,name="",turn=0,colormap='force',linewidth=1.5,draw_robot=True):
    maxs = [10,7]
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.7)
    ground = Block([[0,0,1]],muc=0.7)
    sim = DiscreteSimulator(maxs, 2, [hexagon], 2, 30, 30)
    sim.add_ground(ground,[9,0])
    sim.add_ground(ground, [3,0])
    fig,ax = gr.draw_grid(maxs,h=h,color='none',label_points = False)
    gr.fill_grid(ax,
                 sim.grid,
                 forces_bag=sim.ph_mod,
                 draw_arrows = draw_forces,
                 linewidth=linewidth)
    plt.savefig(f'../graphics/scenario/{name}empty_{colormap}.png')
    sim.put_rel(hexagon, 0, 0, 0, 0,idconsup=0)
    sim.put_rel(hexagon, 0, 0, 0, 0,idconsup=1)
    
    sim.put_rel(hexagon, 0, 0, 2, 4)
    sim.put_rel(hexagon, 0, 0, 3, 4)
    sim.hold(0,3)
    sim.hold(1,4)
    if colormap == 'force':
        use_con=False
        use_forces = True

    if colormap == 'con':
        use_con = True
        use_forces = False

    if colormap == 'id':
        use_forces = False
        use_con = False
        
        
    fig,ax = gr.draw_grid(maxs,h=h,color='none',label_points = False)
    gr.fill_grid(ax,
                 sim.grid,use_con = use_con,
                 use_forces = use_forces,
                 forces_bag=sim.ph_mod,
                 draw_arrows = draw_forces,
                 linewidth=linewidth)
    if draw_robot:
        for i in range(2):
            gr.draw_robot(ax, sim.grid, i, [sim.grid.shape[0]+1,-2])
    plt.tight_layout()
    plt.savefig(f'../graphics/scenario/{name}initial_{colormap}.png')
    plt.show()
    
    sim.leave(turn)
    fig,ax = gr.draw_grid(maxs,h=h,color='none',label_points = False)
    gr.fill_grid(ax,
                 sim.grid,use_con = use_con,
                 use_forces = use_forces,
                 forces_bag=sim.ph_mod,
                 draw_arrows = draw_forces,
                 linewidth=linewidth)
    
    action_args={'sideblock':0,'sidesup':0,'bid_sup':4,'side_ori':2,'idconsup':0}
    gr.draw_action_rel(ax,turn, 'Ph',hexagon,sim.grid,animated=False,multi=False,**action_args)
    if turn == 0:
        sim.put_rel(hexagon, 0, 0, 4, 2)
        sim.hold(0, 5)
    if draw_robot:
        for i in range(2):
            if i == turn:
                gr.draw_robot(ax, sim.grid, i, [sim.grid.shape[0]+1,-2],dash='--',actuator_pos=get_cm(hexagon.parts))
            else:
                gr.draw_robot(ax, sim.grid, i, [sim.grid.shape[0]+1,-2])
    
    plt.tight_layout()
    plt.savefig(f'../graphics/scenario/{name}action_{colormap}.png')
    plt.show()
    
        
    fig,ax = gr.draw_grid(maxs,h=h,color='none',label_points = False)
    gr.fill_grid(ax,
                 sim.grid,use_con = use_con,
                 use_forces = use_forces,
                 forces_bag=sim.ph_mod,
                 draw_arrows = draw_forces,
                 linewidth=linewidth)
    if draw_robot:
        for i in range(2):
            gr.draw_robot(ax, sim.grid, i, [sim.grid.shape[0]+1,-2])
    plt.tight_layout()
    plt.savefig(f'../graphics/scenario/{name}final_{colormap}.png')
    plt.show()
def gen_example_robots(h,rid,linewidth=3):
    
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]])
    ground =  Block([[1,0,1]])
    #link = Block([[3,0,0],[4,0,0],[3,1,1],[3,1,0],[2,2,1],[3,2,1]])
    fig,ax = gr.draw_grid([2,3],h=h,color='none',label_points = False)
    grid = Grid([2,3])
    grid.put(ground, [1,0], 0,floating=True)
    
    gr.draw_put_rel(ax,hexagon,
                     rid,
                     True,)
    gr.fill_grid(ax,grid,fixed_color=plt.cm.plasma(0))
    plt.tight_layout()
    plt.savefig('../graphics/robot_example/Ph.png')
    plt.show()
    
    fig,ax = gr.draw_grid([2,3],h=h,color='none',label_points = False)
    gr.fill_grid(ax,grid,fixed_color=plt.cm.plasma(0))
    gr.draw_put_rel(ax,hexagon,
                     rid,
                     True,)
    grid.put(hexagon, [1,0], 1,holder_rid=rid)
    
    gr.draw_robot(ax,grid,rid, [2.4,-1.5],dash='--')
    plt.tight_layout()
    plt.savefig('../graphics/robot_example/Ph_robot.png')
    plt.show()
    
    fig,ax = gr.draw_grid([2,3],h=h,color='none',label_points = False)
    
    #gr.draw_block(ax, hexagon,facecolor=plt.cm.plasma(0),edgecolor=plt.cm.plasma(0))
    gr.fill_grid(ax,grid,fixed_color=plt.cm.plasma(0))
    plt.tight_layout()
    plt.savefig('../graphics/robot_example/h.png')
    plt.show()
    fig,ax = gr.draw_grid([2,3],h=h,color='none',label_points = False)
    gr.fill_grid(ax,grid,fixed_color=plt.cm.plasma(0))
    gr.draw_robot(ax,grid,rid, [2.4,-1.5])
    plt.tight_layout()
    plt.savefig('../graphics/robot_example/h_robot.png')
    plt.show()
def gen_validation(h):
    muc = 0.5
    hexagon = Block([[1,0,0],[0,1,1],[1,1,1],[1,1,0],[0,2,1],[0,1,0]],muc=muc)
    linkh = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[1,2,1],[2,0,0]],muc=muc)
    linkr = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=muc)
    linkl = Block([[1,0,1],[1,0,0],[0,1,1],[1,1,0],[1,1,1],[2,0,0]],muc=muc)
    
    for gap in range(1,6):
        sim = DiscreteSimulator([gap+4,6], 1, [hexagon,linkh,linkl,linkr], 2, 30, 30)
        build_line(sim,gap,muc=muc)
        fig,ax = gr.draw_grid([gap+4,5],h=h,color='none',linewidth=2,label_points = False)
        gr.fill_grid(ax, sim.grid,forces_bag=sim.ph_mod,linewidth=h)
        plt.tight_layout()
        plt.savefig(f'../graphics/validation/width{gap}.png')
    
    n_blocks = 3
    sim = DiscreteSimulator([8,6], 1, [hexagon,linkh,linkl,linkr], 2, 30, 30)
    build_torque(sim,n_blocks,muc=muc)
    fig,ax = gr.draw_grid([8,5],h=h,color='none',linewidth=2,label_points = False)
    gr.fill_grid(ax, sim.grid,forces_bag=sim.ph_mod,linewidth=h)
    plt.tight_layout()
    plt.savefig(f'../graphics/validation/width_hold.png')
    sim = DiscreteSimulator([8,6], 1, [hexagon,linkh,linkl,linkr], 2, 30, 30)
    #build_line(sim,5,muc=muc)
    build_corner(sim)
    fig,ax = gr.draw_grid([8,6],h=h,color='none',linewidth=2,label_points = False)
    gr.fill_grid(ax, sim.grid,forces_bag=sim.ph_mod,linewidth=h)
    plt.tight_layout()
    plt.savefig(f'../graphics/validation/corner.png')
def gen_forces(h):
    gen_forces_no_robot(h,0.7,"equilibrium07")
    gen_forces_no_robot(h,0.5,"equilibrium05")
    gen_forces_no_robot(h,0.3,"equilibrium03")
    gen_forces_no_robot(h,0,"equilibrium0")
    gen_forces_no_robot_kernel(h,0,"kernel0")
    gen_forces_no_robot_kernel(h,0.2,"kernel02")
    gen_forces_no_robot_kernel(h,0.5,"kernel05")
    gen_forces_no_robot_kernel(h,0,"kernel0_unstable",unstable=True)
    gen_forces_no_robot_kernel(h,0.2,"kernel02_unstable",unstable=True)
    gen_forces_no_robot_kernel(h,0.5,"kernel05_unstable",unstable=True)
    for i in range(5):
        gen_forces_max(h,i,50,f"vert_force{i+1}")
    for i in range(6):
        gen_forces_max_diag(h,i,50,f"diag_force_50N_{i+1}",maxs=[8,8])
        gen_forces_max_diag(h,i,70,f"diag_force_70N_{i+1}",maxs=[8,8])
    color_arrows_example(h)
def gen_color_examples(h):
    gen_forces_colors(h,"stack",6)
def gen_colormaps(h):
    colormap_arrows(0.74*h/0.25,get_w(h,[6,5]))
    colormap_forces(h, h/1.7,font_ratio=3/12,vert = True)
    colormap_bid(h, h/1.7,font_ratio=3/12,vert = True)
    colormap_region(h,h/1.7,font_ratio=3/12,n_region=2)
    colormap_robot(h,  h*1.11,font_ratio=3/12,title_font=15,n_robots = 4)
    
def gen_visualisation(h):
    hexagon = Block([[1,0,0],[0,1,1],[1,1,1],[1,1,0],[0,2,1],[0,1,0]])
    linkr = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]])
    linkl = Block([[1,0,1],[1,0,0],[0,1,1],[1,1,0],[1,1,1],[2,0,0]])
    linkh = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[1,2,1],[2,0,0]])
    sim = DiscreteSimulator([8,6], 1, [hexagon,linkr,linkl,linkh], 2, 30, 30)
    
    build(sim)
    fig,ax = gr.draw_grid([8,6],h=h,color='k',linewidth=2,label_points = False)
    gr.fill_grid(ax, sim.grid,forces_bag=sim.ph_mod,draw_arrows=False,linewidth=h)
    plt.tight_layout()
    plt.savefig(f'../graphics/visualisation/rep.png')
    
    gen_visualisation_type(h,'OI',order_independant=True)
    gen_visualisation_type(h,'OD',order_independant=False)
def draw_mask(h,sim,rid,name='test'):
    mask = generate_mask_no_rot(sim, 1, sim.n_side_oriented, sim.n_side_oriented_sup, True, 3, 2, 'Ph', sim.type_id)
    selected, = np.nonzero(mask)
    mask_show = mask.astype(int)
    mask_show[selected[0]] = 2
    fig,ax=plt.subplots(1,1,figsize=(h/3,h))
    ax.imshow(np.tile(mask_show,(60,1)).T,cmap=plt.cm.RdYlGn,vmin=0,vmax=2,interpolation='none',aspect='auto')
    ax.set_frame_on(False)
    ax.set_yticks([selected[0]], labels=["chosen action"],fontsize=h*1.7)
    ax.axes.get_xaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(f'../graphics/base action set/{name}.png')
def gen_base_action_set(h):
    muc = 0.5
    hexagon = Block([[1,0,0],[0,1,1],[1,1,1],[1,1,0],[0,2,1],[0,1,0]],muc=muc)
    linkh = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[1,2,1],[2,0,0]],muc=muc)
    linkr = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=muc)
    linkl = Block([[1,0,1],[1,0,0],[0,1,1],[1,1,0],[1,1,1],[2,0,0]],muc=muc)
    grid_size = [5,5]
    sim = DiscreteSimulator(grid_size, 1, [hexagon,linkh,linkl,linkr], 2, 30, 30)
    sim.add_ground(Block([[0,0,1]],muc=muc), [grid_size[0]-2,0])
    fig,ax = gr.draw_grid(grid_size,h=h,color='none',label_points = False)
    
    gr.fill_grid(ax, sim.grid,forces_bag=sim.ph_mod)
    
    draw_mask(h,sim,1,name='mask_ground')
    sim.put_rel(hexagon,0,0,0,0,blocktypeid=0,idconsup=0)
    sim.hold(0, 1)
    gr.draw_put_rel(ax,hexagon,
                     1,
                     True,)
    plt.figure(fig)
    plt.tight_layout()
    plt.savefig(f'../graphics/base action set/ground.png')
    
    
    
    draw_mask(h,sim,1,name='mask_hex')
    fig,ax = gr.draw_grid(grid_size,h=h,color='none',label_points = False)
    gr.fill_grid(ax, sim.grid,draw_hold='dot',fixed_color=plt.cm.plasma(0))
    sim.put_rel(hexagon,0,0,1,0,blocktypeid=0,idconsup=0)
    gr.draw_put_rel(ax,hexagon,
                     1,
                     True,)
    plt.figure(fig)
    plt.tight_layout()
    plt.savefig(f'../graphics/base action set/hex.png')
    sim = DiscreteSimulator(grid_size, 1, [hexagon,linkh,linkl,linkr], 2, 30, 30)
    sim.add_ground(Block([[0,0,1]],muc=muc), [grid_size[0]-2,0])
    
    fig,ax = gr.draw_grid(grid_size,h=h,color='none',label_points = False)
   
    sim.put_rel(linkh,0,0,0,0,blocktypeid=1,idconsup=0)
    sim.hold(0, 1)
    gr.fill_grid(ax, sim.grid,draw_hold='dot',fixed_color=plt.cm.plasma(0))
    draw_mask(h,sim,1,name='mask_linkh')
    sim.put_rel(hexagon,0,0,1,0,blocktypeid=0,idconsup=0)
    
    gr.draw_put_rel(ax,hexagon,
                     1,
                     True,)
    plt.figure(fig)
    plt.tight_layout()
    plt.savefig(f'../graphics/base action set/linkh.png')
    
    sim = DiscreteSimulator(grid_size, 1, [hexagon,linkh,linkl,linkr], 2, 30, 30)
    sim.add_ground(Block([[0,0,1]],muc=muc), [grid_size[0]-2,0])
    fig,ax = gr.draw_grid(grid_size,h=h,color='none',label_points = False)
   
    sim.put_rel(linkl,0,0,0,0,blocktypeid=2,idconsup=0)
    sim.hold(0, 1)
    draw_mask(h,sim,1,name='mask_linkl')
    gr.fill_grid(ax, sim.grid,draw_hold='dot',fixed_color=plt.cm.plasma(0))
    sim.put_rel(hexagon,0,0,1,0,blocktypeid=0,idconsup=0)
    
    gr.draw_put_rel(ax,hexagon,
                     1,
                     True,)
    plt.figure(fig)
    plt.tight_layout()
    plt.savefig(f'../graphics/base action set/linkl.png')
    
    sim = DiscreteSimulator(grid_size, 1, [hexagon,linkh,linkl,linkr], 2, 30, 30)
    sim.add_ground(Block([[0,0,1]],muc=muc), [grid_size[0]-2,0])
    fig,ax = gr.draw_grid(grid_size,h=h,color='none',label_points = False)
   
    sim.put_rel(linkr,0,0,0,0,blocktypeid=3,idconsup=0)
    draw_mask(h,sim,1,name='mask_linkr')
    sim.hold(0, 1)
    gr.fill_grid(ax, sim.grid,draw_hold='dot',fixed_color=plt.cm.plasma(0))
    sim.put_rel(hexagon,0,0,1,0,blocktypeid=0,idconsup=0)
    gr.draw_put_rel(ax,hexagon,
                     1,
                     True,)
    plt.figure(fig)
    plt.tight_layout()
    plt.savefig(f'../graphics/base action set/linkr.png')
    
def draw_dist(h):
    muc = 0.5
    hexagon = Block([[1,0,0],[0,1,1],[1,1,1],[1,1,0],[0,2,1],[0,1,0]],muc=muc)
    linkh = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[1,2,1],[2,0,0]],muc=muc)
    linkr = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=muc)
    linkl = Block([[1,0,1],[1,0,0],[0,1,1],[1,1,0],[1,1,1],[2,0,0]],muc=muc)
    grid_size = [10,7]
    sim = DiscreteSimulator(grid_size, 1, [hexagon,linkh,linkl,linkr], 2, 30, 30)
    build(sim,5)
    fig,ax = gr.draw_grid(grid_size,h=h,color='none',label_points = False)
    gr.fill_grid(ax, sim.grid,draw_hold='dot',use_con=True)
    x1,y1 = [4,4]
    x2,y2 = [6,3]
    art = FancyArrowPatch((x1+y1/2,y1*np.sqrt(3)/2), (x2+y2/2, y2*np.sqrt(3)/2), arrowstyle='<|-|>', mutation_scale=20,color='b',label=r'$d^*(g_1,g_2)$')
    ax.add_artist(art)
    x1,y1 = [4,0]
    x2,y2 = [9,0]
    art = FancyArrowPatch((x1+y1/2,y1*np.sqrt(3)/2), (x2+y2/2, y2*np.sqrt(3)/2), arrowstyle='<|-|>', mutation_scale=20,color='k',label=r'$d(g_1,g_2)$')
    ax.add_artist(art)
    ax.legend(fontsize=h*2.2)
    plt.tight_layout()
    plt.savefig("../graphics/distances/dist.png")
    plt.savefig("../graphics/distances/dist.png")
if __name__ == "__main__":
    # gen_grid(h=10)
    # gen_blocks(h=10)
    # gen_forces(h=8)
    # gen_validation(h=8)
    # gen_counterintutive(h=8)
    # gen_visualisation(8)
    # gen_colormaps(8)
    #gen_color_examples(8)
    # gen_structures(h=20)
    # gen_scenario(8,draw_forces=True,name="arrow_",turn=0,linewidth=4,draw_robot=False)
    # gen_scenario(8,draw_forces=True,name="arrow_fail_",turn=1,linewidth=4,draw_robot=False)
    # gen_scenario(8,draw_forces=False,name="robot_",turn=0,linewidth=4,draw_robot=True)
    gen_scenario(8,draw_forces=False,name="robot_fail_",turn=1,linewidth=4,draw_robot=True)
    # gen_example_robots(8,0)
    #gen_base_action_set(h=12)
    # draw_dist(8)