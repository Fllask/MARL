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
import numpy as np
def gen_grid(h=10):
    fig,ax=gr.draw_grid([5,5],label_points=True,steps=1,color='k',h=h,linewidth=1)
    fig.tight_layout()
    plt.savefig('../graphics/grid1.pdf')  
def gen_blocks(h=20):
    fig,ax = gr.draw_grid([6,3],h=h,color='k',linewidth=0.8,label_points = False)
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]])
    link = Block([[3,0,0],[4,0,0],[3,1,1],[3,1,0],[2,2,1],[3,2,1]])
    ground = Block([[5,1,1]])
    gr.draw_block(ax, hexagon, color=plt.cm.plasma(0))
    gr.draw_block(ax, link, color=plt.cm.plasma(0))
    gr.draw_block(ax, ground, color="darkslategrey")
    plt.tight_layout()
    plt.savefig('../graphics/blocks.pdf')
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
    plt.savefig('../graphics/structure/tower.pdf')  
    
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
    plt.savefig('../graphics/structure/small_arch.pdf')
    
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
    plt.savefig('../graphics/structure/big_arch.pdf')
    
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
    plt.savefig('../graphics/structure/disclocated.pdf')
    
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
    plt.savefig('../graphics/structure/asymetry.pdf')
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
    plt.savefig('../graphics/structure/multiple.pdf')
    plt.show()
def gen_forces_no_robot(h,muc,name):
    maxs=[10,10]
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=muc)
    linkr = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=muc) 
    linkl = Block([[0,0,0],[0,1,1],[1,0,0],[0,1,0],[0,0,1],[-1,1,1]],muc=muc) 
    linkh = Block([[1,0,0],[2,0,0],[1,1,1],[1,1,0],[0,2,1],[1,2,1]],muc=muc)
    
    ground = Block([[0,0,1]],muc=0.7)
    sim = DiscreteSimulator(maxs, 1, [hexagon,linkr,linkl,linkh], 1, 30, 30)
    sim.ph_mod.set_max_forces(0,Fx=[-50,50])
    sim.add_ground(ground, [6,1])
    sim.put_rel(hexagon, 0, 0, 0, 0,idconsup=0)
    sim.put_rel(linkr, 0, 0, 1, 0,idconsup=0)
    sim.hold(0,3)
    fig,ax = gr.draw_grid(maxs,h=h,color='none',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, sim.grid,forces_bag=sim.ph_mod,draw_hold=False)
    plt.tight_layout()
    plt.savefig(f'../graphics/forces/{name}.pdf')
    plt.show()
def gen_forces_no_robot_kernel(h,kernel,name,unstable=False):
    maxs=[10,10]
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.7)
    linkr = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=0.7) 
    linkl = Block([[0,0,0],[0,1,1],[1,0,0],[0,1,0],[0,0,1],[-1,1,1]],muc=0.7) 
    linkh = Block([[1,0,0],[2,0,0],[1,1,1],[1,1,0],[0,2,1],[1,2,1]],muc=0.7)
    
    ground = Block([[0,0,1]],muc=0.7)
    sim = DiscreteSimulator(maxs, 1, [hexagon,linkr,linkl,linkh], 1, 30, 30)
    sim.ph_mod.safety_kernel=kernel
    sim.ph_mod.set_max_forces(0,Fx=[-50,50])
    sim.add_ground(ground, [6,1])
    sim.put_rel(hexagon, 0, 0, 0, 0,idconsup=0)
    if unstable:
        sim.put_rel(linkh, 0, 0, 1, 0,idconsup=0)
    else:
        sim.put_rel(linkr, 0, 0, 1, 0,idconsup=0)
    sim.hold(0,3)
    fig,ax = gr.draw_grid(maxs,h=h,color='none',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, sim.grid,forces_bag=sim.ph_mod,draw_hold=False)
    plt.tight_layout()
    plt.savefig(f'../graphics/forces/{name}.pdf')
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
    plt.savefig(f'../graphics/forces/{name}.pdf')
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
    plt.savefig(f'../graphics/forces/{name}.pdf')
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
    for r in range(n_blocks-2):
        sim.hold(r,bid)
        bid+=n_blocks-r-1
        
    sim.ph_mod.set_max_forces(0,M=[-50,50],Fy = [-36*(n_blocks-1),36*(n_blocks-1)])
    fig,ax = gr.draw_grid(maxs,h=h,color='none',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, sim.grid,forces_bag=sim.ph_mod,draw_hold=True,linewidth=2,draw_forces=False)
    plt.tight_layout()
    plt.savefig(f'../graphics/colormaps/{name}_color_forces_strong_noforce.pdf')
    plt.show()
    
    
    sim.ph_mod.set_max_forces(0,M=[-50,50],Fy = [-6*(n_blocks-1),6*(n_blocks-1)])
    fig,ax = gr.draw_grid(maxs,h=h,color='none',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, sim.grid,forces_bag=sim.ph_mod,draw_hold=True,linewidth=2,draw_forces=False)
    plt.tight_layout()
    plt.savefig(f'../graphics/colormaps/{name}_color_forces_weak_noforce.pdf')
    plt.show()
    fig,ax = gr.draw_grid(maxs,h=h,color='none',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, sim.grid,draw_hold=True,linewidth=2)
    plt.tight_layout()
    plt.savefig(f'../graphics/colormaps/{name}_color_bid_noforce.pdf')
    plt.show()
    
    fig,ax = gr.draw_grid(maxs,h=h,color='none',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, sim.grid,draw_hold=True,linewidth=2,use_con=True)
    plt.tight_layout()
    plt.savefig(f'../graphics/colormaps/{name}_color_con_noforce.pdf')
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
    ax.imshow(colorid, cmap=plt.cm.Set2,vmax=7)
    for i in range(n_robots):
        if vert:
            ax.set_yticks(np.arange(50,n_robots*100,100), labels=[f"Robot n°{i}" for i in range(n_robots)],fontsize=h/font_ratio,)
            ax.axes.get_xaxis().set_visible(False)
        else:
            ax.text((i+0.5)/n_robots,0.5, f"Robot n°{i}", va='center', ha='center', fontsize=h/font_ratio,
                    transform=ax.transAxes)
            ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(f'../graphics/colormaps/colormap_robots_{n_robots}.pdf')
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
    plt.savefig(f'../graphics/colormaps/colormap_region_{n_region}.pdf')
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
    plt.savefig(f'../graphics/colormaps/colormap_Max_force.pdf')
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
    plt.savefig(f'../graphics/colormaps/colormap_bid.pdf')
    plt.show()
def colormap_arrows(h,font_ratio =1/3):
    # Create figure and adjust figure height to number of colormaps
    n_colors = 8
    colorid = np.arange(n_colors)
    colors = plt.cm.Pastel1(colorid)
    fig, ax = plt.subplots(1, figsize=(h, h),subplot_kw={'projection': 'polar'})
    #ax.set_title(f'Forces arrow colormap', fontsize=2*h/font_ratio)
    ax.bar(x=np.arange(0,np.pi*2,np.pi*2/n_colors),
       bottom=0.9, height=0.5, width = np.pi*2/n_colors-0.05,
       color=colors, edgecolor='none', linewidth=1, align="edge")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(f'../graphics/colormaps/colormap_arrow_force.pdf')
    plt.show()
def gen_scenario(h,draw_forces=False,name="",turn=0):
    maxs = [10,7]
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.7)
    ground = Block([[0,0,1]],muc=0.7)
    sim = DiscreteSimulator(maxs, 2, [hexagon], 2, 30, 30)
    sim.add_ground(ground,[9,0])
    sim.add_ground(ground, [3,0])
    
    sim.put_rel(hexagon, 0, 0, 0, 0,idconsup=0)
    sim.put_rel(hexagon, 0, 0, 0, 0,idconsup=1)
    
    sim.put_rel(hexagon, 0, 0, 2, 4)
    sim.put_rel(hexagon, 0, 0, 3, 4)
    sim.hold(0,3)
    sim.hold(1,4)
    
    fig,ax = gr.draw_grid(maxs,h=h,color='none',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, sim.grid,forces_bag=sim.ph_mod,draw_hold=not draw_forces,draw_arrows = draw_forces, linewidth=1)
    plt.tight_layout()
    plt.savefig(f'../graphics/scenario/{name}initial_forces.pdf')
    plt.show()
    
    fig,ax = gr.draw_grid(maxs,h=h,color='none',linewidth=1.5,label_points = False)
    gr.fill_grid(ax, sim.grid,forces_bag=sim.ph_mod,draw_hold = not draw_forces,linewidth=1,draw_arrows = draw_forces)
    action_args={'sideblock':0,'sidesup':0,'bid_sup':4,'side_ori':2,'idconsup':0}
    gr.draw_action_rel(ax,turn, 'Ph',hexagon,sim.grid,animated=False,multi=False,**action_args)
    plt.tight_layout()
    plt.savefig(f'../graphics/scenario/{name}action_force.pdf')
    plt.show()
    
    fig,ax = gr.draw_grid(maxs,h=h,color='none',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, sim.grid,use_con=True,draw_hold = not draw_forces,draw_arrows = draw_forces,linewidth=1.5,forces_bag=sim.ph_mod)
    plt.tight_layout()
    plt.savefig(f'../graphics/scenario/{name}initial_con.pdf')
    plt.show()
    
    fig,ax = gr.draw_grid(maxs,h=h,color='none',linewidth=1.5,label_points = False)
    gr.fill_grid(ax, sim.grid,use_con=True,draw_hold = not draw_forces,linewidth=1.5,draw_arrows = draw_forces,forces_bag=sim.ph_mod)
    action_args={'sideblock':0,'sidesup':0,'bid_sup':4,'side_ori':2,'idconsup':0}
    gr.draw_action_rel(ax,turn, 'Ph',hexagon,sim.grid,animated=False,multi=False,**action_args)
    plt.tight_layout()
    plt.savefig(f'../graphics/scenario/{name}action_con.pdf')
    plt.show()
    
    fig,ax = gr.draw_grid(maxs,h=h,color='none',linewidth=1.5,label_points = False)
    gr.fill_grid(ax, sim.grid,use_forces = False,draw_hold = not draw_forces,linewidth=1.5,draw_arrows = draw_forces,forces_bag=sim.ph_mod)
    plt.tight_layout()
    plt.savefig(f'../graphics/scenario/{name}initial_id.pdf')
    plt.show()
    
    fig,ax = gr.draw_grid(maxs,h=h,color='none',linewidth=1.5,label_points = False)
    gr.fill_grid(ax, sim.grid,use_forces = False,draw_hold = not draw_forces,linewidth=1.5,draw_arrows = draw_forces,forces_bag=sim.ph_mod)
    action_args={'sideblock':0,'sidesup':0,'bid_sup':4,'side_ori':2,'idconsup':0}
    gr.draw_action_rel(ax,turn, 'Ph',hexagon,sim.grid,animated=False,multi=False,**action_args)
    plt.tight_layout()
    plt.savefig(f'../graphics/scenario/{name}action_id.pdf')
    plt.show()
    if turn == 0:
        sim.put_rel(hexagon, 0, 0, 4, 2)
        sim.hold(0, 5)
    else:
        sim.leave(1)
    fig,ax = gr.draw_grid(maxs,h=h,color='none',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, sim.grid,forces_bag=sim.ph_mod,draw_hold = True,linewidth=1,draw_arrows = draw_forces)
    plt.tight_layout()
    plt.savefig(f'../graphics/scenario/{name}final_forces.pdf')
    plt.show()
    
    fig,ax = gr.draw_grid(maxs,h=h,color='none',linewidth=0.7,label_points = False)
    gr.fill_grid(ax, sim.grid,use_con=True,draw_hold = not draw_forces,linewidth=1.5,draw_arrows = draw_forces,forces_bag=sim.ph_mod)
    plt.tight_layout()
    plt.savefig(f'../graphics/scenario/{name}final_con.pdf')
    plt.show()
    
    fig,ax = gr.draw_grid(maxs,h=h,color='none',linewidth=1.5,label_points = False)
    gr.fill_grid(ax, sim.grid,use_forces = False, draw_hold = not draw_forces,linewidth=1.5,draw_arrows = draw_forces,forces_bag=sim.ph_mod)
    plt.tight_layout()
    plt.savefig(f'../graphics/scenario/{name}final_id.pdf')
    plt.show()
def gen_example_robots(h,rid,linewidth=3):
    
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]])
    #link = Block([[3,0,0],[4,0,0],[3,1,1],[3,1,0],[2,2,1],[3,2,1]])
    fig,ax = gr.draw_grid([2,3],h=h,color='none',linewidth=0.8,label_points = False)
    gr.draw_block(ax, hexagon, color=plt.cm.Set2(rid))
    coords = np.concatenate([np.tile(hexagon.parts,(3,1)),np.repeat(np.arange(3),hexagon.parts.shape[0])[...,None]],axis=1)
    for coord in coords:
        gr.draw_side(ax, coord,color='k',linewidth=linewidth)
    plt.tight_layout()
    plt.savefig('../graphics/robot_example/Ph.pdf')
    plt.show()
    fig,ax = gr.draw_grid([2,3],h=h,color='none',linewidth=0.8,label_points = False)
    gr.draw_block(ax, hexagon,facecolor='k',edgecolor = 'w')
    for coord in coords:
        gr.draw_side(ax, coord,color=plt.cm.Set2(rid),linewidth=linewidth)
    plt.tight_layout()
    plt.savefig('../graphics/robot_example/h.pdf')
    plt.show()
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
    
    for i in range(9):
        gen_forces_max(h,i,50,f"vert_force{i}")
    for i in range(9):
        gen_forces_max_diag(h,i,50,f"diag_force_50N_{i}")
        gen_forces_max_diag(h,i,70,f"diag_force_70N_{i}")
def gen_color_examples(h):
    gen_forces_colors(h,"stack",7)
def gen_colormaps(h):
    colormap_arrows(h*3)
    colormap_forces(h, h/1.7,font_ratio=3/12,vert = True)
    colormap_bid(h, h/1.7,font_ratio=3/12,vert = True)
    colormap_region(h,h/1.7,font_ratio=3/12)
    colormap_robot(h,  h*0.8,font_ratio=3/12,title_font=15,n_robots = 6)

if __name__ == "__main__":
    gen_grid(h=10)
    #gen_blocks(h=20)
    # gen_forces(h=8)
    #gen_colormaps(8)
    #gen_color_examples(8)
    #gen_structures(h=20)
    #gen_actions(h=8)
    #gen_scenario(8,draw_forces=True,name="arrows_",turn=0)
    #gen_example_robots(8,0)