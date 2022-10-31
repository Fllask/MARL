# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 13:12:51 2022

@author: valla
"""

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle,Wedge,Polygon
import numpy as np

from Blocks import discret_block as Block
from Blocks import Grid

from matplotlib import animation
from IPython.display import HTML


s3 =np.sqrt(3)
base = np.array([[1,0.5],[0,s3/2]])
def draw_grid(maxs,label_points=False,steps=1,color='darkslategrey',h=6,linewidth=1):
    fig,ax = plt.subplots(1,1,figsize=(h,h*(maxs[0]+maxs[1]*0.5)/((maxs[1]-0.5)*s3)))
    xlim = [-0.5,maxs[0]+0.5*maxs[1]]
    ylim = [-s3/2,maxs[1]*s3/2]
    ax.set_aspect('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    #draw the horizontal lines
    y = np.arange(ylim[0],ylim[1],s3/2*steps)
    y = np.stack([y,y])
    ax.plot(np.reshape(xlim,(2,1)),y,color=color,linewidth=linewidth)
    #draw the 60 degrees lines
    y = np.arange(ylim[0]-(xlim[1])*s3,ylim[1]-(xlim[0])*s3,s3*steps)
    y = y-y[np.argmin(abs(y))]
    y = np.stack([y+xlim[0]*s3,y+xlim[1]*s3])    
    
    ax.plot(np.reshape(xlim,(2,1)),y,color=color,linewidth=linewidth)
    #draw the -60 degree lines
    y = np.arange(ylim[0]+(xlim[0])*s3,ylim[1]+(xlim[1])*s3,s3*steps)
    y = y-y[np.argmin(abs(y))]
    y = np.stack([y-xlim[0]*s3,y-xlim[1]*s3])    
    ax.plot(np.reshape(xlim,(2,1)),y,color=color,linewidth=linewidth)
    if label_points:
        for x0 in range(maxs[0]):
            for x1 in range(maxs[1]):
                ax.text(x0+x1/2+0.5,x1*s3/2+0.2,f"({x0},{x1},0)",
                        va='center',
                        ha='center',
                        fontsize=h/2)
                ax.text(x0+x1/2+0.5,x1*s3/2-0.2,f"({x0},{x1},1)",
                        va='center',
                        ha='center',
                        fontsize=h/2)
    ax.axis('off')
    # ax.set_yticklabels(np.arange(-xlim[0],np.floor((ylim[1]-ylim[0])/np.sqrt(3))-xlim[0]+1))
    return fig,ax
def draw_action(ax,rid,action,blocktype,animated=False,**action_args):
    if action == 'Ph':
        art = draw_put(ax,rid,blocktype,hold=True,**action_args,animated=animated)
    elif action == 'Pl':
        art=draw_put(ax,rid,blocktype,hold=False,**action_args,animated=animated)
    elif action == 'H':
        art=draw_hold(ax,rid,**action_args,animated=animated)
    elif action == 'R':
        art=draw_remove(ax,rid,**action_args,animated=animated)
    elif action == 'L':
        art=draw_leave(ax,rid,animated=animated)
    return art
def draw_put(ax,rid,blocktype,hold,pos,ori,blocktypeid,animated=False):
    arts = draw_block(ax, blocktype, pos, ori, color=plt.cm.Set1(rid),animated=animated)
    if hold:
        #the block was already moved by draw_block
        parts = blocktype.parts
        sides = np.concatenate([np.tile(parts,(3,1)),np.repeat(np.arange(3),parts.shape[0])[...,None]],axis=1)
        for side in sides:
            arts+= draw_side(ax, side,color='k',animated=animated)
    return arts
        
def draw_hold(ax,rid,bid,animated=False):
    art = ax.text(0,ax.get_ylim()[1]-1,f"Robot {rid} tried to grab block {bid}",color = plt.cm.Set1(rid),animated=animated)
    return [art]
def draw_remove(ax,rid,bid,animated=False):
    art = ax.text(0,ax.get_ylim()[1]-1,f"Robot {rid} tried to remove block {bid}",color = plt.cm.Set1(rid),animated=animated)
    return [art]
def draw_leave(ax,rid,animated=False):
    art = ax.text(0,ax.get_ylim()[1]-1,f"Robot {rid} tried to leave",color = plt.cm.Set1(rid),animated=animated)
    return [art]
# def draw_block(ax,block,label_corners = False,**kwarg):
#     triangle = np.array([[-0.5,np.sqrt(3)/2],[0,-0.25+np.sqrt(3)/2],[0.5,0.25-np.sqrt(3)/2]])
#     for p in block.parts:
#         if p[0]%2==0:
#             art = Polygon(p+triangle)
#             ax.add_artist(art)
#         else:
#             art = Polygon(p-triangle)
#             ax.add_artist(art)
def rad(alpha):
    return alpha*180/np.pi

def fill_triangle(ax,coord,animated=False,**colorkw):
    coord = np.array(coord)
    
    triangle = np.tile([[0,0],[0.5,np.sqrt(3)/2],[1,0]],(coord.shape[0],1,1))
    down = coord[:,2]==1
    triangle[down,:,1]=-triangle[down,:,1]
    xycoord = (base @ coord[:,:-1].T).T
    
    art = [Polygon(xycoord[i]+triangle[i],animated=animated,**colorkw) for i in range(coord.shape[0])]
    [ax.add_artist(a) for a in art]
    return art
def draw_side(ax,coord,color=None,animated=False,linewidth=2):
    if coord[3]==0:
        p1 = coord[:2]
        p2 = coord[:2]+[1,0]
    elif coord[3]==1:
        p1 = coord[:2]+[1,0]-[coord[2],0]
        p2 = coord[:2]+[0,1]+[coord[2],-2*coord[2]]
    else:
        p1 = coord[:2]+[coord[2],0]
        p2 = coord[:2]+[coord[2],(-2*coord[2]+1)]
    if color is None:
        if coord[2]==1:
            color = 'y'
        else:
            color = 'g'
    p1xy = base @ p1
    p2xy = base @ p2
    art = ax.plot([p1xy[0],p2xy[0]],[p1xy[1],p2xy[1]],color=color,alpha=1,linewidth=linewidth,animated=animated)
    return art
        
def fill_grid(ax,grid,draw_neigh =False,use_con=False,animated=False,draw_hold = True, ground_color='darkslategrey'):
    ids = np.unique(grid.occ)
    arts = []
    if animated:
        for i in ids:
            if i == -1:
                continue
            if i ==0:
                coords = np.array(np.where(grid.occ==i))
                arts += fill_triangle(ax, coords.T,color=ground_color,animated=animated)
            else:
                coords = np.array(np.where(grid.occ==i))
                if grid.connection[coords[0,0],coords[1,0],coords[2,0]] == 0:
                    color = plt.cm.Pastel2(0)
                elif grid.connection[coords[0,0],coords[1,0],coords[2,0]] == 1:
                    color = plt.cm.Pastel2(1)
                else:
                    color = plt.cm.Pastel2(3)
                arts +=fill_triangle(ax, coords.T,color=color,animated=animated)
            
        coords=np.nonzero(grid.neighbours!=-1)
        coords = np.array(coords)
        for i in range(coords.shape[1]):
            arts+=draw_side(ax,coords[:,i],color = 'k')
        if draw_hold:
            for i in np.unique(grid.hold):
                if i ==-1:
                    continue
                coords=np.nonzero(grid.hold==i)
                coords = np.array(coords).T
                coords = np.concatenate([np.tile(coords,(3,1)),np.repeat(np.arange(3),coords.shape[0])[...,None]],axis=1)
                for coord in coords:
                    arts+=draw_side(ax, coord,color=plt.cm.Set1(i),animated=animated)
    else:
            
        for i in ids:
            if i == -1:
                continue
            if i ==0:
                coords = np.array(np.where(grid.occ==i))
                arts += fill_triangle(ax, coords.T,color=ground_color,animated=animated)
            else:
                coords = np.array(np.where(grid.occ==i))
                if use_con:
                    if grid.connection[coords[0,0],coords[1,0],coords[2,0]] == 0:
                        cmap = plt.cm.summer
                    elif grid.connection[coords[0,0],coords[1,0],coords[2,0]] == 1:
                        cmap = plt.cm.winter
                    else:
                        cmap = plt.cm.autumn
                    arts +=fill_triangle(ax, coords.T,color=cmap(i/np.max(ids)),animated=animated)
                else:
                    arts +=fill_triangle(ax, coords.T,color=plt.cm.turbo(i/np.max(ids)),animated=animated)
        if draw_neigh:
            coords=np.nonzero(grid.neighbours!=-1)
            coords = np.array(coords)
            for i in range(coords.shape[1]):
                arts+=draw_side(ax,coords[:,i])
        if draw_hold:
            for i in np.unique(grid.hold):
                if i ==-1:
                    continue
                coords=np.nonzero(grid.hold==i)
                coords = np.array(coords).T
                coords = np.concatenate([np.tile(coords,(3,1)),np.repeat(np.arange(3),coords.shape[0])[...,None]],axis=1)
                for coord in coords:
                    arts+=draw_side(ax, coord,color=plt.cm.Set1(i),animated=animated)
                    
    return arts
def animate(fig,arts_list,sperframe= 0.1):
    ani = animation.ArtistAnimation(fig, arts_list, interval=sperframe*1000, blit=True)
    HTML(ani.to_jshtml())
    return ani
def draw_block(ax,block, pos,ori,draw_neigh=False,animated=False, **colorkw):
    block.turn(ori)
    block.move(pos)
    arts = fill_triangle(ax, block.parts,animated=animated,**colorkw)
    
    if draw_neigh:
        for s in block.neigh:
            arts+=draw_side(ax,s,color='k',animated=animated)
    return arts
def save_anim(ani,name='animation',ext = 'html'):
    if ext == 'html':
        file = open(f"{name}.html","w")
        file.write(ani.to_jshtml())
        file.close()
    elif ext == 'gif':
        writergif = animation.PillowWriter(fps=3) 
        ani.save(f"{name}.gif", writer=writergif)

if __name__ == "__main__":
    print("Start test")
    plt.close()
    
    maxs = [9,6]
    grid = Grid(maxs)
    fig,ax = draw_grid(maxs,steps=1,color='k',label_points=True,h=30,linewidth=0.3)
    t = Block([[0,0,1],[0,0,0]])
    ground = Block([[0,0,0],[2,0,0],[6,0,0],[8,0,0]]+[[i,0,1] for i in range(0,maxs[0])])
    #ground = Block([[0,0,0],[maxs[0]-1,0,0]]+[[i,0,1] for i in range(0,maxs[0])],muc=0.7)
    hinge = Block([[1,0,0],[0,1,1],[1,1,1],[1,1,0],[0,2,1],[0,1,0]])
    link = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]])
    
    grid.put(ground,[0,0],0,1,floating=True)
    grid.put(hinge,[1,0],0,2,holder_rid=2)
    grid.put(link,[0,2],0,3,holder_rid=1)
    grid.put(hinge,[1,3],0,4,holder_rid=0)
    grid.put(link,[1,5],5,5)
    grid.put(hinge,[4,3],0,6)
    grid.put(link,[5,3],4,7)
    grid.put(hinge,[7,0],0,8)
    grid.put(link,[4,5],0,2,floating=True)
        
    fill_grid(ax,grid,draw_neigh=False)
    #
    #draw_block(ax,hinge,[4,4],color = 'none',draw_neigh = True)
    # draw_block(ax, link, [4,5],draw_neigh = True,color = 'r')
    a = 199
    #draw_block(ax,t,[4,4],color = 'yellowgreen',draw_neigh = True)
    #draw_block(ax,t,[4,4],color = 'darkred',draw_neigh = True)
    plt.show()
    print("End test")