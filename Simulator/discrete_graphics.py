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
s3 =np.sqrt(3)
base = np.array([[1,0.5],[0,s3/2]])
def draw_grid(maxs,label_points=False,steps=1,color='darkslategrey',h=6):
    fig,ax = plt.subplots(1,1,figsize=(h,h*(maxs[0]+maxs[1]*0.5)/((maxs[1]-0.5)*s3)))
    xlim = [-0.5,maxs[0]+0.5*maxs[1]]
    ylim = [-s3/2,maxs[1]*s3/2]
    ax.set_aspect('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    #draw the horizontal lines
    y = np.arange(ylim[0],ylim[1],s3/2*steps)
    y = np.stack([y,y])
    ax.plot(np.reshape(xlim,(2,1)),y,color=color)
    #draw the 60 degrees lines
    y = np.arange(ylim[0]-(xlim[1])*s3,ylim[1]-(xlim[0])*s3,s3*steps)
    y = y-y[np.argmin(abs(y))]
    y = np.stack([y+xlim[0]*s3,y+xlim[1]*s3])    
    
    ax.plot(np.reshape(xlim,(2,1)),y,color=color)
    #draw the -60 degree lines
    y = np.arange(ylim[0]+(xlim[0])*s3,ylim[1]+(xlim[1])*s3,s3*steps)
    y = y-y[np.argmin(abs(y))]
    y = np.stack([y-xlim[0]*s3,y-xlim[1]*s3])    
    ax.plot(np.reshape(xlim,(2,1)),y,color=color)
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

def fill_triangle(ax,coord,**colorkw):
    coord = np.array(coord)
    
    triangle = np.tile([[0,0],[0.5,np.sqrt(3)/2],[1,0]],(coord.shape[0],1,1))
    down = coord[:,2]==1
    triangle[down,:,1]=-triangle[down,:,1]
    xycoord = (base @ coord[:,:-1].T).T
    
    art = [Polygon(xycoord[i]+triangle[i],**colorkw) for i in range(coord.shape[0])]
    [ax.add_artist(a) for a in art]
def draw_side(ax,coord):
    if coord[3]==0:
        p1 = coord[:2]
        p2 = coord[:2]+[1,0]
    elif coord[3]==1:
        p1 = coord[:2]+[1,0]-[coord[2],0]
        p2 = coord[:2]+[0,1]+[coord[2],-2*coord[2]]
    else:
        p1 = coord[:2]+[coord[2],0]
        p2 = coord[:2]+[coord[2],(-2*coord[2]+1)]
    if coord[2]==1:
        color = 'y'
    else:
        color = 'g'
    p1xy = base @ p1
    p2xy = base @ p2
    ax.plot([p1xy[0],p2xy[0]],[p1xy[1],p2xy[1]],color=color,alpha=0.5,linewidth=10)
        
def fill_grid(ax,grid,draw_neigh =False):
    ids = np.unique(grid.occ)
    for i in ids:
        if i == 0:
            continue
        else:
            coords = np.array(np.where(grid.occ==i))
            fill_triangle(ax, coords.T,color=plt.cm.turbo(i/np.max(ids)))
    if draw_neigh:
        coords=np.nonzero(grid.neighbours)
        coords = np.array(coords)
        for i in range(coords.shape[1]):
            draw_side(ax,coords[:,i])
            
    
def draw_block(ax,block, pos,draw_neigh=False, **colorkw):
    block.move(pos)
    fill_triangle(ax, block.parts,**colorkw)
    
    if draw_neigh:
        for s in block.neigh:
            draw_side(ax,s)
            

if __name__ == "__main__":
    print("Start test")
    plt.close()
    
    maxs = [9,6]
    grid = Grid(maxs)
    fig,ax = draw_grid(maxs,steps=1,color='k',label_points=True,h=30)
    t = Block([[0,0,1],[0,0,0]])
    ground = Block([[0,0,0],[2,0,0],[6,0,0],[8,0,0]]+[[i,0,1] for i in range(0,maxs[0])])
    #ground = Block([[0,0,0],[maxs[0]-1,0,0]]+[[i,0,1] for i in range(0,maxs[0])],muc=0.7)
    hinge = Block([[1,0,0],[0,1,1],[1,1,1],[1,1,0],[0,2,1],[0,1,0]])
    link = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]])
    grid.put(ground,[0,0],0,1,floating=True)
    grid.put(hinge,[1,0],0,2)
    grid.put(link,[0,2],0,3)
    grid.put(hinge,[1,3],0,4)
    grid.put(link,[1,5],5,5)
    grid.put(hinge,[4,3],0,6)
    grid.put(link,[5,3],5,7)
    grid.put(hinge,[7,0],0,8)
    # grid.put(link,[4,5],0,2,floating=True)
    # grid.put(hinge, [5,6],0,3,floating=False)
    fill_grid(ax,grid,draw_neigh=False)
    #
    #draw_block(ax,hinge,[4,4],color = 'none',draw_neigh = True)
    # draw_block(ax, link, [4,5],draw_neigh = True,color = 'r')
    a = 199
    #draw_block(ax,t,[4,4],color = 'yellowgreen',draw_neigh = True)
    #draw_block(ax,t,[4,4],color = 'darkred',draw_neigh = True)
    plt.show()
    print("End test")