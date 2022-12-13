# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 13:12:51 2022

@author: valla
"""

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon,Circle,FancyArrowPatch
import numpy as np

from discrete_blocks_norot import discret_block_norot as Block
from discrete_blocks_norot import Grid, Graph,FullGraph

from matplotlib import animation
from IPython.display import HTML


s3 =np.sqrt(3)
base = np.array([[1,0.5],[0,s3/2]])
def draw_grid(maxs,label_points=False,steps=1,color='darkslategrey',h=6,linewidth=1):
    fig,ax = plt.subplots(1,1,figsize=(h*(maxs[0]+maxs[1]*0.5)/((maxs[1]-0.5)*s3),h))
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
                        fontsize=h)
                ax.text(x0+x1/2+0.5,x1*s3/2-0.2,f"({x0},{x1},1)",
                        va='center',
                        ha='center',
                        fontsize=h)
    ax.axis('off')
    # ax.set_yticklabels(np.arange(-xlim[0],np.floor((ylim[1]-ylim[0])/np.sqrt(3))-xlim[0]+1))
    return fig,ax
def draw_action(ax,rid,action,blocktype,animated=False,**action_args):
    action_args['rid']=rid
    if action == 'Ph':
        art = draw_put(ax,blocktype,hold=True,**action_args,animated=animated)
    elif action == 'Pl':
        art=draw_put(ax,blocktype,hold=False,**action_args,animated=animated)
    elif action == 'H':
        art=draw_hold(ax,**action_args,animated=animated)
    elif action == 'R':
        art=draw_remove(ax,**action_args,animated=animated)
    elif action == 'L':
        art=draw_leave(ax,**action_args,animated=animated)
    return art
def draw_action_rel(ax,rid,action,blocktype,grid,animated=False,**action_args):
    action_args['rid']=rid
    if action == 'Ph':
        if action_args.get('bid_sup') is None:
            action_args['bid_sup'] = np.max(grid.occ)
        grid.connect(blocktype, None, action_args['sideblock'], action_args['sidesup'],action_args['bid_sup'],action_args['side_ori'],idcon=action_args['idconsup'])
        art = draw_put_rel(ax,blocktype,hold=True,**action_args,animated=animated)
    elif action == 'Pl':
        if action_args.get('bid_sup') is None:
            action_args['bid_sup'] = np.max(grid.occ)
        grid.connect(blocktype, None, action_args['sideblock'], action_args['sidesup'],action_args['bid_sup'],idcon=action_args['idconsup'])
        art=draw_put_rel(ax,blocktype,hold=False,**action_args,animated=animated)
    elif action == 'H':
        art=draw_hold(ax,**action_args,animated=animated)
    elif action == 'R':
        art=draw_remove(ax,**action_args,animated=animated)
    elif action == 'L':
        art=draw_leave(ax,**action_args,animated=animated)
    return art
def draw_put_rel(ax,
                 blocktype,
                 rid,
                 hold,
                 sideblock=None,
                 sidesup = None,
                 bid_sup = None,
                 blocktypeid = None,
                 idconsup=None,
                 side_ori=None,
                 animated=False):
    arts = draw_block(ax, blocktype, color=plt.cm.Set1(rid),animated=animated)
    if hold:
        #the block was already moved by draw_block
        parts = blocktype.parts
        sides = np.concatenate([np.tile(parts,(3,1)),np.repeat(np.arange(3),parts.shape[0])[...,None]],axis=1)
        for side in sides:
            arts+= draw_side(ax, side,color='k',linewidth=0.5,animated=animated)
    return arts
def draw_put(ax,blocktype,rid,hold,pos,ori,blocktypeid,animated=False):
    arts = draw_block(ax, blocktype, pos, ori, color=plt.cm.Set1(rid),animated=animated)
    if hold:
        #the block was already moved by draw_block
        parts = blocktype.parts
        sides = np.concatenate([np.tile(parts,(3,1)),np.repeat(np.arange(3),parts.shape[0])[...,None]],axis=1)
        for side in sides:
            arts+= draw_side(ax, side,color='k',animated=animated)
    return arts
        
def draw_hold(ax,rid,bid=None,pos=None,ori=None,animated=False):
    if bid is None:
        arts = fill_triangle(ax, pos+ori%2, color=plt.cm.Set1(rid),animated=animated)
        side = pos+ori//2
        arts+= draw_side(ax, side,color='k',animated=animated)
    else:
        art = ax.text(0,ax.get_ylim()[1]-1,f"Robot {rid} tried to grab block {bid}",color = plt.cm.Set1(rid),animated=animated)
        arts = [art]
    return arts
def draw_remove(ax,rid,bid=None, pos=None,ori=None,animated=False):
    if bid is None:
        arts = fill_triangle(ax, pos+ori%2, color='k',animated=animated)
        side = pos+ori//2
        arts+= draw_side(ax, side,color=plt.cm.Set1(rid),linewidth=0.5,animated=animated)
    else:
        art = ax.text(0,ax.get_ylim()[1]-1,f"Robot {rid} tried to remove block {bid}",color = plt.cm.Set1(rid),animated=animated)
        arts = [art]
    return arts
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

def fill_triangle(ax,coord,animated=False,text=None,fontsize = 7,**colorkw):
    coord = np.array(coord)
   
        
    triangle = np.tile([[0,0],[0.5,np.sqrt(3)/2],[1,0]],(coord.shape[0],1,1))
    down = coord[:,2]==1
    triangle[down,:,1]=-triangle[down,:,1]
    xycoord = (base @ coord[:,:-1].T).T
    
    art = [Polygon(xycoord[i]+triangle[i],animated=animated,**colorkw) for i in range(coord.shape[0])]
    if text is not None:
        art+=[ax.text(xycoord[i,0]+0.5,xycoord[i,1]-0.3*(2*down[i].astype(int)-1),f"{text}",
                va='center',
                ha='center',
                fontsize=fontsize) for i in range(coord.shape[0])]
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
def add_graph(ax,graph,animated=False,connectivity = 'sparse'):
    arts = []
    if connectivity == 'sparse':
        xys = base @ graph.blocks[2:-1,:]
        xys = np.concatenate([base @ graph.grounds[2:-1,:], xys],axis=1)
    elif connectivity == 'full':
        xys = base @ graph.blocks[2:-2,:]
        xys = np.concatenate([base @ graph.actions[2:-2,:], base @ graph.grounds[2:-2,:], xys],axis=1)
    #remove the unactive nodes
    xys = xys[:,graph.active_nodes]
    #add the grounds
    coords, inv, counts = np.unique(xys,return_counts=True,return_inverse = True,axis=1)
    for doubleid in np.nonzero(counts==2)[0]:
        loc = np.nonzero(inv ==doubleid)[0]
        xys[0,loc[0]]+=0.5
    for xy in xys.T:
        arts.append(Circle(xy,radius = 0.2,animated=animated,color='k'))
        
    #get all the sender nodes
    counts = np.zeros((graph.i_s.shape[1],graph.i_r.shape[1]))
    for s,r in zip(graph.i_s.T,graph.i_r.T):
        if np.sum(s)==0:
            continue
        ids = np.nonzero(s[graph.active_nodes])[0]
        idr = np.nonzero(r[graph.active_nodes])[0]

        
        arts.append(FancyArrowPatch(xys[:,ids].flatten(),xys[:,idr].flatten(),
                                    arrowstyle='->',
                                    mutation_scale = 20,
                                    color=plt.cm.rainbow(counts[ids[0],idr[0]]/5),
                                    connectionstyle=f'arc3,rad={0.5*(counts[ids[0],idr[0]])+0.4}',
                                    linewidth=2,
                                    animated= animated))
        counts[ids,idr]+=1
        #arts+=ax.plot(xys[0,[ids,idr]].flatten(),xys[1,[ids,idr]].flatten())
    #val, = np.unique(posr,return_inverse=True,return_counts = True)
    
    [ax.add_artist(a) for a in arts]
    return arts
def write_state(grid,h,linewidth = 0.5,alpha=0.5):
    fig,axs = plt.subplots(1,3,figsize=(3*h*((grid.shape[0])+(grid.shape[1])*0.5)/(((grid.shape[1])-0.5)*s3),h))
    xlim = [-0.5,grid.shape[0]+0.5*(grid.shape[1])]
    ylim = [-s3/2,(grid.shape[1])*s3/2]
    channels = ['bid','rid','cid']
    for ax,channel in zip(axs,channels):
        #prepare the drawing
        ax.set_aspect('equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.axis('off')
        ax.set_title(f'Channel: {channel}')
        #draw the grid
        
        #draw the horizontal lines
        y = np.arange(ylim[0],ylim[1],s3/2)
        y = np.stack([y,y])
        ax.plot(np.reshape(xlim,(2,1)),y,color='k',linewidth=linewidth)
        #draw the 60 degrees lines
        y = np.arange(ylim[0]-(xlim[1])*s3,ylim[1]-(xlim[0])*s3,s3)
        y = y-y[np.argmin(abs(y))]
        y = np.stack([y+xlim[0]*s3,y+xlim[1]*s3])    
        
        ax.plot(np.reshape(xlim,(2,1)),y,color='k',linewidth=linewidth)
        #draw the -60 degree lines
        y = np.arange(ylim[0]+(xlim[0])*s3,ylim[1]+(xlim[1])*s3,s3)
        y = y-y[np.argmin(abs(y))]
        y = np.stack([y-xlim[0]*s3,y-xlim[1]*s3])    
        ax.plot(np.reshape(xlim,(2,1)),y,color='k',linewidth=linewidth)
        
        if channel == 'bid':
            grid_ar = grid.occ
        elif channel == 'rid':
            grid_ar = grid.hold
        elif channel == 'cid':
            grid_ar = grid.connection
        ids = np.unique(grid_ar)
        for i in ids:
            coords = np.array(np.where(grid_ar[:-1,:-1,:]==i))
            fill_triangle(ax, coords.T,color=plt.cm.turbo((i+1)/(1+ids[-1])),alpha=alpha,text=str(i),fontsize=h*1.5)
            
        
        # ax.set_yticklabels(np.arange(-xlim[0],np.floor((ylim[1]-ylim[0])/np.sqrt(3))-xlim[0]+1))
def fill_grid(ax,
              grid,
              draw_neigh =False,
              use_con=False,
              animated=False,
              draw_hold = True,
              ground_color='darkslategrey',
              graph = None,
              connectivity = None):
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
            arts+=draw_side(ax,coords[:,i],color = 'xkcd:beige',animated=animated)
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
    if graph is not None:
        arts+= add_graph(ax, graph,animated = animated,connectivity=connectivity)
    return arts
def animate(fig,arts_list,sperframe= 0.1):
    ani = animation.ArtistAnimation(fig, arts_list, interval=sperframe*1000, blit=True)
    #HTML(ani.to_jshtml())
    return ani
def draw_block(ax,block, pos=None,ori=None,draw_neigh=True,highlight_ref = False,animated=False, **colorkw):
    if ori is not None:
        block.turn(ori)
    if pos is not None:
        block.move(pos)
    arts = fill_triangle(ax, block.parts,animated=animated,**colorkw)
    if highlight_ref:
        arts+=fill_triangle(ax,[block.parts[0]],animated=animated,color = 'none',text='0')
    if draw_neigh:
        for i,s in enumerate(block.neigh):
            arts+=draw_side(ax,s,color=plt.cm.turbo(i/block.neigh.shape[0]),animated=animated)
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
    
    maxs = [10,10]
    grid = Grid(maxs)
    graph = Graph(2,
                 2,
                 2,
                 10,
                 30,
                 )
    full_graph = FullGraph(2,2,2,1,10)
    t = Block([[0,0,1],[0,0,0]])
    #ground = Block([[0,0,0],[2,0,0],[6,0,0],[8,0,0]]+[[i,0,1] for i in range(0,maxs[0])])
    ground = Block([[0,0,1]])
    #ground = Block([[0,0,0],[maxs[0]-1,0,0]]+[[i,0,1] for i in range(0,maxs[0])],muc=0.7)
    hinge = Block([[1,0,0],[0,1,1],[1,1,1],[1,1,0],[0,2,1],[0,1,0]])
    
    linkr = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]])
    linkl = Block([[1,0,1],[1,0,0],[0,1,1],[1,1,0],[1,1,1],[2,0,0]])
    linkh = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[1,2,1],[2,0,0]])
    _,_,i=grid.put(ground,[1,0],0,floating=True)
    grid.put(ground,[7,0],0,floating=True)
    grid.put(hinge,[1,0],1)
    grid.put(linkr,[0,2],2)
    grid.put(hinge,[1,3],3)
    grid.put(linkh,[2,3],4)
    grid.put(hinge,[4,3],5)
    grid.put(linkl,[5,2],6)
    grid.put(hinge,[7,0],7)
    fig,ax = draw_grid(maxs,steps=1,color='k',label_points=True,h=10,linewidth=0.3)
    draw_block(ax, hinge,[4,0],highlight_ref=True,draw_neigh=True)
    fill_grid(ax,grid,draw_neigh=True,animated=True,connectivity = 'sparse')
    plt.show()
    print("End test")