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
from physics_scipy import side2corners,get_cm

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
def draw_action(ax,rid,action,blocktype,animated=False,multi=False,**action_args):
    action_args['rid']=rid
    if action in  ['Ph','P']:
        art = draw_put(ax,blocktype,hold=True,**action_args,animated=animated,multi=multi)
    elif action == 'Pl':
        art=draw_put(ax,blocktype,hold=False,**action_args,animated=animated,multi=multi)
    elif action == 'H':
        art=draw_hold(ax,**action_args,animated=animated,multi=multi)
    elif action == 'R':
        art=draw_remove(ax,**action_args,animated=animated,multi=multi)
    elif action == 'L':
        art=draw_leave(ax,**action_args,animated=animated,multi=multi)
    elif action == 'S':
        art= draw_stay(ax,**action_args,animated = animated,multi=multi)
    return art
def draw_action_rel(ax,rid,action,blocktype,grid,animated=False,multi=False,**action_args):
    action_args['rid']=rid
    if action in  ['Ph','P']:
        if action_args.get('bid_sup') is None:
            action_args['bid_sup'] = np.max(grid.occ)
        grid.connect(blocktype, None, action_args['sideblock'], action_args['sidesup'],action_args['bid_sup'],action_args['side_ori'],idcon=action_args['idconsup'])
        art = draw_put_rel(ax,blocktype,hold=True,**action_args,animated=animated,multi=multi)
    elif action == 'Pl':
        if action_args.get('bid_sup') is None:
            action_args['bid_sup'] = np.max(grid.occ)
        grid.connect(blocktype, None, action_args['sideblock'], action_args['sidesup'],action_args['bid_sup'],idcon=action_args['idconsup'])
        art=draw_put_rel(ax,blocktype,hold=False,**action_args,animated=animated,multi=multi)
    elif action == 'H':
        art=draw_hold(ax,**action_args,animated=animated,multi=multi)
    elif action == 'R':
        art=draw_remove(ax,**action_args,animated=animated,multi=multi)
    elif action == 'L':
        art=draw_leave(ax,**action_args,animated=animated,multi=multi)
    elif action == 'S':
        art= draw_stay(ax,**action_args,animated = animated,multi=multi)
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
                 animated=False,
                 multi=False):
    if multi:
        arts = draw_block(ax, blocktype, color=plt.cm.Set2(rid),alpha=0.5,animated=animated)
    else:
        arts = draw_block(ax, blocktype, color=plt.cm.Set2(rid),animated=animated)
    if hold:
        #the block was already moved by draw_block
        parts = blocktype.parts
        sides = np.concatenate([np.tile(parts,(3,1)),np.repeat(np.arange(3),parts.shape[0])[...,None]],axis=1)
        for side in sides:
            arts+= draw_side(ax, side,color='k',linewidth=0.5,animated=animated)
    return arts
def draw_put(ax,blocktype,rid,hold,pos,ori,blocktypeid,animated=False,multi=False):
    arts = draw_block(ax, blocktype, pos, ori, color=plt.cm.Set2(rid),animated=animated)
    if hold:
        #the block was already moved by draw_block
        parts = blocktype.parts
        sides = np.concatenate([np.tile(parts,(3,1)),np.repeat(np.arange(3),parts.shape[0])[...,None]],axis=1)
        for side in sides:
            arts+= draw_side(ax, side,color='k',animated=animated)
    return arts
        
def draw_hold(ax,rid,bid=None,pos=None,ori=None,animated=False,multi=False):
    if bid is None:
        arts = fill_triangle(ax, pos+ori%2, color=plt.cm.Set2(rid),animated=animated)
        side = pos+ori//2
        arts+= draw_side(ax, side,color='k',animated=animated)
    else:
        art = ax.text(0,ax.get_ylim()[1]-1,f"Robot {rid} tried to grab block {bid}",color = plt.cm.Set2(rid),animated=animated)
        arts = [art]
    return arts
def draw_remove(ax,rid,bid=None, pos=None,ori=None,animated=False,multi=False):
    if bid is None:
        arts = fill_triangle(ax, pos+ori%2, color='k',animated=animated)
        side = pos+ori//2
        if multi:
            arts+= draw_side(ax, side,color=plt.cm.Set2(rid),linewidth=0.5,alpha = 0.5,animated=animated)
        else:
            arts+= draw_side(ax, side,color=plt.cm.Set2(rid),linewidth=0.5,animated=animated)
    else:
        if multi:
            art = ax.text(0,ax.get_ylim()[1]-1-rid,f"Robot {rid} tried to remove block {bid}",color = plt.cm.Set2(rid),animated=animated)
        else:
            art = ax.text(0,ax.get_ylim()[1]-1,f"Robot {rid} tried to remove block {bid}",color = plt.cm.Set2(rid),animated=animated)
        arts = [art]
    return arts
def draw_leave(ax,rid,animated=False,multi=False):
    if multi:
        art = ax.text(0,ax.get_ylim()[1]-1-rid,f"Robot {rid} tried to leave",color = plt.cm.Set2(rid),animated=animated)
    else:
        art = ax.text(0,ax.get_ylim()[1]-1,f"Robot {rid} tried to leave",color = plt.cm.Set2(rid),animated=animated)
    return [art]
def draw_stay(ax,rid,animated=False,multi=False):
    if multi:
        art = ax.text(0,ax.get_ylim()[1]-1-rid,f"Robot {rid} stayed in placed",color = plt.cm.Set2(rid),animated=animated)
    else:
        art = ax.text(0,ax.get_ylim()[1]-1,f"Robot {rid} stayed in placed",color = plt.cm.Set2(rid),animated=animated)
    return [art]
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
            color = plt.cm.tab10(coord[3]/10)
    p1xy = base @ p1
    p2xy = base @ p2
    #corners = side2corners(np.expand_dims(coord[:-1],0),security_factor=0.5)
    
    art = ax.plot([p1xy[0],p2xy[0]],[p1xy[1],p2xy[1]],color=color,alpha=1,linewidth=linewidth,animated=animated)
    
    
    #art += [ax.scatter(corners[0,:,0],corners[0,:,1],color=color)]
    
    return art
def draw_forces(ax,grid,force_bag,bids=None,rids = None, uniform_scale=1/12,animated=False,solve=True):
    if solve:
        constraints = force_bag.solve(soft=True)
    if force_bag.x_soft is None:
        return []
    blocksid = force_bag.xid
    n_blocks=np.max(blocksid)
    start_idx = force_bag.nr*6
    # draw the external forces
    x = force_bag.x_soft
    posx = force_bag.xpos
        
    faileds, = np.nonzero(force_bag.constraints > 1e-3)
    arts = []
    
    if faileds.shape[0]>0:
        constraints = force_bag.constraints
        for failed in faileds:
            if force_bag.xid[failed]==-1:
                continue
            f_stick = constraints[failed]
            blocks = np.unique(force_bag.roweqid[force_bag.Aeq[:,failed]!=0])
            for block in blocks:
                Aeq_block=force_bag.Aeq[force_bag.roweqid==block]
                pos =posx[failed]
                scale = 1/ np.max(np.append(x[~np.all(Aeq_block==0,axis=0)],force_bag.beq[force_bag.roweqid==block][1]))/2
                arts+=[plt.arrow(pos[0],
                                  pos[1],
                                  -Aeq_block[0,failed]*f_stick*scale,
                                  -Aeq_block[1,failed]*f_stick*scale,
                                  color='r',
                                  width=0.1,
                                  animated= animated,
                                  zorder=100)]
    #     bids_failed = np.unique(blocksid[faileds])
    #     for bid in bids_failed:
    #         mov = -constraints[force_bag.roweqid==bid]
    #         cm = force_bag.cmpos[bid]
    #         n = np.sqrt(mov[0]*mov[0]+mov[1]*mov[1])*2
    #         arts+=[ax.arrow(cm[0],cm[1],
    #                         mov[0]/n,mov[1]/n,
    #                         color='r',
    #                         width=0.2,
    #                         animated= animated,
    #                         zorder=99)]

    #         ax.add_artist(arts[-1])
            # if abs(mov[2])>1e-6:
            #     arts+=ax.plot(cm[0],cm[1],marker=r'$\circlearrowleft$',c='r',zorder=98,ms=50,animated = animated)
            # if abs(mov[2])<-1e-6:
            #     arts+=ax.plot(cm[0],cm[1],marker=r'$\circlearrowright$',c='r',zorder=98,ms=50,animated = animated)
    #draw the forces from the robots:
    if rids is None:
        rids = np.arange(force_bag.nr)
    for r in rids:
        idxr = r*6
        bid_held = np.unique(force_bag.roweqid[force_bag.Aeq[:,idxr]==1])
        if len(bid_held>0):
            fr_x = x[idxr]-x[idxr+1]
            fr_y = x[idxr+2]-x[idxr+3]
            Mr = x[idxr+4]-x[idxr+5]
            cm = force_bag.cmpos[bid_held[0]]
            f2 = fr_x*fr_x+fr_y*fr_y
            
            if uniform_scale is None:
                scale = 1/(2*np.max(
                    np.append(x[~np.all(force_bag.Aeq[force_bag.roweqid==bid_held]==0,axis=0)],
                              force_bag.beq[force_bag.roweqid==bid_held][1])))
            else:
                scale = uniform_scale
                
            if f2 > 1e-6:
                arts+=[ax.arrow(cm[0], cm[1],fr_x*scale,fr_y*scale,
                                            color=plt.cm.Set2(r),
                                            width=0.05,
                                            animated= animated,
                                            zorder=100)]
            if abs(Mr) > 1e-6:
                arts+=[ax.arrow(cm[0]+0.5, cm[1],0,Mr*scale,
                                            color=plt.cm.Set2(r),
                                            width=0.05,
                                            animated= animated,
                                            zorder=100)]
                arts+=[ax.arrow(cm[0]-0.5, cm[1],0,-Mr*scale,
                                            color=plt.cm.Set2(r),
                                            width=0.05,
                                            animated= animated,
                                            zorder=100)]
    if bids is None:
        bids = np.arange(1,n_blocks+1)
    for i in bids:
        Aeq_block = force_bag.Aeq[force_bag.roweqid==i]
        if uniform_scale is None:
            x_blocks = x[~np.all(Aeq_block==0,axis=0)]
            scale = 1/2/np.max(np.append(x_blocks,force_bag.beq[force_bag.roweqid==i][1]))
        else:
            scale = uniform_scale
        #draw the weight
        cm = force_bag.cmpos[i]
        f_g = -force_bag.beq[force_bag.roweqid==i][:2]
        
        arts+=[ax.arrow(cm[0],cm[1],f_g[0]*scale,f_g[1]*scale,
                                    color=plt.cm.Pastel1(i%10),
                                    width=0.05,
                                    animated= animated,
                                    zorder=100)]
        
        
        for idxx in range(start_idx,x.shape[0],6):
            #compute the total support force
            fs = x[idxx]+x[idxx+3]
            #compute the combination of the locations
            if not np.all(Aeq_block[:,idxx]==0) and fs > 1e-6:
                pos = (x[idxx]*posx[idxx]+x[idxx+3]*posx[idxx+3])/fs
                arts+=[plt.arrow(pos[0],
                                  pos[1],
                                  Aeq_block[0,idxx]*fs*scale,
                                  Aeq_block[1,idxx]*fs*scale,
                                  color=plt.cm.Pastel1(i%10),
                                  width=0.05,
                                  animated= animated,
                                  zorder=100)]
                #comput the friction forces
                ff = x[idxx+1]-x[idxx+2]+x[idxx+4]-x[idxx+5]
                if abs(ff) > 1e-6:
                    arts+=[ax.arrow(pos[0],
                                     pos[1],
                                     Aeq_block[0,idxx+1]*ff*scale,
                                     Aeq_block[1,idxx+1]*ff*scale,
                                     color=plt.cm.Pastel1(i%10),
                                     width=0.05,
                                     animated= animated,
                                     zorder=100)]
                    
                
    #[ax.add_artist(a) for a in arts]
    return arts
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
              use_forces=True,
              animated=False,
              draw_hold = True,
              ground_color='darkslategrey',
              graph = None,
              connectivity = None,
              forces_bag=None):
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
                if use_con:
                    if grid.connection[coords[0,0],coords[1,0],coords[2,0]] == 0:
                        color = plt.cm.Pastel2(0)
                    elif grid.connection[coords[0,0],coords[1,0],coords[2,0]] == 1:
                        color = plt.cm.Pastel2(1)
                    else:
                        color = plt.cm.Pastel2(3)
                if use_forces:
                    c = forces_bag.solve(soft=True)
                    if c is not None:
                        x = forces_bag.x_soft[:forces_bag.Aeq.shape[1]]
                        val = np.max(np.append(x[~np.all(forces_bag.Aeq[forces_bag.roweqid==i]==0,axis=0)],forces_bag.beq[forces_bag.roweqid==i][1]))
                        color = plt.cm.plasma(val/np.min(np.abs(forces_bag.b[:forces_bag.nr*6])))
                    else:
                        color=plt.cm.plasma(np.nan)
                arts +=fill_triangle(ax, coords.T,color=color,animated=animated)
            
        coords=np.nonzero(grid.neighbours!=-1)
        coords = np.array(coords)
        for i in range(coords.shape[1]):
            if draw_neigh:
                arts+=draw_side(ax,coords[:,i],animated=animated)
            else:
                arts+=draw_side(ax,coords[:,i],color = 'k',animated=animated)
        if draw_hold:
            for i in np.unique(grid.hold):
                if i ==-1:
                    continue
                coords=np.nonzero(grid.hold==i)
                coords = np.array(coords).T
                coords = np.concatenate([np.tile(coords,(3,1)),np.repeat(np.arange(3),coords.shape[0])[...,None]],axis=1)
                for coord in coords:
                    arts+=draw_side(ax, coord,color=plt.cm.Set2(i),animated=animated)
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
                    arts+=draw_side(ax, coord,color=plt.cm.Set2(i),animated=animated)
    if graph is not None:
        arts+= add_graph(ax, graph,animated = animated,connectivity=connectivity)
    if forces_bag is not None:
        arts+=draw_forces(ax,grid,forces_bag,uniform_scale=None,solve=False,animated=animated)
    return arts
def animate(fig,arts_list,sperframe= 0.1):
    ani = animation.ArtistAnimation(fig, arts_list, interval=sperframe*1000, blit=True)
    #HTML(ani.to_jshtml())
    return ani
def draw_block(ax,block, pos=None,ori=None,draw_neigh=False,highlight_ref = False,animated=False, **colorkw):
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
    fig,ax = draw_grid(maxs,steps=1,color='k',label_points=False,h=10,linewidth=0.3)
    # draw_block(ax, hinge,[4,0],highlight_ref=True,draw_neigh=True,alpha=0.5,color=plt.cm.Set2(0))
    # draw_block(ax, hinge,[4,0],highlight_ref=True,draw_neigh=True,alpha=0.5,color=plt.cm.Set2(1))
    fill_grid(ax,grid,draw_neigh=True,animated=True,connectivity = 'sparse')
    plt.show()
    print("End test")