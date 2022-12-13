# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:12:18 2022

@author: valla
"""
import numpy as np
class Graph():
    def __init__(self,
                 n_blocktype,
                 n_robot,
                 n_reg,
                 maxblocks,
                 maxinterface,
                 ):
        self.bt = n_blocktype
        self.n_robot = n_robot
        self.n_nodes = n_reg+maxblocks
        self.mblock = maxblocks
        self.minter = maxinterface
        self.blocks = np.zeros((5,maxblocks),int)
        self.grounds = np.zeros((5,n_reg),int)
        self.robots = np.zeros((5,n_robot),int)
        self.i_s = np.zeros((self.n_nodes,maxinterface),int)
        self.i_r = np.zeros((self.n_nodes,maxinterface),int)
        self.i_a = np.zeros((1,maxinterface),int)
        self.n_reg = n_reg
        self.active_nodes = np.zeros((self.n_nodes),bool)
        self.active_edges = np.zeros((maxinterface),bool)
        self.n_ground = 0
        self.n_blocks = 0
        self.n_interface = 0
        
        #initialise the robot nodes 
    def add_ground(self,pos,rot):
        #note: it is assumed that all the grounds have are the same block 
        assert self.n_ground < self.n_reg, "Attempt to add more grounds than regions"
        self.grounds[:2,self.n_ground]=-1
        self.grounds[2:,self.n_ground]=[pos[0],pos[1],rot]
        self.active_nodes[self.n_ground]=True
        self.n_ground+=1
    def add_block(self,bid,blocktypeid,pos,rot):
        self.blocks[0,bid-1]=blocktypeid
        self.blocks[1,bid-1]=-1
        self.blocks[2:,bid-1]=[pos[0],pos[1],rot]
        self.active_nodes[self.n_reg+bid-1]=True
        self.n_blocks+=1
    def hold(self,bid,rid):
        self.blocks[1,bid-1]=rid
    def add_rel(self,bid1,bid2,side1,side2,connection2):
        if bid1 >0:
            if bid2 == 0:
                self.i_r[bid1-1+self.n_reg,self.n_interface]=1
                self.i_s[connection2,self.n_interface]=1
                self.i_a[:,self.n_interface]=side1
            else:
                self.i_r[bid1-1+self.n_reg,self.n_interface]=1
                self.i_s[bid2-1+self.n_reg,self.n_interface]=1
                self.i_a[:,self.n_interface]=side1
            self.active_edges[self.n_interface]=True
            self.n_interface+=1
            
        if bid2 > 0:
            self.i_r[bid2-1+self.n_reg,self.n_interface]=1
            self.i_s[bid1-1+self.n_reg,self.n_interface]=1
            self.i_a[:,self.n_interface]=side2
            self.active_edges[self.n_interface]=True
            self.n_interface+=1
    def remove(self,bid):
        assert bid >0, "cannot remove the ground"
        self.blocks[:,bid-1]=0
        self.active_nodes[bid-1]=False
        interfaces_to_delete = np.nonzero((self.i_r[bid-1+self.n_reg,:])|(self.i_s[bid-1+self.n_reg,:]))
        self.active_edges[interfaces_to_delete]=False
        self.i_a[:,interfaces_to_delete]=0
        self.i_r[:,interfaces_to_delete]=0
        self.i_s[:,interfaces_to_delete]=0
class FullGraph():
    def __init__(self,
                 n_blocktype,
                 n_robot,
                 n_reg,
                 n_sim_actions,
                 maxblocks,
                 
                 ):
        self.bt = n_blocktype
        self.n_robot = n_robot
        self.n_nodes = n_sim_actions+n_reg+maxblocks
        self.mblock = maxblocks
        self.n_act = n_sim_actions
        self.actions = np.zeros((6,n_sim_actions),int)
        self.blocks = np.zeros((6,maxblocks),int)
        self.grounds = np.zeros((6,n_reg),int)
        self.i_s = np.zeros((self.n_nodes, self.n_nodes*self.n_nodes),int)
        self.i_s[np.tile(np.arange(self.n_nodes),self.n_nodes),np.arange(self.i_s.shape[1])]=1
        self.i_r = np.zeros((self.n_nodes, self.n_nodes*self.n_nodes),int)
        self.i_r[np.repeat(np.arange(self.n_nodes),self.n_nodes),np.arange(self.i_r.shape[1])]=1
        self.i_a = np.zeros((0,self.n_nodes*self.n_nodes),int)
        self.n_reg = n_reg
        self.active_nodes = np.zeros((self.n_nodes),bool)
        self.active_edges = np.zeros((self.n_nodes*self.n_nodes),bool)
        #initialize the first action
        self.active_nodes[:n_sim_actions]=True
        self.actions[0,:] = -2
        self.actions[1,:] = np.arange(n_sim_actions)
        #leave the other parameters at 0
        
        self.n_ground = 0
        self.n_blocks = 0
        self.n_interface = 0
        
        #initialise the robot nodes 
    def add_ground(self,pos,rot):
        #note: it is assumed that all the grounds have are the same block 
        assert self.n_ground < self.n_reg, "Attempt to add more grounds than regions"
        self.grounds[:2,self.n_ground]=-1
        self.grounds[2:,self.n_ground]=[pos[0],pos[1],np.cos(rot*np.pi/3),np.sin(rot*np.pi/3)]
        self.active_nodes[self.n_act+self.n_ground]=True
        self.active_edges[(self.i_r[self.n_act+self.n_ground,:]==1) | (self.i_s[self.n_act+self.n_ground,:]==1)]=True
        self.n_ground+=1
    def add_block(self,bid,blocktypeid,pos,rot):
        self.blocks[0,bid-1]=blocktypeid
        self.blocks[1,bid-1]=-1
        self.blocks[2:,bid-1]=[pos[0],pos[1],np.cos(rot*np.pi/3),np.sin(rot*np.pi/3)]
        self.active_nodes[self.n_reg+self.n_act+bid-1]=True
        self.active_edges[(self.i_r[self.n_reg+self.n_act+bid-1,:]==1) | (self.i_s[self.n_reg+self.n_act+bid-1,:]==1)]=True
        self.n_blocks+=1
    def hold(self,bid,rid):
        self.blocks[1,bid-1]=rid
class Grid():
    def __init__(self,maxs):
        self.occ = -np.ones((maxs[0]+1,maxs[1]+1,2),dtype=int)
        self.neighbours = -np.ones((maxs[0]+1,maxs[1]+1,2,3,2),dtype=int)
        self.connection = -np.ones((maxs[0]+1,maxs[1]+1,2),dtype=int)
        self.hold = -np.ones((maxs[0]+1,maxs[1]+1,2),dtype = int)
        self.min_dist = np.zeros((0,0))
        self.nreg = 0
        self.shape = maxs
    def put(self,block,pos0,rot,bid,floating=False,holder_rid=None):
        block.turn(rot)
        block.move(pos0)
        #test if the block is in the grid
        if (np.min(block.parts)<0 or
            np.max(block.parts[:,0])>=self.occ.shape[0]-1 or
            np.max(block.parts[:,1])>=self.occ.shape[1]-1):
            return False,None,None
        #test if the is no overlap
        if np.any(self.occ[block.parts[:,0],block.parts[:,1],block.parts[:,2]]!=-1):
            return False,None,None
        else:
            #check if there is a connection (ask for at least 1 points)
            #if floating or np.any(same_side(block.neigh, np.array(np.where(self.neighbours)).T)):
            if not floating:
                candidates = switch_direction(block.neigh)
                interfaces = self.neighbours[candidates[:,0],candidates[:,1],candidates[:,2],candidates[:,3],:]
                if len(interfaces[interfaces[:,0]!=-1,:])==0:
                    return False,None,None
                interfaces =np.concatenate([interfaces,
                                            self.connection[candidates[:,0],candidates[:,1],candidates[:,2]][...,None]],
                                           axis=1)
            #if floating or np.any(switch_direction(block.neigh)np.array(np.where(self.neighbours)).T)):
                #add the block
            self.occ[block.parts[:,0],block.parts[:,1],block.parts[:,2]]=bid
            self.neighbours[block.neigh[:,0],block.neigh[:,1],block.neigh[:,2],block.neigh[:,3],0]=bid
            self.neighbours[block.neigh[:,0],block.neigh[:,1],block.neigh[:,2],block.neigh[:,3],1]=np.arange(len(block.neigh))
            if holder_rid is not None:
                self.hold[block.parts[:,0],block.parts[:,1],block.parts[:,2]]=holder_rid
            
            
            if floating:
                #create a new region
                self.connection[block.parts[:,0],block.parts[:,1],block.parts[:,2]]=self.nreg
                self.min_dist = np.block([[self.min_dist,np.zeros((self.min_dist.shape[0],1))],
                                          [np.zeros((1,self.min_dist.shape[1])),np.zeros((1,1))]])
                for regi in range(self.nreg):
                    partsi = np.array(np.nonzero(self.connection==regi))
                    min_dist = closest_point(block.parts, partsi.T)
                    self.min_dist[regi,self.nreg]=min_dist
                    self.min_dist[self.nreg,regi]=min_dist
                self.nreg = self.nreg+1
                
                return True,None,None
            else:
                #take care of the connectivity of the different regions
                closer=self.update_dist(block)
                
                return True,closer,interfaces

    def update_dist(self,block):
        #add the block to the connected region
        candidates = switch_direction(block.neigh)
        connect_region = np.unique(self.connection[candidates[:,0],candidates[:,1],candidates[:,2]])
        connect_region = np.delete(connect_region,connect_region==-1)
        
        targets = np.delete(np.arange(self.nreg),connect_region[0])
        ndist = np.zeros(self.nreg)
        #distb = np.zeros(targets.shape[0])
        closer = -1
        for i in targets:
            distb = closest_point(block.parts,np.array(np.nonzero(self.connection==i)).T)
            
            #ndist[i] = (distb-self.min_dist[connect_region[0],i])/(distb+self.min_dist[connect_region[0],i]+1e-5)
           
            if distb<self.min_dist[connect_region[0],i]+1e-5:
                closer =1
                self.min_dist[connect_region[0],i]=distb
                self.min_dist[i,connect_region[0]]=distb
                self.shortest_path(i,connect_region[0])
                self.shortest_path(connect_region[0],i)
            elif distb<self.min_dist[connect_region[0],i]-1e-5:
                closer = 0
        self.connection[block.parts[:,0],block.parts[:,1],block.parts[:,2]]=connect_region[0]
        return closer
    def remove(self,bid):
        assert bid > 0
        idcon = self.connection[self.occ==bid]
        if len(idcon)==0:
            return False
        else:
            idcon = idcon[0]
        self.connection[self.occ==bid]=-1
        for i in np.delete(np.arange(self.nreg),idcon):
            
            min_dist= closest_point(np.array(np.nonzero(self.connection==idcon)).T,np.array(np.nonzero(self.connection==i)).T)
            self.min_dist[i,idcon]=min_dist
            self.min_dist[idcon,i]=min_dist
        for i in np.delete(np.arange(self.nreg),idcon):
            self.shortest_path(i,idcon)
            self.shortest_path(idcon,i)
        self.occ[self.occ==bid]=-1
        self.neighbours[self.neighbours[:,:,:,:,0]==bid]=-1
        
        return True
    def absorb_reg(self,old_cid,new_cid):
        self.connection[self.connection==old_cid]=new_cid
        self.min_dist[old_cid,new_cid]=0
        self.min_dist[new_cid,old_cid]=0
    def reset(self):
        self.occ[:,:,:] = -1
        self.neighbours[:,:,:,:] = -1
        self.connection[:,:,:] = -1
        self.hold[:,:,:] = -1
        self.min_dist = np.zeros((0,0))
        self.nreg = 0
    def shortest_path(self,source,sink):
        twosteps = self.min_dist[source,sink]+self.min_dist[sink,:]
        shortest = twosteps < self.min_dist[source,:]
        if np.any(shortest):
            pass
        self.min_dist[source,shortest]=twosteps[shortest]
        self.min_dist[shortest,source]=twosteps[shortest]
    def touch_side(self,block,last):
        '''return the list of all pos ori pair where block touches at least one side''' 
        if last:
            parts = np.array(np.nonzero(self.occ==np.max(self.occ))).T
        else:
            parts = np.array(np.nonzero(self.occ!=-1)).T
        sides_struct = switch_direction(outer_sides(parts))
        sides_struct = sides_struct[(sides_struct[:,0]>=0) & (sides_struct[:,1]>=0)]
        sides_struct = sides_struct[(sides_struct[:,0]<self.occ.shape[0]-1) 
                                    & (sides_struct[:,1]<self.occ.shape[1]-1)]
        pos_ar = np.zeros((0,2),dtype=int)
        ori_ar = np.zeros(0,dtype=int)
        
        for ori in range(6):
            block.turn(ori)
            block.move([0,0])
            for side in block.neigh:
                compatible = (side[2]==sides_struct[:,2]) & (side[3]==sides_struct[:,3])
                pos = sides_struct[compatible][:,:2]-side[:2]
                pos = pos[(pos[:,0]>=0) & (pos[:,1]>=0)]
                pos = pos[(pos[:,0]<self.occ.shape[0]-1) & (pos[:,1]<self.occ.shape[1]-1)]
                pos_ar = np.concatenate([pos_ar,pos])
                ori_ar = np.concatenate([ori_ar,ori*np.ones(len(pos),dtype=int)])
        return (pos_ar,ori_ar)
    def connect(self,block,bid,sideid,support_sideid,support_bid,holder_rid=None,idcon=None):
        if support_bid ==0:
            supportsides = np.array(np.nonzero((self.neighbours[:,:,:,:,0]==support_bid) & 
                                               (self.neighbours[:,:,:,:,1]==support_sideid))).T
            supportsides =  supportsides[self.connection[supportsides[:,0],supportsides[:,1],supportsides[:,2]]==idcon,:]
        else:
            supportsides = np.array(np.nonzero((self.neighbours[:,:,:,:,0]==support_bid) & 
                                               (self.neighbours[:,:,:,:,1]==support_sideid))).T
        if supportsides.shape[0]==0:
            return False,None,None
        target = switch_direction(supportsides)
        block.connect(sideid,target[0])
        if bid is not None:
            if (np.min(block.parts)<0 or
                np.max(block.parts[:,0])>=self.occ.shape[0]-1 or
                np.max(block.parts[:,1])>=self.occ.shape[1]-1):
                return False,None,None
            #test if there is no overlap
            if np.any(self.occ[block.parts[:,0],block.parts[:,1],block.parts[:,2]]!=-1):
                return False,None,None
            else:
                #add the block
                self.occ[block.parts[:,0],block.parts[:,1],block.parts[:,2]]=bid
                self.neighbours[block.neigh[:,0],block.neigh[:,1],block.neigh[:,2],block.neigh[:,3],0]=bid
                self.neighbours[block.neigh[:,0],block.neigh[:,1],block.neigh[:,2],block.neigh[:,3],1]=np.arange(len(block.neigh))
                if holder_rid is not None:
                    self.hold[block.parts[:,0],block.parts[:,1],block.parts[:,2]]=holder_rid
                
                #get the interfaces for the graph
                pot_support = switch_direction(block.neigh)
                interfaces = self.neighbours[pot_support[:,0],pot_support[:,1],pot_support[:,2],pot_support[:,3]]
                #add the connected region to the interfaces:
                interfaces =np.concatenate([interfaces,
                                            self.connection[pot_support[:,0],pot_support[:,1],pot_support[:,2]][...,None]],
                                           axis=1)
                #take care of the connectivity of the different regions
                closer = self.update_dist(block)
                    
                return True,closer,interfaces
def closest_point(parts1,parts2):
    if parts1.shape[0]==0 or parts2.shape[0]==0:
        return np.nan
    p1 = side2corner(outer_sides(parts1))
    p1=np.reshape(p1,(-1,2))
    p2 = side2corner(outer_sides(parts2))
    p2=np.reshape(p2,(-1,2))
    dists = np.sqrt(np.square(p1[:,0][None,...]-p2[:,0][...,None])+np.square(p1[:,1][None,...]-p2[:,1][...,None]))
    closestp1, closestp2 = np.unravel_index(np.argmin(dists), dists.shape)
    mindist = dists[closestp1, closestp2]
    return mindist
def side2corner(sides):
    #return the real coordinates of the two corners of the sides
    #note that p1(side)==p1(switch direction(side))
    
    #switch the direction of all down triangles:
    upsides = np.copy(sides)
    upsides[sides[:,2]==1,:]=switch_direction(sides[sides[:,2]==1,:])
    #first put each corner at the leftest point
    corners = grid2real(upsides)
    corners = np.stack([corners,corners],axis=1)
    corners[upsides[:,3]==0,1,:] = corners[upsides[:,3]==0,1,:]+[1,0]
    corners[upsides[:,3]==1,0,:] = corners[upsides[:,3]==1,0,:]+[1,0]
    corners[upsides[:,3]==1,1,:] = corners[upsides[:,3]==1,1,:]+[0.5,np.sqrt(3)/2]
    corners[upsides[:,3]==2,0,:] = corners[upsides[:,3]==2,0,:]+[0.5,np.sqrt(3)/2]
    return corners
def grid2real(parts):
    #return the coordinate of the leftmost point of the triangle
    r_parts = np.zeros((parts.shape[0],2))
    r_parts[:,0] = parts[:,0]+parts[:,1]*0.5
    r_parts[:,1] = parts[:,1]*np.sqrt(3)/2
    return r_parts
def same_side(sides1,sides2):
    #check if the 3rd and 4rth coord are compatible
    ret = np.zeros((sides1.shape[0],sides2.shape[0]),dtype=bool)
    ret[((np.expand_dims(sides1[:,3],1) == 0) &
          (np.expand_dims(sides2[:,3],0) == 0) &
          (np.expand_dims(sides1[:,0],1) == np.expand_dims(sides2[:,0],0)) &
          (np.expand_dims(sides1[:,1],1) == np.expand_dims(sides2[:,1],0)) )]=True
    ret[((np.expand_dims(sides1[:,3],1) == 1) &
          (np.expand_dims(sides2[:,3],0) == 1) &
          (np.expand_dims(sides1[:,0],1) == np.expand_dims(sides2[:,0],0)) &
          (np.expand_dims(sides1[:,1]-2*sides1[:,2]+1,1) == np.expand_dims(sides2[:,1],0)))]=True
    ret[((np.expand_dims(sides1[:,3],1) == 2) &
          (np.expand_dims(sides2[:,3],0) == 2) &
          (np.expand_dims(sides1[:,0]+sides1[:,2]*2-1,1) == np.expand_dims(sides2[:,0],0)) &
          (np.expand_dims(sides1[:,1]-sides1[:,2]*2+1,1) == np.expand_dims(sides2[:,1],0)))]=True

    ret[np.expand_dims(sides1[:,2],1) == np.expand_dims(sides2[:,2],0)]=False
    return ret
def switch_direction(sides):
    ret = np.zeros((sides.shape[0],4),dtype=int)
    
    ret[sides[:,3]==0,:2]=sides[sides[:,3]==0,:2]
    
    ret[sides[:,3]==1,0]=sides[sides[:,3]==1,0]
    ret[sides[:,3]==1,1]=sides[sides[:,3]==1,1]-2*sides[sides[:,3]==1,2]+1
    
    ret[sides[:,3]==2,0]=sides[sides[:,3]==2,0]+2*sides[sides[:,3]==2,2]-1
    ret[sides[:,3]==2,1]=sides[sides[:,3]==2,1]-2*sides[sides[:,3]==2,2]+1
    
    ret[:,2]=(sides[:,2]+1)%2
    ret[:,3]=sides[:,3]
    return ret

class discret_block():
    def __init__(self,parts,density=1,muc=0,rot=0):
        self.parts = np.array(parts)
        self.neigh = outer_sides(self.parts)
        self.mass = density*self.parts.shape[0]
        self.muc = muc
        self.rot = 0
    def turn(self,time_abs,center=None):
        time = (time_abs-self.rot)%6
        self.rot = time_abs%6
        if center is None:
            center = self.parts[0,:2]
        if time == 0:
            return
        elif time==1:
            rot_mat = np.array([[0,1],
                                [-1,1]])
            rem_zero = np.array([[1,-1]])
            add_one = np.array([[0,0]])
        elif time ==2:
            rot_mat = np.array([[-1,1],
                                [-1,0]])
            rem_zero = np.array([[1,0]])
            add_one = np.array([[-1,1]])
        elif time ==3:
            rot_mat = np.array([[-1,0],
                                [0,-1]])
            
            rem_zero = np.array([[1,0]])
            add_one = np.array([[-1,0]])
        elif time ==4:
            rot_mat = np.array([[0,-1],
                                [1,-1]])
            
            rem_zero = np.array([[-2,2]])
            add_one = np.array([[1,-1]])
        elif time ==5:
            rot_mat = np.array([[1,-1],
                                [1,0]])
            
            rem_zero = np.array([[0,0]])
            add_one = np.array([[0,-1]])
        
        nparts = np.zeros((self.parts.shape[0],3))
        nparts[:,:2] = (self.parts[:,:2] -center) @ rot_mat  + center
        nparts[:,:2]=nparts[:,:2] + rem_zero * np.expand_dims((self.parts[:,2]-1),1) + add_one * np.expand_dims((self.parts[:,2]),1)
        nparts[:,2]=(self.parts[:,2]+time)%2
        self.parts = nparts.astype(int)
        
        #rotate the sides:
        nneigh = np.zeros((self.neigh.shape[0],4))
        nneigh[:,:2] = (self.neigh[:,:2] -center) @ rot_mat  + center
        nneigh[:,:2]=nneigh[:,:2] + rem_zero * np.expand_dims((self.neigh[:,2]-1),1) + add_one * np.expand_dims((self.neigh[:,2]),1)
        nneigh[:,2]=(self.neigh[:,2]+time)%2
        nneigh[:,3]= (self.neigh[:,3]-time)%3
        #sort the neighbours left to right
        #i=np.lexsort((self.neigh[:,3],self.neigh[:,2],self.neigh[:,1],self.neigh[:,0]))
        self.neigh = nneigh.astype(int)
        
    def move(self,new_p):
        
        delta =new_p-self.parts[0,:-1]
        self.parts[:,:-1] = self.parts[:,:-1]+delta
        self.neigh[:,:2]=self.neigh[:,:2]+delta
        #use real coordonates
        #self.cm = self.cm + delta @ 
    def connect(self,sideid,target):
        self.turn(0)
        side = self.neigh[sideid]
        ori = (target[3]-side[3])*2+(target[2]-side[2])*3
        self.turn(ori)
        self.move([0,0])
        side = self.neigh[sideid]
        self.move(target[:2]-side[:2])
def outer_sides(parts):
    neigh = np.zeros((3*parts.shape[0],4),dtype=int)
    neigh[:,:3] =np.tile(parts,(3,1))
    neigh[:parts.shape[0],3] = 0
    neigh[parts.shape[0]:2*parts.shape[0],3] = 1
    neigh[2*parts.shape[0]:,3] = 2
    same = same_side(neigh,neigh)
    return neigh[~np.any(same,axis=0)]