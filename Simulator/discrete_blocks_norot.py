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
                 maxinterface_ground = 12,
                 ):
        self.bt = n_blocktype
        self.n_robot = n_robot
        self.n_nodes = n_reg+maxblocks
        self.mblock = maxblocks
        self.minter = maxinterface
        self.blocks = np.zeros((maxblocks,4),int)
        self.grounds = np.zeros((n_reg,4),int)
        self.robots = np.zeros((n_robot,4),int)
        self.edges_index_bb = np.zeros((2,maxinterface),int)
        self.edges_index_bg = np.zeros((2,maxinterface_ground//2),int)
        self.edges_index_gb = np.zeros((2,maxinterface_ground//2),int)
        self.i_a_bb = np.zeros((maxinterface,3),int) # side ori, side_b, side_sup
        self.i_a_bg = np.zeros((maxinterface_ground//2,3),int) # side ori, side_b, side_sup
        self.i_a_gb = np.zeros((maxinterface_ground//2,3),int) # side ori, side_b, side_sup
        self.n_reg = n_reg
        self.active_grounds = np.zeros((n_reg),bool)
        self.active_blocks = np.zeros((self.mblock),bool)
        self.active_edges_bb = np.zeros((maxinterface),bool)
        self.active_edges_bg = np.zeros((maxinterface_ground//2),bool)
        self.active_edges_gb = np.zeros((maxinterface_ground//2),bool)
        self.n_ground = 0
        self.n_blocks = 0
        self.n_interface_bb = 0
        self.n_interface_bg = 0
        self.n_interface_gb = 0
        #initialise the robot nodes 
    def add_ground(self,pos):
        #note: it is assumed that all the grounds have are the same block 
        assert self.n_ground < self.n_reg, "Attempt to add more grounds than regions"
        self.grounds[self.n_ground,:2]=-1
        self.grounds[self.n_ground,2:]=[pos[0],pos[1]]
        self.active_grounds[self.n_ground]=True
        self.n_ground+=1
    def add_block(self,blocktypeid,pos):
        self.blocks[self.n_blocks,1]=blocktypeid
        self.blocks[self.n_blocks,0]=-1
        self.blocks[self.n_blocks,2:]=[pos[0],pos[1]]
        self.active_blocks[self.n_blocks]=True
        self.n_blocks+=1
    def hold(self,bid,rid):
        self.blocks[bid-1,:]=rid
    def leave(self,rid):
        self.blocks[:,0]=-1
    def add_rel(self,bid1,bid2,ori1,side1,side2,connection1,connection2):
        if bid1 >0:
            if bid2 == 0:
                if self.n_interface_bg == self.edges_index_bg.shape[1]:
                    return False
                self.edges_index_bg[0,self.n_interface_bg]=bid1-1
                self.edges_index_bg[1,self.n_interface_bg]=connection2
                self.i_a_bg[self.n_interface_bg,:]=[ori1,side1,side2]
                self.active_edges_bg[self.n_interface_bg]=True
                self.n_interface_bg+=1
            else:
                if self.n_interface_bb == self.edges_index_bb.shape[1]:
                    return False
                self.edges_index_bb[0,self.n_interface_bb]=bid1-1
                self.edges_index_bb[1,self.n_interface_bb]=bid2-1
                self.i_a_bb[self.n_interface_bb,:]=[ori1,side1,side2]
                self.active_edges_bb[self.n_interface_bb]=True
                self.n_interface_bb+=1
            
        else:
            if bid2>0:
                if self.n_interface_gb == self.edges_index_gb.shape[1]:
                    return False
                self.edges_index_gb[0,self.n_interface_gb]=connection1
                self.edges_index_gb[1,self.n_interface_gb]=bid2-1
                self.i_a_gb[self.n_interface_gb,:]=[ori1,side1,side2]
                self.active_edges_gb[self.n_interface_gb]=True
                self.n_interface_gb+=1
                
            
        return True
    def remove(self,bid):
        assert bid >0, "cannot remove the ground"
        self.blocks[bid-1,:]=0
        self.active_blocks[bid-1]=False
        
        interfaces_to_delete_bb = np.nonzero(np.any(self.edges_index_bb == bid-1,axis=1))
        interfaces_to_delete_bg = np.nonzero(np.any(self.edges_index_bg == bid-1,axis=1))
        interfaces_to_delete_gb = np.nonzero(np.any(self.edges_index_gb == bid-1,axis=1))
        self.active_edges_bb[interfaces_to_delete_bb]=False
        self.i_a_bb[interfaces_to_delete_bb,:]=0
        self.active_edges_bg[interfaces_to_delete_bg]=False
        self.i_a_bg[interfaces_to_delete_bg,:]=0
        self.active_edges_gb[interfaces_to_delete_gb]=False
        self.i_a_gb[interfaces_to_delete_gb,:]=0
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
        self.actions = np.zeros((4,n_sim_actions),int)
        self.blocks = np.zeros((4,maxblocks),int)
        self.grounds = np.zeros((4,n_reg),int)
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
    def add_ground(self,pos):
        #note: it is assumed that all the grounds have are the same block 
        assert self.n_ground < self.n_reg, "Attempt to add more grounds than regions"
        self.grounds[:2,self.n_ground]=-1
        self.grounds[2:,self.n_ground]=[pos[0],pos[1]]
        self.active_nodes[self.n_act+self.n_ground]=True
        self.active_edges[(self.i_r[self.n_act+self.n_ground,:]==1) | (self.i_s[self.n_act+self.n_ground,:]==1)]=True
        self.n_ground+=1
    def add_block(self,bid,blocktypeid,pos,rot):
        self.blocks[0,bid-1]=blocktypeid
        self.blocks[1,bid-1]=-1
        self.blocks[2:,bid-1]=[pos[0],pos[1]]
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
    def put(self,block,pos0,bid,floating=False,holder_rid=None):
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
                candidates_bid = self.neighbours[candidates[:,0],candidates[:,1],candidates[:,2],candidates[:,3],0]
                
                if np.all(candidates_bid==-1):
                    return False,None,None
                
                
                #if floating or np.any(switch_direction(block.neigh)np.array(np.where(self.neighbours)).T)):
                #add the block
            
            self.occ[block.parts[:,0],block.parts[:,1],block.parts[:,2]]=bid
            self.neighbours[block.neigh[:,0],block.neigh[:,1],block.neigh[:,2],block.neigh[:,3],0]=bid
            self.neighbours[block.neigh[:,0],block.neigh[:,1],block.neigh[:,2],block.neigh[:,3],1]=block.sidesid
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
                
                
                
                interfaces_ori = block.neigh[candidates_bid!=-1,2]*3+block.neigh[candidates_bid!=-1,3]
                interfaces_coord = candidates[candidates_bid!=-1]
                interfaces_sideid = block.sidesid[candidates_bid!=-1]
                interfaces_supsideid = self.neighbours[interfaces_coord[:,0],interfaces_coord[:,1],interfaces_coord[:,2],interfaces_coord[:,3],1]
                interfaces_supbid = self.neighbours[interfaces_coord[:,0],interfaces_coord[:,1],interfaces_coord[:,2],interfaces_coord[:,3],0]
                interfaces_con = self.connection[interfaces_coord[:,0],interfaces_coord[:,1],interfaces_coord[:,2]]
                
                #interface: [ori,side1,side2,bid1,bid2,con1,con2]
                interfaces = np.array([interfaces_ori,
                                       interfaces_sideid,
                                       interfaces_supsideid,
                                       bid*np.ones(interfaces_coord.shape[0]),
                                       interfaces_supbid,
                                       interfaces_con,
                                       np.min(interfaces_con)*np.ones(interfaces_coord.shape[0])]).T
                
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
           
            if distb<self.min_dist[connect_region[0],i]-1e-5:
                closer =1
                self.min_dist[connect_region[0],i]=distb
                self.min_dist[i,connect_region[0]]=distb
                self.shortest_path(i,connect_region[0])
                self.shortest_path(connect_region[0],i)
            elif distb<self.min_dist[connect_region[0],i]+1e-5:
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
    def connect(self,block,bid,sideid,support_sideid,support_bid,side_ori,holder_rid=None,idcon=None):
        oriented_neigh = self.neighbours.reshape(self.neighbours.shape[0],self.neighbours.shape[1],6,2)
        
        supportside = np.array(np.nonzero((oriented_neigh[:,:,side_ori-3,0]==support_bid)&
                                          (oriented_neigh[:,:,side_ori-3,1]==support_sideid))).T
        
        #select only the sides that have the right orientation        
        if support_bid ==0:
            supportside =  supportside[self.connection[supportside[:,0],supportside[:,1],1-side_ori//3]==idcon]
        
            
            
        if supportside.shape[0]==0:
            return False,None,None
        assert supportside.shape[0]==1, "multiple definition of side"
        
        target = switch_direction(np.concatenate([supportside,[[1-side_ori//3,side_ori%3]]],axis=1))
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
                self.neighbours[block.neigh[:,0],block.neigh[:,1],block.neigh[:,2],block.neigh[:,3],1]=block.sidesid
                if holder_rid is not None:
                    self.hold[block.parts[:,0],block.parts[:,1],block.parts[:,2]]=holder_rid
                
                #get the interfaces for the graph
                candidates = switch_direction(block.neigh)
                candidates_bid = self.neighbours[candidates[:,0],candidates[:,1],candidates[:,2],candidates[:,3],0]
                interfaces_ori = block.neigh[candidates_bid!=-1,2]*3+block.neigh[candidates_bid!=-1,3]
                interfaces_coord = candidates[candidates_bid!=-1]
                interfaces_sideid = block.sidesid[candidates_bid!=-1]
                interfaces_supsideid = self.neighbours[interfaces_coord[:,0],interfaces_coord[:,1],interfaces_coord[:,2],interfaces_coord[:,3],1]
                interfaces_supbid = self.neighbours[interfaces_coord[:,0],interfaces_coord[:,1],interfaces_coord[:,2],interfaces_coord[:,3],0]
                interfaces_con = self.connection[interfaces_coord[:,0],interfaces_coord[:,1],interfaces_coord[:,2]]
                
                #interface: [ori,side1,side2,bid1,bid2,con1,con2]
                interfaces = np.array([interfaces_ori,
                                       interfaces_sideid,
                                       interfaces_supsideid,
                                       bid*np.ones(interfaces_coord.shape[0]),
                                       interfaces_supbid,
                                       interfaces_con,
                                       np.min(interfaces_con)*np.ones(interfaces_coord.shape[0])]).T
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

class discret_block_norot():
    def __init__(self,parts,density=1,muc=0):
        self.parts = np.array(parts)
        self.neigh = outer_sides(self.parts)
        orientation_neig = self.neigh[:,2]*3+self.neigh[:,3]
        
        self.sidesid = np.zeros(self.neigh.shape[0],dtype=int)
        for ori in range(6):
            self.sidesid[orientation_neig==ori]=np.arange(len(self.sidesid[orientation_neig==ori]))
        self.mass = density*self.parts.shape[0]
        self.muc = muc
    def move(self,new_p):
        delta =new_p-self.parts[0,:-1]
        self.parts[:,:-1] = self.parts[:,:-1]+delta
        self.neigh[:,:2]=self.neigh[:,:2]+delta
    def connect(self,sideid,target):
        self.move([0,0])
        side_valid = self.neigh[(self.neigh[:,2] == target[2]) & (self.neigh[:,3] == target[3])]
        side = side_valid[sideid]
        self.move(target[:2]-side[:2])
def outer_sides(parts):
    neigh = np.zeros((3*parts.shape[0],4),dtype=int)
    neigh[:,:3] =np.tile(parts,(3,1))
    neigh[:parts.shape[0],3] = 0
    neigh[parts.shape[0]:2*parts.shape[0],3] = 1
    neigh[2*parts.shape[0]:,3] = 2
    same = same_side(neigh,neigh)
    return neigh[~np.any(same,axis=0)]