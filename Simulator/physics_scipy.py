# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 09:55:31 2022

@author: valla
"""
import time 
import numpy as np
from scipy.optimize import linprog 
from discrete_blocks import discret_block as Block, Grid, switch_direction,grid2real



class stability_solver_discrete():
    def __init__(self,
                 Fx_robot = [-100,100],
                 Fy_robot = [-100,100],
                 M_robot = [-1000,1000],
                 n_robots = 2,
                 F_max = 100,
                 ):
        self.F_max = F_max
        self.nr = n_robots
        self.A = np.diag(np.ones(self.nr*6))
        
        self.b = np.zeros(self.nr*6)
        self.b[np.arange(self.nr)*6]=Fx_robot[1]
        self.b[np.arange(self.nr)*6+1]=-Fx_robot[0]
        self.b[np.arange(self.nr)*6+2]=Fy_robot[1]
        self.b[np.arange(self.nr)*6+3]=-Fy_robot[0]
        self.b[np.arange(self.nr)*6+4]=M_robot[1]
        self.b[np.arange(self.nr)*6+5]=-M_robot[0]
        
        self.c = np.ones(self.nr*6)
        self.Aeq = np.zeros((0,self.nr*6))
        self.beq = np.zeros(0)
        self.block = np.zeros((0),dtype=int)
        self.roweqid = np.zeros((0),dtype=int)
        self.rowid = -np.ones((self.nr*6),dtype=int)
        self.xid = -np.ones((self.nr*6),dtype=int)
        
        # self.last_sol = np.zeros(0)
        
    def add_block(self,grid,block,bid,interfaces=None):

        #get all the potential supports:
        pot_sup = switch_direction(block.neigh)
        sup = pot_sup[np.nonzero(grid.neighbours[pot_sup[:,0],
                                                 pot_sup[:,1],
                                                 pot_sup[:,2],
                                                 pot_sup[:,3],0]!=-1)]
  
        ncorners = sup.shape[0]*2
        #get a list of all concerned blocks
        list_sup = np.unique(grid.neighbours[sup[:,0],sup[:,1],sup[:,2],sup[:,3],0])
        
        nAeqcorner = np.zeros((3,ncorners*3))
        
        
        sup0idx_s = np.nonzero(sup[:,3]==0)
        sup0idx = np.repeat(6*sup0idx_s[0],2)
        sup0idx[1::2]=sup0idx[::2]+3
        
        sup1idx_s = np.nonzero(sup[:,3]==1)
        sup1idx = np.repeat(6*sup1idx_s[0],2)
        sup1idx[1::2]=sup1idx[::2]+3
        
        sup2idx_s = np.nonzero(sup[:,3]==2)
        sup2idx = np.repeat(6*sup2idx_s[0],2)
        sup2idx[1::2]=sup2idx[::2]+3
        #first modify the Fx components
        #take care of the Fs
        nAeqcorner[0,sup0idx]=0
        nAeqcorner[0,sup1idx]=-np.sqrt(3)/2
        nAeqcorner[0,sup2idx]=np.sqrt(3)/2
        
        #take care of the Ffp
        nAeqcorner[0,sup0idx+1]=1
        nAeqcorner[0,sup1idx+1]=0.5
        nAeqcorner[0,sup2idx+1]=-0.5
        
        #now the Fy components:
        #Fs
        nAeqcorner[1,sup0idx]=1
        nAeqcorner[1,sup1idx]=-0.5
        nAeqcorner[1,sup2idx]=-0.5
        
        #Ffp
        nAeqcorner[1,sup0idx+1]=0
        nAeqcorner[1,sup1idx+1]=np.sqrt(3)/2
        nAeqcorner[1,sup2idx+1]=-np.sqrt(3)/2
        
        #compute the postion of each corner:
        pos = side2corners(sup)
        pos0 = pos[sup0idx_s]
        pos1 = pos[sup1idx_s]
        pos2 = pos[sup2idx_s]
        
        for idxi, posi in zip([sup0idx_s[0],sup1idx_s[0],sup2idx_s[0]],[pos0,pos1,pos2]):
            numberp=len(idxi)
            if numberp==0:
                continue
            for j,b in enumerate(idxi):
                idxj = b*6+np.arange(3)
                nAeqcorner[2,idxj]=nAeqcorner[1,idxj]*posi[j,0,0]-nAeqcorner[0,idxj]*posi[j,0,1]
                nAeqcorner[2,idxj+3]=nAeqcorner[1,idxj]*posi[j,1,0]-nAeqcorner[0,idxj]*posi[j,1,1]
        
        #take care of Ffm for all of the lines
        nAeqcorner[:,2::3]=-nAeqcorner[:,1::3]
        
        #switch the side of the up triangles
        sup_up = np.repeat(np.nonzero(sup[:,2]==0),6)
        sup_up = np.nonzero(sup[:,2]==0)
        sup_up = np.ravel(np.expand_dims(6*sup_up[0],1)+np.expand_dims(np.arange(6),0))
        nAeqcorner[:,sup_up]=-nAeqcorner[:,sup_up]
        
        
        #change the equilibrium of the support blocks
        nAeqcol = np.zeros((self.Aeq.shape[0],ncorners*3))
        
        for supid in list_sup:
            if supid == 0:
                #index reserved to the ground, ignore
                continue
            #dont ask, dont tell, go fast
            idxs = np.nonzero(grid.neighbours[sup[:,0],sup[:,1],sup[:,2],sup[:,3]]==supid)
            idxs = np.ravel(np.expand_dims(6*idxs[0],1)+np.expand_dims(np.arange(6),0))
            for i,row in enumerate(np.nonzero(self.roweqid==supid)[0]):
                nAeqcol[row,idxs] = -nAeqcorner[i,idxs]
        
        
        #fill the matrix
        nAeqrow = np.zeros((3,self.Aeq.shape[1]))
        
        self.Aeq = np.block([[self.Aeq, nAeqcol],[nAeqrow,nAeqcorner]])
        
        
        #add an index of the row:
        self.roweqid = np.concatenate([self.roweqid,bid*np.ones(3,dtype=int)])
        self.xid = np.concatenate([self.xid,bid*np.ones(ncorners*3,dtype=int)])
        self.beq = np.concatenate([self.beq, [0,block.mass,block.mass*get_cm(block.parts)[0]]])
        
        
        
        #add the friction constraints (Ffp+Ffn - Fs*muc <=0)
        Acorner = np.zeros((ncorners,3*ncorners))
        Acorner[np.arange(ncorners),np.arange(ncorners)*3]=-block.muc
        Acorner[np.arange(ncorners),np.arange(ncorners)*3+1]=1
        Acorner[np.arange(ncorners),np.arange(ncorners)*3+2]=1
        self.A = np.block([[self.A, np.zeros((self.A.shape[0],3*ncorners))],
                           [np.zeros((ncorners,self.A.shape[1])),Acorner]])
    
        self.b = np.concatenate([self.b, np.zeros(ncorners)])
        self.rowid = np.concatenate([self.rowid,bid*np.ones(ncorners)])
        self.c = np.concatenate([self.c,np.ones(ncorners*3)])
        
    def remove_block(self,bid):
        #find all columns where Aeq is not zero
        col2del = np.unique(np.nonzero(self.Aeq[self.roweqid==bid])[1])
        #remove the ones attributed to the robot:
        col2del = np.delete(col2del,np.nonzero(col2del<self.nr*6))
        
        self.Aeq = np.delete(self.Aeq,np.nonzero(self.roweqid==bid),axis=0)
        self.beq = np.delete(self.beq,np.nonzero(self.roweqid==bid))
        
        self.A = np.delete(self.A,np.nonzero(self.rowid==bid),axis=0)
        self.b = np.delete(self.b,np.nonzero(self.rowid==bid))
        
        self.Aeq = np.delete(self.Aeq, col2del,axis=1)
        
        self.A = np.delete(self.A, col2del,axis=1)
        
        self.c = np.delete(self.c,col2del)
        
        #clean the indexes
        
        self.roweqid = np.delete(self.roweqid,self.roweqid == bid)
        self.rowid = np.delete(self.rowid,self.rowid == bid)
        self.xid = np.delete(self.xid,col2del)
        
    def hold_block(self,bid,rid):
        row = np.nonzero(self.roweqid==bid)
        if row[0].shape[0] > 0:
            row0 = row[0][0]
        else:
            return False
        self.Aeq[row0:row0+3,rid*6:(rid+1)*6]= [[1,-1,0,0,0,0],
                                               [0,0,1,-1,0,0],
                                               [0,0,0,0,1,-1]]
        return True
    def leave_block(self,rid):
        self.Aeq[:,rid*6:(rid+1)*6]=0
        
    # def bid2boolarr(self,bid,idx='row'):
    #     return np.nonzero(self.block==bid)
    def solve(self):
        res = linprog(self.c,self.A,self.b,self.Aeq,self.beq)
        return res
    def reset(self):
        self.A = np.diag(np.ones(self.nr*6))
        self.b = self.b[:self.nr*6]
        self.c = np.ones(self.nr*6)
        self.Aeq = np.zeros((0,self.nr*6))
        self.beq = np.zeros(0)
        self.block = np.zeros((0),dtype=int)
        self.roweqid = np.zeros((0),dtype=int)
        self.rowid = -np.ones((self.nr*6),dtype=int)
        self.xid = -np.ones((self.nr*6),dtype=int)


# def side2Aeq(sides):
#     #given an array of sides, return an array of coefficients to get Fx and Fy from 
#     #Fs, Ffp and Ffm
#     coefs = np.zeros(sides.shape[0]*3,3)
#     sides0idxs = np.nonzero(sides[:,3]==0)
#     pos0 = np.tile(grid2real(sides[sides0idxs]),)
#     coefs[sides0idxs*3,:]= np.tile([0,1,-1,0,1,-1],sides0idxs.shape[0])
#     coefs[sides0idxs*3+1,:]=np.tile([1,0,0,1,0,0],sides0idxs.shape[0])
#     coefs[sides0idxs*3+2,:]=0
#     return coefs
    
def side2corners(sides,security_factor=0.1):
    #return the real coordinates of the two corners of the sides
    #note that p1(side)==p1(switch direction(side))
    
    #switch the direction of all down triangles:
    upsides = np.copy(sides)
    upsides[sides[:,2]==1,:]=switch_direction(sides[sides[:,2]==1,:])
    #first put each corner at the leftest point
    corners = grid2real(upsides)
    corners = np.stack([corners,corners],axis=1)
    corners[upsides[:,3]==0,0,:] = corners[upsides[:,3]==0,0,:]+[security_factor/2,0]
    corners[upsides[:,3]==0,1,:] = corners[upsides[:,3]==0,1,:]+[1-security_factor/2,0]
    
    corners[upsides[:,3]==1,0,:] = corners[upsides[:,3]==1,0,:]+[1-security_factor/4,np.sqrt(3)/2*security_factor]
    corners[upsides[:,3]==1,1,:] = corners[upsides[:,3]==1,1,:]+[0.5+security_factor/4,np.sqrt(3)/2*(1-security_factor)]
    
    
    corners[upsides[:,3]==2,0,:] = corners[upsides[:,3]==2,0,:]+[0.5-security_factor/4,np.sqrt(3)/2*(1-security_factor)]
    corners[upsides[:,3]==2,1,:] = corners[upsides[:,3]==2,1,:]+[security_factor/4,np.sqrt(3)/2*security_factor]
    return corners

def get_cm(parts):
    # parts = parts - parts[0,:]
    c_parts = np.zeros((parts.shape[0],2))
    c_parts[:,0] = parts[:,0]+parts[:,1]*0.5+0.5
    c_parts[:,1] = parts[:,1]*np.sqrt(3)/2 + (-parts[:,2]*2+1)/(2*np.sqrt(3))
    return np.mean(c_parts,axis=0)

if __name__ == '__main__':
    print("Start physics test")
    maxs = [9,6]
    t00 = time.perf_counter()
    ph_mod = stability_solver_discrete(n_robots=0)
    grid = Grid(maxs)
    t = Block([[0,0,0]])
    tr = Block([[1,1,1],[1,1,0],[1,2,1]],muc=0.7)
    # ground = Block([[i,0,1] for i in range(maxs[0])])
    # hinge = Block([[1,0,0],[0,1,1],[1,1,1],[1,1,0],[0,2,1],[0,1,0]])
    # link = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]])
    ground = Block([[0,0,0],[2,0,0],[6,0,0],[8,0,0]]+[[i,0,1] for i in range(0,maxs[0])],muc=0.5)
    hinge = Block([[1,0,0],[0,1,1],[1,1,1],[1,1,0],[0,2,1],[0,1,0]],muc=0.7)
    link = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=0.7)
    #build an arch: the physics simulator should only result in a pass if 
    #no block is missing
    
    
    # grid.put(hinge,[1,3],0,4)
    # grid.put(link,[1,5],5,5)
    # grid.put(hinge,[4,3],0,6)
    # grid.put(link,[5,3],5,7)
    # grid.put(hinge,[7,0],0,8)
    t01 = time.perf_counter()
    print(f"the setup was done in {t01-t00}s")
    
    t10 = time.perf_counter()
    #grid.put(t,[0,0],0,1,floating=True)
    #ph_mod.add_block(grid,t,1,base = True)
    grid.put(ground,[0,0],0,1,floating=True)
    t11 = time.perf_counter()        
    print(f"The ground was put in {t11-t10}s")
    t20 = time.perf_counter()
    grid.put(hinge,[1,0],0,2)
    ph_mod.add_block(grid,hinge, 2)
    #ph_mod.hold_block(2, 0)
    t21 = time.perf_counter()
    print(f'block 2 put in {t21-t20}s')
    t200 = time.perf_counter()
    res2 = ph_mod.solve()
    t201 = time.perf_counter()
    print(f'block 2 solved in {t201-t200}s ({res2.success=})')
    
    t30 = time.perf_counter()
    grid.put(link,[0,2],0,3)
    ph_mod.add_block(grid,link, 3)
    #ph_mod.hold_block(3, 0)
    res3 = ph_mod.solve()
    t31 = time.perf_counter()
    print(f'block 3 put and solved in {t31-t30}s ({res3.success=})')
    grid.put(hinge,[1,3],0,4)
    ph_mod.add_block(grid,hinge, 4)
    res4 = ph_mod.solve()
    
    grid.put(link,[1,5],5,5)
    ph_mod.add_block(grid,link, 5)
    res5 = ph_mod.solve()
    
    grid.put(hinge,[4,3],0,6)
    ph_mod.add_block(grid,hinge, 6)
    res6 = ph_mod.solve()
    
    grid.put(link,[5,3],5,7)
    ph_mod.add_block(grid,link, 7)
    res7 = ph_mod.solve()
    
    grid.put(hinge,[7,0],0,8)
    ph_mod.add_block(grid,hinge, 8)
    
    res8 = ph_mod.solve()
    print(f'Test Passed: {res8.success}\n total time: {time.perf_counter()-t10}s')
    
    print("End physics test")