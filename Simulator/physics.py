# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 09:55:31 2022

@author: valla
"""
from gekko import GEKKO
from Blocks import discret_block as Block, Grid, switch_direction
import numpy as np
import time 

class stability_solver_discrete():
    def __init__(self,
                 maxs,
                 n_robot = 2,
                 n_block_max = 100,
                 Fx_robot = [-100,100],
                 Fy_robot = [-100,100],
                 M_robot = [-1000,1000],
                 F_max = 100,
                 muc=0
                 ):
        
        self.F_max = F_max
        self.gk = GEKKO(remote=True)
        self.gk.options.LINEAR = 1
        self.gk.options.SOLVER = 1 # APOPT
        self.gk.options.IMODE = 3
        self.gk.options.REDUCE = 0
        self.rFx = np.array([self.gk.CV(0,lb=Fx_robot[0],ub=Fx_robot[1],name=f'rFx{i}') for i in range(n_robot)])
        self.rFy = np.array([self.gk.CV(0,lb=Fx_robot[0],ub=Fx_robot[1],name=f'rFy{i}') for i in range(n_robot)])
        self.rM = np.array([self.gk.CV(0,lb=Fx_robot[0],ub=Fx_robot[1],name=f'rM{i}') for i in range(n_robot)])
        self.gk.Minimize(np.sum(self.rFx*self.rFx)/(Fx_robot[1]-Fx_robot[0])+
                          np.sum(self.rFy*self.rFy)/(Fy_robot[1]-Fy_robot[0])+
                          np.sum(self.rM*self.rM)/(M_robot[1]-M_robot[0]))
        #initialize all the forces
        self.fs = np.reshape([ self.gk.Var(0,lb=0,ub = F_max,name=f"fs{x}_{y}_{ori}_{side}")
                             for x in range(maxs[0]+1) for y in range(maxs[1]+1) for ori in range(2) for side in range(3)],
                            (maxs[0]+1,maxs[1]+1,2,3))
        self.ff=np.reshape([ self.gk.Var(0,lb=-F_max,ub = F_max,name=f"ff{x}_{y}_{ori}_{side}")
                          for x in range(maxs[0]+1) for y in range(maxs[1]+1) for ori in range(2) for side in range(3)],
                          (maxs[0]+1,maxs[1]+1,2,3))
        self.moments = np.reshape([ self.gk.Var(0,lb=-F_max,ub = F_max,name=f"m{x}_{y}_{ori}_{side}")
                          for x in range(maxs[0]+1) for y in range(maxs[1]+1) for ori in range(2) for side in range(3)],
                          (maxs[0]+1,maxs[1]+1,2,3))
        #initialize the array of weights
        self.mass = np.array([self.gk.Param(0,name=f"mass{i}") for i in range(n_block_max)])
        #only the x coodinate is usefull as the force is vertical
        self.xcm = np.array([self.gk.Param(0,name=f"xcm{i}") for i in range(n_block_max)])
        #keep the equations for each block
        self.eqs = np.empty(n_block_max,dtype=object)
        # add the friction constraints:
        for x in range(maxs[0]): 
            for y in range(maxs[1]):
                #add the moment constraints:
                self.gk.Equations([self.moments[x,y,0,0]<=self.fs[x,y,0,0]*(x+1+0.5*y)+
                                                          -self.ff[x,y,0,0]*np.sqrt(3)/2*y,
                                   self.moments[x,y,0,0]>=self.fs[x,y,0,0]*(x+0.5*y)+
                                                          -self.ff[x,y,0,0]*np.sqrt(3)/2*y,
                                   self.moments[x,y,0,1]<=self.fs[x,y,0,0]*((y+1)/2-x/2)+
                                                          self.ff[x,y,0,0]*np.sqrt(3)/2*(x+y+1),
                                   self.moments[x,y,0,1]>=self.fs[x,y,0,0]*(y/2-(x+1)/2)+
                                                          self.ff[x,y,0,0]*np.sqrt(3)/2*(x+y+1),
                                   self.moments[x,y,0,2]<=self.fs[x,y,0,0]*(-y+1-x/2)+
                                                          -self.ff[x,y,0,0]*np.sqrt(3)/2*(x),
                                   self.moments[x,y,0,2]>=self.fs[x,y,0,0]*(-y-x/2)+
                                                          -self.ff[x,y,0,0]*np.sqrt(3)/2*(x)
                                  ])
                for side in range(3):
                    #add the side/side connections:
                    s = np.squeeze(switch_direction(np.array([[x,y,0,side]])))
                    self.gk.Equations([self.fs[s[0],s[1],s[2],s[3]]==-self.fs[x,y,0,side],
                                       self.ff[s[0],s[1],s[2],s[3]]==-self.ff[x,y,0,side],
                                       self.moments[s[0],s[1],s[2],s[3]]==-self.moments[x,y,0,side]])
                    #add the friction constraints
                    self.gk.Equations([self.ff[x,y,0,side] >= - muc*self.fs[x,y,0,side],
                                       self.ff[x,y,0,side] <=   muc*self.fs[x,y,0,side]])
                    #fix all the forces
                    self.gk.fix(self.fs[x,y,0,side],0)
                        
                        
        self.blocks = np.empty(n_block_max,dtype=object)
        self.n_blocks =0
        self.gk.solve()
    def add_block(self,grid,block,bid,base=False,held=False):
        assert not (base and held),"Cannot hold the ground"
        if base:
            return
        self.mass[bid].value = block.mass
        self.xcm[bid].value = get_cm(block.parts)[0]
        
        #free the support forces and the internal forces:
        actneigh = switch_direction(block.neigh)
        support = actneigh[np.nonzero(grid.neighbours[actneigh[:,0],
                                                      actneigh[:,1],
                                                      actneigh[:,2],
                                                      actneigh[:,3]])[0]]
        #only the up triangles are fixed
        support[support[:,2]==1] = switch_direction(support[support[:,2]==1])
        [self.gk.free(self.fs[i[0],i[1],i[2],i[3]]) for i in support]
        #create the internal constraints
        idx0 = block.neigh[block.neigh[:,3]==0]
        idx1 = block.neigh[block.neigh[:,3]==1]
        idx2 = block.neigh[block.neigh[:,3]==2]
        pol = -2*block.neigh[:,2]+1
        if held:
            self.eqs[bid] = self.gk.Equations([
                #sum Fx
                    #fs
                        #side0 = 0
                        #side1
                        -self.gk.sum(self.fs[idx1[:,0],idx1[:,1],idx1[:,2],idx1[:,3]]*np.sqrt(3)/2*pol[block.neigh[:,3]==1])+
                        #side2
                        self.gk.sum(self.fs[idx2[:,0],idx2[:,1],idx2[:,2],idx2[:,3]]*np.sqrt(3)/2*pol[block.neigh[:,3]==2])+
                    #ff
                        #side0
                        self.gk.sum(self.ff[idx0[:,0],idx0[:,1],idx0[:,2],idx0[:,3]]*pol[block.neigh[:,3]==0])+
                        #side1
                        -self.gk.sum(self.ff[idx1[:,0],idx1[:,1],idx1[:,2],idx1[:,3]]*0.5*pol[block.neigh[:,3]==1])+
                        #side2
                        -self.gk.sum(self.ff[idx2[:,0],idx2[:,1],idx2[:,2],idx2[:,3]]*0.5*pol[block.neigh[:,3]==2])+
                    #robot
                        self.rFx[held] == 0,
                #sum Fy
                    #fs
                        #side0
                        self.gk.sum(self.fs[idx0[:,0],idx0[:,1],idx0[:,2],idx0[:,3]]*pol[block.neigh[:,3]==0])+
                        #side1
                        -self.gk.sum(self.ff[idx1[:,0],idx1[:,1],idx1[:,2],idx1[:,3]]*0.5*pol[block.neigh[:,3]==1])+
                        #side2
                        -self.gk.sum(self.ff[idx2[:,0],idx2[:,1],idx2[:,2],idx2[:,3]]*0.5*pol[block.neigh[:,3]==2])+
                    #ff
                        #side0=0
                        #side1
                        -self.gk.sum(self.fs[idx1[:,0],idx1[:,1],idx1[:,2],idx1[:,3]]*np.sqrt(3)/2*pol[block.neigh[:,3]==1])+
                        #side2
                        self.gk.sum(self.fs[idx2[:,0],idx2[:,1],idx2[:,2],idx2[:,3]]*np.sqrt(3)/2*pol[block.neigh[:,3]==2])+
                        
                    #robot
                        self.rFy[held]+
                    #weight 
                        -self.mass[bid]==0,
                #sum M
                    self.gk.sum(self.moments[block.neigh[:,0],
                                        block.neigh[:,1],
                                        block.neigh[:,2],
                                        block.neigh[:,3]])+
                    -self.xcm[bid]*self.mass[bid]+
                    self.rM[held] ==0
                    
                ])
        else:
            #same equations, without the robot force
            self.eqs[bid] = self.gk.Equations([
                #sum Fx
                    #fs
                        #side0 = 0
                        #side1
                        -np.sum(self.fs[idx1[:,0],idx1[:,1],idx1[:,2],idx1[:,3]]*np.sqrt(3)/2*pol[block.neigh[:,3]==1])+
                        #side2
                        np.sum(self.fs[idx2[:,0],idx2[:,1],idx2[:,2],idx2[:,3]]*np.sqrt(3)/2*pol[block.neigh[:,3]==2])+
                    #ff
                        #side0
                        np.sum(self.ff[idx0[:,0],idx0[:,1],idx0[:,2],idx0[:,3]]*pol[block.neigh[:,3]==0])+
                        #side1
                        -np.sum(self.ff[idx1[:,0],idx1[:,1],idx1[:,2],idx1[:,3]]*0.5*pol[block.neigh[:,3]==1])+
                        #side2
                        -np.sum(self.ff[idx2[:,0],idx2[:,1],idx2[:,2],idx2[:,3]]*0.5*pol[block.neigh[:,3]==2])== 0,
                #sum Fy
                    #fs
                        #side0
                        np.sum(self.fs[idx0[:,0],idx0[:,1],idx0[:,2],idx0[:,3]]*pol[block.neigh[:,3]==0])+
                        #side1
                        -np.sum(self.ff[idx1[:,0],idx1[:,1],idx1[:,2],idx1[:,3]]*0.5*pol[block.neigh[:,3]==1])+
                        #side2
                        -np.sum(self.ff[idx1[:,0],idx2[:,1],idx2[:,2],idx2[:,3]]*0.5*pol[block.neigh[:,3]==2])+
                    #ff
                        #side0=0
                        #side1
                        -np.sum(self.fs[idx1[:,0],idx1[:,1],idx1[:,2],idx1[:,3]]*np.sqrt(3)/2*pol[block.neigh[:,3]==1])+
                        #side2
                        np.sum(self.fs[idx2[:,0],idx2[:,1],idx2[:,2],idx2[:,3]]*np.sqrt(3)/2*pol[block.neigh[:,3]==2])+
                        
                    #weight 
                        -self.mass[bid]==0,
                #sum M
                    np.sum(self.moments[block.neigh[:,0],
                                        block.neigh[:,1],
                                        block.neigh[:,2],
                                        block.neigh[:,3]])+
                    -self.xcm[bid]*self.mass[bid]==0
                ])
        self.gk.solve()
    def add_constraint():
        pass
    def remove_constraint():
        pass
    def cleanup(self):
        self.gk.cleanup()
def get_cm(parts):
    # parts = parts - parts[0,:]
    c_parts = np.zeros((parts.shape[0],2))
    c_parts[:,0] = parts[:,0]+parts[:,1]*0.5+0.5
    c_parts[:,1] = parts[:,1]*np.sqrt(3)/2 + (-parts[:,2]*2+1)/(2*np.sqrt(3))
    return np.mean(c_parts,axis=0)
def create_constraints(grid):
    #add all the weights
    pass
def check_stability():
    #use gekko
    pass
if __name__ == '__main__':
    print("Start physics test")
    maxs = [9,6]
    t00 = time.perf_counter()
    ph_mod = stability_solver_discrete(maxs,n_robot=2,n_block_max=10)
    grid = Grid(maxs)
    t = Block([[0,0,0]])
    # ground = Block([[i,0,1] for i in range(maxs[0])])
    # hinge = Block([[1,0,0],[0,1,1],[1,1,1],[1,1,0],[0,2,1],[0,1,0]])
    # link = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]])
    ground = Block([[0,0,0],[2,0,0],[6,0,0],[8,0,0]]+[[i,0,1] for i in range(3,maxs[0])])
    hinge = Block([[1,0,0],[0,1,1],[1,1,1],[1,1,0],[0,2,1],[0,1,0]])
    link = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]])
    #build an arch: the physics simulator should only result in a pass if 
    #no block is missing
    
    
    # grid.put(link,[0,2],0,3)
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
    ph_mod.add_block(grid, ground, 1,base=True)
    t11 = time.perf_counter()        
    print(f"The ground was put in {t11-t10}s")
    t20 = time.perf_counter()
    grid.put(hinge,[1,0],0,2)
    ph_mod.add_block(grid, hinge, 2,held=1)
    t21 = time.perf_counter()
    print(f'block 1 solved in {t21-t20}s')
    print("End physics test")