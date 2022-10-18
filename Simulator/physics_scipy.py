# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 09:55:31 2022

@author: valla
"""
from scipy import 
from Blocks import discret_block as Block, Grid, switch_direction
import numpy as np
import time 

class stability_solver_discrete():
    def __init__(self,
                 maxs,
                 Fx_robot = [-100,100],
                 Fy_robot = [-100,100],
                 M_robot = [-1000,1000],
                 n_robots = 2,
                 F_max = 100,
                 ):
        self.F_max = F_max
        self.nr = n_robots
        self.A = np.diag(np.ones(self.nr*6))
        self.A[1::2,1::2]=-self.A[1::2,1::2]
        self.b = np.zeros((self.nr*6,1))
        self.b[np.arange(self.nr)*6]=Fx_robot[1]
        self.b[np.arange(self.nr)*6+1]=Fx_robot[0]
        self.b[np.arange(self.nr)*6+2]=Fy_robot[1]
        self.b[np.arange(self.nr)*6+3]=Fy_robot[0]
        self.b[np.arange(self.nr)*6+4]=M_robot[1]
        self.b[np.arange(self.nr)*6+5]=M_robot[0]
        
        self.c = np.ones((self.nr*6,1))*10
        self.Aeq = np.zeros((0,self.nr*6))
        self.beq = np.zeros((self.nr*6,1))
        self.block = np.zeros((self.nr*6),dtype=int)
        
        
    def add_block(self,grid,block,bid,base=False,held=False):
        self.block = np.concatenate([self.block,bid*np.ones(2*n_sup)])
        self.beq = np.concatenate([self.beq, [block.mass,get_cm(block.parts)[0]]])
        #chage the equilibrium of the support blocks
        
        sup = np.nonzero(grid.neigh)
        self.A = np.concatenate([self.A, np.zeros(self.A.shape[0],])
    def hold_block(self,bid,rid):
        self.Aeq[:,self.bid2boolarr(bid)]
    def leave_block(self,rid):
        self.A[:,rid:rid+6]=0
        self.Aeq[:,rid:rid+6]=0
        
    def bid2boolarr(self,bid):
        
        
        return np.nonzero(self.block==bid)
    def solve(self):
        pass
def buildA():
    pass
def buildC(rFx,rFy,rm,fs,ff,m):
    c = np.ones(rFx.shape[0]+rFy.shape[0])
def buildB():
    pass
def buildAeq():
    pass
def buildvec(rFx,rFy,rm,fs,ff,m):
    pass
    
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