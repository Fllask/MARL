# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 09:55:31 2022

@author: valla
"""
from Blocks import Block
import numpy as np
def check_moments(block:Block,m_top:float=0,cm_top:np.array=None):
    if block.support is None:
        return True        
    else:
        cm_block = (block.corners[0]+block.corners[1])/2
        if cm_top is not None:
            cm = (cm_top*m_top+cm_block*block.mass)/(m_top+block.mass)
        else:
            cm = cm_block
        print(cm)
        pivot_l = None
        pivot_r = None
        #find the pivot
        for support_el in block.support:
            if pivot_l is None or support_el.corners[0,0]<pivot_l:
                pivot_l = support_el.corners[0,0]
            if pivot_r is None or support_el.corners[1,0] > pivot_r:
                pivot_r = support_el.corners[1,0]
        if pivot_l < block.corners[0,0]:
            pivot_l = block.corners[0,0]
        if pivot_r > block.corners[1,0]:
            pivot_r = block.corners[1,0]
        return cm[0]>pivot_l and cm[0]<pivot_r
def check_support(block, support = None,tol=1e-5):
    if support is None:
        support = block.support
    if block.corners.shape == (2,2):
        
        y = np.min(block.corners[0,1])
        res = np.ones(len(support),dtype=bool)
        
        for i,support_el in enumerate(support):
            if abs(y - support_el.corners[1,1])>tol:
                res[i] = False
            elif support_el.corners[0,0]>block.corners[1,0]:
                res[i] = False
            elif support_el.corners[1,0]<block.corners[0,0]:
                res[i] = False
    assert np.sum(res) <= 2, "system overconstrained"
    return res
def check_stability():
    #use gekko
    pass
if __name__ == '__main__':
    print("Start physics test")
    b1 = Block([[0,0],[1,1]])
    b11 = Block([[-1,-1],[1,1]])
    b2 = Block([[-2.9,1],[3,2]],support=[b1,b11])
    print(check_support(b2))
    print(check_moments(b2))
    print("End physics test")