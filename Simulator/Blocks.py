# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 09:58:11 2022

@author: valla
"""
import numpy as np
import abc
import matplotlib.pyplot as plt
np.seterr(invalid='ignore')


class Block():
    #Building blocks for the tower
    #can be initiated in 3 different ways:
        #-by giving its bottom left corner and top right corner
        #-by giving its bottom left corner, width and height
        #by defining an arbitrary quadrilateral with 4 points:
            #in this case, the corners given will be stored with the closest point 
            #to the origine as 0, and then going counterclockwise 
    def __init__(self,
                 corners: np.array = None,#([[blx,bly],
                                            #[trx,try]])
                 bl: np.array = None,
                 w: float = None,
                 h: float = None,                   
                 colors: dict = {},
                 density:float =1,
                 muc: float = 1,
                 holded:bool = True,
                 support:list = None):
        if corners is not None:
            if type(corners)==list:
                corners = np.array(corners)
            assert corners.shape == (2,2) or corners.shape==(4,2), "Corner error, 2 or 4 points needed"
            assert w is None, "Block over constrained, remove w"
            assert h is None, "Block over constrained, remove h"
            assert bl is None, "Block over constrained, remove bl"
            #rectangular block
            if corners.shape == (2,2):
                self.w = corners[1,0]-corners[0,0]
                self.h = corners[1,1]-corners[0,1]
                self.mass = density*self.h*self.w
                self.corners = np.copy(corners)
                self.angles = np.array([0,np.pi/2,0,np.pi/2])
            #arbitrary 4 sided shape
            else:
                self.w = np.max(corners[:,0])-np.min(corners[:,0])
                self.h = np.max(corners[:,1])-np.min(corners[:,1])
                #reorder the points:
                neworder = np.arange(4)
                    #first find the lowest point, and put it first
                new0 = np.argmin(np.sum(corners, axis=1))
                neworder[new0],neworder[0]=neworder[0],neworder[new0]
                #get the angles to the first point:
                angles0 = np.arctan2(corners[:,1]-corners[new0,1],
                                     corners[:,0]-corners[new0,0])+np.pi
                #order the angles in increasing order (delete the origine point)
                neworder[1:]=neworder[1:][np.argsort(np.delete(angles0,new0))]
                
                #reorder the points and angles:
                corners = corners[neworder]
                angles0 = angles0[neworder]-np.pi
                #get the missing orientations
                angles2 = np.arctan2(corners[:,1]-corners[2,1],
                                     corners[:,0]-corners[2,0])
                self.angles = np.array([angles0[1],
                                        angles2[1]+np.pi,
                                        angles2[3],
                                        angles0[3]+np.pi])
                
                diag = corners[2]-corners[0]
                diag_l = np.sqrt(np.sum(np.square(diag)))
                #subtract the diagonal angle to the side
                angles_sep = np.delete(angles0-angles0[2],[0,2])
                sides = np.delete(corners-corners[0,:],[0,2],axis=0)
                sides_l = np.sqrt(np.sum(np.square(sides),axis=1))
                area = diag_l*np.sum(sides_l*np.abs(np.sin(angles_sep)))/2
                self.mass = area*density
                self.corners = corners
                
        else:
            assert w is not None, "Block underconstraind, add w"
            assert h is not None, "Block underconstrained, add h"
            assert bl is not None, "Block underconstrained, add bl"
            self.w = w
            self.h = h
            self.mass = density*self.h*self.w
            self.corners = np.array([[bl[0],bl[1]],[bl[0]+w,bl[1]+h]])
        self.density = density
        self.muc = muc
        self.holded = holded
        self.colors = colors
        self.support = support
        
    def expand(self):
        if self.corners.shape == (2,2):
            self.corners = np.array([self.corners[0],
                                     [self.corners[1,0],self.corners[0,1]],
                                      self.corners[1],
                                      [self.corners[0,0],self.corners[1,1]]
                                     ])
            self.angles = np.array([0,np.pi/2,np.pi,-np.pi/2])
        return self
    def __str__(self):
        if self.corners.shape == (2,2):
            
            return (f"Rectangle located between ({self.corners[0,0]:.2f},{self.corners[0,1]:.2f})"+
                    f" and ({self.corners[1,0]:.2f},{self.corners[1,1]:.2f})"+
                   f" (color info: {self.colors})")
        else:
            return (f"Block located between ({self.corners[0,0]:.2f},{self.corners[0,1]:.2f}), "+
                    f"({self.corners[1,0]:.2f},{self.corners[1,1]:.2f}), "+
                    f"({self.corners[2,0]:.2f},{self.corners[2,1]:.2f}), "+
                    f" and ({self.corners[3,0]:.2f},{self.corners[3,1]:.2f})"+
                   f" (color info: {self.colors})")
    def move(self,old_p,new_p):
        self.corners = self.corners-old_p+new_p
    def turn(self, angle,center = [0,0]):
        if self.corners.shape == (2,2):
            self.expand()
        self.angles = (self.angles + angle + np.pi)%(2*np.pi)-np.pi
        corners_centered = self.corners-center
        corners_rot = np.stack([corners_centered[:,0]*np.cos(angle)-
                                corners_centered[:,1]*np.sin(angle),
                                corners_centered[:,0]*np.sin(angle)+
                                corners_centered[:,1]*np.cos(angle)],axis=1)
        self.corners = corners_rot+center
class pivot():
    def __init__(self,pos, mode, valid_range, block1,block2):
        self.pos = pos
        self.x = x
        self.valid = False
    def set_mode(self,mode):
        assert mode == 'f' or mode == 'm','unkwnown pivot mode'
        self.mode = mode
        
        
    def validate(self,x):
        if self.mode == 'fpoint':
            if abs(x-self.x)<self.eps:
                self.valid = True

class Interface(metaclass=abc.ABCMeta):
    def __init__(self, blockm:Block, blocksf:list):
        super().__init__()
        self.blockm = blockm
        self.blocksf = blocksf
    @abc.abstractmethod
    def set_x(self,x):
       pass
    @abc.abstractmethod
    def complete(self,blocks:list):
        pass
    @abc.abstractmethod
    def discretize(self,n,add_contact=False,candidates:list = None):
        pass
class Slide(Interface):
    def __init__(self,blockm,blockf,facem,facef,x=None,eps = 1e-5,safe=False):
        super().__init__(blockm,[blockf])
        self.i =0
        self.safe = safe
        self.facem = facem
        self.facef = facef
        self.eps = eps
        self.contact_av = []
        self.max_list = []
        self.min_list = []
        self.slide_list = []
        self.smin_list = []
        self.smax_list =[]
        self.pivot_list = []
        self.n_obst = [0]
        self.alphaf = self.blocksf[0].angles[self.facef]
        self.completed = False
        self.blockm.turn(self.alphaf-self.blockm.angles[self.facem]+np.pi)
        self.c1m = self.blockm.corners[self.facem]
        self.c2m = self.blockm.corners[(self.facem+1)%4]
        self.c1f = self.blocksf[0].corners[self.facef]
        self.c2f = self.blocksf[0].corners[(self.facef+1)%4]
        self.valid_range = None
        
        self.l = self.c2m-self.c1m-self.c2f+self.c1f
        self.start_p = self.c2f-self.c2m+self.c1m
        #store the distances between the origine point(c1m) and all the other corners
        #self.deltam = self.blockm.corners-self.c1m
        self.constraints = []
        if x is not None:
            self.set_x(x)
        else:
            self.x = None
    def set_x(self,x,physical=True):
        if self.safe:
            assert len(self.valid_range)>0, 'Invalid interface: no available range'
            range_l = np.array([i[1]-i[0] for i in self.valid_range])
            tot_range = np.sum(range_l)
            cum_prop = np.zeros(len(range_l))
            for i,r in enumerate(range_l):
                cum_prop[i] = cum_prop[i-1]+r/tot_range
            #find the max cum_prop < x
            base = np.searchsorted(cum_prop, x)
            if base > 0:
                x = (x-cum_prop[base-1])*tot_range+self.valid_range[base][0]
            else:
                x = x*tot_range+self.valid_range[base][0]
        self.blockm.move(self.c1m,self.start_p+x*self.l)
        self.c1m = self.blockm.corners[self.facem]
        self.c2m = self.blockm.corners[(self.facem+1)%4]
        if physical:
            self.x = x
        
        if abs(self.l[0])<self.eps:
            self.slope = self.l[1]/self.eps
        else:
            self.slope = self.l[1]/self.l[0]
    def discretize(self,n,add_contact=False,candidates:list=None):
        discrete_x = np.arange(0,1,n)
        if add_contact:
            assert False, "Not implemented"
        return discrete_x
    def complete(self,blocks:list):
        self.complete = True
        
        pivot_l = []
        for ib,b in enumerate(blocks):
            #tocheck: define wich side are stil to check
            intersect = np.zeros((4,4),dtype=bool)
            tocheck = np.ones((2,4,4),dtype=bool)
            for c in range(4):
                for m in range(4):
                    
                    start_m = self.start_p - self.c1m + self.blockm.corners[m]
                    x = corner_corner_intersection(start_m,
                                                    b.corners[c],
                                                    self.l)
                    
                    if type(x) is not bool:
                        self.update_range(x,
                                           np.arctan2(self.l[1],self.l[0]),
                                           (self.blockm.angles[m-1])%(2*np.pi)-np.pi,
                                           self.blockm.angles[m],
                                           (b.angles[c-1])%(2*np.pi)-np.pi,
                                           b.angles[c])
                        intersect[m,c]=True
                        

            (idxm,idxc)= np.where(intersect)
            
            for c in idxc:
                for m in idxm:
                    if intersect[m-1,c]:
                        #create a mranged pivottocheck[:,m,c]=False
                        tocheck[0,m-1,:]=False
                        
                        
                    if intersect[m,c-1]:
                        #create a franged pivottocheck[:,m,c]=False
                        tocheck[1,:,c-1]=False
                        
                    elif (not intersect[(m+1)%4,c] and not intersect[m,(c+1)%4]
                          and not intersect[m-1,c]):
                        #create a fpoint pivot
                        pass
                
            for c in range(4):
                deltai = b.corners[c]-b.corners[c-1]
                for m in range(4):
                    deltam = self.blockm.corners[m]-self.blockm.corners[m-1]
                    
                          
                        
                    if tocheck[0,m,c]:
                    #--------sideo-cornerm---------
                    
                        x = side_corner_intersection(b.corners[c-1], deltai, 
                                                     start_m, self.l, 1-self.eps,self.eps)
                        
                        if type(x) is not bool:
                            #p = pivot(x,start_m+x*self.l)
                            self.update_range(x,
                                                np.arctan2(self.l[1],self.l[0]),
                                                (self.blockm.angles[m])%(2*np.pi)-np.pi,
                                                self.blockm.angles[m],
                                                (b.angles[c-1])%(2*np.pi)-np.pi,
                                                b.angles[c])
                    if m == 1 and c == 3:
                        pass
                    if tocheck[1,m,c]:
                        #cornero-sidem
                        
                        x = side_corner_intersection(start_m,deltam,
                                                      b.corners[c],-self.l, 1-self.eps,self.eps)
                        if type(x) is not bool:
                            #p=pivot(x,start_m+self.l*x)
                            #print(f"intersection at {x=}: side: n° {(m-1)%4} of mobile, corner {c} of obstacle {ib}")
                            self.update_range(x,
                                                np.arctan2(self.l[1],self.l[0]),
                                                (self.blockm.angles[m-1])%(2*np.pi)-np.pi,
                                                self.blockm.angles[m],
                                                (b.angles[c])%(2*np.pi)-np.pi,
                                                b.angles[c])
                    
    def update_range(self,candidate_x,phi,alpha0m,alpha1m,alpha0o,alpha1o):
        angles = np.array([alpha0m,alpha1o,alpha0o,alpha1m])
        order = np.argsort(angles)
        if np.all(((order-order[0])%4 == [0,1,2,3])):
            #check if phi is equal to a side
            isequal = abs(angles-phi)%(2*np.pi) <self.eps
            if np.sum(isequal) >1:
                return False
            if isequal[0]:
                invphi = (phi)%(2*np.pi)-np.pi
                if (invphi -alpha0o+self.eps)%(2*np.pi)>np.pi:
                    self.min_list.append(candidate_x)
                    return True
                else: 
                    return False
            if isequal[3]:
                invphi = (phi)%(2*np.pi)-np.pi
                if (invphi -alpha1o+self.eps)%(2*np.pi)>np.pi:
                    self.min_list.append(candidate_x)
                    return True
                else: 
                    return False
            if isequal[1]:
                invphi = (phi)%(2*np.pi)-np.pi
                if (invphi -alpha0m-self.eps)%(2*np.pi)>np.pi:
                    self.max_list.append(candidate_x)
                    return True
                else: 
                    return False
            if isequal[2]:
                invphi = (phi)%(2*np.pi)-np.pi
                if (invphi -alpha1m-self.eps)%(2*np.pi)>np.pi:
                    self.max_list.append(candidate_x)
                    return True
                else: 
                    return False
            # if isequal[0] or isequal[3]:
            #     #check if phi+pi is between alpha1o and alpha0o
            #      invphi = (phi)%(2*np.pi)-np.pi
            #      if (((invphi - alpha1m-self.eps)%(2*np.pi)<np.pi and (invphi - alpha0m+self.eps)%(2*np.pi)>np.pi) or
            #          ((invphi - alpha0m-self.eps)%(2*np.pi)<np.pi and (invphi - alpha1m+self.eps)%(2*np.pi)>np.pi)):
            #          self.min_list.append(candidate_x)
            #          return True
            #      else:
            #          return False
            # if isequal[1] or isequal[2]:
            #     #check if phi+pi is between alpha1m and alpha0m
            #      invphi = (phi)%(2*np.pi)-np.pi
            #      if (((invphi - alpha1o-self.eps)%(2*np.pi)<np.pi and (invphi - alpha0o+self.eps)%(2*np.pi)>np.pi) or
            #          ((invphi - alpha0o-self.eps)%(2*np.pi)<np.pi and (invphi - alpha1o+self.eps)%(2*np.pi)>np.pi)):
            #          self.max_list.append(candidate_x)
            #          return True
            #      else:
            #          return False
            
            pos = np.searchsorted(angles[order],phi)%4
            
            if order[pos] == 0:
                #phi is between alpha0m and alpha1m
                self.min_list.append(candidate_x)
                return True
            elif order[pos] == 2:
                #phi is between alpha0o and alpha1o
                self.max_list.append(candidate_x)
                return True
            elif order[pos]==1:
                #phi is between alpha0m and alpha1o
                if (alpha1m-phi+np.pi+self.eps)%(2*np.pi)>np.pi:
                    self.max_list.append(candidate_x)
                    return True
                elif (alpha0o-phi+np.pi-self.eps)%(2*np.pi)<np.pi:
                    self.min_list.append(candidate_x)
                    return True
            elif order[pos]==3:
                if (alpha0m-phi+np.pi+self.eps)%(2*np.pi)>np.pi:
                    self.min_list.append(candidate_x)
                    return True
                elif (alpha1o-phi+np.pi-self.eps)%(2*np.pi)<np.pi:
                    self.max_list.append(candidate_x)
                    return True
        return False
            
                
    def update_bounds(self,x,phi,alpha0m,alpha1m,alpha0o,alpha1o):
        
        #check for degenerate angles
        if abs(alpha0m-alpha1o)%(2*np.pi)<self.eps:
            if abs(alpha1m-alpha0o)%(2*np.pi)<self.eps:
                return False
            else:
                if abs(alpha0m-phi)%(2*np.pi)<self.eps:
                    if abs(alpha0o-alpha1o-np.pi)%(2*np.pi)<self.eps:
                        #if the two obstacle sides are parallel => ignore
                        return False
                    else:
                        self.smin_list.append(x)
                    
                elif abs(alpha0m-phi-np.pi)%(2*np.pi)<self.eps:
                    if abs(alpha0o-alpha1o-np.pi)%(2*np.pi)<self.eps:
                        #if the two obstacle sides are parallel => ignore
                        return False
                    else:
                        self.smax_list.append(x)
                elif (alpha0m-phi)%(2*np.pi)<np.pi:
                    self.min_list.append(x)
                else:
                    self.max_list.append(x)
        elif abs(alpha1m-alpha0o)%(2*np.pi)<self.eps:
            if abs(alpha1m-phi)%(2*np.pi)<self.eps:
                if abs(alpha0o-alpha1o-np.pi)%(2*np.pi)<self.eps:
                    #if the two obstacle sides are parallel => ignore
                    return False
                else:
                    self.smax_list.append(x)
                
            elif abs(alpha1m-phi-np.pi)%(2*np.pi)<self.eps:
                if abs(alpha0o-alpha1o-np.pi)%(2*np.pi)<self.eps:
                    #if the two obstacle sides are parallel => ignore
                    return False
                else:
                    self.smax_list.append(x)
            #the degenration means that the min or max will already be added by another point
            elif (alpha1m-phi)%(2*np.pi)<np.pi:
                self.smax_list.append(x)
            else:
                self.smin_list.append(x)
        elif abs(alpha1m-alpha1o)%(2*np.pi)<self.eps:
            return False
        elif abs(alpha0m-alpha0o)%(2*np.pi)<self.eps:
            return False
        else:
            angles = np.array([alpha0m,alpha1o,alpha0o,alpha1m])
            order = np.argsort(angles)
            if np.all(((order-order[0])%4 == [0,1,2,3])):
                #check for degeneracy with the angle of the dof
                if abs(alpha0m-phi)%(2*np.pi)<self.eps:
                    if (alpha0o-phi)%(2*np.pi)<np.pi+self.eps:
                        self.smin_list.append(x)
                    else:
                        self.min_list.append(x)
                elif abs(alpha1m-phi)%(2*np.pi)<self.eps: 
                    if (alpha1o-phi)%(2*np.pi)>np.pi-self.eps:
                        self.smax_list.append(x)
                    else:
                        self.min_list.append(x)
                elif abs(alpha0o-phi)%(2*np.pi)<self.eps:
                    if abs(alpha1o-phi-np.pi)%(2*np.pi)<self.eps:
                        return False
                    else:
                        if (alpha0m-phi)%(2*np.pi)>np.pi+self.eps:
                            self.max_list.append(x)
                        else:
                            self.smin_list.append(x)
                elif abs(alpha1o-phi)%(2*np.pi)<self.eps: 
                    if (alpha1m-phi)%(2*np.pi)<np.pi-self.eps:
                        self.max_list.append(x)
                    else:
                        self.smin_list.append(x)
                else:
                    pos = np.searchsorted(angles[order],phi)
                    pos = pos%4
                    if order[pos] == 0:
                        #phi is between alpha0m and alpha1m
                        self.min_list.append(x)
                    elif order[pos] == 2:
                        #phi is between alpha0o and alpha1o
                        self.max_list.append(x)
                    elif order[pos]==1:
                        #phi is between alpha0m and alpha1o
                        if (alpha1m-phi+np.pi)%(2*np.pi)>np.pi+self.eps:
                            self.max_list.append(x)
                        elif (alpha0o-phi+np.pi)%(2*np.pi)<np.pi-self.eps:
                            self.min_list.append(x)
                        elif abs(alpha0o-phi+np.pi)%(2*np.pi)<self.eps:
                            self.smin_list.append(x)
                        elif abs(alpha1m-phi+np.pi)%(2*np.pi)<self.eps:
                            self.smax_list.append(x)
                    elif order[pos]==3:
                        if (alpha0m-phi+np.pi)%(2*np.pi)>np.pi+self.eps:
                            self.min_list.append(x)
                        elif (alpha1o-phi+np.pi)%(2*np.pi)<np.pi-self.eps:
                            self.max_list.append(x)
                        elif abs(alpha1o-phi+np.pi)%(2*np.pi)<self.eps:
                            self.smin_list.append(x)
                        elif abs(alpha0m-phi+np.pi)%(2*np.pi)<self.eps:
                            self.smax_list.append(x)
                                                
            else:
                return False
                
               
        
            
      
            
      
        
      
    # def update_bounds_side_corner(self,x,phi,alpha0m,alpha1m,alpha0o,alpha1o):
        
    #     if (((alpha1m-alpha0o)%(np.pi*2) < np.pi-self.eps  or (alpha1m-alpha1o)%(np.pi*2) > np.pi+self.eps) and
    #         ((alpha0m-alpha0o)%(np.pi*2) < np.pi-self.eps  or (alpha0m-alpha1o)%(np.pi*2)>np.pi+self.eps) and
    #         ((alpha0o - phi)%(np.pi*2) < np.pi and (alpha1o - phi)%(np.pi*2) >np.pi)):
       
    #         self.max_list.append(x)
    #     if (((alpha1m-alpha0o)%(np.pi*2) < np.pi-self.eps  or (alpha1m-alpha1o)%(np.pi*2)>np.pi+self.eps) and
    #         ((alpha0m-alpha0o)%(np.pi*2) < np.pi-self.eps  or (alpha0m-alpha1o)%(np.pi*2)>np.pi+self.eps) and
    #         ((alpha0o - phi)%(np.pi*2) > np.pi and (alpha1o - phi)%(np.pi*2) <np.pi)):
    #         self.min_list.append(x)
    #     if abs(alpha0o-phi)<self.eps and alpha1m-phi > self.eps and alpha0m-phi > self.eps:
    #         self.smin_list.append(x)
    #     if abs(abs(alpha1o)-np.pi)< self.eps and alpha1m-phi > self.eps and (alpha0m-phi)%(2*np.pi) < np.pi - self.eps:
    #         self.smax_list.append(x)
    def find_valid(self):
        if len(self.n_obst)==0:
            self.valid_range = [[0,1]]
            return True
        else:
            #create a virtual obstacle at 0 and 1 to limit the ranges
            val = np.concatenate([np.ones(len(self.max_list)+1),-np.ones(len(self.min_list)+1)])
            self.ip = np.array(self.max_list+[1]+self.min_list+[0])
            self.n_obst = np.zeros(len(self.ip)+1)
            self.n_obst[0]=1
            order = np.argsort(self.ip)
            self.ip = self.ip[order]
            val_sort = val[order]
            for i,k in enumerate(val_sort):
                self.n_obst[i+1]=self.n_obst[i]+k
            indexes = np.where((self.n_obst==0))[0]
            #print(self.n_obst)
            if self.n_obst[-1]==0:
                pass
            self.valid_range = [[self.ip[i-1],self.ip[i]] for i in indexes]
            return len(self.valid_range)>0
    
def corner_corner_intersection(xm,xo,dof,tol=1e-10):
    deltax = xo-xm
    
    if (abs(cross(deltax,dof))>tol):
        return False
    else:
        ax = np.argmax(abs(dof))
        xdof = deltax[ax]/dof[ax]
        return xdof
    
def side_corner_intersection(side_s, deltas, corner, dof_l,x_max,x_min,tol=1e-10):
    if abs(deltas[0])>tol:
        ax1=0
        ax2=1
    else:
        if abs(deltas[1])<tol:
            return False
        ax1 = 1
        ax2 = 0
    slops = deltas[ax2]/deltas[ax1]
    d_in = side_s-corner
    den = slops*dof_l[ax1]-dof_l[ax2]
    if abs(den)<tol:
        #the dof and side are colinear
        if abs(d_in[ax2]-d_in[ax1]*slops)>tol:
            #print("C")
            return False
        else:
            #all three of the dof and the direction are colinear
            #send a flag to resolve the face-face interaction
            
            return True
    candidate_x = (d_in[ax1]*slops-d_in[ax2])/den
    #if candidate_x < x_max and candidate_x > x_min:
    candidate_y = (candidate_x*dof_l[ax1]-d_in[ax1])/deltas[ax1]
    if candidate_y < x_max-tol and candidate_y > x_min+tol:
        return candidate_x
        
    return False

    
    
class Hang(Interface):
    def __init__(self,blockm,blocksf,facesm,cornersf,x=None,safe=False,tol=1e-5):
        super().__init__(blockm,blocksf)
        self.safe = safe
        self.facesm = facesm
        self.cornersf = cornersf
        #abs is allowed as the polygones are convex
        self.theta = abs((blockm.angles[facesm[0]]-blockm.angles[facesm[1]])%(2*np.pi)-np.pi)
        assert self.theta % np.pi > tol, "The faces are parallel"
            
        self.d = blocksf[1].corners[cornersf[1]]-blocksf[0].corners[cornersf[0]]
        self.c0 = blocksf[0].corners[cornersf[0]]
        self.c1 = blocksf[1].corners[cornersf[1]]
        self.radius = np.squeeze(np.array([-self.d[0]-1/np.tan(self.theta)*self.d[1],
                                -self.d[1]+1/np.tan(self.theta)*self.d[0]])/2)
        self.r = norm(self.radius)
        self.center = self.c0-self.radius
        self.alphad = np.arctan2(self.d[1],self.d[0])%(2*np.pi)
        #compute the intersection point of the two sides
        diff_face = facesm[0]-facesm[1]
        delta0 = blockm.corners[(facesm[0]+1)%4]-blockm.corners[facesm[0]]
        delta1 = blockm.corners[(facesm[1]+1)%4]-blockm.corners[facesm[1]]
    
        if diff_face%4==1:
            self.interp = blockm.corners[facesm[0]]
            l0 = [0,norm(delta0)]
            l1 = [0,norm(delta1)]
            if self.safe:
                #new angle facesm[0]: np.pi+t/2+self.theta+self.alphad
                #
                pass
        elif diff_face%4==3:
            self.interp = blockm.corners[facesm[1]]
            l0 = [0,norm(delta0)]
            l1 = [0,norm(delta1)]
            if self.safe:
                a0 = [(blocksf[0].angles[(cornersf[0]-1)%4]%(2*np.pi)-self.theta-self.alphad)*2,
                      (blocksf[0].angles[cornersf[0]]%(2*np.pi)-self.theta-self.alphad)*2]
                a1 = [(blocksf[1].angles[(cornersf[1]-1)%4]%(2*np.pi)-self.alphad-np.pi)*2,
                      (blocksf[1].angles[cornersf[1]]%(2*np.pi)-self.alphad-np.pi)*2]
            
        else:
            deltad = blockm.corners[facesm[0]]-blockm.corners[facesm[1]]
            k = ((deltad[1]-deltad[0]*delta0[1]/delta0[0])/
                 (delta1[0]*delta0[1]/delta0[0]-delta1[1]))
        
            self.interp = blockm.corners[facesm[1]]-k*delta1
            #can still be optimized
            l0 = [norm(self.interp-blockm.corners[(facesm[0]+1)%4]),
                  norm(self.interp-blockm.corners[facesm[0]])]
            l0.sort()
            l1 = [norm(self.interp-blockm.corners[(facesm[1]+1)%4]),
                  norm(self.interp-blockm.corners[facesm[1]])]
            # print(f"{l1=}")
            l1.sort()
            # print(f"{l1=}")
            # print(f"{l0=}")
       
            #alphamin1 = self.theta+self.alphad
        #as l1 = l0+d:
        l0const=np.arccos(1-np.square(l0/self.r)/2)
        print(self.theta)
        
        l1const=np.arccos(1-np.square(l1/self.r)/2)-self.theta*2
        print(l0const)
        print(l1const)
        #self.tmin = np.nanmax([l0const[0],l1const[1]])
        #self.tmax = np.nanmin([2*(np.pi-self.theta),l0const[1],l1const[0]])
        #self.tmin = np.nanmax([0,l0const[0],2*np.pi-l1const[1]])%(np.pi*2)
        if self.safe:
            print(f"{a0=}")
            print(np.array(a0)%(2*np.pi))
            print(np.array(a1)%(2*np.pi))
            self.tmin = np.nanmax([0,                   
                                  (a0[0]-self.alphad)%(2*np.pi)+self.alphad,
                                  (a1[0]-self.alphad)%(2*np.pi)+self.alphad])
            self.tmax = np.nanmin([2*(np.pi-self.theta),
                                  (a0[1]-self.alphad)%(2*np.pi)+self.alphad,
                                  (a1[1]-self.alphad)%(2*np.pi)+self.alphad])
        else:
            self.tmin = np.nanmax([0])
            self.tmax = np.nanmin([2*(np.pi-self.theta)])
        # print(f"{l0const=}")
        # print(f"{l1const=}")
        print(self.tmin)
        print(f"Valid: {self.tmax>self.tmin}")
        #assert self.tmax>self.tmin, "invalid interface"
        if x is not None:
            self.set_x(x)
        else:
            self.x = None
        self.diff_face = diff_face%4
    def set_x(self,x):
        # if self.diff_face < 2:
        #     x = 1-x
        t = x*(self.tmax-self.tmin)+self.tmin
        # if self.theta >0:
        #     t = x*(self.tmax-self.tmin)+self.tmin
        # else:
        #     t = x*(-2*np.pi-2*self.theta)
        
        newangle = np.pi+t/2+self.theta+self.alphad
            
        newinter = self.c0+[(np.cos(t)-1)*self.radius[0]-np.sin(t)*self.radius[1],
                            (np.cos(t)-1)*self.radius[1]+np.sin(t)*self.radius[0]]
        self.blockm.move(self.interp,newinter)
        
        if self.diff_face < 2:
            self.blockm.turn(newangle-self.blockm.angles[self.facesm[1]],center=newinter)
        elif self.diff_face == 2:
            self.blockm.turn(newangle-self.blockm.angles[self.facesm[0]],center=newinter)
        else:
            self.blockm.turn(newangle-self.blockm.angles[self.facesm[0]],center=newinter)
        self.interp = newinter
        self.x = x
    def plot_l_vs_t(self):
        fig,ax = plt.subplots(1)
        x = np.linspace(0,np.pi*2-2*self.theta,200)
        l1 = np.sqrt(2-2*np.cos(x))*self.r
        l2 = np.sqrt(2*(1-np.cos(2*np.pi-2*self.theta-x)))*self.r
        ax.plot(x,l1)
        ax.plot(x,l2)
    def plot_angle_vs_t(self):
        fig,ax = plt.subplots(1)
        x = np.linspace(0,np.pi*2-2*self.theta,200)
        alpha1 = (np.pi+x/2+self.theta+self.alphad)%(2*np.pi)
        alpha2 = (np.pi*2+x/2+self.alphad)
        ax.plot(x,alpha1)
        ax.plot(x,alpha2)
    def discretize(self,n,add_contact=False,candidates:list=None):
        pass
    def complete(self,blocks:list):
        pass
class Grid():
    def __init__(self,maxs):
        self.occ = np.zeros((maxs[0],maxs[1],2),dtype=int)
        self.neighbours = np.zeros((maxs[0]+1,maxs[1]+1,2,3),dtype=int)
    def put(self,block,pos0,rot,bid,floating=False):
        #if bid < 0: the block is held by robot -bid
        block.turn(rot)
        block.move(pos0)
        #test if the block is in the grid
        if (np.min(block.parts)<0 or
            np.max(block.parts[:,0])>=self.occ.shape[0] or
            np.max(block.parts[:,1])>=self.occ.shape[1]):
            return False
        #test if the is no overlap
        if np.any(self.occ[block.parts[:,0],block.parts[:,1],block.parts[:,2]]):
            return False
        else:
            #check if there is a connection (ask for at least 1 points)
            #if floating or np.any(same_side(block.neigh, np.array(np.where(self.neighbours)).T)):
            if not floating:
                candidates = switch_direction(block.neigh)
                if not np.any(self.neighbours[candidates[:,0],candidates[:,1],candidates[:,2],candidates[:,3]]):
                    return False
            #if floating or np.any(switch_direction(block.neigh)np.array(np.where(self.neighbours)).T)):
                #addd the block
            self.occ[block.parts[:,0],block.parts[:,1],block.parts[:,2]]=bid
            
            self.neighbours[block.neigh[:,0],block.neigh[:,1],block.neigh[:,2],block.neigh[:,3]]=bid
            return True
            # else:
            #     return False
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
    def __init__(self,parts,density=1,muc=0):
        self.parts = np.array(parts)
        self.neigh = np.zeros((3*self.parts.shape[0],4),dtype=int)
        self.neigh[:,:3] =np.tile(self.parts,(3,1))
        self.neigh[:self.parts.shape[0],3] = 0
        self.neigh[self.parts.shape[0]:2*self.parts.shape[0],3] = 1
        self.neigh[2*self.parts.shape[0]:,3] = 2
        same = same_side(self.neigh,self.neigh)
        self.neigh = self.neigh[~np.any(same,axis=0)]
        self.mass = density*self.parts.shape[0]
        self.muc = muc
    def turn(self,time,center=None):
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
        self.neigh = nneigh.astype(int)
        
    def move(self,new_p):
        
        delta =new_p-self.parts[0,:-1]
        self.parts[:,:-1] = self.parts[:,:-1]+delta
        self.neigh[:,:2]=self.neigh[:,:2]+delta
        #use real coordonates
        #self.cm = self.cm + delta @ 
def norm(x):
    return np.sqrt(np.sum(np.square(x)))
def cross(x,y):
    return x[0]*y[1]-x[1]*y[0]
def fold_angle(angles,min_valid = 0, max_valid=2*np.pi):
    angles = np.clip(angles,min_valid,max_valid)
    return angles
if __name__ == '__main__':
    print("Test blocks")
    b1 = Block([[0,0],[1,1]]).expand()
    b2 = Block([[2,0],[3,1]]).expand()
    b3 = Block([[10,10],[11,11]]).expand()
    
    i = Hang(b3,np.array([b1,b2]),np.array([0,1]),[2,3])
    print(b1)
    i.set_x(0.5)
    print(b1)
    print("End test blocks")