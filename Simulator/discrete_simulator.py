# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 20:11:43 2022

@author: valla
"""
import time
import copy
import matplotlib.pyplot as plt
import numpy as np


import discrete_graphics as gr
from discrete_blocks import discret_block as Block, Grid,Graph
from physics_scipy import stability_solver_discrete as ph
class Backup():
    def __init__(self,maxs):
        self.grid = Grid(maxs)
        self.matrices = {}
class Transition():
    def __init__(self,state,action,reward,new_state):
        self.state = state
        self.a = action
        self.r = reward
        self.new_state = new_state
class DiscreteSimulator():
    def __init__(self,maxs,n_robots,block_choices,n_reg,maxblocks,maxinterface):
        self.grid = Grid(maxs)
        self.blocks = block_choices
        self.graph = Graph(len(block_choices),
                       	  n_robots,
                           n_reg,
                           maxblocks,
                           maxinterface,)
        # self.prev = Backup(maxs)
        self.ph_mod = ph(n_robots = n_robots)
        self.nbid=1
        self.prev = None
    def setup_anim(self,h=6):
        self.frames = []
        self.fig,self.ax = gr.draw_grid(self.grid.occ.shape[:2],color='none',h=h)
    def add_ground(self,block,pos,rot):
        self.grid.put(block,pos,rot,0,floating=True)
        self.graph.add_ground(pos, rot)
    def put(self,block,pos,rot,blocktypeid=None):
        if blocktypeid is not None:
            block = self.blocks[blocktypeid]
        else:
            blocktypeid = -1
        valid, closer,interfaces = self.grid.put(block, pos, rot, self.nbid)
        if valid:
            self.graph.add_block(self.nbid,blocktypeid, pos, rot)
            for side1,interface in enumerate(interfaces):
                if interface[0]==-1:
                    continue
                self.graph.add_rel(self.nbid,interface[0],side1,interface[1])
            self.ph_mod.add_block(self.grid,block,self.nbid)
            self.nbid+=1

        return valid,closer
    def put_rel(self,block,idsideblock,idsidesup,idblocksup,blocktypeid=None,idconsup=None):
        if blocktypeid is not None:
            block = self.blocks[blocktypeid]
        else:
            blocktypeid = -1
            
        if idsideblock >= block.neigh.shape[0]:
            pass
        valid, closer,interfaces = self.grid.connect(block, self.nbid, idsideblock, idsidesup,idblocksup,idcon=idconsup)
        if valid:
            self.graph.add_block(self.nbid, blocktypeid, block.parts[0,:2], block.rot)
            for side1,interface in enumerate(interfaces):
                if interface[0]==-1:
                    continue
                self.graph.add_rel(self.nbid,interface[0],side1,interface[1],interface[2])
            self.ph_mod.add_block(self.grid,block,self.nbid)
            self.nbid+=1
        return valid,closer
    def remove(self,bid,save=True):
        if bid < 1 or  bid >=self.nbid:
            #cannot remove the ground or a block already put way before
            return False
        if not np.any(self.grid.occ==bid):
            #cannot remove the same block twice
            return False
        
        if save:
            self.save()
        if bid == self.nbid-1:
            self.nbid -= 1
        self.grid.remove(bid)
        self.graph.remove(bid)
        self.ph_mod.remove_block(bid)
        return True
    def remove_loc(self,pos,ori):
        bid = self.grid.occ[pos[0],pos[1],ori%2]
        if bid < 0:
            return False
        

        self.prev = (copy.deepcopy(self.grid),copy.deepcopy(self.ph_mod),self.nbid)
        self.grid.remove(bid)
        self.ph_mod.remove_block(bid)
        return True
    def save(self):
        self.prev = (copy.deepcopy(self.grid),copy.deepcopy(self.ph_mod),self.nbid)
    def undo(self):
        assert self.prev is not None, "no state was saved"
        self.grid,self.ph_mod,self.nbid = self.prev
        self.prev = None
    def leave(self,rid):
        bid = np.unique(self.grid.occ[self.grid.hold==rid])
        if len(bid)==0:
            return None
        self.grid.hold[self.grid.hold==rid]=-1
        self.ph_mod.leave_block(rid)
        return bid
    def hold(self,rid,bid):
        '''hold the block bid'''
        if bid is None:
            return True
        if bid < 1 or  bid >=self.nbid:
            #cannot remove the ground
            return False
        #self.prev.grid = copy.deepcopy(self.grid)
        self.grid.hold[self.grid.occ==bid]=rid
        exist = self.ph_mod.hold_block(bid, rid)
        return exist
    
    def hold_loc(self,rid,pos,ori):
        bid = self.grid.occ[pos[0],pos[1],ori%2]
        if bid < 1:
            #cannot remove the ground
            return False
        self.grid.hold[self.grid.occ==bid]=rid
        return True
    def check(self):
        res = self.ph_mod.solve()
        assert res.status in [0,2],"error in the static solver"
        return res.status == 0
    def add_frame(self):
        '''add a frame to later animate the simulation'''
        self.frames.append(gr.fill_grid(self.ax, self.grid,animated=True))
    def draw_act(self,rid,action,blocktype,**action_params):
        state = gr.fill_grid(self.ax, self.grid,animated=True)
        if 'sideblock' in action_params.keys():
            self.frames.append(state+gr.draw_action_rel(self.ax,rid,action,blocktype,animated=True,**action_params))
        else:
            self.frames.append(state+gr.draw_action(self.ax,rid,action,blocktype,animated=True,**action_params))
    def animate(self):
        anim = gr.animate(self.fig, self.frames)
        plt.close()
        return anim
    def reset(self):
        '''remove all blocks from the sim'''
        self.grid.reset()
        self.ph_mod.reset()
    def draw_state_debug(self):
        fig,ax = gr.draw_grid(self.grid.occ.shape[:2],color='none',h=12)
        gr.fill_grid(ax, self.grid,graph=self.graph)
        plt.show()
def scenario1(maxs, n_block = 10,maxtry=100,draw=False):
    #try to fill the grid with hexagones
    arts = []
    block_list = [Block([[0,1,1],[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0]])]
    
    
    grid = Grid(maxs)
    grid.put(block_list[0],[maxs[0]//2,maxs[1]//2],0,1,floating=True)
    bid = 2
    trys=0
    if draw:
        fig,ax = gr.draw_grid(maxs,h=7,color='none')
    while bid < n_block+1 and trys < maxtry:
        block = np.random.choice(block_list)
        pos = np.random.randint(maxs)
        valid,*_ = grid.put(block,pos,0,bid)
        if valid:
            if draw:
                arts.append(gr.fill_grid(ax, grid,animated=True))
            bid +=1
            trys=0
        else:
            trys+=1
    if draw:
        print("drawing")
        ani = gr.animate(fig, arts,sperframe= 0.1)
        return grid,bid-1,ani
    return grid,bid-1,None
def scenario2(maxs, n_block = 10,maxtry=100):
    #try to fill the grid with hexagones
    block_list = [Block([[0,1,1],[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0]]),
                  Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]])]
    
    
    grid = Grid(maxs)
    grid.put(block_list[0],[maxs[0]//2,maxs[1]//2],0,0,floating=True)
    bid = 2
    trys=0
    
    while bid < n_block+1 and trys < maxtry:
        block =block_list[bid%2]
        pos = np.random.randint(maxs)
        rot = np.random.randint(6)
        if grid.put(block,pos,rot,bid):
            bid +=1
            trys=0
        else:
            trys+=1
    return grid,bid-1
def scenario3(maxs,n_block,maxtry = 100,mode='triangle',draw=False,physon=True):
    #pill up some shapes
    
    block_list = [Block([[0,1,1],[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0]],muc=0.7),
                  Block([[0,0,0]],muc=0.7)]
    ground = Block([[0,2,0],[maxs[0]-1,2,0]]+[[i,2,1] for i in range(0,maxs[0])],muc=0.7)
    grid = Grid(maxs)
    grid.put(ground, [0,0], 0, 0,floating=True)
    arts = []
    phys = ph(maxs,n_robots = 0)
    trys = 0
    bid = 1
    if draw:
        fig,ax = gr.draw_grid(maxs,h=10,color='none')
        arts.append(gr.fill_grid(ax, grid,animated=True))
    while bid < n_block and trys < maxtry:
        if mode == 'triangle':
            block = block_list[1]
        elif mode == 'hex':
            block = block_list[0]
        else:
            block = np.random.choice(block_list)
        pos = np.random.randint(maxs)
        rot = np.random.randint(6)
        valid, *_ = grid.put(block,pos,rot,bid)
        if valid:
            if physon:    
                phys.add_block(grid, block, bid)
                res = phys.solve()
                if res.status==0:
                    if draw:
                        arts.append(gr.fill_grid(ax, grid,animated=True))
                    bid+=1
                    trys=0
                    
                else:
                    trys+=1
                    grid.remove(bid)
                    phys.remove_block(bid)
                    
            else:
                if draw:
                    arts.append(gr.fill_grid(ax, grid,animated=True))
                bid+=1
                trys=0
                
        else:
            trys+=1
    if draw:
        print("drawing")
        ani = gr.animate(fig, arts,sperframe= 0.1)
        return grid,bid-1,ani
    return grid,bid-1,None
def scenario4(maxs,n_block,maxtry = 100,mode='triangle',draw=False,physon=False,scale=0.0):
    #try to bias the postion of the blocks so that they try to connect
    
    block_list = [Block([[0,1,1],[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0]],muc=0.7),
                  Block([[0,0,0]],muc=0.7)]
    
    grid = Grid(maxs)
    grid.put(block_list[0], [10,maxs[1]//2], 0, 0,floating=True)
    grid.put(block_list[0], [maxs[0]-11,maxs[1]//2], 0, 0,floating=True)
    if physon:
        phys = ph(maxs,n_robots = 0)
    trys = 0
    bid = 1
    while bid < n_block+1 and trys < maxtry:
        if mode == 'triangle':
            block = block_list[1]
        elif mode == 'hex':
            block = block_list[0]
        else:
            block = np.random.choice(block_list)
        pos = np.random.randint(maxs)
        rot = np.random.randint(6)
        valid,dist,con = grid.put(block,pos,rot,bid)
        if valid:
            if draw:
                fig,ax = gr.draw_grid(maxs,h=30,label_points=True)
                gr.fill_grid(ax,grid)
            if physon:    
                phys.add_block(grid, block, bid)
                res = phys.solve()
                if res.status==0:
                    remove = np.random.random(1)*2-1
                    if remove > dist[0]:
                        grid.remove_block(bid)
                    else:
                        bid+=1
                        trys=0
                    
                else:
                    trys+=1
                    grid.remove(bid)
                    if draw:
                        fig,ax = gr.draw_grid(maxs,h=30,label_points=True)
                        gr.fill_grid(ax,grid)
                    phys.remove_block(bid)
                    
                    
            else:
                if con is not None:
                    print("connected")
                    break
                remove = (np.random.random(1)*2-1)*scale
                con_dist = np.argmax(dist)
                if remove < dist[con_dist]:
                    grid.remove(bid)
                else:
                    bid+=1
                    trys=0
                
        else:
            trys+=1
    return grid,bid-1
def scenario5(maxs,n_block,maxtry = 100,mode='triangle',draw=False,scale=0.0):
    #try to bias the postion of the blocks so that they try to connect, with 3 grounds
    
    block_list = [Block([[0,1,1],[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0]],muc=0.7),
                  Block([[0,0,0]],muc=0.7)]
    if draw:
        fig,ax = gr.draw_grid(maxs,h=7,color='none')
        arts = []
    grid = Grid(maxs)
   
    grid.put(block_list[1], [10,maxs[1]//4], 0, 0,floating=True)
    grid.put(block_list[1], [maxs[0]-11,maxs[1]//4], 0, 0,floating=True)
    grid.put(block_list[1], [maxs[0]//3+1,3*maxs[1]//4], 0, 0,floating=True)
    trys = 0
    bid = 1
    #regtoconnect = 2
    while bid < n_block+1 and trys < maxtry:
        if mode == 'triangle':
            block = block_list[1]
        elif mode == 'hex':
            block = block_list[0]
        else:
            block = np.random.choice(block_list)
        pos = np.random.randint(maxs)
        rot = np.random.randint(6)
        # valid,dist,con = grid.put(block,pos,rot,bid)
        valid,closer,con = grid.put(block,pos,rot,bid)
        if valid:
            # if con is not None:     
            # #if np.sum(dist==0)==dist.shape[0]-regtoconnect+1:
            #     #grid.absorb_reg(con[1], con[0])
            #     #regtoconnect -=1
                
            #     print("connected")
            #     if draw:
            #         arts.append(gr.fill_grid(ax, grid,animated=True,use_con=True))
            #     #if regtoconnect==0:
            #     if np.all(grid.min_dist<1e-5):
            #         break
                
            #     bid+=1
            #     trys=0
            # else:
            #remove = (np.random.random(1)*2-1)*scale
            #dist[dist==0] = np.nan
            # con_dist = np.nanargmin(dist)
            # if remove < dist[con_dist]+1e-5:
            if not closer:# or grid.connection[grid.occ==bid]==0:
                grid.remove(bid)
            else:
                if draw:
                    arts.append(gr.fill_grid(ax, grid,animated=True,use_con=True))
                if np.all(grid.min_dist<1e-5):
                    print("success")
                    break
                bid+=1
                trys=0
            
        else:
            trys+=1
    if draw:
        print("drawing")
        ani = gr.animate(fig, arts,sperframe= 0.1)
        return grid,bid-1,ani
    return grid,bid-1,None
def scenario6(sim):
    link = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=0.7)
    hexagon = Block([[0,1,1],[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0]],muc=0.7)
    sim.setup_anim()
    sim.add_ground(Block([[0,0,0]]),[sim.grid.occ.shape[0]-2,0],0)
    sim.add_frame()
    #agent0 does Ph
    oldbid = sim.leave(0)
    if oldbid is not None:
        stable = sim.check()
        if not stable:
            #the robot cannot move from there
            valid = False
            sim.hold(0,oldbid)
    else:
        valid,closer = sim.put(link,[sim.grid.occ.shape[0]-3,1],0)
        if valid:
            sim.hold(0,sim.nbid-1)
    action_args={'pos':[sim.grid.occ.shape[0]-3,1],'ori':0,'blocktypeid':1}
    sim.draw_act(0,'Ph',link,**action_args)
    sim.add_frame()
    sim.draw_act(1,'Pl',hexagon,pos=[sim.grid.occ.shape[0]-3,3],ori=0,blocktypeid=2)
    sim.put(hexagon,[sim.grid.occ.shape[0]-3,3],0)
    sim.add_frame()
    #r0 remove its block
    sim.draw_act(0, 'R', blocktype=None, bid=1)
    oldbid = sim.leave(0)
    if sim.remove(1):
        stable = sim.check()
        valid = stable
        if not stable:
            sim.undo()
            sim.hold(0,oldbid)
    else:
        sim.hold(0,oldbid)
    sim.add_frame()
    anim=sim.animate()
    return anim
def demo_action_rel(sim,groundid,block_choices=[Block([[0,1,1],[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0]]),#hexagone
                                                Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]])]):
    sim.setup_anim()
    sim.add_ground(block_choices[groundid],[sim.grid.shape[0]//2,sim.grid.shape[1]//2-1],0)
    sim.add_frame()
    nside = [block.neigh.shape[0] for block in block_choices]
    max_blocks = 3
    rid=0
    mask = generate_mask(sim.grid,0,nside,True,max_blocks,2)
    # actions = [('Ph',{'rid':0,'blocktypeid':1,'sideblock':0,'sidesup':1}),
    #            ('Ph',{'rid':0,'blocktypeid':1,'sideblock':1,'sidesup':0}),
    #            ('Ph',{'rid':0,'blocktypeid':1,'sideblock':2,'sidesup':1}),
    #             ('Ph',{'rid':0,'blocktypeid':1,'sideblock':3,'sidesup':2}),
    #             ('Ph',{'rid':0,'blocktypeid':1,'sideblock':4,'sidesup':3}),
    #             ('Ph',{'rid':0,'blocktypeid':1,'sideblock':5,'sidesup':4}),
    #             ('Ph',{'rid':0,'blocktypeid':1,'sideblock':0,'sidesup':5}),
    #             ('Ph',{'rid':1,'blocktypeid':0,'sideblock':1,'sidesup':0}),
    #             ('Ph',{'rid':1,'blocktypeid':0,'sideblock':1,'sidesup':1}),
    #             ('Ph',{'rid':1,'blocktypeid':0,'sideblock':1,'sidesup':2}),
    #             ('Ph',{'rid':1,'blocktypeid':0,'sideblock':1,'sidesup':3}),
    #             ('Ph',{'rid':1,'blocktypeid':0,'sideblock':1,'sidesup':4}),
    #             ('Ph',{'rid':1,'blocktypeid':0,'sideblock':1,'sidesup':5}),
               
   #     ]
    for actionid in np.nonzero(mask)[0]:
        action,param =int2act_sup(actionid,nside,True,max_blocks)
        valid,_,bt = act_rel(sim,action,**param,draw=True)
        sim.draw_act(param.pop('rid'),action,bt,**param)
        sim.add_frame()
        sim.remove(sim.nbid-1,save=False)
    anim=sim.animate()
    return anim
def act_rel(simulator,action,
        rid=None,
        sideblock=None,
        sidesup = None,
        bid_sup = None,
        blocktype = None,
        blocktypeid=None,
        choices = None,
        draw= False
        ):
    valid,closer = None,None
    if blocktypeid is not None:
        blocktype= choices[blocktypeid]
    
    if bid_sup is None:
        bid_sup = simulator.nbid-1
    if action in {'Ph','Pl'}:
        oldbid = simulator.leave(rid)
        if oldbid is not None:
            stable = simulator.check()
            if not stable:
                if draw:
                    #if draw, place the block on the grid and remove it. this way the block is located at the right position
                    valid_pos,closer = simulator.put_rel(blocktype,sideblock,sidesup,bid_sup)
                    if valid_pos:
                        simulator.remove(simulator.nbid-1,save=False)
                    simulator.hold(rid,oldbid)
                    return False,None,blocktype
                else:
                    #the robot cannot move from there
                    simulator.hold(rid,oldbid)
                    return False,None,blocktype
            
        valid,closer = simulator.put_rel(blocktype,sideblock,sidesup,bid_sup)
        if valid:
            if action == 'Ph':
                simulator.hold(rid,simulator.nbid-1)
            if action == 'Pl':
                stable = simulator.check()
                if not stable:
                    simulator.remove(simulator.nbid-1,save=False)
                    valid = False
        if not valid:
            simulator.hold(rid,oldbid)
                        
    elif action == 'L':
        oldbid = simulator.leave(rid)
        if oldbid is not None:
            stable = simulator.check()
            valid = stable
            if not stable:
                simulator.hold(rid,oldbid)
        else:
            valid = False
    else:
        assert False,'Unknown action'
    return valid,closer,blocktype
def int2act_sup(action_id,n_side,last_only,max_blocks):
    maxs = max(n_side)
    sums = sum(n_side)
    cumsum = np.cumsum(n_side)
    if last_only:
        r_id = action_id//(2*maxs*sums+1)
        action_id = action_id%(2*maxs*sums+1)
        action_type = action_id//(maxs*sums)
        action = ['Ph','Pl','L'][action_type]
        if action != 'L':
            action_id = action_id%(maxs*sums)
            
            side_support = action_id//sums
            action_id = action_id%sums
            blocktypeid = np.searchsorted(cumsum,action_id,side='right')
            if blocktypeid > 0:
                action_id -= cumsum[blocktypeid-1]
            side_block = action_id
            action_params = {'rid':r_id,
                             'blocktypeid':blocktypeid,
                             'sideblock':side_block,
                             'sidesup':side_support
                              }
   
        else:
            action_params = {'rid':r_id,
                              }
        return action,action_params
def generate_mask(state,rid,n_side,last_only,max_blocks,n_robots):
    if last_only:
        n_actions = (2*max(n_side)*sum(n_side)+1)
    else:
        n_actions = (2*max(n_side)*sum(n_side)+1)*max_blocks
    #only ph,pl and l
    
    mask = np.zeros(n_actions*n_robots,dtype=bool)
        
    base_idx = rid*n_actions
    #get the ids of the feasible put actions (note that the are not all hidden)
    if last_only:
        mask[base_idx:base_idx+n_actions]=True
        n_side_last = np.sum(state.neighbours==np.max(state.neighbours))
        #hide out the remaining indices )if the last block had 2 sides less than the max, hide out these sides
        for i in range(n_side_last,max(n_side)):
            mask[base_idx+i*sum(n_side):base_idx+(i+1)*sum(n_side)]=False
            mask[base_idx+n_actions//2+i*sum(n_side):base_idx+n_actions//2+(i+1)*sum(n_side)]=False
    else:
        #only allows the ids that are already present:
        n_current= np.max(state.neigbours)
        mask[base_idx:base_idx+n_current*(n_actions-1)//max_blocks]=True
        for bid in range(n_current+1):
            n_side_bid = np.sum(state.neigbours==bid)
            for i in range(n_side_bid,max(n_side)):
                mask[base_idx+i*sum(n_side)+bid*sum(n_side)*max(n_side):base_idx+(i+1)*sum(n_side)+bid*sum(n_side)*max(n_side)]=False
                mask[base_idx+n_actions//2+i*sum(n_side)+bid*sum(n_side)*max(n_side):base_idx+n_actions//2+(i+1)*sum(n_side)+bid*sum(n_side)*max(n_side)]=False
    #leave
    mask[base_idx+n_actions-1]=rid in state.hold
        
    return mask
if __name__ == '__main__':
    print("Start test simulator")
    maxs = [9,6]
    choices = [Block([[0,1,1],[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0]],muc=0.7),#hexagone
               Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=0.7)]#link
    sim = DiscreteSimulator(maxs,2,choices,1,30,5)
    time0 = time.perf_counter()
    
    #grid,bid,ani = scenario1(maxs,n_block=200,maxtry=10000,draw=True)
    #ani = demo_action_rel(sim,0)
    grid,bid,ani = scenario5(maxs,n_block=600,maxtry=200000,mode='triangle',draw=True)
    #ani = scenario6(sim)
    time1 = time.perf_counter()
    #print(f"time needed to put {bid} blocks: {time1-time0} ")
    if ani is not None:
        gr.save_anim(ani,"test scenario")
    fig,ax = gr.draw_grid(maxs,h=30,label_points=False,color='none')
    #gr.fill_grid(ax, grid,use_con=True)
    plt.show()
    print("End test simulator")
    