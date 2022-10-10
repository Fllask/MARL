# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 09:32:41 2022

@author: valla
"""

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle,Wedge
import numpy as np

from Blocks import Block,Interface,Slide,Hang

def draw_block(ax,block,label_corners = False,**kwarg):
    if len(kwarg)== 0:
        if block.corners.shape == (2,2):
            art = Rectangle(block.corners[0],block.w,block.h,**block.colors)
            ax.add_artist(art)
        else:
            art = ax.fill(block.corners[:,0],block.corners[:,1],**block.colors)
    else:
        if block.corners.shape == (2,2):
            art = Rectangle(block.corners[0],block.w,block.h,**kwarg)
            ax.add_artist(art)
        else:
            art = ax.fill(block.corners[:,0],block.corners[:,1],**kwarg)
    if label_corners:
        art = [art]
        for i,c in enumerate(block.corners):
            art.append(ax.annotate(str(i),c))
    return art
def draw_contact(ax,block):
    pass
def draw_interface(ax,inter:Interface,n=None,label_corners=False):
    art = []
    if inter.x is not None:
        art.append(draw_block(ax,inter.blocksf[0],facecolor='b',
                              label_corners=label_corners))
        art.append(draw_block(ax,inter.blockm,facecolor='g',alpha = 0.4,
                              label_corners=label_corners))
    else:
        
        if type(inter) is Slide or type(inter) is Hang:
            if inter.x is None:
                b = inter.blockm
                for i in range(n+1):
                    
                    if inter.valid_range is not None:
                        if len(inter.valid_range)==0:
                            inter.safe = False
                            inter.set_x(i/n,physical=False)
                            art.append(draw_block(ax,b,facecolor='none',edgecolor='k',alpha=0.3))

                        else:
                            inter.set_x(i/n,physical=False)
                            art.append(draw_block(ax,b,facecolor='none',edgecolor=plt.cm.turbo(i/n)))
                    else:
                        inter.safe = False
                        inter.set_x(i/n,physical=False)
                        art.append(draw_block(ax,b,facecolor='none',edgecolor='b',alpha=0.3))
                    if type(inter) is Hang:
                        art.append(ax.scatter(inter.interp[0],inter.interp[1],s=1,color='k'))
                if type(inter) is Hang:
                    art.append(ax.scatter(inter.center[0],inter.center[1]))
        # elif type(inter) is Hang:
        #     if n ==0:
        #         alphad= inter.alphad
        #         art.append(Wedge(inter.center,inter.r,rad(alphad+inter.theta*2),rad(alphad),width=0,edgecolor='k'))
        #         ax.add_artist(art[0])
    
        else:
            assert False, "Unkwown interface"
    return art
def rad(alpha):
    return alpha*180/np.pi
def check_intersection(ax):
    #create a sliding interface
    blockf = Block([[4,4],[6,5]]).expand()
    a = -0*np.pi/2
    blockf.turn(a,[4.5,4.5])
    blockm = Block([[0,0],[2,0],[1,1],[0,1]])
    # obstacles = [
    #     Block([[5.2,5],[6,6]]).expand(),
    #     Block([[3,4.5],[4,5.5]]).expand(),
    #     Block([[6,4],[7,6]]).expand()
    #             ]
    # [b.turn(np.pi/4,[3.9,4.5]) for b in obstacles]
    obstacles= [ 
        Block([[6,4],[7,5]]).expand(),
        Block([[2,5],[3,6]]).expand(),
        Block([[7,4.5],[8,6]]).expand(),
        Block([[3,4],[4,5]]).expand()
        ]
    [b.turn(0*np.pi/4,[6,5]) for b in obstacles]
    [b.turn(a,[4.5,4.5]) for b in obstacles]
    i = Slide(blockm,blockf,0,2,safe = True)
    i.complete(obstacles)
    print(f"{i.min_list=}")
    print(f"{i.max_list=}")
    print(f"{i.smin_list=}")
    print(f"{i.smax_list=}")
    i.find_valid()
    [draw_block(ax,b,facecolor='r',label_corners = False) for b in obstacles]
    draw_block(ax,blockf,facecolor='b')
    #i.set_x(0)
    draw_interface(ax,i,n=40)
    return i
def check_intersection2(ax):
    #create a sliding interface
    blockf = Block([[0,0],[3,0],[2,1],[1,1]])
    a = -0*np.pi/2
    c=0.6
    blockf.move([0,0],[4,4])
    blockf.turn(a,[4.5,4.5])
    blockm = Block([[0,0],[3+c,0],[2+c,1],[1,1]])
    # obstacles = [
    #     Block([[5.2,5],[6,6]]).expand(),
    #     Block([[3,4.5],[4,5.5]]).expand(),
    #     Block([[6,4],[7,6]]).expand()
    #             ]
    # [b.turn(np.pi/4,[3.9,4.5]) for b in obstacles]
    obstacles= [ 
        Block([[0,0],[3,0],[2,1],[1,1]]),
        #Block([[0,0],[3,0],[2,1],[1,1]]),
        #Block([[0,0],[3,0],[2,1],[1,1]]),
        ]
    # [b.turn(-np.pi/2,[4.5,4.5]) for b in obstacles]
    # [b.move([0,6],[4,4]) for b in obstacles]
    [b.turn(np.pi) for b in obstacles]
    [b.move([-3,0],[6,5]) for b in obstacles]
    [b.turn(a,[4.5,4.5]) for b in obstacles]
    i = Slide(blockm,blockf,2,2,safe = True)
    i.complete(obstacles)
    print(f"{i.min_list=}")
    print(f"{i.max_list=}")
    print(f"{i.smin_list=}")
    print(f"{i.smax_list=}")
    i.find_valid()
    [draw_block(ax,b,facecolor='r',label_corners = False) for b in obstacles]
    draw_block(ax,blockf,facecolor='b')
    #i.set_x(0)
    draw_interface(ax,i,n=40)
    return i
def check_intersection3(ax):
    #create a sliding interface
    blockf = Block([[ 2.78154171, -2.38088075],
     [ 2.78154171, -5.38088075],
     [ 3.78154171, -4.38088075],
     [ 3.78154171 ,-3.38088075]])
    a =  1.5*np.pi/10
    blockm = Block([[ 9.23913045, -5.07887244],
     [ 9.23913045, -2.07887244],
     [ 8.23913045, -3.07887244],
     [ 8.23913045, -4.07887244],
                    ])
  
    # [b.turn(np.pi/4,[3.9,4.5]) for b in obstacles]
    obstacles= [ 
        Block([[ 4. , -5.05 ],
                [ 5.01 , -5.01 ],
                [ 5.4 , -4.6],
                [ 4. , -4.6]]),
       
        # Block([[-2.57908657, -0.68151772],
        # [-0.45776622, -2.80283806],
        # [-0.45776622, -1.3886245 ],
        # [-1.164873  , -0.68151772]]),
        ]
    [b.turn(a,[4,-4]) for b in obstacles]
    #[b.move([0,0],[1,0]) for b in obstacles]
    blockf.turn(a,[4,-4])
    i = Slide(blockm,blockf,0,0,safe = True)

    i.complete(obstacles)
    print(f"{i.min_list=}")
    print(f"{i.max_list=}")
    print(f"{i.smin_list=}")
    print(f"{i.smax_list=}")
    i.find_valid()
    [draw_block(ax,b,facecolor='r',label_corners = False) for b in obstacles]
    draw_block(ax,blockf,facecolor='b')
    #i.set_x(0)
    draw_interface(ax,i,n=40)
    return i



if __name__ == "__main__":
    print("Start test")
    plt.close()
    fig,ax = plt.subplots(1,1,figsize=(6,6))
    ax.set_xlim([0,11])
    ax.set_ylim([0,11])
    i = check_intersection(ax)
    #test the slide interface
    #block1 = Block([[0,0],[3,0],[2,1],[1,1]])
    #block2 = Block([[0,0],[3,0],[2,1],[1,1]])
    # block2 = Block([[0,0],[0,1],[2,1],[3,0]])
    # block2.move([0,0],[5,5])
    
    # draw_block(ax,block1,label_corners=False)
    #i = Slide(block2,block1,1,2)
    # i.set_x(0.6)
    # _ = draw_interface(ax,i,100,label_corners=True)
    #test the hang interface    
    # block1 = Block([[4,4],[5,5]]).expand()
    # block2 = Block([[5.5,3.3],[8.5,5.5]]).expand()
    # block3 = Block([[0,0],[1,1]]).expand()
    # #block3 = Block([[0,0],[2,0],[1,1],[0,1]])
    # draw_block(ax,block1)
    # draw_block(ax,block2)
    # draw_block(ax,block3,label_corners=True)
    # i = Hang(block3,np.array([block1,block2]),[1,2],[2,3],safe = False)
    # i.plot_angle_vs_t()
    #x = (np.pi-i.tmin)/(i.tmax-i.tmin)
    # i.set_x(0.3)
    #draw_interface(ax, i,100,label_corners=True)
    plt.show()
    print("End test")