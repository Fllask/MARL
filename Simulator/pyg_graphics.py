# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:38:37 2023

@author: valla
"""

import networkx as nx
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import animation
from IPython.display import HTML
from physics_scipy import side2corners,get_cm
from discrete_blocks_norot import discret_block_norot as Block
from geometric_internal_model_one_hot import create_sparse_graph
from torch_geometric import utils
import numpy as np



def draw_graph(pyg_graph,maxs = [10,10],h=8,dist=0.75):
    fig,ax = plt.subplots(1,figsize=(h,h))
    ax.set_xlim(-0.5,maxs[0]+0.5)
    ax.set_ylim(-1,maxs[1]+1.5)
    
    
    
    graph = pyg_graph.to_homogeneous()
    
    robot_nodes, = np.nonzero(graph.node_type.numpy()==2)
    nr = len(robot_nodes)
    robot_coords = {robot_nodes[i]: [i*maxs[0]/(nr-1),maxs[1]] for i in range(nr)}
    
    ground_nodes, = np.nonzero(graph.node_type.numpy()==0)
    ng = len(ground_nodes)
    ground_coords = {ground_nodes[i]: pyg_graph['ground'].x[i,-2:].numpy() for i in range(ng)}
    
    block_nodes, = np.nonzero(graph.node_type.numpy()==1)
    nb= len(block_nodes)
    block_coords = {block_nodes[i]: pyg_graph['block'].x[i,-2:].numpy() for i in range(nb)}
    
    
    #lookup_ori = np.array([np.pi/2,-np.pi*5/6,-np.pi/6,-np.pi/2,np.pi*5/6,np.pi/6])
    #block_side,n_sides_b = np.unique(pyg_graph['block','action_desc','side_sup'].edge_index[0,:].numpy(),return_counts = True)
    sides_sup = pyg_graph['side_sup'].x.to(int).numpy()
    #lookup_ori = np.array([0,1,2,3,4,5])#+0.5
    lookup_ori = np.array([1,3,5,4,0,2])
    #lookup_ori = np.array([1,0,2,3,4,5])
    inv_lookup_ori = np.argsort(lookup_ori)
    _,sides_sup_id = np.nonzero(sides_sup[:,:-1])
    angle = lookup_ori[sides_sup_id]*np.pi/3+np.pi/6
    order_side_sup = np.lexsort((sides_sup[:,-1],
                                 lookup_ori[sides_sup_id],
                                 ))
    inv_order_side_sup = np.argsort(order_side_sup)
    
    for i in range(6):
        nsides_ori = np.max(sides_sup[sides_sup[:,i]==1,-1])
        angle[sides_sup_id==i] += (sides_sup[sides_sup[:,i]==1,-1]-nsides_ori/2)/(1+nsides_ori)*np.pi/3
        
    
    
    
    
    side_sup_nodes, =  np.nonzero(graph.node_type.numpy()==3)
    nss = len(side_sup_nodes)
    # angles = np.linspace(0,np.pi*2,nss+1)+np.pi/nss
    # angle = angles[inv_order_side_sup]
    sides_sup_coords = {side_sup_nodes[i]: block_coords[block_nodes[int(pyg_graph['block','action_desc','side_sup'].edge_index[0,i])]]+
                                           dist*np.array([np.cos(angle[i]),
                                                          np.sin(angle[i])])
                                                          
                                           for i in range(nss)}
    
    
    new_block_nodes, = np.nonzero(graph.node_type.numpy()==4)
    nnb= len(new_block_nodes)
    # angle_dif = np.pi*2/(nnb)
    # angle = np.pi/(nnb)
    angles = np.linspace(0,np.pi*2,nnb+1)+np.pi/nnb
    #angles = np.arange(nnb)
    ind = np.lexsort((
                     pyg_graph['new_block'].x[:,1],
                     pyg_graph['new_block'].x[:,0],
                     inv_order_side_sup[pyg_graph['side_sup','put_against','new_block'].edge_index[0,:]],
                     
                     ))
    
    angles=angles[np.argsort(ind)]
        
    new_block_coords = {new_block_nodes[i]: block_coords[block_nodes[int(pyg_graph['block','action_desc','side_sup'].edge_index[
                                                                    0,pyg_graph['side_sup','put_against','new_block'].edge_index[0,i]])]]+
                                              2*dist*np.array([np.cos(angles[i]),np.sin(angles[i])])
                                              for i in range(nnb)}
    
    
    graph_nx = utils.to_networkx(graph)
    reach_ground, =np.nonzero(graph.edge_type.numpy()==9)
    
    robot_color = plt.cm.Set2(np.arange(nr))
    #robot_color[:,3]=0
    ground_color = np.tile(mcolors.to_rgba_array('darkslategrey'),(ng,1))
    block_color = np.tile(plt.cm.plasma(0),(nb,1))
    side_sup_color = np.tile(mcolors.to_rgba_array('grey'),(nss,1)) 
    side_sup_color =plt.cm.tab10(sides_sup_id/10)
    new_block_color = np.tile(mcolors.to_rgba_array('gold'),(nnb,1))#plt.cm.Set1(pyg_graph['new_block'].x[:,0].numpy().astype(int))
    #new_block_color[:,3]=0
    
    
    
    pos = robot_coords
    pos.update(ground_coords)
    pos.update(block_coords)
    pos.update(sides_sup_coords)
    pos.update(new_block_coords)
    
    node_size = 100*np.ones(graph.node_type.shape[0])
    
    put_against_edges, = np.nonzero(graph.edge_type.numpy()==2)
    edge_list_put_against = [(node[0],node[1]) for node in graph.edge_index[:,put_against_edges].numpy().T]
    
    action_descb_edges, = np.nonzero(graph.edge_type.numpy()==0)
    edge_list_action_descb = [(node[0],node[1]) for node in graph.edge_index[:,action_descb_edges].numpy().T]
    
    action_descg_edges, = np.nonzero(graph.edge_type.numpy()==1)
    edge_list_action_descg = [(node[0],node[1]) for node in graph.edge_index[:,action_descg_edges].numpy().T]
    
    choses_edges, = np.nonzero(graph.edge_type.numpy()==3)
    edge_list_choses = [(node[0],node[1]) for node in graph.edge_index[:,choses_edges].numpy().T]
    
    holds_edges, = np.nonzero(graph.edge_type.numpy()==4)
    edge_list_holds = [(node[0],node[1]) for node in graph.edge_index[:,holds_edges].numpy().T]
    
    touchesb_edges, = np.nonzero(graph.edge_type.numpy()==5)
    edge_list_touchesb = [(node[0],node[1]) for node in graph.edge_index[:,touchesb_edges].numpy().T]
    
    touchesg_edges, = np.nonzero((graph.edge_type.numpy()==6) | (graph.edge_type.numpy()==7))
    edge_list_touchesg = [(node[0],node[1]) for node in graph.edge_index[:,touchesg_edges].numpy().T]
    
    reachesb_edges, = np.nonzero(graph.edge_type.numpy()==8)
    edge_list_reachesb = [(node[0],node[1]) for node in graph.edge_index[:,reachesb_edges].numpy().T]
    
    reachesg_edges, = np.nonzero(graph.edge_type.numpy()==9)
    edge_list_reachesg = [(node[0],node[1]) for node in graph.edge_index[:,reachesg_edges].numpy().T]
    
    communicates_edges, = np.nonzero(graph.edge_type.numpy()==10)
    edge_list_communicates = [(node[0],node[1]) for node in graph.edge_index[:,communicates_edges].numpy().T]
    
    
    
    
    edge_width = 10*np.ones(graph.edge_index.shape[1])
    edge_width[choses_edges]=0.1
    edge_width[reachesg_edges]=0.1
    edge_width[reachesb_edges]=0.1
    
    edge_color = np.zeros((graph.edge_index.shape[1],4))
    #edge_color[:,3]=1
    edge_width[put_against_edges]=1
    edge_color[put_against_edges]=mcolors.to_rgba_array('orange')
    
    edge_width[action_descb_edges]=2
    edge_color[action_descb_edges]=mcolors.to_rgba_array('grey')
    
    edge_width[action_descg_edges]=2
    edge_color[action_descg_edges]=mcolors.to_rgba_array('grey')
    
    edge_width[choses_edges]=0.5
    edge_color[choses_edges]=mcolors.to_rgba_array('darkorange')
    
    edge_width[holds_edges]=4
    edge_color[holds_edges]=mcolors.to_rgba_array('darkred')
    
    edge_width[touchesb_edges]=2
    edge_color[touchesb_edges]=plt.cm.plasma(0)
    
    edge_width[touchesg_edges]=2
    edge_color[touchesg_edges]=mcolors.to_rgba_array('darkslategrey')
    
    edge_width[reachesb_edges]=1
    edge_color[reachesb_edges]=plt.cm.plasma(0)
    
    edge_width[reachesg_edges]=1
    edge_color[reachesg_edges]=mcolors.to_rgba_array('darkslategrey')
    
    edge_width[communicates_edges]=5
    edge_color[communicates_edges]=mcolors.to_rgba_array('darkorange')
    
    nx.draw(graph_nx,ax=ax,
            node_color=np.vstack([robot_color,ground_color,block_color,side_sup_color,new_block_color]),
            nodelist =list(np.hstack([robot_nodes,ground_nodes,block_nodes,side_sup_nodes,new_block_nodes])),
            edgelist= edge_list_action_descb+
                      edge_list_action_descg+
                      edge_list_put_against+
                      edge_list_choses+
                      edge_list_holds+
                      edge_list_touchesb+
                      edge_list_touchesg+
                      edge_list_reachesb+
                      edge_list_reachesg+
                      edge_list_communicates,
            width=edge_width,
            edge_color=edge_color,
            pos = pos,
            node_size = node_size,
            arrows = False)
    plt.show()
    return graph_nx
if __name__ ==  "__main__":
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.5)
    linkr = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=0.5) 
    linkl = Block([[0,0,0],[0,1,1],[1,0,0],[0,1,0],[0,0,1],[-1,1,1]],muc=0.5) 
    linkh = Block([[0,0,0],[0,1,1],[1,0,0],[-1,2,1],[0,1,0],[0,2,1]],muc=0.5)
    triangle = Block([[0,0,1]],muc=0.5)
    from discrete_simulator_norot import DiscreteSimulator
    sim = DiscreteSimulator([10,10], 2, [hexagon,linkl,linkr,linkh], 2, 30, 100,ground_blocks=[triangle])
    sim.add_ground(triangle, [1,0])
    sim.add_ground(triangle, [8,0])
    sim.put_rel(hexagon, 0, 0, 0, 0,idconsup=1,blocktypeid=0)
    sim.put_rel(linkl, 0, 0, 1, 0,blocktypeid=1)
    sim.put_rel(hexagon, 0, 0, 2, 0,blocktypeid=0)
    sim.put_rel(hexagon, 0, 0, 3, 1,blocktypeid=3)
    sim.hold(0, 2)
    sim.hold(1, 3)
    pyg_graph = create_sparse_graph(sim, 1, ['Ph'], 'cpu', sim.n_side_oriented_sup, sim.n_side_oriented,last_only=True)
    
    nx_graph = draw_graph(pyg_graph)