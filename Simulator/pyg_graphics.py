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
from geometric_internal_model import create_sparse_graph
from torch_geometric import utils
import numpy as np

robot_colors = plt.cm.Wistia(np.linspace(0.2,1,4))
def draw_graph(pyg_graph,maxs = [10,10],h=8,dist=0.6,ax=None,draw='full'):
    if ax is None:
        fig,ax = plt.subplots(1,figsize=(h*(maxs[0]+1)/((maxs[1])*np.sqrt(3)/2+2.5),h))
        ax.set_xlim(-0.5,maxs[0]+0.5)
        ax.set_ylim(-1,(maxs[1])*np.sqrt(3)/2+1.5)
    
    graph = pyg_graph.to_homogeneous()
    edge_index,edge_type = utils.remove_self_loops(graph.edge_index,graph.edge_type)
    graph.edge_index=edge_index
    graph.edge_type=edge_type
    robot_nodes, = np.nonzero(graph.node_type.numpy()==2)
    nr = len(robot_nodes)
    robot_coords = {robot_nodes[i]: np.array([i*maxs[0]/(nr-1),maxs[1]]) for i in range(nr)}
    
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
    sides_sup_coords = {side_sup_nodes[int(pyg_graph['block','action_desc','side_sup'].edge_index[1,i])]: 
                        pyg_graph['block'].x[int(pyg_graph['block','action_desc','side_sup'].edge_index[0,i]),-2:].numpy()+
                                           dist*np.array([np.cos(angle[pyg_graph['block','action_desc','side_sup'].edge_index[1,i]]),
                                                          np.sin(angle[pyg_graph['block','action_desc','side_sup'].edge_index[1,i]])])
                                                          
                                           for i in range(pyg_graph['block','action_desc','side_sup'].edge_index.shape[1])}
    
    sides_sup_coords.update({side_sup_nodes[int(pyg_graph['ground','action_desc','side_sup'].edge_index[1,i])]: pyg_graph['ground'].x[int(pyg_graph['ground','action_desc','side_sup'].edge_index[0,i]),-2:].numpy()+
                                           dist*np.array([np.cos(angle[pyg_graph['ground','action_desc','side_sup'].edge_index[1,i]]),
                                                          np.sin(angle[pyg_graph['ground','action_desc','side_sup'].edge_index[1,i]])])
                                                          
                                           for i in range(pyg_graph['ground','action_desc','side_sup'].edge_index.shape[1])})
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
    sides_block = pyg_graph['block','action_desc','side_sup'].edge_index[1,:]
    put_against_block = pyg_graph['side_sup','put_against','new_block'].edge_index[1,np.isin(pyg_graph['side_sup','put_against','new_block'].edge_index[0,:],sides_block)]
    new_block_coords = {new_block_nodes[j]: block_coords[block_nodes[int(pyg_graph['block','action_desc','side_sup'].edge_index[0,
                                                                         pyg_graph['block','action_desc','side_sup'].edge_index[1,:]==pyg_graph['side_sup','put_against','new_block'].edge_index[0,j]])]]+
                                              2*dist*np.array([np.cos(angles[j]),np.sin(angles[j])])
                                              for i,j in enumerate(put_against_block)}
    
    
    
    sides_ground = pyg_graph['ground','action_desc','side_sup'].edge_index[1,:]
    put_against_ground = pyg_graph['side_sup','put_against','new_block'].edge_index[1,np.isin(pyg_graph['side_sup','put_against','new_block'].edge_index[0,:],sides_ground)]
    new_block_coords.update({new_block_nodes[j]: ground_coords[ground_nodes[int(pyg_graph['ground','action_desc','side_sup'].edge_index[0,
                                                                         pyg_graph['ground','action_desc','side_sup'].edge_index[1,:]==pyg_graph['side_sup','put_against','new_block'].edge_index[0,j]])]]+
                                              2*dist*np.array([np.cos(angles[j]),np.sin(angles[j])])
                                              for i,j in enumerate(put_against_ground)})
    
    #add the stay node above each robot
    
    
    graph_nx = utils.to_networkx(graph)
    reach_ground, =np.nonzero(graph.edge_type.numpy()==9)
    
    robot_color = robot_colors[np.arange(nr)]
    #robot_color[:,3]=0
    ground_color = np.tile(mcolors.to_rgba_array('darkslategrey'),(ng,1))
    block_color = np.tile(plt.cm.plasma(0),(nb,1))
    side_sup_color = np.tile(mcolors.to_rgba_array('grey'),(nss,1)) 
    #side_sup_color =plt.cm.tab10(sides_sup_id/10)
    new_block_color = np.tile(mcolors.to_rgba_array('gold'),(nnb,1))#plt.cm.Set1(pyg_graph['new_block'].x[:,0].numpy().astype(int))
    #new_block_color[:,3]=0
    
    
    
    pos = robot_coords
    pos.update(ground_coords)
    pos.update(block_coords)
    pos.update(sides_sup_coords)
    pos.update(new_block_coords)
    
    
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
    
    node_size = 100*np.ones(graph.node_type.shape[0])
    node_size[new_block_nodes]=50
    
    
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
    
    edge_width[communicates_edges]=1
    edge_color[communicates_edges]=mcolors.to_rgba_array('darkorange')
    
    
    if draw=='full':
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
    else:
        if 'robot_nodes' in draw:
            nx.draw(graph_nx,ax=ax,
                    node_color=robot_color,
                    nodelist =robot_nodes,
                    edgelist=[],
                    pos = robot_coords,
                    node_size = node_size[robot_nodes])
        if 'ground_nodes' in draw:
            nx.draw(graph_nx,ax=ax,
                    node_color=ground_color,
                    nodelist =ground_nodes,
                    edgelist=[],
                    pos = ground_coords,
                    node_size = node_size[ground_nodes])
        if 'block_nodes' in draw:
            nx.draw(graph_nx,ax=ax,
                    node_color=block_color,
                    nodelist =block_nodes,
                    edgelist=[],
                    pos = block_coords,
                    node_size = node_size[block_nodes])
        if 'new_block_nodes' in draw:
            nx.draw(graph_nx,ax=ax,
                    node_color=new_block_color,
                    nodelist =new_block_nodes,
                    edgelist=[],
                    pos = new_block_coords,
                    node_size = node_size[new_block_nodes])
        if 'new_block_nodes' in draw:
            nx.draw(graph_nx,ax=ax,
                    node_color=new_block_color,
                    nodelist =new_block_nodes,
                    edgelist=[],
                    pos = new_block_coords,
                    node_size = node_size[new_block_nodes])
        if 'side_sup_nodes' in draw:
            nx.draw(graph_nx,ax=ax,
                    node_color=side_sup_color,
                    nodelist =side_sup_nodes,
                    edgelist=[],
                    pos = sides_sup_coords,
                    node_size = node_size[side_sup_nodes])
        if 'action_descb' in draw:
            nx.draw(graph_nx,ax=ax,
                    nodelist =[],
                    edgelist= edge_list_action_descb,
                    width=edge_width[action_descb_edges],
                    edge_color=edge_color[action_descb_edges],
                    pos = pos,
                    arrows = False)
        if 'action_descg' in draw:
            nx.draw(graph_nx,ax=ax,
                    nodelist =[],
                    edgelist= edge_list_action_descg,
                    width=edge_width[action_descg_edges],
                    edge_color=edge_color[action_descg_edges],
                    pos = pos,
                    arrows = False)
        if 'put_against' in draw:
            nx.draw(graph_nx,ax=ax,
                    nodelist =[],
                    edgelist= edge_list_put_against,
                    width=edge_width[put_against_edges],
                    edge_color=edge_color[put_against_edges],
                    pos = pos,
                    arrows = False)
        if 'choses' in draw:
            nx.draw(graph_nx,ax=ax,
                    nodelist =[],
                    edgelist= edge_list_choses,
                    width=edge_width[choses_edges],
                    edge_color=edge_color[choses_edges],
                    pos = pos,
                    arrows = False)
        if 'holds' in draw:
            nx.draw(graph_nx,ax=ax,
                    nodelist =[],
                    edgelist= edge_list_holds,
                    width=edge_width[holds_edges],
                    edge_color=edge_color[holds_edges],
                    pos = pos,
                    arrows = False)
        if 'touchesb' in draw:
            nx.draw(graph_nx,ax=ax,
                    nodelist =[],
                    edgelist= edge_list_touchesb,
                    width=edge_width[touchesb_edges],
                    edge_color=edge_color[touchesb_edges],
                    pos = pos,
                    arrows = False)
        if 'touchesg' in draw:
            nx.draw(graph_nx,ax=ax,
                    nodelist =[],
                    edgelist= edge_list_touchesg,
                    width=edge_width[touchesg_edges],
                    edge_color=edge_color[touchesg_edges],
                    pos = pos,
                    arrows = False)
        if 'reachesb' in draw:
            nx.draw(graph_nx,ax=ax,
                    nodelist =[],
                    edgelist= edge_list_reachesb,
                    width=edge_width[reachesb_edges],
                    edge_color=edge_color[reachesb_edges],
                    pos = pos,
                    arrows = False)
        if 'reachesg' in draw:
            nx.draw(graph_nx,ax=ax,
                    nodelist =[],
                    edgelist= edge_list_reachesg,
                    width=edge_width[reachesg_edges],
                    edge_color=edge_color[reachesg_edges],
                    pos = pos,
                    arrows = False)
        if 'communicates' in draw:
            nx.draw_networkx_edges(graph_nx,ax=ax,
                    nodelist =[],
                    edgelist= edge_list_communicates,
                    width=edge_width[communicates_edges],
                    edge_color=edge_color[communicates_edges],
                    pos = pos,
                    connectionstyle='arc,rad=0',
                    min_source_margin=1,
                    arrows = False)
    return ax
if __name__ ==  "__main__":
    import discrete_graphics as gr
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.5)
    linkr = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=0.5) 
    linkl = Block([[0,0,0],[0,1,1],[1,0,0],[0,1,0],[0,0,1],[-1,1,1]],muc=0.5) 
    linkh = Block([[0,0,0],[0,1,1],[1,0,0],[-1,2,1],[0,1,0],[0,2,1]],muc=0.5)
    triangle = Block([[0,0,1]],muc=0.5)
    from discrete_simulator_norot import DiscreteSimulator
    sim = DiscreteSimulator([10,7], 2, [hexagon,linkl,linkr,linkh], 2, 30, 100,ground_blocks=[triangle])
    sim.add_ground(triangle, [2,0])
    sim.add_ground(triangle, [8,0])
    sim.put_rel(hexagon, 0, 0, 0, 0,idconsup=1,blocktypeid=0)
    sim.put_rel(linkl, 0, 0, 1, 0,blocktypeid=1)
    sim.put_rel(hexagon, 0, 0, 2, 0,blocktypeid=0)
    sim.put_rel(hexagon, 0, 0, 3, 1,blocktypeid=3)
    sim.hold(0, 4)
    sim.hold(1, 3)
    pyg_graph = create_sparse_graph(sim, 1, ['Ph'], 'cpu', sim.n_side_oriented_sup, sim.n_side_oriented,last_only=True)
    #ax=draw_robots_nodes(pyg_graph)
    #ax = draw_bg(pyg_graph,ax=ax,draw_touches=True)
    #ax = draw_actions(pyg_graph,ax=ax,node_size=70)
    #ax = draw_graph(pyg_graph,draw=['ground_nodes','block_nodes'],maxs=[10,5])
    #ax = draw_graph(pyg_graph,draw=['robot_nodes'],maxs=[10,5])
    # ax = draw_graph(pyg_graph,draw=['side_sup_nodes','new_block_nodes'],maxs=[10,5])
    # ax = draw_graph(pyg_graph,draw=['ground_nodes','block_nodes','robot_nodes','side_sup_nodes','new_block_nodes'],maxs=[10,5])
    ax = draw_graph(pyg_graph,draw=['ground_nodes','block_nodes','robot_nodes','reachesg','reachesb','communicates'],maxs=[10,5])
    # ax = draw_graph(pyg_graph,draw=['ground_nodes','block_nodes','robot_nodes','holds'],maxs=[10,5])
    # ax = draw_graph(pyg_graph,draw=['block_nodes','side_sup_nodes','new_block_nodes','put_against','action_descb'],maxs=[10,5])
    # ax = draw_graph(pyg_graph,draw=['robot_nodes','new_block_nodes','choses'],maxs=[10,5])
    #ax = draw_graph(pyg_graph,draw=['ground_nodes','block_nodes','touchesg','touchesb'],maxs=[10,5])
    # ax = draw_graph(pyg_graph,draw='full',maxs=[10,5])
    plt.tight_layout()
    plt.savefig(f'../graphics/visualisation/vis_reaches.pdf')
    #fig,ax = gr.draw_grid([10,5],color='none')
    #gr.fill_grid(ax, sim.grid,fixed_color=plt.cm.plasma(0))
    # plt.tight_layout()
    # plt.savefig(f'../graphics/visualisation/graph_true.pdf')