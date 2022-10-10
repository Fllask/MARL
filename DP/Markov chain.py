# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:58:45 2022

@author: valla
"""
import numpy as np
import matplotlib.pyplot as plt
def next_x(x,u,d,C):
    return np.clip(x+u-d,0,C)

p = np.array([0.1,0.2,0.7])
w = np.array([0,1,2])
C=6
T = 100
x0 = C
x = np.zeros(T+1)
x[0]=x0
u = np.zeros(T)
d = np.zeros(T)
for t in range(T):
    d[t]=np.random.choice(w,p=p)
    u[t]=(C-x[t])*(x[t]<=1)
    x[t+1] =next_x(x[t],u[t],d[t],C)
plt.plot(range(T+1),x)
#%% Dynamic programming: finite state and horizon
n = 500
m=25
T=50
P=np.random.sample((m,n,n))
P = P/np.sum(P,axis=2)[:,:,None]
pi = np.zeros((T,n))
J = np.zeros((T+1,n))
gT = np.random.sample(n)
def g(t,x,u):
    return x

J[1,:]=gT
for t in reversed(range(T)):
    for i in range(n):
        J_pi = np.zeros((m,n))
        for j in range(m):
            J_pi[j,:]=g(t,i,j)+P[j,i,:].T*J[t+1,:]
        J[t,:] = np.min(J_pi,axis=0)
        pi[t,:]= np.argmin(g(t,i,j)+P[:,i,:]*J[t+1,:],axis=0)
#%% Value iteration: dynamic programming with finite state and infinite horizon

n = 20*20
eps = 1e-5
m=25
T=50
def P_f(x,u,x0):
    if 
P = np.zeros((n,2,n))
P[:20]=
pi = np.zeros(n)
J = np.zeros(n)
def g(t,x,u):
    if u == 0:
        return 0
    else:
        if x == 
    return x

J[1,:]=gT
while abs(J-newJ) > np.ones(n)*eps:
    for i in range(n):
        J_pi = np.zeros((m,n))
        for j in range(m):
            J_pi[j,:]=g(t,i,j)+P[j,i,:].T*J[t+1,:]
        J[t,:] = np.min(J_pi,axis=0)
        pi[t,:]= np.argmin(g(t,i,j)+P[:,i,:]*J[t+1,:],axis=0)
