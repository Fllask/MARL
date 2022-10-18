# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 13:19:57 2022

@author: valla
"""
import scipy
import numpy as np
n=5
m=2
T=30
x0 = 40*np.random.sample((1,n))
A = np.random.sample((n,n))
A = A / np.max(abs(np.linalg.eig(A)[0]))
B = np.random.sample((n,m))
Q = np.diag(np.ones(n))
QT = np.diag(5*np.ones(n))
W = 0.1*np.diag(np.ones(n))

P = np.zeros((T+1,n,n))
r = np.zeros((T+1,1))
K = np.zeros((T+1,m,n))
P[T,:,:] = QT
for t in reversed(range(T)):
    P[t,:,:]= Q + A.T@(P[t+1,:,:]-P[t+1,:,:]@B@np.linalg.inv(B.T@P[t+1,:,:]@B)@B.T@P[t+1,:,:]@A)
    K[t,:,:]= -np.linalg.inv(B.T@P[t+1,:,:]@B)@B.T@P[t+1,:,:]@A
#%%
n=1
m=1
T=30
x0 = 40*np.random.sample((1,n))
A = np.array([[1]])
B = np.array([[1]])
Q = np.diag(np.ones(n))
QT = np.diag(0*np.ones(n))
W = 1*np.diag(np.ones(n))

P = np.zeros((T+1,n,n))
r = np.zeros((T+1,1))
K = np.zeros((T+1,m,n))
P[T,:,:] = QT
for t in reversed(range(T)):
    P[t,:,:]= Q + A.T@(P[t+1,:,:]-P[t+1,:,:]@B@np.linalg.inv(B.T@P[t+1,:,:]@B)@B.T@P[t+1,:,:]@A)
    K[t,:,:]= -np.linalg.inv(B.T@P[t+1,:,:]@B)@B.T@P[t+1,:,:]@A
#%%
scipy.linalg.solve_discrete_are(A, B, Q, np.zeros((m,m)))

    
    
    
    
    