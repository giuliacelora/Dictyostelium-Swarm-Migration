#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 14:05:40 2022

@author: celora
"""


import sys  

from dolfin import *
import csv
from numpy import pi,min, inf,empty,zeros,savetxt,array,sqrt
import numpy as np
from ufl import tanh,sinh,cosh
import matplotlib.pyplot as plt
import pdb
import os
from mpi4py import MPI
import importlib.util
from mpi4py import MPI
import pandas as pd
from scipy.interpolate import griddata
nsim=2
H=2
L=10.0
namedic="Figure_activity_"+str(nsim)+"_version_vel"
dic_param=pd.read_csv(namedic+'/parmeter_simulation.csv',index_col=0)
#%%
hdf5 = HDF5File(MPI.COMM_WORLD, 'simulation_output'+str(nsim)+'.h5','r')
mesh=Mesh()
hdf5.read(mesh,'mesh',False)
Q = FunctionSpace(mesh, "Lagrange", 2) # or whatever it is
Vel=VectorElement("Lagrange",mesh.ufl_cell(),2,dim=2)
V=FunctionSpace(mesh,Vel) 
phi_fun = Function(Q)
b_fun   = Function(Q)
vel_fun = Function(V)
nx=100
ny=100
xx=np.linspace(0,L,401)
yy=np.linspace(0,H,101)
xfun = interpolate(Expression("x[0]",degree=1), Q)
loc_droplet=[]

phi=[]
bac_concentration=[]
vel_x=[]
vel_y=[]
hx=0.05
for it in range(50):
    hdf5.read(phi_fun, "phase_field/iter"+str(it))
    index_vec=np.where(np.abs(phi_fun.vector() - 0.5)<1e-1)[0]
    approx_new_loc_droplet=np.max(xfun.vector()[index_vec])
    XX=np.linspace(approx_new_loc_droplet-hx,approx_new_loc_droplet+hx)
    value=[]
    for point in XX:
        value.append(phi_fun((point,0)))
    value=np.array(value)
    loc_droplet.append(XX[np.argmin(np.abs(value-0.5))])
    #%%
vel_drop=np.diff(loc_droplet)/0.1
for it in range(50,52,5):
    hdf5.read(phi_fun, "phase_field/iter"+str(it))
    hdf5.read(b_fun, "bacterium_concentration/iter"+str(it))
    hdf5.read(vel_fun, "velocity/iter"+str(it))
    M=project(grad(b_fun),V)
    new_vel_fun=vel_fun+M
    
    nx=0
    U1=[]
    U2=[]
    U3a=[]
    U3b=[]
    for x in xx:
        temp1=[]
        temp2=[]
        temp3a=[]
        temp3b=[]

        for y in yy:
            temp1.append(phi_fun((x,y)))
            temp2.append(b_fun((x,y)))
            temp_u=new_vel_fun((x,y))
            temp3a.append(temp_u[0]-0.15)
            temp3b.append(temp_u[1])
        U1.append(temp1)
        U2.append(temp2)
        U3a.append(temp3a)
        U3b.append(temp3b)

    phi.append(np.array(U1))
    bac_concentration.append(np.array(U2))

    vel_x.append(np.array(U3a))
    vel_y.append(np.array(U3b))

#%%
X,Y=np.meshgrid(xx,yy)
plt.contour(xx, yy,phi[0].transpose(),levels=[0.5])
plt.streamplot(X,Y, vel_x[0].transpose(), vel_y[0].transpose(), linewidth=2,
                         cmap='autumn',density=[2., 1.])
plt.ylim(0,0.5)
plt.xlim(0,L/2)

#%%

plt.contourf(xx, yy,vel_y[0].transpose(),levels=[0.5])
