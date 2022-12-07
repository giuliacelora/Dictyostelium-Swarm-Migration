#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:09:01 2022

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
import importlib.util
from mpi4py import MPI
import pandas as pd
nsim=5
norm_activity_vec=[2.5,5.]
#define the model parameters
mu=1
g=0.0#1
E=0.25
eps=0.1
C=1.
norm_activity=50.#norm_activity_vec[nsim]
gamma=0.1
chi=0.#1.0/gamma
radius_droplet=1.

lmb=0.27/4
omega=lmb/eps**2
D_b=1e-4
L_slip=1.#5.
# adaptive time step
timestep=0.1
dt = Constant(timestep) # saves the current time step
dt2= Constant(1.0) # time step for the more accurate approximation for time-stepping dt2=dt/2
slip=True
rank=MPI.COMM_WORLD.Get_rank()
namedic="Figure_activity_"+str(nsim)+"_version_vel"

if rank==0:
    dic_par={'R0':radius_droplet,'L_splip':L_slip,'activity':norm_activity,'mu':mu,'g':g,'E':E,'chi':chi,'gamma':gamma,'lambda':lmb,'D_b':D_b}
    try:
        os.makedirs(namedic)
        print('Folder created')
    except:
        print('Folder already exist')
    df=pd.DataFrame(dic_par,index=[0])
    df.to_csv(namedic+'/parmeter_simulation.csv')
    print("saved parameters")
    

#%%%
# define the mesh, with restriction to the boundary in contact with the bath
inlet=1
outlet=2
bottom=3
top=4

H=2
L=10.0
mesh=RectangleMesh(Point(0,0),Point(L,H),int(L/0.05),int(H/0.05))
class Border(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0, 1.) 

Border = Border()

# Number of refinements
nor = 0
for i in range(nor):
    cell_markers = MeshFunction("bool", mesh,mesh.topology().dim())
    cell_markers.set_all(False)
    Border.mark(cell_markers, True)

    mesh=refine(mesh, cell_markers)
    
#%%
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
 

class InterfaceIn(SubDomain):
    def inside(self,x,on_boundary):
        return near(x[0],0)   

class InterfaceOut(SubDomain):
    def inside(self,x,on_boundary):
        return near(x[0],L)  
class Floor(SubDomain):
    def inside(self,x,on_boundary):
        return near(x[1],0) 
class Top(SubDomain):
    def inside(self,x,on_boundary):
        return near(x[1],H)  
InterfaceIn().mark(boundaries, inlet)
InterfaceOut().mark(boundaries, outlet)
Floor().mark(boundaries, bottom)
Top().mark(boundaries, top)



#%%%%% Initialisation of the solution with the homogeneous solution

xcentre=1.5
ycentre=0.0
class ICond_n(UserExpression):
    def eval(self, values, x):
        i=0
        for el in values:
            values[i]=0.0
            i+=1
        radius=np.sqrt((x[0]-xcentre)**2+(x[1]-ycentre)**2)
        
        values[0] = tanh((radius_droplet-radius)/0.1)
     

    def value_shape(self):
        return (2,)

class ICond_b(UserExpression):
    def eval(self, values, x):
       
        values[0] = (tanh((x[0]-xcentre-radius_droplet/2)/0.5)+1)/2*(tanh((1*radius_droplet/2-x[1])/0.1)+1)/2

    def value_shape(self):
        return ()


class BC_Cond(UserExpression):
    def __init__(self,phi,gradb,**kwargs):       
        super().__init__(**kwargs)

        self.phi=phi
        self.gradb=gradb        

    def eval(self, values, x):
        values[0]=0.0
        if near(x[1],0):
            gradb_x=self.gradb(x)
            values[0] = (self.phi(x)+1)/2*C*gradb_x[0]
            
        values[1]=0.0

    def value_shape(self):
        return (2,)
#%% Definition of the measures
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

#%% Definition of Functional space and functions
Sel=FiniteElement("Lagrange",mesh.ufl_cell(),2)
Vel=VectorElement("Lagrange",mesh.ufl_cell(),2,dim=2)
Pel=FiniteElement("Lagrange",mesh.ufl_cell(),1)

V_b=FunctionSpace(mesh,Sel) # function space for bacteria
V_n=FunctionSpace(mesh,MixedElement([Sel,Sel])) # function space for n=[phi,psi]
V=FunctionSpace(mesh,Vel) 
V_u=FunctionSpace(mesh,MixedElement([Vel,Pel])) # functions space for u=[vel,p]

u = Function(V_u) # new iterates with time step dt
uhigh = Function(V_u) # new iterates with time step dt

nnew= Function(V_n) # contain solution at the current time-step
nold= Function(V_n) # contain solution at the previous time-step
noldhigh= Function(V_n) # contain solution at the previous time-step
nhigh = Function(V_n)
vel_save=Function(V)
bnew = Function(V_b)
bold = Function(V_b)
boldhigh = Function(V_b)
dul = TrialFunction(V_u)
dn = TrialFunction(V_n)
db = TrialFunction(V_b)
v_u= TestFunction(V_u)
v_n= TestFunction(V_n)
v_b= TestFunction(V_b)


#%%%%% set the initial conditions
nold.interpolate(ICond_n())

bold.interpolate(ICond_b())
boldhigh.interpolate(ICond_b())
assign(nnew,nold)
#%%
def D_operator(u):
    return sym(grad(u))
def Stress_operator(u,p):
    
    return 2*mu*D_operator(u)-p*Identity(len(u))
def Gamma(phi,b):
    return g*b*(phi+1)/2
def activity_1(b,v):
    norm_b=(inner(grad(b),grad(b)))**(1/2)
    director_vector= grad(b)/norm_b
    I=Identity(len(director_vector))
    term_activity=norm_activity*norm_b*(outer(director_vector,director_vector)-I/2)
    return inner(term_activity*(1+phi)**2/4,D_operator(v))
def activity_2(b,v):
    return inner(norm_activity*grad(b)*(1+phi)**2/4,v)
def residual_u(u,n,b):
    (vel,p)=split(u) 
    (v_vel,v_p)=split(v_u)  

    (phi,psi)=split(n) #access the previous solution
    F_u= inner(Stress_operator(vel,p),D_operator(v_vel))
    F_p=div(vel)*v_p
    L_u=inner((psi+chi*b)*grad(phi),v_vel)+activity_2(b,v_vel)
    L_p=Gamma(phi,b)*v_p
    if slip:
        return [(F_u+F_p)*dx+vel[0]/L_slip*v_vel[0]*ds(bottom),(L_p+L_u)*dx(metadata={'quadrature_degree': 2})]
    else:
        return [(F_u+F_p)*dx,(L_p+L_u)*dx]
def motility(phi):
    return gamma*(1+phi)**2/4
def dFdphi(phi):
    x=variable(phi)
    f=(x+1)**2*(1-x)**2/4
    return diff(f,x)#*(phi+1)/2
def residual_n(n,nprev,dt,u,b):

    (vel,_)=split(u) #access the new solution

    (phi_prev,psi_prev)=split(nprev) #access the previous solution
    (phi,psi)=split(n)
    (v_phi,v_psi)=split(v_n)   #access the trial functions

   
    
    #definition of the variational form for chemical pontential
    F_psi=psi*v_psi-lmb*inner(grad(phi),grad(v_psi))-(dFdphi(phi_prev)+2*(phi-phi_prev))*v_psi*omega+chi*b*v_psi
    
    # definition of the variational form for time dependent evolution of the phase field parameter
    
    F_phi= phi*v_phi+dt*inner(motility(phi_prev)*grad(psi),grad(v_phi))+dt*(inner(vel,grad(phi))+phi*Gamma(phi_prev,b))*v_phi
    
    L_phi=phi_prev*v_phi+Gamma(phi_prev,b)*v_phi*dt
   # L_psi=(phi_prev**3*v_psi-3*phi_prev*v_psi)*omega
        
    return F_psi*dx+F_phi*dx-L_phi*dx # full residual

    
def residual_b(b,b_prev,dt,n,nold):

    (phi,psi)=split(n)
    (phi_old,_)=split(nold)

    #definition of the variational form for bacteria concentration
    
    F_b=b*v_b+dt*(E*(1+phi)/2*b*v_b+D_b*inner(grad(b),grad(v_b)))

    L_b=b_prev*v_b
    return [F_b*dx,L_b*dx] # full residual
#$%%
Ftemp=residual_n(nnew,nold,Constant(0),u,bold)
Jtemp=derivative(Ftemp,nnew,dn)
P_temp=NonlinearVariationalProblem(Ftemp,nnew,[],Jtemp);
solver_temp=NonlinearVariationalSolver(P_temp)
solver_temp.solve()
solver_temp.parameters['nonlinear_solver']='snes'

solver_temp.parameters['snes_solver']['krylov_solver']["absolute_tolerance"] = 1e-15
solver_temp.parameters['snes_solver']['krylov_solver']["relative_tolerance"] = 1e-8
solver_temp.parameters['snes_solver']['krylov_solver']["maximum_iterations"] = 20
solver_temp.parameters['snes_solver']['linear_solver']= 'gmres'
nhigh.assign(nnew)
noldhigh.assign(nnew)
nold.assign(nnew)
initial_mass=assemble(nold.sub(0)*dx)
#%% initialisation of the simulation

t=0.0 # current timepoint
set_log_active(False) # avoid printing lots of info 
Toll=0.05 # tolerance for the time-stepping
dtmax=0.5 # maximum time-step
dtmin=1e-6 # minimum time-step
timestep=1e-3 # intial time step
tend=100 # final time for the simulation

dt = Constant(timestep) # saves the current time step
dt2= Constant(1.0)      # saves the refined time step


#%% Definition of the variational problems
# variational prolem for the phase field and chemical potential 
# (I use an adaptive time step method that is why I define it twice)
F_n_low=residual_n(nnew,nold,dt,u,bold)
J_n=derivative(F_n_low,nnew,dn)
P_n=NonlinearVariationalProblem(F_n_low,nnew,[],J_n);
solver_n_Low= NonlinearVariationalSolver(P_n)
solver_n_Low.parameters["nonlinear_solver"] ="newton"
solver_n_Low.parameters['newton_solver']['linear_solver']="mumps"
prm=solver_n_Low.parameters['newton_solver']['krylov_solver']
prm["absolute_tolerance"] = 1e-6
prm["relative_tolerance"] = 1e-4
prm["maximum_iterations"] = 20

F_n_high=residual_n(nhigh,noldhigh,dt2,uhigh,boldhigh)
J_n_high=derivative(F_n_high,nhigh,dn)
P_n_high=NonlinearVariationalProblem(F_n_high,nhigh,[],J_n_high);
solver_n_High= NonlinearVariationalSolver(P_n_high)
solver_n_High.parameters["nonlinear_solver"] ="newton"
solver_n_High.parameters['newton_solver']["linear_solver"] ="mumps"
prm=solver_n_High.parameters['newton_solver']['krylov_solver']
prm["absolute_tolerance"] = 1e-6
prm["relative_tolerance"] = 1e-4
prm["maximum_iterations"] = 20

#%% variational problem for the bacteria
F_b,L_b=residual_b(db,boldhigh,dt2,nhigh,nold)
#J_b=derivative(F_b,bnew,db)
P_b=LinearVariationalProblem(F_b,L_b,bnew,[]);
solver_b= LinearVariationalSolver(P_b)

solver_b.parameters["linear_solver"] ="gmres"


# variational problem for the velocity and pressure is inside the for loop 
# since the boundary conditions are non-constant
#%% variables to save the solution
namedic="Figure_activity_"+str(nsim)+"_version_vel"
hdf = HDF5File(mesh.mpi_comm(), '/tmp/simulation_output'+str(nsim)+'.h5','w')
hdf.write(mesh, "mesh")
file_save_phi=File(namedic+"/droplet.pvd")
file_save_velocity=File(namedic+"/vel_field.pvd")
file_save_bac=File(namedic+"/bacterium.pvd")

#%% Time loop with time-step controll
print('initiation simulation')
tlast=-1
arrest=False
(phi,_)=nold.split()
M=project(grad(bold),V)
fun_boundary=BC_Cond(phi,M)
uboundary=interpolate(fun_boundary,V)
  

# boundary condition for the velocity
bcs1 = DirichletBC(V_u.sub(0),Constant((0,0)),boundaries,top)
if slip:
    bcs2=DirichletBC(V_u.sub(0).sub(1),Constant((0)),boundaries,bottom)
else:
    bcs2 = DirichletBC(V_u.sub(0),uboundary,boundaries,bottom)

#  bcs2 = DirichletBC(V_u.sub(1), Constant(0), boundaries, bottom)

bcs=[bcs1,bcs2]
F_u,L_u=residual_u(dul,nhigh,boldhigh)
P_u=LinearVariationalProblem(F_u,L_u,uhigh,bcs);
solver_u=LinearVariationalSolver(P_u)
solver_u.parameters["linear_solver"] ="mumps"
prm=solver_u.parameters["krylov_solver"]
prm["absolute_tolerance"]=1e-6
prm["relative_tolerance"]=1e-4
prm["maximum_iterations"]=20
prm["nonzero_initial_guess"] = True
#sol ver_u.parameters["preconditioner"] ="ilu"
nsave=0
solver_u.solve()  
u.assign(uhigh) 
print('solved u') 
#%% 
time_vec=[]
while t<tend:

    # solve for the velocity and pressure
   

    retry=True
    # solve for n=(phi,psi) using adaptive time-stepping
    while retry:
        dt.assign(timestep)        
        dt2.assign(timestep/2)
        try:
            solver_n_Low.solve()#solve(F_n_low==0,nnew,[],solver_parameters=dict(linear_solver="gmres"))
            Conv=True
        except:
            Conv=False
            print('Problem convergence_1')

        if Conv:
            try:

                solver_n_High.solve()#(F_n_high==0,nhigh,[],solver_parameters=dict(linear_solver="gmres"))
                Conv=True
                solver_b.solve()
                boldhigh.assign(bnew)
                noldhigh.assign(nhigh)
                if not slip:
                    
                    (phi,_)=nhigh.split()
                    M=project(grad(boldhigh),V)
                    fun_boundary=BC_Cond(phi,M)
                    uboundary=interpolate(fun_boundary,V)
                    bcs2 = DirichletBC(V_u.sub(0),uboundary,boundaries,bottom)
              #  bcs2 = DirichletBC(V_u.sub(1), Constant(0), boundaries, bottom)

                    bcs=[bcs1,bcs2]
                    F_u,L_u=residual_u(dul,nhigh,boldhigh)
                    P_u=LinearVariationalProblem(F_u,L_u,uhigh,bcs);
                    solver_u=LinearVariationalSolver(P_u)
                    solver_u.parameters["linear_solver"] ="gmres"
                    solver_u.parameters["preconditioner"] ="amg"
                    
                solver_u.solve()  
                solver_n_High.solve()#solve(F_n_high==0,nhigh,[],solver_parameters=dict(linear_solver="gmres"))
                solver_b.solve()
                boldhigh.assign(bnew)
            except:
                Conv=False
                print('Problem convergence_2')

        
        
        nvals=bnew.vector().get_local()
        if np.sum(nvals<0)>1:
            Conv=False
            print('Problem positivity')
        if Conv:
    
            eta=np.sqrt(assemble(dot(nhigh-nnew,nhigh-nnew)*dx)/(H*L))
            
            
            if (eta<Toll) or timestep==dtmin:
                t=t+timestep
        
                timestep=np.max([np.min([dtmax,Toll/2/eta*timestep]),dtmin])  
                # successful step: update with the new solution 
                assign(nold, nhigh) 
                assign(noldhigh, nhigh) # update with the new soltion
                assign(nnew, nhigh) # update with the new soltion
                assign(bold,bnew)
                assign(boldhigh,bnew)
                assign(u,uhigh)

                retry=False
            else:
                timestep=np.max([np.min([dtmax,Toll/2/eta*timestep]),dtmin])   
                # step failed: reset solution and retry
                assign(nnew, nold) 
                assign(nhigh,nold)
                assign(noldhigh, nold)
                assign(bnew,bold)
                assign(boldhigh,bold)
                assign(uhigh,u)


        else:
            if timestep==dtmin:
                print("Problem: Minimum Time Step reached")
                arrest=True
                break
            else:
                # step failed: reset solution and retry

                timestep=np.max([np.min([dtmax,timestep/2]),dtmin])   
                assign(nnew, nold) 
                assign(nhigh,nold)
                assign(noldhigh, nold)
                assign(bnew,bold)
                assign(boldhigh,bold)
                assign(uhigh,u)
    if arrest:
        break
    
        

    # boundary condition for the velocity
    if not slip:
        
        (phi,_)=nold.split()
        M=project(grad(bold),V)
        fun_boundary=BC_Cond(phi,M)
        uboundary=interpolate(fun_boundary,V)
        bcs2 = DirichletBC(V_u.sub(0),uboundary,boundaries,bottom)
  #  bcs2 = DirichletBC(V_u.sub(1), Constant(0), boundaries, bottom)

        bcs=[bcs1,bcs2]
        F_u,L_u=residual_u(dul,high,boldhigh)
        P_u=LinearVariationalProblem(F_u,L_u,uhigh,bcs);
        solver_u=LinearVariationalSolver(P_u)
        solver_u.parameters["linear_solver"] ="gmres"
        solver_u.parameters["preconditioner"] ="amg"
        
        
    solver_u.solve() 
    u.assign(uhigh)   
    print("done with time:", t)
    # solving for the distribution of bacteria 
    if int(t*10)>int(tlast*10):
        (vel,_)=u.split()
        (phi,_)=nnew.split()
        hdf.write(vel, 'velocity/iter'+str(nsave))
        hdf.write(phi, 'phase_field/iter'+str(nsave))
        hdf.write(bnew, 'bacterium_concentration/iter'+str(nsave))
        vel_save.interpolate(u.sub(0))
        file_save_velocity<< (vel_save,t)
        file_save_phi<< (nold.sub(0),t)
        file_save_bac<< (bold,t)
        time_vec.append(t)
        print('Time:',t)     
        tlast =t 
        nsave+=1
        
        #%%
import csv
if rank==0:

    with open(namedic+'time.csv', 'w') as csvfile:
            csv.writer(csvfile, delimiter=' ').writerow(tvec)
    csvfile.close() 