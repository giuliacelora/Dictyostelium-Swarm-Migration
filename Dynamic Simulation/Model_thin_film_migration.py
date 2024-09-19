#%%
from dolfin import *
import csv
from numpy import pi,min, inf,empty,zeros,savetxt,array,sqrt
import numpy as np
from ufl import tanh,sinh,cosh,cos,sin,log
import matplotlib.pyplot as plt
import pdb
import os
import importlib.util
import sys
import pickle

namedic='save_solution/'
L=40.
nx=2000

x0=-5
K_vec=[4.,4.5,5.,5.25,5.5,5.75,6.,6.5,7.,7.5,8.]
mesh=IntervalMesh(nx,x0,L+x0)
step_x=L/nx

"""
definition of model and simulation paraemters
"""
timestep=1e-3
dt = Constant(timestep) 
A_vec=[3.,3.5,4.,4.5,5.,5.25,5.5,5.75,6.,6.5,7.,7.5,8.]
A=A_vec[int(sys.argv[1])] # active capillary number (Ca_\xi)
E=0.335 # rate of bacteria consumption 
Lsd=0.12 # slip length
Kappa=K_vec[int(sys.argv[2])] # inverse of the capillary number (Ca_k)^{-1}

g=Constant(0.0)
gmax=0.025 # proliferation rate of cells
tstart_prol=25 # time at which proliferation is activated in the model
scaling_A=2/(3*sqrt(3)) # scaling of the alignment order parameter
D_b=0.0001 # diffusion coefficient bacteria
D_c=9.6 # diffusion coefficient chemoattractant
E_c=9.6 # decay rate chemoattractant
alpha=0.02 # steepness sensing gradient \alpha
delta=5/65 # control length scale disjoining pressure H_\delta
b0=0.05 # threshold bacteria concentration for cell proliferation arrest
nb=2. # rhill coefficient for the proliferation rate of cells
tend=500 # end time for the simulations

"""
definition simulation mesh
"""

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
ds=Measure('ds',domain=mesh,subdomain_data=boundaries)
n = FacetNormal(mesh)

front=1
class Front_Fun(SubDomain):
    def inside(self,x,on_boundary):
        return near(x[0],L)   
Front_Fun().mark(boundaries, front)


"""
weak formulation of the model
"""

def psi(h):
    return -3*delta**2/(h+delta)**3*(1-delta/(h+delta)) # disjoining pressure

def alignment_strenght(x):
    S = tanh(alpha*x)**(2)/alpha/scaling_A # alignment function up to a constant. Since we are only interested in gradients of S, the constant can be neglected
    return S*A

def residual(u,v,h,s,b): # weak form for the swarm height
    h_dot,theta=split(u)
    v1,v2=split(v)
   
    Mot=(h/3+Lsd)*h**2
    
    F_h_dot= h_dot*v1+Mot*Kappa*inner(grad(theta),grad(v1))-g*b**nb/(b0**nb+b**nb)*dt*h_dot*v1
    F_pi = (theta)*v2 - dt*inner(grad(h_dot),grad(v2))
    L_hdot=g*h*b**nb/(b0**nb+b**nb)*v1*dx
    L_pi = inner(grad(h),grad(v2))*dx-(psi(h)+s)*v2*dx
    return [(F_pi+F_h_dot)*dx,L_pi+L_hdot]

def residual_B(B,Bold,vB,H): # weak form for the bacteria concentration
    F_b=B*vB +dt*inner(D_b*grad(B),grad(vB)) + E*dt*B*H*vB
    L_b=Bold*vB
    return [F_b*dx,L_b*dx]

def residual_C(C,Cold,vC,b): # weak form for the chemoattractant concentration
    F_c=C*vC +dt*inner(D_c*grad(C),grad(vC)) + E_c*dt*C*vC
    L_c=(Cold+E_c*b*dt)*vC
    return [F_c*dx,L_c*dx]

"""
Definition of the functional space for the solution
"""
Fel=FiniteElement("Lagrange",mesh.ufl_cell(),1) # finite element
F_B=FunctionSpace(mesh,Fel) # function space for bacteria
F_u=FunctionSpace(mesh,MixedElement([Fel,Fel])) # function space for n=[phi,pi]

u = Function(F_u) # new iterates with time step dt
H= Function(F_B) # contains the swarm height solution at the current time-step
S= Function(F_B) # contains the alignment fucntion S at the current time-step
B= Function(F_B) # contains bacterial field at the current time-step
C=Function(F_B) # contains chemoattractant field at the current time-step

Cold=Function(F_B) # contains chemoattractant field at the previous time-step
Bold = Function(F_B) # contains bacterial field at the previous time-step
Hold = Function(F_B) # contains swarm height at the previous time-step
F_vel=Function(F_B) # contains the flux for the height at the current time

dul = TrialFunction(F_u)
db = TrialFunction(F_B)
v_u= TestFunction(F_u)
v_b= TestFunction(F_B)

"""
set up of the iteration problem for height, bacterial and chemoattractant field
"""
res_u,L_u=residual(dul,v_u,H,S,B)
P_u=LinearVariationalProblem(res_u,L_u,u,[]);
solver_u=LinearVariationalSolver(P_u) # solve height problem

res_B,L_B=residual_B(db,Bold,v_b,H)
bcsB = DirichletBC(F_B,Constant((1.)),boundaries,front)
P_B=LinearVariationalProblem(res_B,L_B,B,[]);
solver_B=LinearVariationalSolver(P_B) # solve bacterial field problem

res_C,L_C=residual_C(db,Cold,v_b,B)
P_C=LinearVariationalProblem(res_C,L_C,C,[]);
solver_C=LinearVariationalSolver(P_C) # solve chemoattractant field problem


#%%
"""
set up of the intiial condition
"""
R0=1 # location of the initial clump
t=0.0 

t_vec=[]
sol_vec=[]
B_vec=[]

(h_dot,theta)=split(u)

x1=0

class ICond_mass(UserExpression):
    def eval(self, values, x):
       
        values[0] = 2.*exp(-(x[0]-R0/2)**2/0.2)

    def value_shape(self):
        return ()
    
class ICond_bac(UserExpression):
    def eval(self, values, x):
        
        values[0] = np.min([exp(3.*x[0]),1]) 

    def value_shape(self):
        return ()
class ICond_shift(UserExpression):
    def add_fun(self,fun,xmax):
        self.fun=fun
        self.xmax=xmax
    def eval(self, values, x):
        oldcoord = x[0] + self.xmax
        if oldcoord > L+x0:
            values[0]= self.fun(L+x0)
        elif oldcoord<-2:
            values[0]= self.fun(L+x0)

        else:
            values[0] = self.fun(oldcoord)

    def value_shape(self):
        return ()
    
H.interpolate(ICond_mass())
Hold.assign(H)
B.interpolate(ICond_bac())
Bold.assign(B)
C.assign(B)
Cold.assign(B)


#%% saving the solution 
X0_vec=[]
S_vec=[]
F_vel_vec=[]

boundary_check=True
flag=False
tlast=-1 # time at which the solution has been last saved
KH=ICond_shift() # function used to shift the solution when approcing the domain boundary
Mcrit=6.#4
restart = False
X0=0 # oring of the system updated when shifting the solution to avoid contact with the boundary
step_fig=25. # step size for saving images of the solution (must be a multiple of step_save_sol)
step_save_sol=0.2 # step size for saving the solution 

parameters={'inhibition_growth':b0,'Diffusion_B':D_b,'Diffusion_C':D_c,'sens_min':xmin,'sens_max':xmax,'slip':Lsd,'K_over_A':Kappa,'growth_rate':g((0,)),
            'chemo_consumption':E_c,'bacteria_consumption':E,'delta':delta,'domain_size':L}

try:
    os.makedirs(namedic)
    print('Folder created')
except:
    print('Folder already exist')
    exit()

    
with open(namedic+'/saved_dictionary.pkl', 'wb') as f:
        pickle.dump(parameters, f)
f.close()

set_log_active(False) # avoid printing lots of info 
print("Start of the simulation\n")
while t<tend:
    if t>tstart_prol:
        g.assign(gmax)
    bx=project(Dx(C,0),F_B)
    Stemp=project(alignment_strenght(bx),F_B) # evaluation of the alignment function
    S.assign(Stemp)
    
    solver_u.solve() # solving for \partial_t h
    H.assign(H+dt*project(h_dot,F_B)) # updating the swarm heigh - h

    solver_B.solve() # updating the bacterial field B
    solver_C.solve() # updating the chemoattractant field C   
   
    t+=timestep # updating time

    """
    save/plot the solution
    """
    if int(t/step)>int(tlast/step):
        t_vec.append(t) 
        # adaptive time stepping
        sol_vec.append(H.vector()[::-1])
        B_vec.append(B.vector()[::-1])
        S_vec.append(S.vector()[::-1])
        solver_vel.solve()
        F_vel_vec.append(F_vel.vector()[::-1])
        
        if int(t/step_fig)>int(tlast/step_fig):
            plt.figure()
            plot(C)
            plot(B)

            plot(H)
            
            plt.ylim(0,4)
            print('done with time '+str(t))

            plt.savefig(namedic+"/time"+str(t)+".png")
        tlast=t
        X0_vec.append(X0)
   
    """
    Computation distance swarm front from the RHS boundary. If smaller than a certain tolerance the reference frame is shifted to allow the
    swarm to coninue migration
    """
    temp=H.vector()[::-1]
    indexes=np.where(temp>delta*2.)[0]
    if len(indexes)>0:
        if indexes[-1]>nx-100:
            # decision of new origin to acoid cutting parts of the swarm
            initial_points=[n for n in indexes if n+1 not in indexes]
            initial_points=[int(initial_points[0]/2)]+initial_points
            final_points=[n for n in indexes if n-1 not in indexes]
            new=np.argmin(temp[initial_points[-2]:final_points[-1]])+initial_points[-2]
            distance=(final_points[-1]-initial_points[-2])/nx*L
            location=(new)/nx*L

            KH.add_fun(H,location)
            H.interpolate(KH)
            KH.add_fun(B,location)
            B.interpolate(KH)
            KH.add_fun(C,location)
            C.interpolate(KH)
            X0+=location # new origin of the domain in the "lab" reference frame
 


    Bold.assign(B)
    Cold.assign(C)


#%%
#save solution
import csv
M_B=np.array(B_vec)
M_S=np.array(S_vec)
M_F=np.array(F_vel_vec)
M_H=np.array(sol_vec)
X0_vec=np.array(X0_vec)
t_vec=np.array(t_vec)
import os
print('solving solution at the destination '+namedic)

with open(namedic+'/time.csv', 'w') as csvfile:
        csv.writer(csvfile, delimiter=' ').writerow(t_vec)
csvfile.close() 
with open(namedic+'/X0.csv', 'w') as csvfile:
        csv.writer(csvfile, delimiter=' ').writerow(X0_vec)
csvfile.close() 
with open(namedic+'/height.csv', 'w') as csvfile:
        csv.writer(csvfile, delimiter=' ').writerows(M_H)
csvfile.close() 
with open(namedic+'/bacteria.csv', 'w') as csvfile:
        csv.writer(csvfile, delimiter=' ').writerows(M_B)
csvfile.close() 
with open(namedic+'/alignment.csv', 'w') as csvfile:
        csv.writer(csvfile, delimiter=' ').writerows(M_S)
csvfile.close() 
with open(namedic+'/pressure.csv', 'w') as csvfile:
        csv.writer(csvfile, delimiter=' ').writerows(M_F)
csvfile.close() 

# %%
