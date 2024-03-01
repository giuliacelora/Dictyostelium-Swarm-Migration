module Extra_Migration
export FH, DS, save_solution

using LinearAlgebra, SparseArrays,Parameters, Setfield,DelimitedFiles,Plots

xmax=2.
xmin=0.02
function DS(DCDx,par)
    
    f(el)=(tanh((abs(el)-xmin)/0.25/xmin)+tanh((xmax-abs(el))/0.1/xmax))
    S_x=f.(DCDx)
    return S_x
end

function FH(res,H,W,U,M,p,N)
    @unpack K,A,Lsd,θ,mu0,E,γ,expDC,S=p
    θ1=sqrt(-2*U*mu0+θ^2)
    θ2=sqrt(2*U*mu0+θ^2)
    dx=1/N
    DC=10^expDC    
    
    H_with_BC=vcat(0.0,H,0.0) # dirichlet BC

    ############################################################################################
    # definition differential operators
    D3=diagm.(0=>3. .*ones(N+1))-diagm.(1=>3. .*ones(N))+diagm.(2=>ones(N-1))-diagm.(-1=>ones(N))
    D3[end-1,end]=-2
    D3H=1/(dx)^3/W^2 .*D3[2:end-1,:]*H_with_BC
    D3H[end]-=θ2/dx^2/W^2

    D2=diagm.(0=>-2. .*ones(N+1))+diagm.(-1=>ones(N))+diagm.(1=>ones(N))
    D2*=DC/(dx*W)^2
    D1=diagm.(0=>-1. .*ones(N+1))+diagm.(1=>ones(N))
    D1*=1/(dx*W)
    ############################################################################################  
    
    ############################################################################################
    # solving for the chemoattractant
    H_int=[0]
    for n in 2:length(H_with_BC)
        H_int=[H_int;H_int[end]+(H_with_BC[n-1]+H_with_BC[n])/2]
    end
    H_int=H_int*dx*W^2
    H_int=H_int.-H_int[end]
    xx=collect(range(0,1,step=dx))*W
    F=exp.(E/U*H_int) # source for the concentration
    
    λp=(-U+sqrt(U^2+4*γ*DC))/DC/2 
    λm=(-U-sqrt(U^2+4*γ*DC))/DC/2 
    
    Q=D2+U*D1-γ*diagm.(0=>ones(N+1))
    Q[1,1:2]=[-1/dx/W-λp 1/dx/W]
    Q[end,end-1:end]=[-1/dx/W +1/dx/W-λm]
    RHS=[-λp*exp(-E*M/U);-γ*F[2:end-1];-λm]

    C=Q\RHS # solve for the chemoattractant
    ############################################################################################
   
    mH=(W*H./3 .+Lsd).*H*W # motility
    ###### active component of the velocity
    DxC=D1*C
    DxC[end]=λm*(C[end]-1)
    Sprime=S(DxC[2:end],p)
    F_A=A*Sprime[2:end].*(DxC[3:end]-DxC[2:end-1])/dx/W  
    ##################
    
    v_i= K*(D3H.+F_A).*mH # total velocity
    res[1]=H[end]-dx*θ2 # strong imposure contact angle
    res[2:N]=v_i[1:end].-U 
    res[N]=H[1]-dx*θ1 # strong imposure contact angle
    res[N+1]=dx*sum(H)*W^2-M # constrain volume
end


"""
Sets the initial condition either from a given file
"""

function initial_cond(N,p,fromfile)
    @unpack filename=p
    x0=zeros(N+1,)

    if fromfile
        x0=readdlm(filename, '\t')
    else
        println("No initial condition available. Default given")
        dx=1/N

        xx=range(dx,1-dx,step=dx)
        x0[1:N-1].=xx.*(1 .-xx)
        x0[N]=0.5
        mass=1/2-1/3
        x0[N+1]=mass
    end

    return x0
end

function export_h(H_no_bound,W)
    
    H=vcat(0,H_no_bound,0).*W
   
    return H
end

function export_c(H,U,W,p,N,M)
    @unpack E,γ,expDC,S=p

    dx=1/N
    DC=10^expDC    

    H_with_ghost=vcat(0.0,H,0) 
    H_int=[0]
    for n in 2:length(H_with_ghost)
        H_int=[H_int;H_int[end]+(H_with_ghost[n-1]+H_with_ghost[n])/2]
    end
    H_int=H_int*dx*W^2
    H_int=H_int.-H_int[end]
    xx=collect(range(0,1,step=dx))*W
    F=exp.(E/U*H_int) # source for the concentration
   
    D2=diagm.(0=>-2. .*ones(N+1))+diagm.(-1=>ones(N))+diagm.(1=>ones(N))
    D2*=DC/(dx*W)^2
    D1=diagm.(0=>-1. .*ones(N+1))+diagm.(1=>ones(N))
    D1*=1/(dx*W)
    
    λp=(-U+sqrt(U^2+4*γ*DC))/DC/2 #E*W/U
    λm=(-U-sqrt(U^2+4*γ*DC))/DC/2 #E*W/U
    
    Q=D2+U*D1-γ*diagm.(0=>ones(N+1))
    Q[1,1:2]=[-1/dx/W-λp 1/dx/W]
    Q[end,end-1:end]=[-1/dx/W +1/dx/W-λm]
    RHS=[-λp*exp(-E*M/U);-γ*F[2:end-1];-λm]

    C=Q\RHS
   
    return C
end
function export_flux(H,U,p)
    @unpack Lsd=p
    mH=(H./3 .+Lsd) # 

    flux2=U./mH
    return flux2
end

function save_solution_A(namedir,branch,Avalue,N,par_mod)
    U_vec= Vector{Float64}()
    W_vec= Vector{Float64}()
    M_vec= Vector{Float64}()
    solution_C=[]
    solution_H=[]
    solution_F=[]


    for el in reverse(branch.sol)
        append!(U_vec,el.x[end-1])    
        append!(W_vec,el.p)    

        append!(M_vec,el.x[end])
        H=export_h(el.x[1:end-2],W_vec[end])
        append!(solution_H,H)
        C=export_c(el.x[1:end-2],U_vec[end],W_vec[end],par_mod,N,M_vec[end])
        append!(solution_C,C)
        append!(solution_F,export_flux(H,U_vec[end],par_mod))

    end
   
    Nx=length(branch.sol[1].x)
    solution_H=reshape(solution_H,Nx,length(W_vec));
    solution_C=reshape(solution_C,Nx,length(W_vec));
    solution_F=reshape(solution_F,Nx,length(W_vec));

  

    folder=namedir*"/A_value_"*string(Avalue)*"/"
    try
        mkdir(folder)
    catch
        println("Folder already exists")
    end

    M=hcat(M_vec,W_vec,U_vec)

    open(folder*"summary.txt","w") do io
        writedlm(io,M,',')
    end
    open(folder*"solution_H.txt","w") do io
        writedlm(io,solution_H,',')
    end
    open(folder*"solution_C.txt","w") do io
        writedlm(io,solution_C,',')
    end
    open(folder*"solution_Flux.txt","w") do io
        writedlm(io,solution_F,',')
    end
end

function save_solution(namedir,sol_dic,N,par_mod)
     try
        mkdir(namedir)
    catch
        println("Folder already exists")
    end
    for A in keys(sol_dic)
        save_solution_A(namedir,sol_dic[A],A,N,par_mod)
    end
end

end