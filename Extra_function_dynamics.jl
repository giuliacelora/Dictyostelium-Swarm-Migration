module ExtraFun
using Interpolations,SparseArrays,LinearAlgebra,Parameters
function interpolation_grid()
    xii=([-1/3*sqrt(5+2*sqrt(10/7)) -1/3*sqrt(5-2*sqrt(10/7)) 0 1/3*sqrt(5-2*sqrt(10/7)) 1/3*sqrt(5+2*sqrt(10/7))] .+1)/2;
    wii=[(322-13*sqrt(70))/900 (322+13*sqrt(70))/900 128/225 (322+13*sqrt(70))/900 (322-13*sqrt(70))/900]/2;
    return xii, wii
end
function hydrodynamics_problem(h,x,b,xb,dt,param,nd)
    @unpack L_slip,η_slip,saturation_alignment=param
    # construct system matrices
    ndof=length(x)
    S,Sw,M,Dx=build_FE_matrices(h,x,nd,dt,L_slip) # script: matrices A,S,M,Dx for FEM
    dh,P,I,X=build_ALE_matrices(h,x,M,Dx)  # script: matrices for ALE decomposition


    Mbd = sparse(vec([1 2]),vec([1 ndof]),vec([1 1]));
    ZZ  = spzeros(2,ndof);
    ZZ1 = spzeros(2,2);
    Sbd = η_slip*[abs(dh[1])^2 0;0 abs(dh[end])^2];
    g1=  ExtraFun.project_b_to_g(b,xb,x,saturation_alignment)

    # FE problem: build right-hand-side rhs & solve

    rhs=[S*h-M*g1*4;zeros(ndof+2,1);];
    rhs[1]+=(1+(dh[1]^2)/2)/(dh[1]);
    rhs[ndof]-=(1+(dh[end]^2)/2)/(dh[end]);

    # u = (hdot,pi,xdot,lambda)
     A = [-dt*S  M Mbd';M Sw  ZZ';Mbd ZZ Sbd];


    # solve
    u = A\rhs;

    # perform ALE decomposition & update solution
    U = (I-P)\u[1:ndof]; # select only u, forget p

    h = h + dt*I*U;      # update h
    x = x + dt*X*U;      # update x
    return h,x
end
function proliferation_problem(h,dt,param)
    @unpack proliferation_rate=param
    h = h + dt*proliferation_rate*h;      # update h
    return h
end
function build_FE_matrices(h,x,nd,dt,Lslip)
    # build matrices for the finite element method
    npoint = size(x)[1]
    nelement = npoint -1
    ndof = npoint;                # number of degrees of freedoms

    edet = x[nd[1:end,2]]-x[nd[1:end,1]]; # determinant of transformation
    ii  = zeros(nelement,4); # integer array of indices
    jj  = zeros(nelement,4); # integer array of indices
    S_  = zeros(nelement,4); # real  array of matrix values
    Ms_ = zeros(nelement,4); # real  array of matrix values
    Sw_ = zeros(nelement,4); # real  array of matrix values
    Dx_ = zeros(nelement,4); # real  array of matrix values
    # explicit integration for mobility with alpha=2
    mobi = zeros(nelement,1);
    xii,wii=interpolation_grid()

    for i in 1:length(xii)
    
        hval = (1-xii[i])*h[nd[1:end,1]] .+ xii[i] * h[nd[1:end,2]];   

        mobi = mobi .+ wii[i] * hval.^2 .*(hval/3 .+Lslip);
    
    end
    local_mass_p1   = [1/3 1/6;1/6 1/3]; 
    # build global matrices from local matrices 
    for k in 1:nelement    
        dphi = [-1 1]/edet[k];         # local gradient   
        # build local matrices    
        sloc = (dphi'*dphi)  * edet[k];# stiffness matrix (is it here to add the surface tension)

        mloc = local_mass_p1 * edet[k];# mass matrix
        cloc = [dphi;dphi]/2 * edet[k];# derivative matrix
        # generate indices for sparse matrix
        ii[k,:] = [nd[k,1] nd[k,2] nd[k,1] nd[k,2]]; 
        jj[k,:] = [nd[k,1] nd[k,1] nd[k,2] nd[k,2]];    
        # generate values for sparse matrix
        S_[k,:] =         sloc[1:end];
        Ms_[k,:] =         mloc[1:end];
        Sw_[k,:] = mobi[k]*sloc[1:end];
        Dx_[k,:] =         cloc[1:end];

    end        

    # generate sparse matrices, e.g. S(ii(k),jj(k))=S_(k)
    S  = sparse(vec(ii),vec(jj),vec(S_));
    Sw = sparse(vec(ii),vec(jj),vec(Sw_));
    M  = sparse(vec(ii),vec(jj),vec(Ms_));
    Dx = sparse(vec(ii),vec(jj),vec(Dx_));
    # build 2x2 block matrices from ndof x ndof matrices

    return S,Sw,M,Dx
end
function build_ALE_matrices(h,x,M,Dx)
    npoint = size(x)[1]
    ndof = npoint;
    P  = spzeros(ndof,ndof);
    I  = spzeros(ndof,ndof);
    X  = spzeros(ndof,ndof);

    dh = M\(Dx*h);
    xi = (x .-x[1])./(x[end].-x[1]);

    for i in 1:ndof
        P[i,1]     =( 1 .-xi[i]) * dh[i];
        P[i,npoint]=(   xi[i]) * dh[i];
        X[i,1]     =( 1. -xi[i]) ;
        X[i,npoint]=(   xi[i]) ;
    end
    for i in 2:npoint-1
        I[i,i]=1;
    end
    return dh,P,I,X
end

function project_h_to_b(droplets,xb)
    x_lawn=droplets[1].x
    h_lawn=droplets[1].h
    for n in 2:solution.dictyostelium.n_droplet
        x_lawn=[x_lawn;dictyostelium_droplets[n].x]
        h_lawn=[h_lawn;dictyostelium_droplets[n].h]
    end
    if xb[1]<x_lawn[1]
        x_lawn=[xb[1];x_lawn]
        h_lawn=[0;h_lawn]
    end
    if xb[end]>x_lawn[end]
        x_lawn=[x_lawn;xb[end]]
        h_lawn=[h_lawn;0]
    end
    interp_linear = linear_interpolation(x_lawn, h_lawn)

    return interp_linear.(xb)
end

function update_bacteria(droplets,solution,param,dt)
    @unpack Diffusion_coeff_B,consumption_B=param
    B=solution.B
    xB=solution.x
    if consumption_B<0
        xcut=droplets[end].x[end]

        Bnew=zeros(length(B))#project_h_to_b(Solution,xb)
        index_change=findall(xB.>xcut)
        for el in index_change
            Bnew[el]=1.
        end
    else
        h=project_h_to_b(droplers,xB)
        dx=xB[2]-xB[1]

        dl = 1.0*ones(length(xb)-1);
        d = -2.0*ones(length(xb))
        MDxx= copy(Tridiagonal(dl, d, dl))

        MDxx[end,1:end].=0.

        MDxx[1,1:2].=[-2.,2.]

        MDxx=MDxx/dx^2
        A= diagm(1.0.+ consumption_B.*hnew*dt)-Diffusion_coeff_B*MDxx*dt

        Bnew=A\B
    end
    return Bnew
end

function update_chemoattractant(B,C,xb,param,dt)
    @unpack Diffusion_coeff_C,decay_C=param

    dx=xb[2]-xb[1]
    dl = 1.0*ones(length(xb)-1);
    d = -2.0*ones(length(xb))
    MDxx= copy(Tridiagonal(dl, d, dl))

    MDxx[end,1:end].=0.

    MDxx[1,1:2].=[-2.,2.]

    MDxx=MDxx/dx^2
    A= diagm(ones(length(xb)).+ decay_C*dt)-Diffusion_coeff_C*MDxx*dt

    bnew=A\(C+decay_C*dt*B)
    return bnew
end

function increase_domain_b(Solution,b,xb,DB,E,dt)
    # construct the laplacian
    hnew=project_h_to_b(Solution,xb)
    dx=xb[2]-xb[1]

    dl = -1.0*ones(length(xb)-1);
    d = 2.0*ones(length(xb))
    MDxx= copy(Tridiagonal(dl, d, dl))

    MDxx[end,1:end].=0.

    MDxx[1,1:2].=[2.,-2.]

    MDxx=MDxx/dx^2
    A= (diagm(1.0.+ E*dt*hnew).+DB*MDxx*dt)

    bnew=A\b
    return bnew
end
function project_b_to_g(b,xb,x,alpha)
    dx=xb[2]-xb[1]
    DxB=(b[2:end]-b[1:end-1])/dx
    S=(DxB.^2) ./(alpha^2 .+(DxB.^2))
    interp_linear = linear_interpolation(xb[2:end], S)

    return  interp_linear.(x)
end
function project_b_to_h(b,xb,x)

    interp_linear = linear_interpolation(xb, b)

    return  interp_linear.(x)
end

end