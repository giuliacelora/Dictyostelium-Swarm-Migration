module Dynamics_Dictyostelium_group
using Interpolations,SparseArrays,LinearAlgebra, Parameters, DelimitedFiles,Plots, ColorSchemes,Revise
includet("Extra_function_dynamics.jl")
using ..ExtraFun
mutable struct Dictyostelium_droplet
    h::Vector{Float64}
    x::Vector{Float64}
end
mutable struct Dictyostelium_solution
    n_droplet::Int64
    droplets::Vector{Dictyostelium_droplet}
end
mutable struct Chemoattractant_solution
    C::Vector{Float64}
    x::Vector{Float64}
end
mutable struct Bacteria_solution
    B::Vector{Float64}
    x::Vector{Float64}
end
mutable struct Solution
    chemoattractant::Chemoattractant_solution
    dictyostelium::Dictyostelium_solution
    bacteria::Bacteria_solution
end
function add_droplet(sld::Dictyostelium_solution,x_drop::Vector{Float64},h_drop::Vector{Float64})
    setfield!(sld, :n_droplet,sld.n_droplet+1)
    setfield!(sld, :droplets,push!(sld.droplets,Dictyostelium_droplet(h_drop,x_drop)))
end
function save_solution(time,solution,save_directory)
    M=hcat(solution.chemoattractant.x,solution.chemoattractant.C,solution.bacteria.B)
    open(save_directory*"/chemoattractant/time"*string(time)*".txt","w") do io
        writedlm(io,M,',')
    end
    dictyostelium_droplets=solution.dictyostelium.droplets
    x_lawn=dictyostelium_droplets[1].x
    h_lawn=dictyostelium_droplets[1].h
    for n in 2:solution.dictyostelium.n_droplet
        x_lawn=[x_lawn;dictyostelium_droplets[n].x]
        h_lawn=[h_lawn;dictyostelium_droplets[n].h]
    end
    M=hcat(x_lawn,h_lawn)
    open(save_directory*"/dictyostelium/time"*string(time)*".txt","w") do io
        writedlm(io,M,',')
    end

end

function check_splitting(solution,threshold)
    droplet_new=Vector{Dictyostelium_droplet}()
    for n in 1:solution.n_droplet
        h=solution.droplets[n].h
        x=solution.droplets[n].x

        index_up=[y for y in 1:length(h)-1 if (h[y]<threshold) & (h[y+1]>threshold)] 
        index_down=[y for y in 1:length(h)-1 if (h[y]>threshold) & (h[y+1]<threshold)] 
        index_up[1]=1
        index_down[end]=length(h)
        npoint=length(x)
        if length(index_up)>1
            for m in 1:length(index_up)
                index_cut_start=minimum([index_up[m],index_down[m]])
                index_cut_end=maximum([index_up[m],index_down[m]])
                h1=h[index_cut_start:index_cut_end]
                h1[end]=0
                h1[1]=0
                x1=range(x[index_cut_start],x[index_cut_end],npoint)
                interp_linear = linear_interpolation(x[index_cut_start:index_cut_end], h1)
                h1=interp_linear.(x1)
                push!(droplet_new,Dictyostelium_droplet(h1,x1))
            end
            
        else
            push!(droplet_new,Dictyostelium_droplet(h,x))
        end
    end
    setfield!(solution,:droplets, droplet_new)
    setfield!(solution,:n_droplet, length(droplet_new))
end
function check_boundary(solution,L)
    x_lawn=solution.chemoattractant.x
    C_lawn=solution.chemoattractant.C
    B_lawn=solution.bacteria.B
    droplets=solution.dictyostelium.droplets
    front_location=droplets[end].x[end]
    if x_lawn[end]-front_location<L
        if solution.dictyostelium.n_droplet>1
            xcut=(droplets[end-1].x[end]+droplets[end].x[1])/2
        else
            xcut=(x_lawn[1]+droplets[end].x[1])/2
        end

        setfield!(solution.dictyostelium,:n_droplet,1)
        setfield!(solution.dictyostelium,:droplets,solution.dictyostelium.droplets[end:end])

        size_lawn=x_lawn[end]-x_lawn[1]
        x_lawn=[x_lawn;x_lawn[end]+size_lawn]
        x_lawn_new=collect(range(xcut,xcut+size_lawn,length(x_lawn)))
        print("xlawn",x_lawn[1],x_lawn[end],"\n")
        print("xlawn_new",x_lawn_new[1],x_lawn_new[end],"\n")

        B_lawn=[B_lawn;1]
        C_lawn=[C_lawn;1]
        interp_linear = linear_interpolation(x_lawn, C_lawn)
        C=interp_linear(x_lawn_new)
        setfield!(solution.chemoattractant,:C,C)
        setfield!(solution.chemoattractant,:x,x_lawn_new)
        interp_linear = linear_interpolation(x_lawn, B_lawn)
        B=interp_linear(x_lawn_new)
        setfield!(solution.bacteria,:B,B)
        setfield!(solution.bacteria,:x,x_lawn_new)
        print("splitting")
        return true
    else
        return false
    end

end
function time_dynamics(physical_par,simulation_par,save_directory)
    @unpack initial_group_length,initial_time,final_time,time_step,npoint_droplet,npoint_lawn,domain_size_lawn,save_every,plot_every=simulation_par
    ### definition of the meshes ####
    X_droplet=collect(range(0,initial_group_length,npoint_droplet));           # mesh droplet
    X_lawn=collect(range(-2,domain_size_lawn,npoint_lawn+1));
    #################################
    
    ### initial conditions ###
    t=initial_time
    h_i  = X_droplet.*(initial_group_length.-X_droplet);
    C_i = exp.(2.5*(X_lawn.-initial_group_length));
    indeces=findall(C_i.>1.)
    C_i[indeces].=1.
    B_i=copy(C_i)
    current_solution=(Solution(Chemoattractant_solution(C_i,X_lawn),
        Dictyostelium_solution(1,[Dictyostelium_droplet(h_i,X_droplet)]),Bacteria_solution(B_i,X_lawn));)
    ##########################
    
    ### define useful variables ###
    nelement=npoint_droplet-1
    nd=Array{Int64}(undef, npoint_droplet-1, 2)
    nd[1:nelement,1]=1:npoint_droplet-1;           # id left point of an element
    nd[1:nelement,2]=2:npoint_droplet;             # id right point of an element
    ##############################
    
    ### make folder to save solution ###
    try
        mkdir(save_directory)   
    catch
        println("Folder already exists")
    end
    try
        mkdir(save_directory*"/chemoattractant")
    catch
        println("subfolder already exists")
    end
    try
        mkdir(save_directory*"/dictyostelium")
    catch
        println("subfolder already exists")
    end
    cmap_C=cgrad(:davos100,rev=:true)
    ##############################
    threshold=0.025
    niter=0
    dummy=false
    while t<final_time
        number_droplets=current_solution.dictyostelium.n_droplet
        current_C=current_solution.chemoattractant.C
        current_X_lawn=current_solution.chemoattractant.x
        for n in 1:number_droplets-1
            h,x=current_solution.dictyostelium.droplets[n].h,current_solution.dictyostelium.droplets[n].x
            h,x=ExtraFun.hydrodynamics_problem(h,x,current_C,current_X_lawn,time_step,physical_par,nd)
            setfield!(current_solution.dictyostelium.droplets[n], :h, h)
            setfield!(current_solution.dictyostelium.droplets[n], :x, x)
        end
        n=number_droplets
        ### front droplet -- splitting proliferation and hydrodynamics
        h,x=current_solution.dictyostelium.droplets[n].h,current_solution.dictyostelium.droplets[n].x


        h,x=ExtraFun.hydrodynamics_problem(h,x,current_C,current_X_lawn,time_step/2,physical_par,nd)
        h=ExtraFun.proliferation_problem(h,time_step,physical_par)
        h,x=ExtraFun.hydrodynamics_problem(h,x,current_C,current_X_lawn,time_step/2,physical_par,nd)

        setfield!(current_solution.dictyostelium.droplets[n], :h, h)
        setfield!(current_solution.dictyostelium.droplets[n], :x, x)
        
        ### update bacteria ##
        B=ExtraFun.update_bacteria(current_solution.dictyostelium.droplets,current_solution.bacteria,physical_par,time_step)
        setfield!(current_solution.bacteria, :B, B)
        ### update the chemoattractant
        C=ExtraFun.update_chemoattractant(B,current_C,current_X_lawn,physical_par,time_step)
        setfield!(current_solution.chemoattractant, :C, C)
        
        if mod(niter,20)==0 
            dummy=check_splitting(current_solution.dictyostelium,threshold)
        end
        dummy=check_boundary(current_solution,initial_group_length)
        niter+=1
        t+=time_step
        
        # plot numerical solution
        if mod(niter,plot_every)==0 || number_droplets<current_solution.dictyostelium.n_droplet
            droplets=current_solution.dictyostelium.droplets
            xinitial,xfinal=droplets[1].x[1]*0.9,droplets[end].x[end]*1.1
            function_chemoattratant=linear_interpolation(current_solution.chemoattractant.x, current_solution.chemoattractant.C)
           # plot(current_solution.chemoattractant.x,current_solution.chemoattractant.C)
            plot([xinitial,xinitial],[threshold,threshold])
            for n in 1:current_solution.dictyostelium.n_droplet
                colors=function_chemoattratant.(droplets[n].x)
                plot!(droplets[n].x,droplets[n].h,fillrange=droplets[n].h*0.,c=cmap_C,lw=2.,lc=:black,fill_z=colors,legend=:false,colorbar=:false,plot_title="time: "*string(round(t)))
            end
            display(plot!(xlim=(xinitial,xfinal),ylim=(0,2.5)))
        end
        # save numerical solution
        if mod(niter,save_every)==0
            save_solution(t,current_solution,save_directory)
        end
        
    end
    return current_solution
end

end