# Modelo SEIRD determinístico de tempo contínuo
#import Pkg; Pkg.build("GR")
using OrdinaryDiffEq, PyPlot, Distributions, DataFrames, CSV, Dates
include("leitura_dados.jl")

# dS/dt = -beta*S*I/N
# dE/dt = beta*S*I/N - alpha*E
# dI/dt = alpha*E - gamma*I
# dR/dt = gamma*(1-eta)*I
# dD/dt = gamma*eta*I

function ODE_SEIRD_det(du,u,p,t)
    N, beta, alpha, gamma, eta = p
    du[1] = -(beta/N)*u[1]*u[3]
    du[2] = (beta/N)*u[1]*u[3] - alpha*u[2]
    du[3] = alpha*u[2] - gamma*u[3]
    du[4] = gamma*(1-eta)*u[3]
    du[5] = gamma*eta*u[3]
end

function plot_solution(sol)
    figure(1)
    plot(sol.t, sol[1,:], label = "S(t)")
    plot(sol.t, sol[2,:], label = "E(t)")
    plot(sol.t, sol[3,:], label = "I(t)")
    plot(sol.t, sol[4,:], label = "R(t)")
    plot(sol.t, sol[5,:], label = "D(t)")
    title("Evolução da epidemia simulada\n Modelo SEIRD determinístico")
    xlabel("t (dias)")
    ylabel("Número de indivíduos")
    legend()
    savefig("plotsol_SEIRD_det.png")
    #savefig("plotsol_SEIRD_det_SP.png")

    #delay = 40
    #D_obs = zeros(Int, 60+delay)
    #D_obs[1+delay:45+delay] = le_SIVEP_Gripe("INFLUD22-14-02-2022.csv")
    #D_obs[1+delay:60+delay] = le_SIVEP_Gripe("INFLUD22-09-03-2022.csv")
    #D_obs = le_SIVEP_Gripe_2("INFLUD21-14-02-2022.csv", "INFLUD22-09-03-2022.csv")

    #Medida do estado D(t)
    #D_obs = Int64.(round.(rand.(Uniform.(0, 2*(sol[5,2:end].-sol[5,1:end-1])))))
    #D_obs = rand.(Poisson.(sol[5,2:end].-sol[5,1:end-1]))
    #D_obs = rand.(Binomial.(Int64.(round.(sol[3,2:end])), 0.3*0.01))
    #D_obs = rand.(NegativeBinomial.(Int64.(round.(sol[3,2:end])), 1-0.3*0.01))

    figure(2)
    plot(sol.t[2:end], sol[5,2:end] - sol[5,1:end-1], label = "Modelo")
    #plot(sol.t[2:end], D_obs, label = "Dados")
    title("Mortes diárias\n Modelo SEIRD determinístico")
    xlabel("t (dias)")
    ylabel("Número de indivíduos")
    legend()
    savefig("mortdia_SEIRD_det.png")
    #savefig("mortdia_SEIRD_det_SP.png")

    #EQM = mean((D_obs - (sol[5,2:end] - sol[5,1:end-1])).^2)
    #println(" ")
    #println("EQM = ", EQM)
end

function main_prog()

    #Dados de teste
    N = 1_000_000   #população
    E0 = 5.0        #número inicial de expostos
    I0 = 2.0        #número inicial de infectados
    beta = 0.6      #susceptibilidade
    alpha = 0.2     #inverso do período médio de latência
    gamma = 0.3     #inverso do tempo médio de recuperação
    eta = 0.01      #taxa de mortalidade (IFR)

    #Dados reais SP
    #N = 12_325_232      #população absoluta (SP capital)
    #I0 = 2.0            #número inicial de infectados
    #beta = 0.8373
    #beta = 0.9063       #susceptibilidade
    #alpha = 1/3       #inverso do período médio de latência
    #gamma = 1/5         #inverso do tempo médio de recuperação
    #eta = 0.00013       #taxa de mortalidade (IFR) - NetoNature2021

    p = [N; beta; alpha; gamma; eta]
    u0 = [N-E0-I0; E0; I0; 0.0; 0.0]
    tspan = (0.0, 365.0)

    prob = ODEProblem(ODE_SEIRD_det, u0, tspan, p)
    sol = solve(prob, Tsit5(), reltol=1e-16, abstol=1e-16, saveat=1)

    #Número de reprodução básico
    R0 = beta/gamma
    println("R0 = ", R0)
    println(" ")

    #Estados no fim da simulação
    println("No último instante simulado:")
    println("S = ", Int64.(round.(sol[1,end])))
    println("E = ", Int64.(round.(sol[2,end])))
    println("I = ", Int64.(round.(sol[3,end])))
    println("R = ", Int64.(round.(sol[4,end])))
    println("D = ", Int64.(round.(sol[5,end])))
    
    println(" ")

    println("Valor máximo de E = ", Int64(round(maximum(sol[2,:]))), " no dia ", argmax(sol[2,:])-1)
    println("Valor máximo de I = ", Int64(round(maximum(sol[3,:]))), " no dia ", argmax(sol[3,:])-1)
    println("Variação máxima de R = ", Int64(round(maximum(sol[4,2:end]-sol[4,1:end-1]))), " no dia ", argmax(sol[4,2:end]-sol[4,1:end-1]))
    println("Variação máxima de D = ", Int64(round(maximum(sol[5,2:end]-sol[5,1:end-1]))), " no dia ", argmax(sol[5,2:end]-sol[5,1:end-1]))

    plot_solution(sol)
end 

main_prog()