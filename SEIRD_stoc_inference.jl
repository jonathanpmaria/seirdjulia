using Turing, Random

#Funções adaptadas para modelo SEIRD 
"""
    SEIRDinference(Do, Δt, N)
Função para inferência bayesiana de modelos SEIRD 
simples no Turing.
`Do` => Número de óbitos observados
`Δt` => Passo do dados (em dias)
`N`  => População total da região
"""
@model function SEIRDinference(Do, Δt, N, ::Type{T} = Int64) where{T}
    # Parâmetros do modelo

    # Taxa de infecção
    #β ~ Gamma(1.5, 2.0)
    β ~ Uniform(0.0, 10.0)

    # Taxa de incubação
    #σ ~ Beta(2.0, 2.0)
    #σ ~ Uniform(0.05, 0.5)
    σ = 0.2
    #σ = 1/3.42

    # Taxa de recuperação
    #μ ~ Beta(2.0, 2.0)
    #μ ~ Uniform(0.05, 0.5)
    μ = 0.3
    #μ = 0.1*0.2 + (1-0.1)*0.3 #assint.
    #μ = 1/14

    # Taxa de mortalidade (IFR)
    #eta ~ Uniform(0.0, 0.1)
    eta = 0.01
    #eta = 0.1*0.05 #assint.
    #eta = 0.0002

    # Probabilidade de reportagem de óbito
    #p ~ Beta(1.2, 1.0)
    p ~ Uniform(0.8, 1.0)
    #p = 1
    
    # Condição inicial
    E0 ~ DiscreteUniform(0, 10) # Uniforme discreta
    I0 ~ DiscreteUniform(1, 10) # Uniforme discreta
    #E0 = 5
    #I0 = 2
    
    # Número de observações
    Nsteps = length(Do)
     
    # Compartimentos
    Δse = Vector(undef, Nsteps)
    Δei = Vector(undef, Nsteps)
    Δir = Vector(undef, Nsteps)
    Δid = Vector(undef, Nsteps)
    
    # Condições iniciais
    s = N - E0 - I0
    e = E0 
    i = I0
    r = 0
    d = 0
    Do[1] ~ Binomial(d, p)
    
    # Modelo
    for n in 1:Nsteps-1
        if s > 0
            Δse[n] ~ Binomial(s, 1 - exp(-(Δt/N)*β*i))
        else
            #Δse[n] ~ Dirac(0.0)
            Δse[n] ~ Binomial(0, 0.0)
            s = 0
        end
        if e ≥ 0
            Δei[n] ~ Binomial(e, 1 - exp(-σ*Δt))
        else
            #Δei[n] ~ Dirac(0.0)
            Δei[n] ~ Binomial(0, 0.0)
            e = 0
        end
        if i > 0
            Δir[n] ~ Binomial(i, 1 - exp(-μ*Δt))
            Δid[n] ~ Binomial(Δir[n], eta)
        else
            #Δir[n] ~ Dirac(0.0)
            #Δid[n] ~ Dirac(0.0)
            Δir[n] ~ Binomial(0, 0.0)
            Δid[n] ~ Binomial(0, 0.0)
            i = 0
        end
        
        s = s - Δse[n]
        e = e + Δse[n] - Δei[n]
        i = i + Δei[n] - Δir[n]
        r = r + Δir[n] - Δid[n]
        d = d + Δid[n]
        Do[n+1] ~ Binomial(d, p)
    end
    return Δse, Δei, Δir, Δid, E0, I0 
end

function SEIRD_next_step(u, p, dt)
    N, beta, alpha, gamma, eta = p

    Delta_SE = rand(Binomial(u[1], 1-exp(-(beta/N)*u[3]*dt)))
    Delta_EI = rand(Binomial(u[2], 1-exp(-alpha*dt)))
    Delta_IR = rand(Binomial(u[3], 1-exp(-gamma*dt)))
    Delta_ID = rand(Binomial(Delta_IR, eta))

    S = u[1] - Delta_SE
    E = u[2] + Delta_SE - Delta_EI
    I = u[3] + Delta_EI - Delta_IR
    R = u[4] + Delta_IR - Delta_ID
    D = u[5] + Delta_ID

    #return [S; E; I; R; D]
    return [max(S, 0); max(E, 0); max(I, 0); max(R, 0); max(D, 0)]
end

function gera_cumulativo(x)
    x_acum = zeros(Int,1,length(x))
    x_acum[1] = x[1]
    for i = 2:length(x)
        x_acum[i] = x_acum[i-1] + x[i]
    end
    return x_acum
end

function BayesianSEIRD()
    Δt = 1
    tspan = (0, 200)
    Nsteps = floor(Int, (tspan[2] - tspan[1]) / Δt) + 1
    u = zeros(Int64, 5, Nsteps)
    t = zeros(Nsteps)

    S_obs = zeros(Int, Nsteps)
    E_obs = zeros(Int, Nsteps)
    I_obs = zeros(Int, Nsteps)
    R_obs = zeros(Int, Nsteps)
    D_obs = zeros(Int, Nsteps)

    #Dados de teste
    N = 1_000_000   #população
    # #N = 12_325_232
    # E0 = 5          #número inicial de expostos
    # I0 = 2          #número inicial de infectados
    # beta = 0.6      #susceptibilidade
    # alpha = 0.2     #período médio de latência (1/alpha)
    # gamma = 0.3     #tempo médio de recuperação (1/gamma)
    # eta = 0.01      #taxa de mortalidade (IFR)
    
    # p = [N; beta; alpha; gamma; eta]
    
    # while R_obs[end] ≤ N/4 # Procurar uma simulação em que há epidemia 
    #     u = zeros(Int64, 5, Nsteps)
    #     u[:,1] = [N-E0-I0; E0; I0; 0; 0]

    #     for i = 2:Nsteps
    #        u[:,i] = SEIRD_next_step(u[:,i-1], p, Δt)
    #     end

    #     S_obs = u[1,:]
    #     E_obs = u[2,:]
    #     I_obs = u[3,:]
    #     R_obs = u[4,:]
    #     D_obs = u[5,:]
    # end

    #D_obs de referência (SEIRD simples) - mortes cumulativo
    D_ref = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 8, 9, 9, 10, 10, 10, 10, 13, 13, 15, 15, 15, 16, 17, 19, 22, 25, 28, 32, 33, 34, 35, 38, 42, 49, 54, 59, 61, 65, 71, 74, 78, 87, 95, 112, 130, 152, 171, 190, 212, 236, 265, 310, 350, 382, 414, 475, 541, 594, 656, 705, 769, 838, 950, 1049, 1134, 1239, 1375, 1498, 1626, 1768, 1927, 2100, 2276, 2433, 2618, 2813, 3040, 3224, 3431, 3652, 3865, 4066, 4278, 4497, 4726, 4938, 5144, 5339, 5554, 5743, 5932, 6121, 6296, 6466, 6629, 6788, 6951, 7083, 7209, 7337, 7452, 7528, 7628, 7722, 7802, 7871, 7950, 8008, 8054, 8124, 8179, 8220, 8252, 8285, 8321, 8349, 8380, 8405, 8429, 8452, 8471, 8484, 8501, 8513, 8531, 8545, 8561, 8572, 8582, 8587, 8596, 8602, 8612, 8617, 8619, 8626, 8631, 8634, 8638, 8641, 8643, 8646, 8647, 8648, 8648, 8648, 8651, 8652, 8653, 8654, 8654, 8655, 8656, 8656, 8657, 8659, 8660, 8660, 8660, 8661, 8662, 8663, 8663, 8663, 8663, 8664, 8665, 8666, 8667, 8667, 8667, 8668, 8668]
    
    #D_obs de referência (SEIRD com assintomáticos) - mortes diárias
    #D_ref = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 1, 0, 1, 2, 3, 1, 1, 2, 4, 2, 1, 5, 5, 3, 6, 11, 8, 12, 10, 17, 21, 16, 21, 26, 25, 22, 43, 31, 48, 57, 48, 70, 70, 85, 91, 96, 106, 114, 136, 131, 122, 132, 129, 132, 154, 149, 131, 119, 148, 119, 131, 129, 104, 118, 104, 108, 109, 99, 79, 74, 67, 64, 64, 48, 51, 57, 58, 42, 37, 27, 34, 21, 27, 26, 24, 19, 21, 12, 11, 12, 6, 12, 5, 6, 4, 9, 7, 4, 3, 3, 6, 3, 5, 1, 2, 3, 1, 1, 1, 1, 0, 0, 0, 1, 2, 2, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    #D_obs de referência (SEIRD com assintomáticos) - mortes cumulativo
    #D_ref = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 2 2 3 3 4 4 6 7 7 8 10 13 14 15 17 21 23 24 29 34 37 43 54 62 74 84 101 122 138 159 185 210 232 275 306 354 411 459 529 599 684 775 871 977 1091 1227 1358 1480 1612 1741 1873 2027 2176 2307 2426 2574 2693 2824 2953 3057 3175 3279 3387 3496 3595 3674 3748 3815 3879 3943 3991 4042 4099 4157 4199 4236 4263 4297 4318 4345 4371 4395 4414 4435 4447 4458 4470 4476 4488 4493 4499 4503 4512 4519 4523 4526 4529 4535 4538 4543 4544 4546 4549 4550 4551 4552 4553 4553 4553 4553 4554 4556 4558 4559 4559 4559 4560 4560 4561 4561 4562 4562 4562 4562 4562 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563 4563]
    
    #D_obs de referência (dados reais ômicron, 180 dias) - mortes diárias
    #N = 12_325_232
    #D_ref = [0, 0, 2, 1, 1, 2, 7, 3, 0, 1, 5, 3, 2, 5, 8, 7, 0, 7, 10, 6, 3, 8, 4, 6, 3, 5, 8, 19, 9, 12, 12, 11, 12, 15, 15, 12, 18, 17, 13, 15, 27, 14, 27, 23, 20, 22, 29, 28, 27, 43, 35, 48, 45, 52, 49, 57, 66, 69, 74, 54, 66, 60, 90, 93, 85, 69, 77, 73, 62, 81, 73, 74, 71, 59, 57, 65, 62, 71, 72, 70, 57, 53, 42, 47, 53, 38, 37, 38, 40, 38, 34, 26, 25, 17, 24, 37, 30, 28, 26, 20, 16, 13, 24, 22, 15, 23, 14, 12, 11, 14, 16, 12, 10, 12, 12, 5, 9, 5, 17, 8, 13, 14, 3, 6, 6, 4, 6, 8, 7, 8, 3, 2, 6, 4, 9, 9, 5, 6, 4, 3, 10, 6, 8, 9, 4, 7, 3, 2, 8, 1, 5, 6, 2, 4, 2, 7, 10, 4, 4, 2, 4, 3, 4, 4, 3, 3, 3, 2, 4, 4, 3, 2, 3, 3, 5, 3, 4, 3, 4, 2]
    #D_ref[1:10] = zeros(1,10)

    #D_obs = D_ref[2:end] - D_ref[1:end-1]
    D_obs = D_ref
    #D_obs = gera_cumulativo(D_ref)

    n_chains = 4

    #chns = sample(SEIRDinference(D_obs[1:100], Δt, N), SMC(100), 1000)
    #chns = mapreduce(chn -> sample(SEIRDinference(D_obs[1:100], Δt, N), SMC(10000), 1000), chainscat, 1:n_chains)

    g = Gibbs(HMC(0.02, 20, :p, :σ, :μ, :eta), PG(50, :β, :Δse, :Δei, :Δir, :Δid, :E0, :I0))
    #g = Gibbs(HMC(0.02, 50, :p, :μ, :eta), PG(100, :β, :σ, :Δse, :Δei, :Δir, :Δid, :E0, :I0))
    #g = Gibbs(HMC(0.02, 50, :p, :σ, :eta), PG(100, :β, :μ, :Δse, :Δei, :Δir, :Δid, :E0, :I0))
    #g = Gibbs(HMC(0.02, 50, :p, :σ, :μ), PG(100, :β, :eta, :Δse, :Δei, :Δir, :Δid, :E0, :I0))
    #g = Gibbs(HMC(0.02, 50, :p, :eta), PG(100, :β, :σ, :μ, :Δse, :Δei, :Δir, :Δid, :E0, :I0))
    #g = Gibbs(HMC(0.02, 50, :p, :μ), PG(100, :β, :σ, :eta, :Δse, :Δei, :Δir, :Δid, :E0, :I0))
    #g = Gibbs(HMC(0.02, 50, :p, :σ), PG(100, :β, :μ, :eta, :Δse, :Δei, :Δir, :Δid, :E0, :I0))
    #g = Gibbs(HMC(0.02, 50, :p), PG(100, :β, :σ, :μ, :eta, :Δse, :Δei, :Δir, :Δid, :E0, :I0))

    #chns = sample(SEIRDinference(D_obs[1:100], Δt, N), g, 200)
    chns = mapreduce(chn -> sample(SEIRDinference(D_obs[1:80], Δt, N), g, 200), chainscat, 1:n_chains)

    #return chns, S_obs, E_obs, I_obs, R_obs, D_obs, p
    return chns
end

function main_prog()
    #chn, S_obs, E_obs, I_obs, R_obs, D_obs, p = BayesianSEIRD()
    chns = BayesianSEIRD()

    #println(D_obs)

    chns |> display

    #plot(chns)
    #savefig("PMCMC_all.png")

end

main_prog()