using Distributions, PyPlot

# function SEIRD_next_step(u, p, dt)
#     N, beta, alpha, gamma, eta = p

#     Delta_SE = rand(Binomial(u[1], 1-exp(-(beta/N)*u[3]*dt)))
#     Delta_EI = rand(Binomial(u[2], 1-exp(-alpha*dt)))
#     Delta_IR = rand(Binomial(u[3], (1-eta)*(1-exp(-gamma*dt))))
#     Delta_ID = rand(Binomial(u[3], eta*(1-exp(-gamma*dt))))

#     S = u[1] - Delta_SE
#     E = u[2] + Delta_SE - Delta_EI
#     I = u[3] + Delta_EI - Delta_IR - Delta_ID
#     R = u[4] + Delta_IR
#     D = u[5] + Delta_ID

#     #return [S; E; I; R; D]
#     return [max(S, 0); max(E, 0); max(I, 0); max(R, 0); max(D, 0)]
# end

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

function plot_solution(t, u, D_obs)
    figure(1)
    plot(t, u[1,:], label = "S[n]")
    plot(t, u[2,:], label = "E[n]")
    plot(t, u[3,:], label = "I[n]")
    plot(t, u[4,:], label = "R[n]")
    plot(t, u[5,:], label = "D[n]")
    title("Evolução da epidemia simulada\n Modelo SEIRD estocástico")
    xlabel("n (dias)")
    ylabel("Número de indivíduos")
    legend()
    savefig("plotsol_SEIRD_stoc.png")
    #savefig("plotsol_SEIRD_stoc_SP.png")

    figure(2)
    plot(t[2:end], u[5,2:end] - u[5,1:end-1], label = "Modelo")
    plot(t[2:end], D_obs[2:end] - D_obs[1:end-1], label = "Dados")
    title("Mortes diárias\n Modelo SEIRD estocástico")
    xlabel("n (dias)")
    ylabel("Número de indivíduos")
    legend()
    savefig("mortdia_SEIRD_stoc.png")
    #savefig("mortdia_SEIRD_stoc_SP.png")
end

function gera_cumulativo(x)
    x_acum = zeros(Int,1,length(x)+1)
    for i = 1:length(x)
        x_acum[i+1] = x_acum[i] + x[i]
    end
    #println(x_acum)
    return x_acum
end

function main_prog()

    #Dados de teste
    N = 1_000_000   #população
    E0 = 5          #número inicial de expostos
    I0 = 2          #número inicial de infectados
    beta = 0.6      #susceptibilidade
    alpha = 0.2     #período médio de latência (1/alpha)
    gamma = 0.3     #tempo médio de recuperação (1/gamma)
    eta = 0.01       #taxa de mortalidade (IFR)

    #Dados de teste (modelo com assintomáticos)
    #N = 1_000_000   
    #E0 = 1        
    #I0 = 9         
    #beta = 0.7534     
    #alpha = 0.2     
    #gamma = 0.1*0.2 + (1-0.1)*0.3 
    #eta = 0.1*0.05

    #Dados reais SP
    #N = 12_325_232      #população (SP capital)
    #E0 = 20
    #I0 = 2       
    #beta = 0.7251     
    #alpha = 1/3.42
    #alpha = 0.8078       
    #gamma = 1/14
    #gamma = 0.5669       
    #eta = 0.0081*0.8252
    #eta = 0.0002

    p = [N; beta; alpha; gamma; eta]

    dt = 1
    N_days = 180
    t = Vector(0:dt:N_days)
    T = length(t)

    N_sim = 100

    u = zeros(Int64, 5, T)

    for k = 1:N_sim
        while u[4,end] ≤ N/4 # Procurar uma simulação em que há epidemia
            u = zeros(Int64, 5, T)
            u[:,1] = [N-E0-I0; E0; I0; 0; 0]
            for i = 2:T
                u[:,i] = SEIRD_next_step(u[:,i-1], p, dt)
            end
        end
        println(u[5,:])
    end
    
    #Número de reprodução básico
    #R0 = beta/gamma
    #println("R0 = ", R0)

    #Estados no fim da simulação
    println("No último instante simulado:")
    println("S = ", u[1,end])
    println("E = ", u[2,end])
    println("I = ", u[3,end])
    println("R = ", u[4,end])
    println("D = ", u[5,end])
    
    println(" ")

    println("Valor máximo de E = ", maximum(u[2,:]), " no dia ", argmax(u[2,:])-1)
    println("Valor máximo de I = ", maximum(u[3,:]), " no dia ", argmax(u[3,:])-1)
    println("Variação máxima de R = ", maximum(u[4,2:end]-u[4,1:end-1]), " no dia ", argmax(u[4,2:end]-u[4,1:end-1]))
    println("Variação máxima de D = ", maximum(u[5,2:end]-u[5,1:end-1]), " no dia ", argmax(u[5,2:end]-u[5,1:end-1]))

    #D_obs para variação de parâmetros (cumulativo)
    #D_obs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 8, 9, 9, 10, 10, 10, 10, 13, 13, 15, 15, 15, 16, 17, 19, 22, 25, 28, 32, 33, 34, 35, 38, 42, 49, 54, 59, 61, 65, 71, 74, 78, 87, 95, 112, 130, 152, 171, 190, 212, 236, 265, 310, 350, 382, 414, 475, 541, 594, 656, 705, 769, 838, 950, 1049, 1134, 1239, 1375, 1498, 1626, 1768, 1927, 2100, 2276, 2433, 2618, 2813, 3040, 3224, 3431, 3652, 3865, 4066, 4278, 4497, 4726, 4938, 5144, 5339, 5554, 5743, 5932, 6121, 6296, 6466, 6629, 6788, 6951, 7083, 7209, 7337, 7452, 7528, 7628, 7722, 7802, 7871, 7950, 8008, 8054, 8124, 8179, 8220, 8252, 8285, 8321, 8349, 8380, 8405, 8429, 8452, 8471, 8484, 8501, 8513, 8531, 8545, 8561, 8572, 8582, 8587, 8596, 8602, 8612, 8617, 8619, 8626, 8631, 8634, 8638, 8641, 8643, 8646, 8647, 8648, 8648, 8648, 8651, 8652, 8653, 8654, 8654, 8655, 8656, 8656, 8657, 8659, 8660, 8660, 8660, 8661, 8662, 8663, 8663, 8663, 8663, 8664, 8665, 8666, 8667, 8667, 8667, 8668, 8668]

    #D_obs para modelo com assintomáticos (diário)
    #D_ref = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 1, 0, 1, 2, 3, 1, 1, 2, 4, 2, 1, 5, 5, 3, 6, 11, 8, 12, 10, 17, 21, 16, 21, 26, 25, 22, 43, 31, 48, 57, 48, 70, 70, 85, 91, 96, 106, 114, 136, 131, 122, 132, 129, 132, 154, 149, 131, 119, 148, 119, 131, 129, 104, 118, 104, 108, 109, 99, 79, 74, 67, 64, 64, 48, 51, 57, 58, 42, 37, 27, 34, 21, 27, 26, 24, 19, 21, 12, 11, 12, 6, 12, 5, 6, 4, 9, 7, 4, 3, 3, 6, 3, 5, 1, 2, 3, 1, 1, 1, 1, 0, 0, 0, 1, 2, 2, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #D_obs = gera_cumulativo(D_ref)

    #D_obs para dados reais da ômicron, 180 dias (diário)
    D_ref = [0, 0, 2, 1, 1, 2, 7, 3, 0, 1, 5, 3, 2, 5, 8, 7, 0, 7, 10, 6, 3, 8, 4, 6, 3, 5, 8, 19, 9, 12, 12, 11, 12, 15, 15, 12, 18, 17, 13, 15, 27, 14, 27, 23, 20, 22, 29, 28, 27, 43, 35, 48, 45, 52, 49, 57, 66, 69, 74, 54, 66, 60, 90, 93, 85, 69, 77, 73, 62, 81, 73, 74, 71, 59, 57, 65, 62, 71, 72, 70, 57, 53, 42, 47, 53, 38, 37, 38, 40, 38, 34, 26, 25, 17, 24, 37, 30, 28, 26, 20, 16, 13, 24, 22, 15, 23, 14, 12, 11, 14, 16, 12, 10, 12, 12, 5, 9, 5, 17, 8, 13, 14, 3, 6, 6, 4, 6, 8, 7, 8, 3, 2, 6, 4, 9, 9, 5, 6, 4, 3, 10, 6, 8, 9, 4, 7, 3, 2, 8, 1, 5, 6, 2, 4, 2, 7, 10, 4, 4, 2, 4, 3, 4, 4, 3, 3, 3, 2, 4, 4, 3, 2, 3, 3, 5, 3, 4, 3, 4, 2]
    D_obs = gera_cumulativo(D_ref)

    plot_solution(t, u, D_obs)
end

main_prog()