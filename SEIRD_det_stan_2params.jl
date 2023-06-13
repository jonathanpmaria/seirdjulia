using StanSample, Distributions, OrdinaryDiffEq, MCMCChains, StatsPlots
#include("leitura_dados.jl")

SEIRD_model_beta_alpha_u0 = "
functions {
    real[] seird_det(real t, real[] u, real[] p, real[] x_r, int[] x_i) {
        
        real N = x_i[1];
        real beta = p[1];
        real alpha = p[2];
        real gamma = x_r[1];
        real eta = x_r[2];

        real S = u[1];
        real E = u[2];
        real I = u[3];
        real R = u[4];
        real D = u[5];

        real dS = -beta*S*I/N;
        real dE = beta*S*I/N - alpha*E;
        real dI = alpha*E - gamma*I;
        real dR = gamma*(1-eta)*I;
        real dD = gamma*eta*I;
        return {dS, dE, dI, dR, dD};
    }

}
data {
    int<lower=1> T;
    real t0;
    real ts[T];
    int D_obs[T];
    int N;
    real<lower=0> gamma;
    real<lower=0> eta;
}
transformed data {
    real x_r[2] = {gamma, eta};
    int x_i[1] = { N };
}
parameters {
    real<lower=0> beta;
    real<lower=0> alpha;
    real<lower=0> phi_nb;

    real<lower=0> E0;
    real<lower=0> I0;
}
transformed parameters {
    real p[2] = {beta, alpha};
}
model {
    real u0[5] = {N-E0-I0, E0, I0, 0, 0};
    real u[T, 5] = integrate_ode_rk45(seird_det, u0, t0, ts, p, x_r, x_i);

    //priors
    beta ~ uniform(0, 10);
    alpha ~ uniform(0, 10);
    phi_nb ~ uniform(0, 100);
    E0 ~ uniform(0, 10);
    I0 ~ uniform(0, 10);

    //sampling distribution (binomial)
    //for (t in 1:T)
    //    D_obs[t] ~ binomial(round(u[t,3]), gamma*eta);

    //sampling distribution (negative binomial 1)
    //for (t in 1:T)
    //    D_obs[t] ~ neg_binomial_2(u[t,3], phi_nb);

    //sampling distribution (negative binomial 2)
    D_obs[1] ~ neg_binomial_2(u[1,5], phi_nb);
    for (t in 2:T)
        D_obs[t] ~ neg_binomial_2(u[t,5] - u[t-1,5], phi_nb);

}
generated quantities {
}
"

SEIRD_model_beta_gamma_u0 = "
functions {
    real[] seird_det(real t, real[] u, real[] p, real[] x_r, int[] x_i) {
        
        real N = x_i[1];
        real beta = p[1];
        real alpha = x_r[1];
        real gamma = p[2];
        real eta = x_r[2];

        real S = u[1];
        real E = u[2];
        real I = u[3];
        real R = u[4];
        real D = u[5];

        real dS = -beta*S*I/N;
        real dE = beta*S*I/N - alpha*E;
        real dI = alpha*E - gamma*I;
        real dR = gamma*(1-eta)*I;
        real dD = gamma*eta*I;
        return {dS, dE, dI, dR, dD};
    }

}
data {
    int<lower=1> T;
    real t0;
    real ts[T];
    int D_obs[T];
    int N;
    real<lower=0> alpha;
    real<lower=0> eta;
}
transformed data {
    real x_r[2] = {alpha, eta};
    int x_i[1] = { N };
}
parameters {
    real<lower=0> beta;
    real<lower=0> gamma;
    real<lower=0> phi_nb;

    real<lower=0> E0;
    real<lower=0> I0;
}
transformed parameters {
    real p[2] = {beta, gamma};
}
model {
    real u0[5] = {N-E0-I0, E0, I0, 0, 0};
    real u[T, 5] = integrate_ode_rk45(seird_det, u0, t0, ts, p, x_r, x_i);

    //priors
    beta ~ uniform(0, 10);
    gamma ~ uniform(0, 10);
    phi_nb ~ uniform(0, 100);
    E0 ~ uniform(0, 10);
    I0 ~ uniform(0, 10);

    //sampling distribution (binomial)
    //for (t in 1:T)
    //    D_obs[t] ~ binomial(round(u[t,3]), gamma*eta);

    //sampling distribution (negative binomial 1)
    //for (t in 1:T)
    //    D_obs[t] ~ neg_binomial_2(u[t,3], phi_nb);

    //sampling distribution (negative binomial 2)
    D_obs[1] ~ neg_binomial_2(u[1,5], phi_nb);
    for (t in 2:T)
        D_obs[t] ~ neg_binomial_2(u[t,5] - u[t-1,5], phi_nb);

}
generated quantities {
}
"

SEIRD_model_beta_eta_u0 = "
functions {
    real[] seird_det(real t, real[] u, real[] p, real[] x_r, int[] x_i) {
        
        real N = x_i[1];
        real beta = p[1];
        real alpha = x_r[1];
        real gamma = x_r[2];
        real eta = p[2];

        real S = u[1];
        real E = u[2];
        real I = u[3];
        real R = u[4];
        real D = u[5];

        real dS = -beta*S*I/N;
        real dE = beta*S*I/N - alpha*E;
        real dI = alpha*E - gamma*I;
        real dR = gamma*(1-eta)*I;
        real dD = gamma*eta*I;
        return {dS, dE, dI, dR, dD};
    }

}
data {
    int<lower=1> T;
    real t0;
    real ts[T];
    int D_obs[T];
    int N;
    real<lower=0> alpha;
    real<lower=0> gamma;
}
transformed data {
    real x_r[2] = {alpha, gamma};
    int x_i[1] = { N };
}
parameters {
    real<lower=0> beta;
    real<lower=0> eta;
    real<lower=0> phi_nb;

    real<lower=0> E0;
    real<lower=0> I0;
}
transformed parameters {
    real p[2] = {beta, eta};
}
model {
    real u0[5] = {N-E0-I0, E0, I0, 0, 0};
    real u[T, 5] = integrate_ode_rk45(seird_det, u0, t0, ts, p, x_r, x_i);

    //priors
    beta ~ uniform(0, 10);
    eta ~ uniform(0, 1);
    phi_nb ~ uniform(0, 100);
    E0 ~ uniform(0, 10);
    I0 ~ uniform(0, 10);

    //sampling distribution (binomial)
    //for (t in 1:T)
    //    D_obs[t] ~ binomial(round(u[t,3]), gamma*eta);

    //sampling distribution (negative binomial 1)
    //for (t in 1:T)
    //    D_obs[t] ~ neg_binomial_2(u[t,3], phi_nb);

    //sampling distribution (negative binomial 2)
    D_obs[1] ~ neg_binomial_2(u[1,5], phi_nb);
    for (t in 2:T)
        D_obs[t] ~ neg_binomial_2(u[t,5] - u[t-1,5], phi_nb);

}
generated quantities {
}
"

function ODE_SEIRD_det(du,u,p,t)
    N, beta, alpha, gamma, eta = p
    du[1] = -(beta/N)*u[1]*u[3]
    du[2] = (beta/N)*u[1]*u[3] - alpha*u[2]
    du[3] = alpha*u[2] - gamma*u[3]
    du[4] = gamma*(1-eta)*u[3]
    du[5] = gamma*eta*u[3]
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

    #Dados reais SP 2021 (variante gama)
    #N = 12_325_232      #população absoluta (SP capital)
    #I0 = 1.0            #número inicial de infectados
    #beta = 1.0         #susceptibilidade
    #alpha = 1/2         #inverso do período médio de latência
    #gamma = 1/3         #inverso do tempo médio de recuperação
    #eta = 0.012         #taxa de mortalidade (IFR)
    #q = 0.9325          #índice entrópico

    #Dados reais SP 2021/2022 (ômicron)
    #N = 12_325_232      #população absoluta (SP capital - IBGE)
    #I0 = 2.0            #número inicial de infectados
    #beta = 1.0          #susceptibilidade 
    #alpha = 1/3         #inverso do período médio de latência 
    #gamma = 1/5         #inverso do tempo médio de infecção
    #eta = 0.0001        #taxa de mortalidade (IFR)
    #q = 0.8392          #índice entrópico

    dt = 1.0
    #t_end = 100.0
    t_end = 200.0
    tspan = (0.0, t_end)
    t = Vector(dt:dt:t_end)
    T = length(t)

    u0 = [N-E0-I0; E0; I0; 0; 0]

    p = [N; beta; alpha; gamma; eta]
    prob = ODEProblem(ODE_SEIRD_det, u0, tspan, p)
    sol = solve(prob, Tsit5(), reltol=1e-16, abstol=1e-16, saveat=1)

    #Medida do estado D(t) - limpo
    #D_obs = Int64.(round.(sol[5,2:end].-sol[5,1:end-1]))

    #Medida do estado D(t) - distribuição uniforme
    #D_obs = Int64.(round.(rand.(Uniform.(0, 2*sol[5,2:end]))))
    #D_obs = Int64.(round.(rand.(Uniform.(0, 2*(sol[5,2:end].-sol[5,1:end-1])))))

    #Medida do estado D(t) - distribuição de Poisson
    #D_obs = rand.(Poisson.(sol[5,2:end]))
    #D_obs = rand.(Poisson.(sol[5,2:end].-sol[5,1:end-1]))

    #Medida do estado D(t) - distribuição binomial 
    D_obs = rand.(Binomial.(Int64.(round.(sol[3,2:end])), gamma*eta))

    #Medida do estado D(t) - distribuição binomial negativa
    #D_obs = rand.(NegativeBinomial.(Int64.(round.(sol[3,2:end])), 1-gamma*eta))

    #Medida do estado D(t) - dados SIVEP-Gripe
    #D_obs = le_SIVEP_Gripe("INFLUD21-30-05-2022.csv", 1)
    #D_obs = le_SIVEP_Gripe_2("INFLUD21-30-05-2022.csv", "INFLUD22-30-05-2022.csv")
    #D_obs[1:30] = zeros(Int, 30)

    #Aprender beta e alpha
    seird_data = Dict("T" => T, "u0" => u0, "t0" => 0.0, "ts" => t, "D_obs" => D_obs, "N" => N, "gamma" => gamma, "eta" => eta)

    #Aprender beta e gamma
    #seird_data = Dict("T" => T, "u0" => u0, "t0" => 0.0, "ts" => t, "D_obs" => D_obs, "N" => N, "alpha" => alpha, "eta" => eta)

    #Aprender beta e eta
    #seird_data = Dict("T" => T, "u0" => u0, "t0" => 0.0, "ts" => t, "D_obs" => D_obs, "N" => N, "alpha" => alpha, "gamma" => gamma)

    # Keep tmpdir across multiple runs to prevent re-compilation
    tmpdir = joinpath(@__DIR__, "tmp")

    #Escolher de acordo com a combinação de parâmetros
    sm = SampleModel("sm_seird_beta_alpha_u0", SEIRD_model_beta_alpha_u0, tmpdir)
    #sm = SampleModel("sm_seird_beta_gamma_u0", SEIRD_model_beta_gamma_u0, tmpdir)
    #sm = SampleModel("sm_seird_beta_eta_u0", SEIRD_model_beta_eta_u0, tmpdir)
    
    rc = stan_sample(sm; data = seird_data, num_samples = 10000)

    if success(rc)
        chns = read_samples(sm, :mcmcchains)
        chns |> display

        StatsPlots.plot(chns)
        StatsPlots.savefig("MCMC_2params_u0.png")
    end
end

main_prog()