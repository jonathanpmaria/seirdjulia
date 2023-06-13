using StanSample, Distributions, OrdinaryDiffEq, MCMCChains, DataFrames, CSV, Dates, StatsPlots
include("modelos_eqdiff.jl")
include("leitura_dados.jl")

SEIRD_frac_model_beta_u0 = "
functions {
    real[] seird_det(real t, real[] u, real[] p, real[] x_r, int[] x_i) {
        
        real N = x_i[1];
        real beta = p[1];
        real alpha = x_r[1];
        real gamma = x_r[2];
        real eta = x_r[3];
        real q = x_r[4];

        real S = u[1];
        real E = u[2];
        real I = u[3];
        real R = u[4];
        real D = u[5];

        real dS = -beta*S*(I^q)/N;
        real dE = beta*S*(I^q)/N - alpha*E;
        real dI = alpha*E - gamma*(I^q);
        real dR = gamma*(1-eta)*(I^q);
        real dD = gamma*eta*(I^q);
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
    real<lower=0> eta;
    real<lower=0> q;
}
transformed data {
    real x_r[4] = {alpha, gamma, eta, q};
    int x_i[1] = { N };
}
parameters {
    real<lower=0> beta;
    real<lower=0> phi_nb;

    real<lower=0> E0;
    real<lower=0> I0;
}
transformed parameters {
    real p[1] = {beta};
}
model {
    real u0[5] = {N-E0-I0, E0, I0, 0, 0};
    real u[T, 5] = integrate_ode_rk45(seird_det, u0, t0, ts, p, x_r, x_i);

    //priors
    beta ~ uniform(0, 10);
    phi_nb ~ uniform(0, 100);
    E0 ~ uniform(0, 10);
    I0 ~ uniform(0, 10);

    //sampling distribution (binomial)
    //for (t in 1:T)
    //    D_obs[t] ~ binomial(round(u[t,3]), gamma*eta);

    //sampling distribution (negative binomial 1)
    //for (t in 1:T)
    //    D_obs[t] ~ neg_binomial_2(u[t,5], phi_nb);

    //sampling distribution (negative binomial 2)
    D_obs[1] ~ neg_binomial_2(u[1,5], phi_nb);
    for (t in 2:T)
        D_obs[t] ~ neg_binomial_2(u[t,5] - u[t-1,5], phi_nb);

}
"

function main_prog()
    #Dados de teste
    #N = 1_000_000   #população
    #E0 = 5.0        #número inicial de expostos
    #I0 = 2.0        #número inicial de infectados

    #Parâmetros de referência (SEIRD fractal, ou SEIRD simples se q = 1)
    #beta = 0.6      #susceptibilidade
    #alpha = 0.2     #inverso do período médio de latência
    #gamma = 0.3     #inverso do tempo médio de recuperação
    #eta = 0.01      #taxa de mortalidade (IFR)  
    #q = 1           #índice entrópico

    #Parâmetros de referência (SEIRD com assintomáticos)
    #beta_s = 0.6    #susceptibilidade (sintomáticos)
    #beta_a = 0.8    #susceptibilidade (assintomáticos)
    #alpha = 0.2     #inverso do período médio de latência
    #phi = 0.1       #probabilidade da infecção ser sintomática
    #gamma_s = 0.2   #inverso do tempo médio de recuperação (sintomáticos)
    #gamma_a = 0.3   #inverso do tempo médio de recuperação (assintomáticos)
    #delta = 0.05    #taxa de mortalidade entre casos sintomáticos

    #Dados reais SP 2021/2022 (ômicron)
    N = 12_325_232      #população absoluta (SP capital)
    #beta = 0.9830       #susceptibilidade
    alpha = 1/3.42         #inverso do período médio de latência
    gamma = 1/14         #inverso do tempo médio de recuperação
    eta = 0.0002        #taxa de mortalidade (IFR)
    q = 1

    dt = 1.0
    t_end = 180.0
    t = Vector(dt:dt:t_end)
    T = length(t)

    #Solução (SEIRD simples)
    #sol = SEIRD_det(t_end, N, E0, I0, beta, alpha, gamma, eta)
    #D_obs = Int64.(round.(sol[5,2:end].-sol[5,1:end-1]))
    #D_obs = rand.(Binomial.(Int64.(round.(sol[3,2:end])), gamma*eta))

    #Solução (SEIRD fractal)
    #sol = SEIRD_det_frac(t_end, N, E0, I0, beta, alpha, gamma, eta, q)
    #D_obs = Int64.(round.(sol[5,2:end].-sol[5,1:end-1]))
    #D_obs = rand.(Binomial.(Int64.(round.(sol[3,2:end].^q)), gamma*eta))

    #Solução (SEIRD com assintomáticos)
    #sol = SEIRD_det_asymp(t_end, N, E0, I0, beta_s, beta_a, alpha, phi, gamma_s, gamma_a, delta)
    #D_obs = Int64.(round.(sol[6,2:end].-sol[6,1:end-1]))
    #D_obs = rand.(Binomial.(Int64.(round.(sol[3,2:end])), gamma_s*delta))

    #Dados reais SIVEP-Gripe
    D_obs = le_SIVEP_Gripe_2("INFLUD21-30-05-2022.csv", "INFLUD22-30-05-2022.csv", 2)

    #Parâmetros passados ao Stan (SEIRD simples ou fractal)
    alpha_stan = alpha
    gamma_stan = gamma
    eta_stan = eta
    q_stan = q

    #Parâmetros passados ao Stan (SEIRD com assintomáticos)
    #alpha_stan = alpha
    #gamma_stan = phi*gamma_s + (1-phi)*gamma_a
    #eta_stan = phi*delta
    #q_stan = 1

    seird_data = Dict("T" => T, "t0" => 0.0, "ts" => t, "D_obs" => D_obs, "N" => N, "alpha" => alpha_stan, "gamma" => gamma_stan, "eta" => eta_stan, "q" => q_stan)

    # Keep tmpdir across multiple runs to prevent re-compilation
    tmpdir = joinpath(@__DIR__, "tmp")

    sm = SampleModel("sm_seird_frac_beta_u0", SEIRD_frac_model_beta_u0, tmpdir)
    rc = stan_sample(sm; data = seird_data, num_samples = 10000)

    if success(rc)
        chns = read_samples(sm, :mcmcchains)
        chns |> display

        StatsPlots.plot(chns)
        StatsPlots.savefig("MCMC_beta_frac_u0.png")
    end
end

main_prog()