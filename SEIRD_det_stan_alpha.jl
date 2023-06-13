using StanSample, Distributions, OrdinaryDiffEq, MCMCChains
include("leitura_dados.jl")

SEIRD_model_alpha = "
functions {
    real[] seird_det(real t, real[] u, real[] p, real[] x_r, int[] x_i) {
        
        real N = x_i[1];
        real beta = x_r[1];
        real alpha = p[1];
        real gamma = x_r[2];
        real eta = x_r[3];

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
    real u0[5];
    real t0;
    real ts[T];
    int y[T];
    int N;
    real<lower=0> beta;
    real<lower=0> gamma;
    real<lower=0> eta;
}
transformed data {
    real x_r[3] = {beta, gamma, eta};
    int x_i[1] = { N };
}
parameters {
    real<lower=0> alpha;
}
transformed parameters {
    real p[1] = {alpha};
}
model {
    real u[T, 5] = integrate_ode_rk45(seird_det, u0, t0, ts, p, x_r, x_i);

    //priors
    alpha ~ uniform(1, 3);

    //sampling distribution (uniform)
    //for (t in 1:T)
    //    y[t] ~ uniform(0, u[t,5]);

    //sampling distribution (Poisson 1)
    for (t in 1:T)
        y[t] ~ poisson(u[t,5]);

    //sampling distribution (Poisson 2 - diff.)
    //y[1] ~ poisson(u[1,5]);
    //for (t in 2:T)
    //    y[t] ~ poisson(u[t,5] - u[t-1,5]);

    //sampling distribution (binomial with p_obs)
    //for (t in 1:T)
    //    y[t] ~ binomial(u[t,5], 0.8);

}
generated quantities {
    //real R0 = beta/gamma;
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
    N = 12_325_232      #população absoluta (SP capital)
    I0 = 1.0       
    u0 = [N-I0; 0; I0; 0; 0]

    dt = 1.0
    tspan = (0.0, 150.0)
    t = Vector(dt:dt:150.0)
    T = length(t)

    beta = 2.9/9.2      #susceptibilidade (beta = R0*gamma) para R0 = 2.9 em SP
    alpha = 1/0.5       #inverso do período médio de latência - NetoNature2021 => 2.0000
    gamma = 1/9.2       #inverso do tempo médio de recuperação - NetoNature2021
    eta = 0.0053        #taxa de mortalidade (IFR) - NetoNature2021

    p = [N; beta; alpha; gamma; eta]

    prob = ODEProblem(ODE_SEIRD_det, u0, tspan, p)
    sol = solve(prob, Tsit5(), rel_tol=1e-16, abs_tol=1e-16, saveat=1)

    #Medida do estado D(t) - distribuição uniforme
    #y = Int64.(round.(rand.(Uniform.(0, sol[5,2:end]))))

    #Medida do estado D(t) - distribuição de Poisson
    y = rand.(Poisson.(sol[5,2:end]))
    #y = rand.(Poisson.(sol[5,2:end].-sol[5,1:end-1]))

    #Medida do estado D(t) - distribuição binomial com p_obs
    #y = rand.(Binomial.(Int64.(round.(sol[5,2:end])), 0.8))

    seird_data = Dict("T" => T, "u0" => u0, "t0" => 0.0, "ts" => t, "y" => y, "N" => N, "beta" => beta, "gamma" => gamma, "eta" => eta)
    #init = Dict("alpha" => alpha)

    # Keep tmpdir across multiple runs to prevent re-compilation
    tmpdir = joinpath(@__DIR__, "tmp")

    sm = SampleModel("sm_seird_alpha", SEIRD_model_alpha, tmpdir)
    rc = stan_sample(sm; data = seird_data, num_samples = 10000)

    if success(rc)
        cnhs = read_samples(sm, :mcmcchains)
        cnhs |> display
    end
end

main_prog()