# Modelos epidemiológicos implementados em Julia
using OrdinaryDiffEq


# Modelo SIR determinístico de tempo contínuo

# dS/dt = -beta*S*I/N
# dI/dt = beta*S*I/N - gamma*I
# dR/dt = gamma*I

function ODE_SIR_det(du,u,p,t)
    N, beta, gamma = p
    du[1] = -(beta/N)*u[1]*u[2]
    du[2] = (beta/N)*u[1]*u[2] - gamma*u[2]
    du[3] = gamma*u[2]
end

function SIR_det(n_days, N, I0, beta, gamma)
    p = [N; beta; gamma]
    u0 = [N-I0; I0; 0.0]
    tspan = (0.0, n_days)

    prob = ODEProblem(ODE_SIR_det, u0, tspan, p)
    sol = solve(prob, Tsit5(), reltol=1e-16, abstol=1e-16, saveat=1)

    return sol
end


# Modelo SEIR determinístico de tempo contínuo

# dS/dt = -beta*S*I/N
# dE/dt = beta*S*I/N - alpha*E
# dI/dt = alpha*E - gamma*I
# dR/dt = gamma*I

function ODE_SEIR_det(du,u,p,t)
    N, beta, alpha, gamma = p
    du[1] = -(beta/N)*u[1]*u[3]
    du[2] = (beta/N)*u[1]*u[3] - alpha*u[2]
    du[3] = alpha*u[2] - gamma*u[3]
    du[4] = gamma*u[3]
end

function SEIR_det(n_days, N, E0, I0, beta, alpha, gamma)
    p = [N; beta; alpha; gamma]
    u0 = [N-E0-I0; E0; I0; 0.0]
    tspan = (0.0, n_days)

    prob = ODEProblem(ODE_SEIR_det, u0, tspan, p)
    sol = solve(prob, Tsit5(), reltol=1e-16, abstol=1e-16, saveat=1)

    return sol
end


# Modelo SEIRD determinístico de tempo contínuo

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

function SEIRD_det(n_days, N, E0, I0, beta, alpha, gamma, eta)
    p = [N; beta; alpha; gamma; eta]
    u0 = [N-E0-I0; E0; I0; 0.0; 0.0]
    tspan = (0.0, n_days)

    prob = ODEProblem(ODE_SEIRD_det, u0, tspan, p)
    sol = solve(prob, Tsit5(), reltol=1e-16, abstol=1e-16, saveat=1)

    return sol
end


# Modelo SEIRD determinístico de tempo contínuo, com fractal

# dS/dt = -beta*S*I^q/N
# dE/dt = beta*S*I^q/N - alpha*E
# dI/dt = alpha*E - gamma*I^q
# dR/dt = gamma*(1-eta)*I^q
# dD/dt = gamma*eta*I^q

function ODE_SEIRD_det_frac(du,u,p,t)
    N, beta, alpha, gamma, eta, q = p
    du[1] = -(beta/N)*u[1]*u[3]^q
    du[2] = (beta/N)*u[1]*u[3]^q - alpha*u[2]
    du[3] = alpha*u[2] - gamma*u[3]^q
    du[4] = gamma*(1-eta)*u[3]^q
    du[5] = gamma*eta*u[3]^q
end

function SEIRD_det_frac(n_days, N, E0, I0, beta, alpha, gamma, eta, q)
    p = [N; beta; alpha; gamma; eta; q]
    u0 = [N-E0-I0; E0; I0; 0.0; 0.0]
    tspan = (0.0, n_days)

    prob = ODEProblem(ODE_SEIRD_det_frac, u0, tspan, p)
    sol = solve(prob, Tsit5(), reltol=1e-16, abstol=1e-16, saveat=1)

    return sol
end


# Modelo SEIRD determinístico de tempo contínuo com assintomáticos

# dS/dt = - beta_s*S*I_s/N - beta_a*S*I_a/N
# dE/dt = beta_s*S*I_s/N + beta_a*S*I_a/N - alpha*E
# dI_s/dt = phi*alpha*E - gamma_s*I_s
# dI_a/dt = (1-phi)*alpha*E - gamma_a*I_a
# dR/dt = gamma_s*(1-delta)*I_s + gamma_a*I_a
# dD/dt = gamma_s*delta*I_s

function ODE_SEIRD_det_asymp(du,u,p,t)
    N, beta_s, beta_a, alpha, phi, gamma_s, gamma_a, delta = p
    du[1] = - (beta_s/N)*u[1]*u[3] - (beta_a/N)*u[1]*u[4]
    du[2] = (beta_s/N)*u[1]*u[3] + (beta_a/N)*u[1]*u[4] - alpha*u[2]
    du[3] = phi*alpha*u[2] - gamma_s*u[3]
    du[4] = (1-phi)*alpha*u[2] - gamma_a*u[4]
    du[5] = gamma_s*(1-delta)*u[3] + gamma_a*u[4]
    du[6] = gamma_s*delta*u[3]
end

function SEIRD_det_asymp(n_days, N, E0, I0, beta_s, beta_a, alpha, phi, gamma_s, gamma_a, delta)
    p = [N; beta_s; beta_a; alpha; phi; gamma_s; gamma_a; delta]
    u0 = [N-E0-I0; E0; phi*I0; (1-phi)*I0; 0.0; 0.0]
    tspan = (0.0, n_days)

    prob = ODEProblem(ODE_SEIRD_det_asymp, u0, tspan, p)
    sol = solve(prob, Tsit5(), reltol=1e-16, abstol=1e-16, saveat=1)

    return sol
end