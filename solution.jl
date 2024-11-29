# julia for solution.py
using DifferentialEquations
using Unitful
using UnitfulAstro
using UnitfulAstro: pc
using Unitful: Gyr, s, km

distance_pc = 1u"pc"
distance_km = uconvert(km, distance_pc)
time_gyr = 1u"Gyr"
time_s = uconvert(s, time_gyr)
transfer = 1e6 * (distance_km / time_s).val

function function!(dz, z, p, t)
    kC1, O10, H0 = p
    dz[1] = z[2]
    numerator = (
        H0^4 * kC1 * O10^2 * (z[1]^4 + 1) +
        3 * H0^4 * O10^2 * z[1]^2 * (2 * kC1 - 3 * z[2]) +
        H0^4 * O10^2 * z[1]^3 * (4 * kC1 - 3 * z[2]) -
        3 * H0^4 * O10^2 * z[2] +
        5 * H0^2 * O10 * z[2]^3 -
        kC1 * z[2]^4 +
        H0^2 * O10 * z[1] * (4 * H0^2 * kC1 * O10 - 9 * H0^2 * O10 * z[2] + 5 * z[2]^3)
    )
    denominator = 2 * H0^2 * O10 * (1 + z[1])^2 * z[2]
    dz[2] = numerator / denominator
    return nothing
end

function solution(log_kC1, O20, H0)
    kC1 = 10^log_kC1 * transfer
    O10 = 1 - O20
    t0 = 1 / H0
    tspan = (t0, 0.0)
    tn = range(t0, stop=0.0, length=100000)
    zt0 = [0.0, -H0]

    prob = ODEProblem(function!, zt0, tspan, (kC1, O10, H0))
    sol = solve(prob, Tsit5(), saveat=tn)
    return sol
end

function result(log_kC1, O20, H0)
    sol = solution(log_kC1, O20, H0)
    z = view(sol.u, 1:length(sol.t))
    sol_z_matrix = reduce(hcat, z)'
    z0 = sol_z_matrix[:, 1]
    z1 = sol_z_matrix[:, 2]
    return z0, z1
end




