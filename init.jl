#
#           By Gabriel Ferreira
#               Orientation: Thiago de Lima Prado
#                            Sérgio Roberto Lopes
#
include("lib/rp.jl")
using .RP

using JLD2
using Flux
using CairoMakie
using Statistics
using LinearAlgebra

const β_values = [2.59, 2.99, 3.59, 3.99, 4.59, 4.99, 5.59, 5.99, 6.59, 6.99]
const timeseries_size = 500
const motif_size = 2
const epochs = 30

const ε = range(0, 0.9995, 100)
const pvec = power_vector(motif_size)

include("src/beta-x.jl")
include("src/main.jl")

function init()
    C = range(2, length(ε))
    R = range(1, length(ε) - 1)
    M = []
    for c in C
        r_vals = findall(r -> r < c, R)
        for r in r_vals
            push!(M, (r, c))
        end
    end
    part_int = length(M) ÷ Threads.nthreads()
    part_rest = length(M) % Threads.nthreads()
    M_intervals = [range(1, part_int + (part_rest > 0 ? 1 : 0))]
    part_rest -= 1
    for th = 2:Threads.nthreads()
        interval = range(1 + M_intervals[th-1][end], M_intervals[th-1][end] + part_int + (part_rest > 0 ? 1 : 0))
        part_rest -= 1
        push!(M_intervals, interval)
    end
end

init()