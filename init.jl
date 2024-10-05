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
const motif_size = 3
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

    #       - Gera os dados...
    xo_to_entropy = range(0.00001, 0.99999, 5)
    xo_to_train_mlp = rand(Float64, 720)
    xo_to_test_mlp = rand(Float64, floor(Int, length(xo_to_train_mlp) / 3))
    for i in eachindex(xo_to_test_mlp)
        while (xo_to_test_mlp[i] in xo_to_train_mlp)
            new_value = rand(Float64, 1)
            while (new_value in xo_to_test_mlp)
                new_value = rand(Float64, 1)
            end
            xo_to_test_mlp[i] = new_value
        end
    end

    mlp_samples = Threads.nthreads()

    global serie_to_entropy = β(xo_to_entropy)
    global serie_to_test_mlp = β(xo_to_test_mlp)
    global serie_to_train_mlp = β(xo_to_train_mlp)
    global entropy_data = zeros(Float64, length(ε), length(ε), length(β_values))
    global accuracy_data = zeros(Float64, length(ε), length(ε), epochs, mlp_samples)

    _train_partitions = []
    _test_partitions = []

    _test_sz = size(serie_to_test_mlp, 3) ÷ Threads.nthreads()
    _train_sz = size(serie_to_train_mlp, 3) ÷ Threads.nthreads()

    for i = 1:Threads.nthreads()
        _test_interval = 1+(i-1)*_test_sz:i*_test_sz
        _train_interval = 1+(i-1)*_train_sz:i*_train_sz
        push!(_test_partitions, _test_interval)
        push!(_train_partitions, _train_interval)
    end

    global train_partition = _train_partitions
    global test_partition = _test_partitions

    #       Monta os labels para usar no cálculo da accuracy...
    _train_labels = ones(Float64, size(serie_to_train_mlp, 3) * length(β_values))
    _test_labels = ones(Float64, size(serie_to_test_mlp, 3) * length(β_values))

    for beta in eachindex(β_values)
        _train_labels[1+(beta-1)*size(serie_to_train_mlp, 3):beta*size(serie_to_train_mlp, 3)] *= β_values[beta]
        _test_labels[1+(beta-1)*size(serie_to_test_mlp, 3):beta*size(serie_to_test_mlp, 3)] *= β_values[beta]
    end

    global train_labels = Flux.onehotbatch(_train_labels, β_values)
    global test_labels = Flux.onehotbatch(_test_labels, β_values)

    for i = 1:mlp_samples
        model = Chain(
            Dense(2^(motif_size * motif_size) => 128, identity),
            Dense(128 => 64, selu),
            Dense(64 => 32, selu),
            Dense(32 => length(β_values)),
            softmax
        )

        save_object(string("obj/net-", i, ".mlp"), f64(model))
    end
end

init()