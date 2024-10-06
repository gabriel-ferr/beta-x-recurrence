include("rp.jl")

using .RP
using JLD2
using Flux
using Statistics
using ProgressMeter
using LinearAlgebra
using BenchmarkTools

const β_values = [2.99, 3.59, 3.99, 4.59, 4.99, 5.59, 5.99, 6.59]
const timeseries_size = 500
const mlp_samples = 1
const motif_size = 2
const epochs = 40

const ε = range(0, 0.9995, 100)
const pvec = power_vector(motif_size)


function main()
    if (!isfile("status.dat"))
        save_object("status.dat", [1])
        xo_to_entropy = range(0.00001, 0.99999, 5)
        xo_to_train_mlp = rand(Float64, 480)
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
        save_object("out/entropy.dat", zeros(Float64, length(ε), length(ε), length(β_values)))
        save_object("out/accuracy.dat", zeros(Float64, length(ε), length(ε), epochs, mlp_samples))
        save_object("out/etr_serie.dat", β(xo_to_entropy))
        save_object("out/etr_train.dat", β(xo_to_train_mlp))
        save_object("out/etr_test.dat", β(xo_to_test_mlp))

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

    global serie_to_entropy = load_object("out/etr_serie.dat")
    global serie_to_test_mlp = load_object("out/etr_test.dat")
    global serie_to_train_mlp = load_object("out/etr_train.dat")
    global entropy_data = load_object("out/entropy.dat")
    global accuracy_data = load_object("out/accuracy.dat")

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

    _train_labels = ones(Float64, size(serie_to_train_mlp, 3) * length(β_values))
    _test_labels = ones(Float64, size(serie_to_test_mlp, 3) * length(β_values))

    for beta in eachindex(β_values)
        _train_labels[1+(beta-1)*size(serie_to_train_mlp, 3):beta*size(serie_to_train_mlp, 3)] *= β_values[beta]
        _test_labels[1+(beta-1)*size(serie_to_test_mlp, 3):beta*size(serie_to_test_mlp, 3)] *= β_values[beta]
    end

    global train_labels = Flux.onehotbatch(_train_labels, β_values)
    global test_labels = Flux.onehotbatch(_test_labels, β_values)

    C = range(2, length(ε))
    R = range(1, length(ε) - 1)
    M = []
    for c in C
        r_vals = findall(r -> r < c, R)
        for r in r_vals
            push!(M, (r, c))
        end
    end

    status = load_object("status.dat")
    for i = status:length(M)
        m = M[i]
        mlp_probs_to_train = zeros(Float64, 2^(motif_size * motif_size), size(serie_to_train_mlp, 3), length(β_values))
        mlp_probs_to_test = zeros(Float64, 2^(motif_size * motif_size), size(serie_to_test_mlp, 3), length(β_values))
        for beta in eachindex(β_values)
            #       Primeiro, calcula as entropias...
            tasks = []
            for sp = 1:length(xo_to_entropy)
                push!(tasks, Threads.@spawn calc_entropy(serie_to_entropy[:, :, sp, beta], m))
            end

            #       Na mesma linha de pensamento, vamos calcular as probabilidades que serão usadas na rede neural...
            train_tasks = []
            for sp in train_partition
                push!(train_tasks, Threads.@spawn calc_probs(serie_to_train_mlp[:, :, sp, beta], m))
            end

            test_tasks = []
            for sp in test_partition
                push!(test_tasks, Threads.@spawn calc_probs(serie_to_test_mlp[:, :, sp, beta], m))
            end

            async_result = fetch.(tasks)

            test_async_result = fetch.(test_tasks)
            for sp in eachindex(test_async_result)
                mlp_probs_to_test[:, test_partition[sp], beta] .= test_async_result[sp]
            end

            train_async_result = fetch.(train_tasks)
            for sp in eachindex(train_async_result)
                mlp_probs_to_train[:, train_partition[sp], beta] .= train_async_result[sp]
            end

            global entropy_data[m[1], m[2], beta] = mean(async_result)
        end
        test_data = reshape(mlp_probs_to_test, (2^(motif_size * motif_size), size(serie_to_test_mlp, 3) * length(β_values)))
        train_data = reshape(mlp_probs_to_train, (2^(motif_size * motif_size), size(serie_to_train_mlp, 3) * length(β_values)))

        loader = Flux.DataLoader((train_data, train_labels), batchsize=50, shuffle=true)

        #       Separa as redes em tasks - 1 POR THREAD NO MÁXIMO PRA NÃO DAR RUIM!!
        mlp_tasks = []
        for th = 1:mlp_samples
            push!(mlp_tasks, Threads.@spawn mlp_train(loader, test_data, th, m))
        end

        wait.(mlp_tasks)

        save_object("out/entropy.dat", entropy_data)
        save_object("out/accuracy.dat", accuracy_data)
        save_object("status.dat", [i + 1])
    end
end


function calc_accuracy(predict, trusty)
    conf = zeros(Int, length(β_values), length(β_values))
    sz = size(predict, 2)

    for i = 1:sz
        mx_prd = findmax(predict[:, i])
        mx_trt = findmax(trusty[:, i])
        conf[mx_prd[2], mx_trt[2]] += 1
    end

    return tr(conf) / sum(conf)
end

function mlp_train(loader, test_data, th_index, m)
    model = load_object(string("obj/net-", th_index, ".mlp"))
    opt = Flux.setup(Flux.Adam(0.0001), model)

    for epc = 1:epochs
        for (x, y) in loader
            _, grads = Flux.withgradient(model) do m
                y_hat = m(x)
                Flux.logitcrossentropy(y_hat, y)
            end
            Flux.update!(opt, model, grads[1])
        end
        accuracy_data[m[1], m[2], epc, th_index] = calc_accuracy(model(test_data), test_labels)
    end
end

function calc_probs(data, m)
    sz = size(data, 3)
    result = zeros(Float64, 2^(motif_size * motif_size), sz)
    for i = 1:sz
        rp = recurrence_matrix(data[:, :, i], (ε[m[1]], ε[m[2]]); recurrence=RP.corridor_recurrence)
        probs, _ = motifs_probabilities(rp, motif_size; power_vector=pvec)
        result[collect(keys(probs)), i] .= collect(values(probs))
    end
    return result
end

function calc_entropy(data, m)
    rp = recurrence_matrix(data, (ε[m[1]], ε[m[2]]); recurrence=RP.corridor_recurrence)
    probs, _ = motifs_probabilities(rp, motif_size; power_vector=pvec)
    return entropy(probs)
end

function β(x; transient=round(Int, (10 * timeseries_size)))
    serie = zeros(Float64, (1, timeseries_size, length(x), length(β_values)))

    for β_index in eachindex(β_values)
        for index in eachindex(x)
            before = x[index]

            for time = 1:(timeseries_size+transient)
                after = before * β_values[β_index]

                while (after > 1.0)
                    after = after - 1.0
                end

                before = after

                if (time > transient)
                    serie[1, time-transient, index, β_index] = before
                end
            end

        end
    end

    return serie
end

@time main()