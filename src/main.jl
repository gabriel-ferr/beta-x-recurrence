function async_main(M, th_id)
    for m in M
        mlp_probs_to_train = zeros(Float64, 2^(motif_size * motif_size), size(serie_to_train_mlp, 3), length(β_values))
        mlp_probs_to_test = zeros(Float64, 2^(motif_size * motif_size), size(serie_to_test_mlp, 3), length(β_values))

        for beta in eachindex(β_values)
            #       Primeiro, calcula as entropias...
            tasks = []
            for sp = 1:size(serie_to_entropy, 3)
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

        #       Prepara para treinar as redes neurais...
        test_data = reshape(mlp_probs_to_test, (2^(motif_size * motif_size), size(serie_to_test_mlp, 3) * length(β_values)))
        train_data = reshape(mlp_probs_to_train, (2^(motif_size * motif_size), size(serie_to_train_mlp, 3) * length(β_values)))

        loader = Flux.DataLoader((train_data, train_labels), batchsize=50, shuffle=true)

        #       Separa as redes em tasks - 1 POR THREAD NO MÁXIMO PRA NÃO DAR RUIM!!
        mlp_tasks = []
        for th = 1:mlp_samples
            push!(mlp_tasks, Threads.@spawn mlp_train(loader, test_data, th, m))
        end

        wait.(mlp_tasks)
        progress[th_id] += 1
    end
end