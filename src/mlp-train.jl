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