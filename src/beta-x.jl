#
#           Bernoulli-Shift Generelized function
#       This will to generate our data =3
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