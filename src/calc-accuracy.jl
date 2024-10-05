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