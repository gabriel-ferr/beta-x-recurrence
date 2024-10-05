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