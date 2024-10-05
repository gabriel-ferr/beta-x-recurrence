function calc_entropy(data, m)
    rp = recurrence_matrix(data, (ε[m[1]], ε[m[2]]); recurrence=RP.corridor_recurrence)
    probs, _ = motifs_probabilities(rp, motif_size; power_vector=pvec)
    return entropy(probs)
end