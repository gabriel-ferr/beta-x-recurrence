function progress_helper(m)
    last_printed = 0.0
    while (running_progress)
        progress_sum = sum(progress) / (m * 1.0)
        if ((progress_sum - last_printed) * 100.0 >= 1.0)
            last_printed = progress_sum
            println(last_printed)
        end
    end
end