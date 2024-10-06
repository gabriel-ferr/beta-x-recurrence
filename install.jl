import Pkg

Pkg.add("JLD2")
Pkg.add("Flux")
Pkg.add("Statistics")
Pkg.add("ProgressMeter")
Pkg.add("LinearAlgebra")
Pkg.add("BenchmarkTools")

if (!isdir("out"))
    mkdir("out")
end

if (!isdir("obj"))
    mkdir("obj")
end
