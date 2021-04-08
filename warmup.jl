using Plots
plotlyjs()
using PlutoUI
using FFTW
using MRIReco
using LinearAlgebra
using IterativeSolvers

p = scatter(rand(10), rand(10));
display(p)
println("Warm-up done.")