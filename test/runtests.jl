using Test
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using TumorSegmentation

include("test_preprocessing.jl")
include("test_dataset.jl")
include("test_unet.jl")
include("test_metrics.jl")
include("test_predict.jl")
