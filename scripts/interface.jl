using Pkg
Pkg.activate(dirname(@__DIR__))

using TumorSegmentation

serve_interface(project_root = dirname(@__DIR__))
