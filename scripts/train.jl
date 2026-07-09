using Pkg
Pkg.activate(dirname(@__DIR__))

using TumorSegmentation

function main()
    cfg = load_config()
    checkpoint = run_training!(cfg)
    println("Treinamento concluído. Modelo salvo em: $checkpoint")
end

main()
