using Pkg
Pkg.activate(dirname(@__DIR__))

using TumorSegmentation

function main()
    cfg = load_config()
    println("Avaliando modelo no conjunto de teste...")
    metrics = run_evaluation!(cfg)
    println()
    println("=== Resultados no conjunto de teste ===")
    println("Dice:      $(round(metrics.dice, digits=4))")
    println("IoU:       $(round(metrics.iou, digits=4))")
    println("Acurácia:  $(round(metrics.accuracy, digits=4))")
end

main()
