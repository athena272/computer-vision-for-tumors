using Pkg
Pkg.activate(dirname(@__DIR__))

using TumorSegmentation

function parse_args()
    image_path = nothing
    checkpoint_path = joinpath("outputs", "checkpoints", "best_model.bson")
    output_path = nothing

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--image" && i < length(ARGS)
            image_path = ARGS[i + 1]
            i += 2
        elseif arg == "--checkpoint" && i < length(ARGS)
            checkpoint_path = ARGS[i + 1]
            i += 2
        elseif arg == "--output" && i < length(ARGS)
            output_path = ARGS[i + 1]
            i += 2
        else
            i += 1
        end
    end

    image_path === nothing && error("Uso: julia --project scripts/predict.jl --image <caminho_da_imagem> [--checkpoint <modelo.bson>] [--output <saida.png>]")
    return image_path, checkpoint_path, output_path
end

function main()
    image_path, checkpoint_path, output_path = parse_args()
    model, cfg = load_model_for_inference(checkpoint_path)

    println("Gerando predição para: $image_path")
    saved_path, _, _ = predict_and_save(model, image_path; cfg = cfg, output_path = output_path)
    println("Máscara salva em: $saved_path")

    gt_path = replace(image_path, r"\.png$" => "_mask.png")
    if isfile(gt_path)
        comparison_path = replace(saved_path, "_pred.png" => "_comparison.png")
        save_comparison(
            image_path,
            predict_image(model, image_path; cfg = cfg),
            comparison_path;
            ground_truth_path = gt_path,
            target_size = cfg.image_size,
        )
        println("Comparação visual salva em: $comparison_path")
    end
end

main()
