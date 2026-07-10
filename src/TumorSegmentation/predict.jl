using Images: Gray, colorview

"""
Escolhe dispositivo para inferência de 1 imagem.
Prefere CPU se a VRAM estiver apertada (comum logo após o treino na GPU).
"""
function select_inference_device(cfg::Config)
    device = select_device(cfg)
    device === cpu && return cpu

    try
        free_mib = round(Int, CUDA.free_memory() / 2^20)
        if free_mib < 1500
            @info "VRAM livre baixa ($(free_mib) MiB); predição em CPU para evitar erro."
            return cpu
        end
    catch
        return cpu
    end
    return device
end

function predict_image(model, image_path::String; cfg::Config = Config())
    image = load(image_path)
    x = preprocess_image(image; target_size = cfg.image_size)
    batch = reshape(x, cfg.image_size, cfg.image_size, 1, 1)

    device = select_inference_device(cfg)
    # Checkpoint BSON sempre carrega na CPU — mover modelo e lote juntos
    model = model |> device
    batch = batch |> device

    y_logits = model(batch) |> cpu
    return sigmoid.(from_flux_batch(y_logits, 1))
end

function predict_and_save(
    model,
    image_path::String;
    cfg::Config = Config(),
    output_path::Union{Nothing,String} = nothing,
)
    probabilities = predict_image(model, image_path; cfg = cfg)
    mask = postprocess_mask(probabilities; threshold = cfg.prediction_threshold)

    if output_path === nothing
        ensure_output_dirs!(cfg)
        basename_img = splitext(basename(image_path))[1]
        output_path = joinpath(cfg.predictions_dir, "$(basename_img)_pred.png")
    else
        mkpath(dirname(abspath(output_path)))
    end

    save(output_path, colorview(Gray, mask))
    return output_path, mask, probabilities
end

function load_model_for_inference(path::String = joinpath("outputs/checkpoints", "best_model.bson"))
    return load_model_with_config(path)
end
