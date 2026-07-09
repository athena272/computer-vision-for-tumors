function dice_coefficient(y_pred, y_true; threshold::Float64 = 0.5, smooth::Float32 = 1.0f-6)
    pred = Float32.(y_pred .>= threshold)
    true_mask = Float32.(y_true .>= threshold)
    intersection = sum(pred .* true_mask)
    union = sum(pred) + sum(true_mask)
    return Float64((2.0f0 * intersection + smooth) / (union + smooth))
end

function iou_coefficient(y_pred, y_true; threshold::Float64 = 0.5, smooth::Float32 = 1.0f-6)
    pred = Float32.(y_pred .>= threshold)
    true_mask = Float32.(y_true .>= threshold)
    intersection = sum(pred .* true_mask)
    union = sum(pred) + sum(true_mask) - intersection
    return Float64((intersection + smooth) / (union + smooth))
end

function pixel_accuracy(y_pred, y_true; threshold::Float64 = 0.5)
    pred = y_pred .>= threshold
    true_mask = y_true .>= threshold
    return Float64(mean(pred .== true_mask))
end

function run_evaluation!(cfg::Config = load_config(); on_progress = nothing)
    notify(progress) = on_progress !== nothing && on_progress(progress)

    notify((; phase = :prepare, percent = 5.0, message = "Carregando modelo..."))
    checkpoint_path = joinpath(cfg.checkpoint_dir, "best_model.bson")
    isfile(checkpoint_path) || error("Modelo não encontrado em $checkpoint_path. Treine o modelo primeiro.")

    model, cfg = load_model_with_config(checkpoint_path)
    notify((; phase = :prepare, percent = 15.0, message = "Preparando conjunto de teste..."))
    samples = list_samples(cfg.dataset_path; max_per_class = cfg.max_samples_per_class)
    splits = split_samples(
        samples;
        train_ratio = cfg.train_ratio,
        val_ratio = cfg.val_ratio,
        seed = cfg.random_seed,
    )
    isempty(splits.test) && error("Conjunto de teste vazio.")

    loaders = create_dataloaders(
        splits;
        image_size = cfg.image_size,
        batch_size = cfg.batch_size,
        shuffle_train = false,
    )
    return evaluate_model(
        model,
        loaders.test;
        threshold = cfg.prediction_threshold,
        on_progress = on_progress,
    )
end

function evaluate_model(
    model,
    dataloader;
    threshold::Float64 = 0.5,
    on_progress = nothing,
    tumor_only::Bool = false,
)
    notify(progress) = on_progress !== nothing && on_progress(progress)
    dice_scores = Float64[]
    iou_scores = Float64[]
    acc_scores = Float64[]
    total_batches = max(1, length(dataloader))
    batch_idx = 0

    for (x, y) in dataloader
        batch_idx += 1
        y_pred = sigmoid.(model(x))
        for i in 1:size(x, 4)
            true_i = y[:, :, :, i]
            tumor_only && sum(true_i) == 0 && continue
            pred_i = y_pred[:, :, :, i]
            push!(dice_scores, dice_coefficient(pred_i, true_i; threshold = threshold))
            push!(iou_scores, iou_coefficient(pred_i, true_i; threshold = threshold))
            push!(acc_scores, pixel_accuracy(pred_i, true_i; threshold = threshold))
        end
        percent = 20.0 + (batch_idx / total_batches) * 75.0
        notify((; phase = :batch, percent, message = "Avaliando lote $batch_idx/$total_batches..."))
        batch_idx % 2 == 0 && yield()
    end

    notify((; phase = :finished, percent = 100.0, message = "Avaliação concluída."))
    return (
        dice = isempty(dice_scores) ? 0.0 : mean(dice_scores),
        iou = isempty(iou_scores) ? 0.0 : mean(iou_scores),
        accuracy = isempty(acc_scores) ? 0.0 : mean(acc_scores),
    )
end

