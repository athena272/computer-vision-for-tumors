using Flux
using FileIO
using Statistics
using TumorSegmentation

const BENIGN_SAMPLE = "docs/material/Dataset_BUSI_with_GT/benign/benign (1).png"

function mask_dice(model, image_path::String, cfg::Config)
    probs = predict_image(model, image_path; cfg = cfg)
    gt_path = replace(image_path, ".png" => "_mask.png")
    isfile(gt_path) || return (dice = 0.0, prob_min = minimum(probs), prob_max = maximum(probs), prob_mean = mean(probs), white_pixels = 0)
    gt = preprocess_mask(load(gt_path); target_size = cfg.image_size)
    dice = dice_coefficient(probs, gt; threshold = cfg.prediction_threshold)
    white = sum(probs .>= cfg.prediction_threshold)
    return (dice = dice, prob_min = minimum(probs), prob_max = maximum(probs), prob_mean = mean(probs), white_pixels = white)
end

function validate_quick_training(;
    image_size::Int = 128,
    max_samples_per_class::Int = 20,
    epochs::Int = 8,
    batch_size::Int = 4,
)
    cfg = Config(
        image_size = image_size,
        max_samples_per_class = max_samples_per_class,
        epochs = epochs,
        batch_size = batch_size,
        learning_rate = 3e-4,
        tumor_sample_weight = 4.0,
        pos_pixel_weight = 20.0,
        prediction_threshold = 0.35,
        early_stopping_patience = 6,
    )

    println("=== Training validation proxy ($(image_size)px) ===")
    samples = list_samples(cfg.dataset_path; max_per_class = cfg.max_samples_per_class)
    splits = split_samples(samples; train_ratio = cfg.train_ratio, val_ratio = cfg.val_ratio, seed = cfg.random_seed)
    loaders = create_dataloaders(splits; image_size = cfg.image_size, batch_size = cfg.batch_size)

    model = TumorSegmentation.build_unet_for_config(cfg)
    opt = Flux.setup(Adam(cfg.learning_rate), model)

    before = mask_dice(model, BENIGN_SAMPLE, cfg)
    println("\nBefore training (benign 1):")
    println("  prob min/max/mean: $(before.prob_min) / $(before.prob_max) / $(round(before.prob_mean, digits=4))")
    println("  white pixels: $(before.white_pixels)")

    losses = Float32[]
    val_dices = Float64[]
    for epoch in 1:cfg.epochs
        epoch_losses = Float32[]
        for (x, y) in loaders.train
            loss, grads = Flux.withgradient(model) do m
                TumorSegmentation.training_loss(m, x, y, cfg)
            end
            Flux.update!(opt, model, grads[1])
            push!(epoch_losses, Float32(loss))
        end
        train_loss = mean(epoch_losses)
        push!(losses, train_loss)
        metrics = evaluate_model(model, loaders.val; threshold = cfg.prediction_threshold, tumor_only = true)
        push!(val_dices, metrics.dice)
        println("Epoch $epoch/$epochs | loss $(round(train_loss, digits=4)) | val dice $(round(metrics.dice, digits=4))")
    end

    after = mask_dice(model, BENIGN_SAMPLE, cfg)
    println("\nAfter training (benign 1):")
    println("  prob min/max/mean: $(after.prob_min) / $(after.prob_max) / $(round(after.prob_mean, digits=4))")
    println("  white pixels: $(after.white_pixels)")
    println("  dice@threshold: $(round(after.dice, digits=4))")

    not_collapsed = after.prob_max > 0.01 && after.prob_min < 0.99
    learned_something = val_dices[end] > 0.05 || after.dice > 0.05
    loss_reasonable = losses[end] < 1.5

    checks = [
        ("loss below 3.0", losses[end] < 3.0),
        ("loss below 1.5", loss_reasonable),
        ("output not collapsed", not_collapsed),
        ("dice above 0.05", learned_something),
    ]

    println("\n=== Checks ===")
    all_ok = true
    for (label, ok) in checks
        status = ok ? "PASS" : "FAIL"
        println("  [$status] $label")
        all_ok &= ok
    end

    if all_ok
        println("\n[PASS] Pipeline learns. Safe to start full quick training at 256px.")
        return true
    else
        println("\n[FAIL] Do not start long training yet.")
        return false
    end
end

validate_quick_training()
