using Statistics
using FileIO
using TumorSegmentation

model, cfg = load_model_with_config("outputs/checkpoints/best_model.bson")
println("Checkpoint image_size: ", cfg.image_size)
println("prediction_threshold: ", cfg.prediction_threshold)

img = "docs/material/Dataset_BUSI_with_GT/benign/benign (1).png"
probs = predict_image(model, img; cfg = cfg)
println("prob min=", minimum(probs), " max=", maximum(probs), " mean=", mean(probs))
println("pixels >= 0.5: ", sum(probs .>= 0.5), " / ", length(probs))
println("pixels >= 0.3: ", sum(probs .>= 0.3), " / ", length(probs))
println("pixels >= 0.1: ", sum(probs .>= 0.1), " / ", length(probs))

gt_path = replace(img, ".png" => "_mask.png")
if isfile(gt_path)
    gt = preprocess_mask(load(gt_path); target_size = cfg.image_size)
    dice = dice_coefficient(probs, gt; threshold = cfg.prediction_threshold)
    dice03 = dice_coefficient(probs, gt; threshold = 0.3)
    println("Dice@0.5 on this image: ", round(dice, digits = 4))
    println("Dice@0.3 on this image: ", round(dice03, digits = 4))
end

println()
println("--- Mascaras por classe (amostra) ---")
for class_name in ("benign", "malignant", "normal")
    samples = list_samples(cfg.dataset_path; max_per_class = 1)
    class_samples = filter(s -> s.label == class_name, samples)
    isempty(class_samples) && continue
    gt = preprocess_mask(load(class_samples[1].mask_path); target_size = cfg.image_size)
    println(class_name, ": ", sum(gt), " pixels de tumor em ", cfg.image_size, "x", cfg.image_size)
end
