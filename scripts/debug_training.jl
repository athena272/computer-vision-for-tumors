using Flux
using Statistics
using TumorSegmentation

cfg = Config(
    image_size = 128,
    max_samples_per_class = 20,
    batch_size = 4,
    tumor_sample_weight = 4.0,
    pos_pixel_weight = 20.0,
    prediction_threshold = 0.35,
)
samples = list_samples(cfg.dataset_path; max_per_class = cfg.max_samples_per_class)
splits = split_samples(samples; train_ratio = cfg.train_ratio, val_ratio = cfg.val_ratio, seed = cfg.random_seed)
loaders = create_dataloaders(splits; image_size = cfg.image_size, batch_size = cfg.batch_size)
model = TumorSegmentation.build_unet_for_config(cfg)
opt = Flux.setup(Adam(3e-4), model)

for epoch in 1:5
    losses = Float32[]
    for (x, y) in loaders.train
        loss, grads = Flux.withgradient(model) do m
            TumorSegmentation.training_loss(m, x, y, cfg)
        end
        Flux.update!(opt, model, grads[1])
        push!(losses, Float32(loss))
    end
    metrics = evaluate_model(model, loaders.val; threshold = cfg.prediction_threshold, tumor_only = true)
    println("epoch $epoch | loss $(round(mean(losses), digits=4)) | dice tumor $(round(metrics.dice, digits=4))")
end

img = "docs/material/Dataset_BUSI_with_GT/benign/benign (1).png"
probs = predict_image(model, img; cfg = cfg)
println("benign (1) prob min=$(minimum(probs)) max=$(maximum(probs)) mean=$(mean(probs))")
println("pixels >= 0.35: ", sum(probs .>= 0.35))
