struct Config
    dataset_path::String
    image_size::Int
    train_ratio::Float64
    val_ratio::Float64
    test_ratio::Float64
    random_seed::Int
    epochs::Int
    batch_size::Int
    learning_rate::Float64
    early_stopping_patience::Int
    use_gpu::Bool
    max_samples_per_class::Int
    prediction_threshold::Float64
    checkpoint_dir::String
    predictions_dir::String
    tumor_sample_weight::Float64
    pos_pixel_weight::Float64
end

function Config(;
    dataset_path::String = "docs/material/Dataset_BUSI_with_GT",
    image_size::Int = 256,
    train_ratio::Float64 = 0.70,
    val_ratio::Float64 = 0.15,
    test_ratio::Float64 = 0.15,
    random_seed::Int = 42,
    epochs::Int = 30,
    batch_size::Int = 4,
    learning_rate::Float64 = 0.001,
    early_stopping_patience::Int = 5,
    use_gpu::Bool = false,
    max_samples_per_class::Int = 0,
    prediction_threshold::Float64 = 0.5,
    checkpoint_dir::String = "outputs/checkpoints",
    predictions_dir::String = "outputs/predictions",
    tumor_sample_weight::Float64 = 1.0,
    pos_pixel_weight::Float64 = 1.0,
)
    total = train_ratio + val_ratio + test_ratio
    abs(total - 1.0) > 1e-6 && error("As proporções de split devem somar 1.0 (atual: $total)")
    return Config(
        dataset_path,
        image_size,
        train_ratio,
        val_ratio,
        test_ratio,
        random_seed,
        epochs,
        batch_size,
        learning_rate,
        early_stopping_patience,
        use_gpu,
        max_samples_per_class,
        prediction_threshold,
        checkpoint_dir,
        predictions_dir,
        tumor_sample_weight,
        pos_pixel_weight,
    )
end

function load_config(path::String = "config/default.toml")::Config
    if !isfile(path)
        return Config()
    end
    data = TOML.parsefile(path)
    return Config(;
        dataset_path = get(data, "dataset_path", "docs/material/Dataset_BUSI_with_GT"),
        image_size = Int(get(data, "image_size", 256)),
        train_ratio = Float64(get(data, "train_ratio", 0.70)),
        val_ratio = Float64(get(data, "val_ratio", 0.15)),
        test_ratio = Float64(get(data, "test_ratio", 0.15)),
        random_seed = Int(get(data, "random_seed", 42)),
        epochs = Int(get(data, "epochs", 30)),
        batch_size = Int(get(data, "batch_size", 4)),
        learning_rate = Float64(get(data, "learning_rate", 0.001)),
        early_stopping_patience = Int(get(data, "early_stopping_patience", 5)),
        use_gpu = Bool(get(data, "use_gpu", false)),
        max_samples_per_class = Int(get(data, "max_samples_per_class", 0)),
        prediction_threshold = Float64(get(data, "prediction_threshold", 0.5)),
        checkpoint_dir = get(data, "checkpoint_dir", "outputs/checkpoints"),
        predictions_dir = get(data, "predictions_dir", "outputs/predictions"),
    )
end

function uses_weighted_training(cfg::Config)
    return cfg.tumor_sample_weight > 1.0 || cfg.pos_pixel_weight > 1.0
end

function ensure_output_dirs!(cfg::Config)
    mkpath(cfg.checkpoint_dir)
    mkpath(cfg.predictions_dir)
end

function select_device(cfg::Config)
    if cfg.use_gpu
        @warn "GPU solicitada na configuração, mas este projeto usa CPU por padrão."
    end
    return cpu
end
