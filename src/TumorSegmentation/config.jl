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
        tumor_sample_weight = Float64(get(data, "tumor_sample_weight", 1.0)),
        pos_pixel_weight = Float64(get(data, "pos_pixel_weight", 1.0)),
    )
end

function uses_weighted_training(cfg::Config)
    return cfg.tumor_sample_weight > 1.0 || cfg.pos_pixel_weight > 1.0
end

function ensure_output_dirs!(cfg::Config)
    mkpath(cfg.checkpoint_dir)
    mkpath(cfg.predictions_dir)
end

"""
Limite suave de VRAM para notebooks (ex.: RTX 3060 6 GB).
Deixa folga para o Windows/display e evita esgotar a memória da placa.
"""
const GPU_SOFT_MEMORY_LIMIT = "4GiB"

function _cudnn_bin_dir()
    for (pid, mod) in Base.loaded_modules
        if pid.name == "CUDNN_jll" && isdefined(mod, :artifact_dir)
            bin = joinpath(getfield(mod, :artifact_dir), "bin")
            return isdir(bin) ? bin : nothing
        end
    end
    return nothing
end

"""Garante que as DLLs do cuDNN estejam no PATH (necessário no Windows)."""
function _ensure_cudnn_path!()
    bin = _cudnn_bin_dir()
    bin === nothing && return
    path = get(ENV, "PATH", "")
    if !occursin(bin, path)
        ENV["PATH"] = bin * ";" * path
    end
end

function _apply_gpu_soft_memory_limit!()
    if !haskey(ENV, "JULIA_CUDA_MEMORY_LIMIT")
        ENV["JULIA_CUDA_MEMORY_LIMIT"] = GPU_SOFT_MEMORY_LIMIT
    end
end

"""
Smoke test leve: 1 convolução 3×3 em GPU via Flux.
Se falhar (ex.: cuDNN), o treino cai para CPU em vez de quebrar no meio.
"""
function _gpu_smoke_ok()::Bool
    try
        _ensure_cudnn_path!()
        layer = Conv((3, 3), 1 => 1, pad = SamePad()) |> gpu
        x = CUDA.rand(Float32, 16, 16, 1, 1)
        y = layer(x)
        CUDA.synchronize()
        return size(y) == (16, 16, 1, 1)
    catch e
        @warn "Smoke test da GPU falhou; usando CPU." exception = e
        return false
    end
end

function select_device(cfg::Config)
    if !cfg.use_gpu
        return cpu
    end

    try
        if !CUDA.functional()
            @warn "GPU solicitada, mas CUDA não está funcional neste sistema. Usando CPU."
            return cpu
        end

        _ensure_cudnn_path!()
        _apply_gpu_soft_memory_limit!()

        if !_gpu_smoke_ok()
            return cpu
        end

        free_mib = round(Int, CUDA.free_memory() / 2^20)
        total_mib = round(Int, CUDA.total_memory() / 2^20)
        limit = get(ENV, "JULIA_CUDA_MEMORY_LIMIT", GPU_SOFT_MEMORY_LIMIT)
        @info "GPU ativa: $(CUDA.name(CUDA.device())) | VRAM livre $(free_mib)/$(total_mib) MiB | limite suave $limit"
        return gpu
    catch e
        @warn "Falha ao inicializar GPU; usando CPU." exception = e
        return cpu
    end
end

"""
Batch menor na GPU em resoluções altas — U-Net 256px consome bastante VRAM
e batches grandes aquecem mais o notebook sem ganho proporcional.
"""
function effective_batch_size(cfg::Config, device)
    if device === cpu
        return cfg.batch_size
    end
    if cfg.image_size >= 256
        return min(cfg.batch_size, 2)
    elseif cfg.image_size >= 128
        return min(cfg.batch_size, 4)
    end
    return min(cfg.batch_size, 8)
end
