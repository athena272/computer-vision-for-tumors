const DATASET_CLASSES = ("benign", "malignant", "normal")

struct Sample
    image_path::String
    mask_path::String
    label::String
end

function is_primary_mask(filename::String)
    return endswith(filename, "_mask.png") && !occursin("_mask_", filename[1:end-9])
end

function is_image_file(filename::String)
    return endswith(lowercase(filename), ".png") &&
           !occursin("_mask", lowercase(filename))
end

function mask_path_for(image_path::String)
    if endswith(image_path, ".png")
        return replace(image_path, ".png" => "_mask.png")
    end
    error("Caminho de imagem inválido: $image_path")
end

function list_samples(dataset_path::String; max_per_class::Int = 0)::Vector{Sample}
    samples = Sample[]
    for class_name in DATASET_CLASSES
        class_dir = joinpath(dataset_path, class_name)
        isdir(class_dir) || continue

        class_samples = Sample[]
        for filename in sort(readdir(class_dir))
            is_image_file(filename) || continue
            image_path = joinpath(class_dir, filename)
            mask_path = mask_path_for(image_path)
            isfile(mask_path) || continue
            push!(class_samples, Sample(image_path, mask_path, class_name))
        end

        if max_per_class > 0 && length(class_samples) > max_per_class
            class_samples = class_samples[1:max_per_class]
        end
        append!(samples, class_samples)
    end
    return samples
end

function split_samples(
    samples::Vector{Sample};
    train_ratio::Float64 = 0.70,
    val_ratio::Float64 = 0.15,
    seed::Int = 42,
)
    isempty(samples) && error("Nenhuma amostra encontrada para dividir.")
    rng = MersenneTwister(seed)

    by_class = Dict{String, Vector{Sample}}()
    for sample in samples
        push!(get!(by_class, sample.label, Sample[]), sample)
    end

    train = Sample[]
    val = Sample[]
    test = Sample[]

    for class_samples in values(by_class)
        shuffled = shuffle(rng, class_samples)
        n = length(shuffled)
        n_train = max(1, floor(Int, n * train_ratio))
        n_val = n > 2 ? max(1, floor(Int, n * val_ratio)) : 0
        n_test = n - n_train - n_val
        n_test < 0 && (n_test = 0; n_val = n - n_train)

        append!(train, shuffled[1:n_train])
        if n_val > 0
            append!(val, shuffled[n_train+1:n_train+n_val])
        end
        if n_test > 0
            append!(test, shuffled[n_train+n_val+1:end])
        end
    end

    return (
        train = shuffle(rng, train),
        val = shuffle(rng, val),
        test = shuffle(rng, test),
    )
end

function load_sample(sample::Sample; image_size::Int = 256)
    image = load(sample.image_path)
    mask = load(sample.mask_path)
    x = preprocess_image(image; target_size = image_size)
    y = preprocess_mask(mask; target_size = image_size)
    return x, y
end

function _collate_batch(batch::Vector{Tuple{Matrix{Float32}, Matrix{Float32}}})
    images = [item[1] for item in batch]
    masks = [item[2] for item in batch]
    return to_flux_batch(images), to_flux_batch(masks)
end

function create_dataloaders(
    splits::NamedTuple;
    image_size::Int = 256,
    batch_size::Int = 4,
    shuffle_train::Bool = true,
)
    function make_loader(samples::Vector{Sample})
        data = [load_sample(s; image_size = image_size) for s in samples]
        return Flux.DataLoader(
            data;
            batchsize = batch_size,
            shuffle = false,
            collate = _collate_batch,
        )
    end

    train_data = [load_sample(s; image_size = image_size) for s in splits.train]
    train_loader = Flux.DataLoader(
        train_data;
        batchsize = batch_size,
        shuffle = shuffle_train,
        collate = _collate_batch,
    )

    val_loader = isempty(splits.val) ? nothing : make_loader(splits.val)
    test_loader = isempty(splits.test) ? nothing : make_loader(splits.test)

    return (train = train_loader, val = val_loader, test = test_loader)
end
