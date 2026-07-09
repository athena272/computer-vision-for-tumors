function run_training!(cfg::Config = load_config(); on_progress = nothing)
    notify(progress) = on_progress !== nothing && on_progress(progress)

    notify((; phase = :prepare, percent = 2.0, message = "Indexando imagens do dataset BUSI..."))
    ensure_output_dirs!(cfg)
    samples = list_samples(cfg.dataset_path; max_per_class = cfg.max_samples_per_class)
    splits = split_samples(
        samples;
        train_ratio = cfg.train_ratio,
        val_ratio = cfg.val_ratio,
        seed = cfg.random_seed,
    )
    loaders = create_dataloaders(
        splits;
        image_size = cfg.image_size,
        batch_size = cfg.batch_size,
        on_progress = on_progress,
    )
    notify((; phase = :prepare, percent = 10.0, message = "Inicializando rede U-Net..."))
    model = build_unet_for_config(cfg)
    train_model!(model, loaders.train, loaders.val; cfg = cfg, on_progress = on_progress)
    notify((; phase = :finished, percent = 100.0, message = "Treinamento finalizado."))
    return joinpath(cfg.checkpoint_dir, "best_model.bson")
end

function train_model!(model, train_loader, val_loader; cfg::Config = Config(), on_progress = nothing)
    ensure_output_dirs!(cfg)
    device = select_device(cfg)
    model = model |> device

    opt = Flux.setup(Adam(cfg.learning_rate), model)
    best_val_dice = -Inf
    patience_counter = 0
    checkpoint_path = joinpath(cfg.checkpoint_dir, "best_model.bson")
    batches_per_epoch = max(1, length(train_loader))

    notify(progress) = on_progress !== nothing && on_progress(progress)
    val_every = cfg.max_samples_per_class > 0 ? 2 : 1

    for epoch in 1:cfg.epochs
        epoch_losses = Float32[]
        batch_idx = 0
        for (x, y) in train_loader
            batch_idx += 1
            x = x |> device
            y = y |> device
            loss, grads = Flux.withgradient(model) do m
                training_loss(m, x, y, cfg)
            end
            Flux.update!(opt, model, grads[1])
            push!(epoch_losses, Float32(loss))
            notify((
                phase = :batch,
                epoch = epoch,
                total_epochs = cfg.epochs,
                batch = batch_idx,
                batches_per_epoch = batches_per_epoch,
            ))
            batch_idx % 2 == 0 && yield()
        end
        train_loss = isempty(epoch_losses) ? 0.0f0 : mean(epoch_losses)

        val_dice = 0.0
        run_validation = val_loader !== nothing && (epoch % val_every == 0 || epoch == cfg.epochs)
        if run_validation
            tumor_only = uses_weighted_training(cfg)
            metrics = evaluate_model(
                model,
                val_loader;
                threshold = cfg.prediction_threshold,
                tumor_only = tumor_only,
            )
            yield()
            val_dice = metrics.dice
            dice_label = tumor_only ? "Dice validação (com tumor)" : "Dice validação"
            println("Época $epoch/$(cfg.epochs) | loss treino: $(round(train_loss, digits=4)) | $dice_label: $(round(val_dice, digits=4))")

            notify((
                phase = :epoch,
                epoch = epoch,
                total_epochs = cfg.epochs,
                batches_per_epoch = batches_per_epoch,
                train_loss = Float64(train_loss),
                val_dice = Float64(val_dice),
            ))

            if val_dice > best_val_dice
                best_val_dice = val_dice
                patience_counter = 0
                _save_checkpoint(checkpoint_path, model, cfg)
                println("  -> Novo melhor modelo salvo (Dice = $(round(val_dice, digits=4)))")
            else
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience
                    println("Early stopping na época $epoch.")
                    notify((
                        phase = :early_stop,
                        epoch = epoch,
                        total_epochs = cfg.epochs,
                        batches_per_epoch = batches_per_epoch,
                        train_loss = Float64(train_loss),
                        val_dice = Float64(val_dice),
                    ))
                    break
                end
            end
        elseif val_loader !== nothing
            println("Época $epoch/$(cfg.epochs) | loss treino: $(round(train_loss, digits=4)) (validação na próxima época)")
            notify((
                phase = :epoch,
                epoch = epoch,
                total_epochs = cfg.epochs,
                batches_per_epoch = batches_per_epoch,
                train_loss = Float64(train_loss),
                val_dice = nothing,
            ))
        else
            println("Época $epoch/$(cfg.epochs) | loss treino: $(round(train_loss, digits=4))")
            notify((
                phase = :epoch,
                epoch = epoch,
                total_epochs = cfg.epochs,
                batches_per_epoch = batches_per_epoch,
                train_loss = Float64(train_loss),
                val_dice = nothing,
            ))
            _save_checkpoint(checkpoint_path, model, cfg)
        end
    end

    if isfile(checkpoint_path)
        model = load_model(checkpoint_path) |> device
    end

    return model
end

function _save_checkpoint(path::String, model, cfg::Config)
    mkpath(dirname(path))
    cpu_model = cpu(model)
    BSON.@save path model=cpu_model config=cfg
end

function load_model(path::String)
    model, _ = load_model_with_config(path)
    return model
end

function load_model_with_config(path::String)
    isfile(path) || error("Checkpoint não encontrado: $path")
    BSON.@load path model config
    if !@isdefined(config) || config === nothing
        config = Config()
    end
    return model, config
end
