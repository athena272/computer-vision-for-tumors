using Test
using Images
using FileIO
using BSON
using Flux
using CUDA

@testset "Predict pipeline" begin
    model = build_unet(in_channels = 1, out_channels = 1)
    cfg = Config(image_size = 64, predictions_dir = mktempdir(), checkpoint_dir = mktempdir(), use_gpu = false)

    temp_dir = mktempdir()
    image_path = joinpath(temp_dir, "sample.png")
    img = colorview(Gray, rand(Float32, 80, 100))
    save(image_path, img)

    output_path, mask, probs = predict_and_save(model, image_path; cfg = cfg)
    @test isfile(output_path)
    @test size(mask) == (cfg.image_size, cfg.image_size)
    @test all(0.0f0 .<= probs .<= 1.0f0)

    checkpoint = joinpath(cfg.checkpoint_dir, "test_model.bson")
    BSON.@save checkpoint model=model config=cfg
    loaded_model, loaded_cfg = load_model_with_config(checkpoint)
    @test loaded_cfg.image_size == cfg.image_size
    pred = predict_image(loaded_model, image_path; cfg = loaded_cfg)
    @test size(pred) == (cfg.image_size, cfg.image_size)
    @test all(0.0f0 .<= pred .<= 1.0f0)
end

@testset "Predict device consistency" begin
    # Simula o bug: modelo na CPU + use_gpu=true (como após carregar BSON)
    model = build_unet(in_channels = 1, out_channels = 1)  # CPU
    cfg = Config(
        image_size = 64,
        predictions_dir = mktempdir(),
        checkpoint_dir = mktempdir(),
        use_gpu = true,
        prediction_threshold = 0.35,
    )

    temp_dir = mktempdir()
    image_path = joinpath(temp_dir, "sample.png")
    save(image_path, colorview(Gray, rand(Float32, 64, 64)))

    # Não deve lançar Scalar indexing / device mismatch
    probs = predict_image(model, image_path; cfg = cfg)
    @test size(probs) == (64, 64)
    @test all(isfinite, probs)
    @test all(0.0f0 .<= probs .<= 1.0f0)

    output_path, mask, _ = predict_and_save(model, image_path; cfg = cfg)
    @test isfile(output_path)
    @test size(mask) == (64, 64)
end

@testset "Predict GPU smoke (se CUDA disponível)" begin
    if !CUDA.functional()
        @test true  # skip sem falhar
    else
        model = build_unet(in_channels = 1, out_channels = 1)
        cfg = Config(
            image_size = 64,
            predictions_dir = mktempdir(),
            use_gpu = true,
            prediction_threshold = 0.35,
        )
        temp_dir = mktempdir()
        image_path = joinpath(temp_dir, "gpu_sample.png")
        save(image_path, colorview(Gray, rand(Float32, 64, 64)))

        # Força caminho GPU se houver VRAM (select_inference_device pode cair para CPU)
        device = TumorSegmentation.select_device(cfg)
        model_d = model |> device
        batch = rand(Float32, 64, 64, 1, 1) |> device
        y = model_d(batch) |> cpu
        @test size(y) == (64, 64, 1, 1)
        @test all(isfinite, y)

        probs = predict_image(model, image_path; cfg = cfg)
        @test size(probs) == (64, 64)
        @test all(0.0f0 .<= probs .<= 1.0f0)
    end
end
