using Test
using Images
using FileIO
using BSON

@testset "Predict pipeline" begin
    model = build_unet(in_channels = 1, out_channels = 1)
    cfg = Config(image_size = 64, predictions_dir = mktempdir(), checkpoint_dir = mktempdir())

    temp_dir = mktempdir()
    image_path = joinpath(temp_dir, "sample.png")
    img = colorview(Gray, rand(Float32, 80, 100))
    save(image_path, img)

    output_path, mask, probs = predict_and_save(model, image_path; cfg = cfg)
    @test isfile(output_path)
    @test size(mask) == (cfg.image_size, cfg.image_size)
    @test size(probs) == (cfg.image_size, cfg.image_size)

    checkpoint = joinpath(cfg.checkpoint_dir, "test_model.bson")
    BSON.@save checkpoint model=model config=cfg
    loaded_model, loaded_cfg = load_model_with_config(checkpoint)
    @test loaded_cfg.image_size == cfg.image_size
    pred = predict_image(loaded_model, image_path; cfg = loaded_cfg)
    @test size(pred) == (cfg.image_size, cfg.image_size)
end
