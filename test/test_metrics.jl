using Test
using Flux

@testset "Metrics" begin
    perfect = ones(Float32, 32, 32)
    @test dice_coefficient(perfect, perfect) ≈ 1.0 atol = 1e-5
    @test iou_coefficient(perfect, perfect) ≈ 1.0 atol = 1e-5
    @test pixel_accuracy(perfect, perfect) ≈ 1.0 atol = 1e-5

    empty_mask = zeros(Float32, 32, 32)
    @test dice_coefficient(empty_mask, empty_mask) ≈ 1.0 atol = 1e-5

    pred = zeros(Float32, 32, 32)
    pred[1:10, 1:10] .= 1.0f0
    true_mask = zeros(Float32, 32, 32)
    true_mask[5:14, 5:14] .= 1.0f0
    @test 0.0 < dice_coefficient(pred, true_mask) < 1.0
end

@testset "Losses" begin
    y_pred = rand(Float32, 16, 16, 1, 1)
    y_true = rand(Float32, 16, 16, 1, 1)
    @test bce_loss(y_pred, y_true) >= 0
    @test dice_loss(y_pred, y_true) >= 0
    @test combined_loss(y_pred, y_true) >= 0
    @test TumorSegmentation.weighted_bce_loss(y_pred, y_true; pos_weight = 5.0f0) >= 0
    @test TumorSegmentation.batch_combined_loss(
        y_pred,
        y_true;
        tumor_sample_weight = 3.0f0,
        pos_pixel_weight = 5.0f0,
    ) >= 0

    model = build_unet(in_channels = 1, out_channels = 1)
    cfg = Config(tumor_sample_weight = 4.0, pos_pixel_weight = 12.0)
    x = rand(Float32, 32, 32, 1, 2)
    y_batch = rand(Float32, 32, 32, 1, 2)
    loss, grads = Flux.withgradient(model) do m
        TumorSegmentation.training_loss(m, x, y_batch, cfg)
    end
    @test loss > 0
    @test grads[1] !== nothing
end
