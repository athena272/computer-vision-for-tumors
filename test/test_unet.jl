using Test
using Flux

@testset "U-Net forward pass" begin
    model = build_unet(in_channels = 1, out_channels = 1)

    for img_size in (64, 128, 256)
        x = rand(Float32, img_size, img_size, 1, 2)
        y = model(x)
        @test size(y) == (img_size, img_size, 1, 2)
        @test all(isfinite, y)
    end
end

@testset "U-Net training step" begin
    model = build_unet(in_channels = 1, out_channels = 1)
    opt = Flux.setup(Adam(0.001), model)

    x = rand(Float32, 64, 64, 1, 2)
    y_true = rand(Float32, 64, 64, 1, 2)

    loss, grads = Flux.withgradient(model) do m
        combined_loss(m(x), y_true)
    end
    @test loss > 0
    Flux.update!(opt, model, grads[1])
end
