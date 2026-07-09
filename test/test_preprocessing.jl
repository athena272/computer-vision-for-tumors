using Test

@testset "Preprocessing" begin
    img = rand(Float32, 100, 80)
    resized = resize_array(img, 64)
    @test size(resized) == (64, 64)

    normalized = normalize_image(img)
    @test minimum(normalized) >= 0.0f0
    @test maximum(normalized) <= 1.0f0

    constant = fill(42.0f0, 10, 10)
    @test normalize_image(constant) == zeros(Float32, 10, 10)

    mask = zeros(Float32, 32, 32)
    mask[10:20, 10:20] .= 1.0f0
    binary = postprocess_mask(mask; threshold = 0.5)
    @test sum(binary) == 11 * 11

    batch = to_flux_batch([ones(Float32, 32, 32), zeros(Float32, 32, 32)])
    @test size(batch) == (32, 32, 1, 2)
    @test from_flux_batch(batch, 1) == ones(Float32, 32, 32)
end
