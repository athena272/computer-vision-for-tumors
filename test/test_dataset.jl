using Test

@testset "Dataset indexing" begin
    @test is_image_file("malignant (1).png")
    @test !is_image_file("malignant (1)_mask.png")
    @test !is_image_file("malignant (1)_mask_1.png")

    @test mask_path_for("benign (3).png") == "benign (3)_mask.png"
end

@testset "Dataset split" begin
    samples = [
        Sample("a/1.png", "a/1_mask.png", "benign"),
        Sample("a/2.png", "a/2_mask.png", "benign"),
        Sample("a/3.png", "a/3_mask.png", "benign"),
        Sample("b/1.png", "b/1_mask.png", "malignant"),
        Sample("b/2.png", "b/2_mask.png", "malignant"),
        Sample("b/3.png", "b/3_mask.png", "malignant"),
        Sample("c/1.png", "c/1_mask.png", "normal"),
        Sample("c/2.png", "c/2_mask.png", "normal"),
        Sample("c/3.png", "c/3_mask.png", "normal"),
    ]

    splits = split_samples(samples; train_ratio = 0.7, val_ratio = 0.15, seed = 42)
    total = length(splits.train) + length(splits.val) + length(splits.test)
    @test total == length(samples)
    @test !isempty(splits.train)
end

@testset "BUSI dataset (se disponível)" begin
    dataset_path = "docs/material/Dataset_BUSI_with_GT"
    if isdir(dataset_path)
        samples = list_samples(dataset_path)
        @test length(samples) > 0
        @test all(isfile(s.image_path) for s in samples)
        @test all(isfile(s.mask_path) for s in samples)
    else
        @test true
    end
end
