using Images: colorview, Gray, hcat

function save_comparison(
    image_path::String,
    prediction::AbstractMatrix{<:Real},
    output_path::String;
    ground_truth_path::Union{Nothing, String} = nothing,
    target_size::Int = 256,
)
    image = preprocess_image(load(image_path); target_size = target_size)
    pred_matrix = resize_array(Float32.(prediction), target_size)
    pred_view = colorview(Gray, postprocess_mask(pred_matrix))

    panels = [colorview(Gray, image), pred_view]

    if ground_truth_path !== nothing && isfile(ground_truth_path)
        gt = preprocess_mask(load(ground_truth_path); target_size = target_size)
        push!(panels, colorview(Gray, gt))
    end

    comparison = hcat(panels...)
    mkpath(dirname(abspath(output_path)))
    save(output_path, comparison)
    return output_path
end
