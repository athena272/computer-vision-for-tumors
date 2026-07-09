using Images: imresize, Gray, colorview

function _to_grayscale_matrix(img)
    try
        return Float64.(Gray.(img))
    catch
        return Float64.(img)
    end
end

function resize_array(arr::AbstractMatrix{<:Real}, target_size::Int)
    h, w = size(arr, 1), size(arr, 2)
    if h == target_size && w == target_size
        return Float32.(arr)
    end
    resized = imresize(Float32.(arr), (target_size, target_size))
    return resized
end

function normalize_image(arr::AbstractMatrix{<:Real})
    arr32 = Float32.(arr)
    minval = minimum(arr32)
    maxval = maximum(arr32)
    if maxval ≈ minval
        return zeros(Float32, size(arr32)...)
    end
    return (arr32 .- minval) ./ (maxval - minval)
end

function preprocess_image(img; target_size::Int = 256)
    gray = _to_grayscale_matrix(img)
    resized = resize_array(gray, target_size)
    return normalize_image(resized)
end

function preprocess_mask(mask; target_size::Int = 256, threshold::Float64 = 0.5)
    gray = _to_grayscale_matrix(mask)
    resized = resize_array(gray, target_size)
    return Float32.(resized .>= threshold)
end

function postprocess_mask(probabilities::AbstractMatrix{<:Real}; threshold::Float64 = 0.5)
    return Float32.(probabilities .>= threshold)
end

function to_flux_batch(images::Vector{<:AbstractMatrix{Float32}})
    h, w = size(images[1])
    batch = Array{Float32}(undef, h, w, 1, length(images))
    for (i, img) in enumerate(images)
        batch[:, :, 1, i] = img
    end
    return batch
end

function from_flux_batch(batch::AbstractArray{<:Real,4}, index::Int = 1)
    return Float32.(batch[:, :, 1, index])
end
