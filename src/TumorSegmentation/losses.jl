function bce_loss(y_pred, y_true)
    return Flux.Losses.binarycrossentropy(y_pred, y_true)
end

function weighted_bce_loss(y_pred, y_true; pos_weight::Float32 = 1.0f0)
    ε = eps(Float32)
    pred = clamp.(y_pred, ε, 1.0f0 - ε)
    weights = 1.0f0 .+ (pos_weight - 1.0f0) .* y_true
    bce = -(y_true .* log.(pred) .+ (1.0f0 .- y_true) .* log.(1.0f0 .- pred))
    return sum(weights .* bce) / sum(weights)
end

function dice_loss(y_pred, y_true; smooth::Float32 = 1.0f-6)
    pred = Float32.(y_pred)
    true_mask = Float32.(y_true)
    intersection = sum(pred .* true_mask)
    union = sum(pred) + sum(true_mask)
    dice = (2.0f0 * intersection + smooth) / (union + smooth)
    return 1.0f0 - dice
end

function combined_loss(y_pred, y_true; bce_weight::Float32 = 0.5f0)
    return bce_weight * bce_loss(y_pred, y_true) + (1.0f0 - bce_weight) * dice_loss(y_pred, y_true)
end

function sample_combined_loss(
    y_pred,
    y_true;
    bce_weight::Float32 = 0.5f0,
    pos_pixel_weight::Float32 = 1.0f0,
)
    bce = pos_pixel_weight > 1.0f0 ?
          weighted_bce_loss(y_pred, y_true; pos_weight = pos_pixel_weight) :
          bce_loss(y_pred, y_true)
    return bce_weight * bce + (1.0f0 - bce_weight) * dice_loss(y_pred, y_true)
end

function batch_combined_loss(
    y_pred,
    y_true;
    bce_weight::Float32 = 0.5f0,
    tumor_sample_weight::Float32 = 1.0f0,
    pos_pixel_weight::Float32 = 1.0f0,
)
    batch_size = size(y_pred, 4)
    batch_size == 0 && return 0.0f0

    total = 0.0f0
    weight_sum = 0.0f0
    for i in 1:batch_size
        pred_i = y_pred[:, :, :, i:i]
        true_i = y_true[:, :, :, i:i]
        has_tumor = sum(true_i) > 0
        sample_weight = has_tumor ? tumor_sample_weight : 1.0f0
        loss_i = sample_combined_loss(
            pred_i,
            true_i;
            bce_weight = bce_weight,
            pos_pixel_weight = pos_pixel_weight,
        )
        total += sample_weight * loss_i
        weight_sum += sample_weight
    end
    return total / weight_sum
end

function training_loss(model, x, y, cfg::Config)
    y_pred = model(x)
    if uses_weighted_training(cfg)
        return batch_combined_loss(
            y_pred,
            y;
            tumor_sample_weight = Float32(cfg.tumor_sample_weight),
            pos_pixel_weight = Float32(cfg.pos_pixel_weight),
        )
    end
    return combined_loss(y_pred, y)
end
