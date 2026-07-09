function logit_bce_elementwise(logits, y_true)
    z = logits
    t = y_true
    return max.(z, 0) .- z .* t .+ log.(1.0f0 .+ exp.(-abs.(z)))
end

function foreground_pos_weight(y_true; max_weight::Float32 = 200.0f0)
    fg = sum(Float32.(y_true))
    fg <= 0 && return 1.0f0
    bg = length(y_true) - fg
    return min(max_weight, Float32(bg / fg))
end

function effective_pos_weight(y_true, pos_pixel_weight::Float32)
    return max(pos_pixel_weight, foreground_pos_weight(y_true))
end

function bce_loss(logits, y_true)
    return mean(logit_bce_elementwise(logits, y_true))
end

function weighted_bce_loss(logits, y_true; pos_weight::Float32 = 1.0f0)
    weights = 1.0f0 .+ (pos_weight - 1.0f0) .* y_true
    bce = logit_bce_elementwise(logits, y_true)
    return sum(weights .* bce) / sum(weights)
end

function dice_loss(logits, y_true; smooth::Float32 = 1.0f-6)
    pred = sigmoid.(logits)
    true_mask = Float32.(y_true)
    intersection = sum(pred .* true_mask)
    union = sum(pred) + sum(true_mask)
    dice = (2.0f0 * intersection + smooth) / (union + smooth)
    return 1.0f0 - dice
end

function combined_loss(logits, y_true; bce_weight::Float32 = 0.5f0)
    return bce_weight * bce_loss(logits, y_true) + (1.0f0 - bce_weight) * dice_loss(logits, y_true)
end

function sample_combined_loss(
    logits,
    y_true;
    bce_weight::Float32 = 0.5f0,
    pos_pixel_weight::Float32 = 1.0f0,
)
    pos_w = effective_pos_weight(y_true, pos_pixel_weight)
    bce = pos_w > 1.0f0 ?
          weighted_bce_loss(logits, y_true; pos_weight = pos_w) :
          bce_loss(logits, y_true)
    return bce_weight * bce + (1.0f0 - bce_weight) * dice_loss(logits, y_true)
end

function batch_combined_loss(
    logits,
    y_true;
    bce_weight::Float32 = 0.5f0,
    tumor_sample_weight::Float32 = 1.0f0,
    pos_pixel_weight::Float32 = 1.0f0,
)
    batch_size = size(logits, 4)
    batch_size == 0 && return 0.0f0

    total = 0.0f0
    weight_sum = 0.0f0
    for i in 1:batch_size
        pred_i = logits[:, :, :, i:i]
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
    logits = model(x)
    if uses_weighted_training(cfg)
        return batch_combined_loss(
            logits,
            y;
            bce_weight = 0.25f0,
            tumor_sample_weight = Float32(cfg.tumor_sample_weight),
            pos_pixel_weight = Float32(cfg.pos_pixel_weight),
        )
    end
    return combined_loss(logits, y)
end
