function bce_loss(y_pred, y_true)
    return Flux.Losses.binarycrossentropy(y_pred, y_true)
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
