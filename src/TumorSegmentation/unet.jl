struct ConvBlock
    layers
end

Flux.@layer ConvBlock

function ConvBlock(in_ch::Int, out_ch::Int)
    return ConvBlock(Chain(
        Conv((3, 3), in_ch => out_ch, pad = SamePad()),
        relu,
        Conv((3, 3), out_ch => out_ch, pad = SamePad()),
        relu,
    ))
end

function (block::ConvBlock)(x)
    return block.layers(x)
end

struct DownBlock
    conv::ConvBlock
    pool
end

Flux.@layer DownBlock

function DownBlock(in_ch::Int, out_ch::Int)
    return DownBlock(ConvBlock(in_ch, out_ch), MaxPool((2, 2)))
end

function (block::DownBlock)(x)
  conv_out = block.conv(x)
    return block.pool(conv_out), conv_out
end

struct UpBlock
    upsample
    conv::ConvBlock
end

Flux.@layer UpBlock

function UpBlock(in_ch::Int, out_ch::Int)
    return UpBlock(
        Upsample(:nearest; scale = 2),
        ConvBlock(in_ch, out_ch),
    )
end

function (block::UpBlock)(x, skip)
    x = block.upsample(x)
    sh, sw = size(skip, 1), size(skip, 2)
    xh, xw = size(x, 1), size(x, 2)
    if sh != xh || sw != xw
        x = x[1:sh, 1:sw, :, :]
    end
    merged = cat(x, skip; dims = 3)
    return block.conv(merged)
end

struct UNet
    down1::DownBlock
    down2::DownBlock
    down3::DownBlock
    down4::DownBlock
    bottleneck::ConvBlock
    up1::UpBlock
    up2::UpBlock
    up3::UpBlock
    up4::UpBlock
    final_conv
end

Flux.@layer UNet

function build_unet(; in_channels::Int = 1, out_channels::Int = 1, features::Vector{Int} = [64, 128, 256, 512])
    length(features) == 4 || error("U-Net requer exatamente 4 níveis de features.")
    f1, f2, f3, f4 = features

    return UNet(
        DownBlock(in_channels, f1),
        DownBlock(f1, f2),
        DownBlock(f2, f3),
        DownBlock(f3, f4),
        ConvBlock(f4, f4 * 2),
        UpBlock(f4 * 2 + f4, f4),
        UpBlock(f4 + f3, f3),
        UpBlock(f3 + f2, f2),
        UpBlock(f2 + f1, f1),
        Conv((1, 1), f1 => out_channels),
    )
end

function (model::UNet)(x)
    x1, skip1 = model.down1(x)
    x2, skip2 = model.down2(x1)
    x3, skip3 = model.down3(x2)
    x4, skip4 = model.down4(x3)

    bottleneck = model.bottleneck(x4)

    u1 = model.up1(bottleneck, skip4)
    u2 = model.up2(u1, skip3)
    u3 = model.up3(u2, skip2)
    u4 = model.up4(u3, skip1)

    logits = model.final_conv(u4)
    return logits
end

"""Converte logits da U-Net em probabilidades (0–1)."""
function model_probabilities(model, x)
    return sigmoid.(model(x))
end

UNet(; kwargs...) = build_unet(; kwargs...)

"""Canais menores para imagens pequenas — treina bem mais rápido em CPU."""
function unet_features_for_size(image_size::Int)
    if image_size <= 64
        return [32, 64, 128, 256]
    elseif image_size <= 128
        return [48, 96, 192, 384]
    end
    return [64, 128, 256, 512]
end

function build_unet_for_config(cfg::Config)
    return build_unet(
        in_channels = 1,
        out_channels = 1,
        features = unet_features_for_size(cfg.image_size),
    )
end
