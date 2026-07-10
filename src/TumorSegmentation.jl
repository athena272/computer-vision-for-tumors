module TumorSegmentation

using Flux
using CUDA
using cuDNN
using Images
using FileIO
using BSON
using Statistics
using Random
using TOML
using Logging: with_logger, current_logger, Info
using LoggingExtras: ActiveFilteredLogger

include("TumorSegmentation/config.jl")
include("TumorSegmentation/preprocessing.jl")
include("TumorSegmentation/dataset.jl")
include("TumorSegmentation/unet.jl")
include("TumorSegmentation/losses.jl")
include("TumorSegmentation/metrics.jl")
include("TumorSegmentation/train.jl")
include("TumorSegmentation/predict.jl")
include("TumorSegmentation/visualize.jl")
include("TumorSegmentation/interface.jl")

export Config, load_config
export preprocess_image, preprocess_mask, postprocess_mask, resize_array, normalize_image, to_flux_batch, from_flux_batch
export list_samples, split_samples, load_sample, create_dataloaders, Sample, is_image_file, mask_path_for
export UNet, build_unet
export combined_loss, dice_loss, bce_loss
export dice_coefficient, iou_coefficient, pixel_accuracy
export train_model!, evaluate_model, run_training!, run_evaluation!
export predict_image, predict_and_save, load_model, load_model_with_config, load_model_for_inference
export save_comparison
export serve_interface, InterfaceState

function __init__()
    try
        _ensure_cudnn_path!()
    catch
        # PATH do cuDNN é opcional até alguém pedir GPU
    end
end

end
