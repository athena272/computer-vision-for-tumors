using Flux
using TumorSegmentation
using CUDA

println("Julia ", VERSION)
println("use_gpu no config: ", load_config().use_gpu)
println("CUDA.functional(): ", CUDA.functional())
flush(stdout)

if CUDA.functional()
    println("GPU: ", CUDA.name(CUDA.device()))
    println("VRAM total: ", round(Int, CUDA.total_memory() / 2^20), " MiB")
    println("VRAM livre: ", round(Int, CUDA.free_memory() / 2^20), " MiB")
    flush(stdout)

    device = TumorSegmentation.select_device(load_config())
    using_gpu = !(device === cpu)
    println("select_device -> ", using_gpu ? "gpu" : "cpu")
    bs = TumorSegmentation.effective_batch_size(load_config(), device)
    println("effective_batch_size (256px): ", bs)
    flush(stdout)

    if using_gpu
        println("Rodando smoke test na GPU (1a vez pode levar 1-3 min compilando CUDA)...")
        flush(stdout)
        t0 = time()
        model = TumorSegmentation.build_unet_for_config(Config(image_size = 64, use_gpu = true)) |> device
        x = rand(Float32, 64, 64, 1, 2) |> device
        y = rand(Float32, 64, 64, 1, 2) |> device
        loss, grads = Flux.withgradient(model) do m
            TumorSegmentation.training_loss(m, x, y, Config(tumor_sample_weight = 4.0, pos_pixel_weight = 20.0))
        end
        elapsed = round(time() - t0; digits = 1)
        println("smoke train step loss: ", round(Float64(loss), digits = 4), " (", elapsed, " s)")
        println("smoke OK on GPU")
    else
        println("GPU solicitada, mas o smoke test caiu para CPU (seguro).")
    end
else
    println("CUDA nao funcional - o treino usara CPU mesmo com use_gpu = true.")
end
