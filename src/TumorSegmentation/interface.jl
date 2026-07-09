using HTTP
using URIs

const INTERFACE_PORT = 8765

mutable struct InterfaceState
    project_root::String
    cfg::Config
    training_status::String
    training_message::String
    last_metrics::Union{Nothing,NamedTuple}
end

function InterfaceState(project_root::String)
    InterfaceState(project_root, load_config(), "idle", "", nothing)
end

function html_escape(text::AbstractString)
    replace(string(text), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;")
end

function layout(title::String, body::String)
    """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>$(html_escape(title))</title>
      <style>
        :root { color-scheme: light dark; font-family: Segoe UI, system-ui, sans-serif; }
        body { margin: 0; background: #0f172a; color: #e2e8f0; }
        .wrap { max-width: 960px; margin: 0 auto; padding: 24px; }
        h1, h2 { color: #f8fafc; }
        a, .btn { color: white; text-decoration: none; }
        .card { background: #1e293b; border-radius: 12px; padding: 20px; margin: 16px 0; box-shadow: 0 8px 24px rgba(0,0,0,.25); }
        .grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
        .btn { display: inline-block; background: #2563eb; padding: 10px 16px; border-radius: 8px; border: 0; cursor: pointer; font-size: 1rem; }
        .btn.secondary { background: #475569; }
        .btn.warn { background: #b45309; }
        label { display: block; margin: 12px 0 6px; font-weight: 600; }
        select, input[type=text] { width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #334155; background: #0f172a; color: #e2e8f0; }
        .images { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
        .images img { width: 100%; border-radius: 8px; background: #000; }
        .muted { color: #94a3b8; }
        .status { padding: 10px 12px; border-radius: 8px; background: #334155; }
        .ok { background: #14532d; }
        .err { background: #7f1d1d; }
        nav a { margin-right: 12px; }
        code { background: #334155; padding: 2px 6px; border-radius: 4px; }
      </style>
    </head>
    <body>
      <div class="wrap">
        <nav class="card">
          <a href="/">Início</a>
          <a href="/predict">Gerar máscara</a>
          <a href="/train">Treinar</a>
          <a href="/evaluate">Avaliar</a>
        </nav>
        $body
      </div>
    </body>
    </html>
    """
end

function project_relative(state::InterfaceState, path::String)
    full = abspath(joinpath(state.project_root, path))
    root = abspath(state.project_root)
    startswith(full, root) || error("Caminho fora do projeto: $path")
    return full
end

function rel_path(state::InterfaceState, path::String)
    return replace(relpath(path, state.project_root), '\\' => '/')
end

function list_example_images(state::InterfaceState)
    samples = list_samples(state.cfg.dataset_path; max_per_class = 8)
    return [s.image_path for s in samples]
end

function model_ready(state::InterfaceState)
    return isfile(project_relative(state, joinpath(state.cfg.checkpoint_dir, "best_model.bson")))
end

function page_home(state::InterfaceState)
    ready = model_ready(state)
    status = ready ? "Modelo treinado encontrado." : "Ainda não há modelo treinado."
    status_class = ready ? "status ok" : "status"
    body = """
    <h1>Segmentação de Tumores (U-Net)</h1>
    <p class="muted">Interface local para treinar, gerar máscaras e avaliar o modelo sem decorar comandos no terminal.</p>
    <div class="$status_class">$status</div>
    <div class="grid">
      <div class="card">
        <h2>1. Treinar</h2>
        <p class="muted">Ensina a rede com as imagens do dataset BUSI.</p>
        <a class="btn" href="/train">Abrir treino</a>
      </div>
      <div class="card">
        <h2>2. Gerar máscara</h2>
        <p class="muted">Escolhe uma imagem e vê a predição da U-Net.</p>
        <a class="btn" href="/predict">Abrir predição</a>
      </div>
      <div class="card">
        <h2>3. Avaliar</h2>
        <p class="muted">Calcula Dice, IoU e acurácia no conjunto de teste.</p>
        <a class="btn secondary" href="/evaluate">Abrir avaliação</a>
      </div>
    </div>
    """
    layout("U-Net BUSI", body)
end

function page_predict_form(state::InterfaceState, message::String = "")
    options = String[]
    for path in list_example_images(state)
        rel = rel_path(state, path)
        push!(options, "<option value=\"$(html_escape(rel))\">$(html_escape(rel))</option>")
    end
    alert = isempty(message) ? "" : "<div class=\"status err\">$(html_escape(message))</div>"
    body = """
    <h1>Gerar máscara</h1>
    $alert
    <div class="card">
      <form method="post" action="/predict">
        <label for="image">Imagem de ultrassom</label>
        <select id="image" name="image" required>
          $(join(options, "\n"))
        </select>
        <p class="muted" style="margin-top:16px">A saída será salva em <code>outputs/predictions/</code>.</p>
        <button class="btn" type="submit" style="margin-top:12px">Gerar máscara</button>
      </form>
    </div>
    """
    layout("Gerar máscara", body)
end

function page_predict_result(state::InterfaceState, image_rel::String, pred_rel::String, comparison_rel::Union{Nothing,String})
    img = "/file?path=$(URIs.escapeuri(image_rel))"
    pred = "/file?path=$(URIs.escapeuri(pred_rel))"
    comparison_block = ""
    if comparison_rel !== nothing
        cmp = "/file?path=$(URIs.escapeuri(comparison_rel))"
        comparison_block = """
        <div>
          <h3>Comparação</h3>
          <img src="$cmp" alt="Comparação">
        </div>
        """
    end
    body = """
    <h1>Resultado</h1>
    <p class="muted">Imagem analisada: <code>$(html_escape(image_rel))</code></p>
    <div class="card images">
      <div>
        <h3>Entrada</h3>
        <img src="$img" alt="Entrada">
      </div>
      <div>
        <h3>Máscara prevista</h3>
        <img src="$pred" alt="Predição">
      </div>
      $comparison_block
    </div>
    <a class="btn secondary" href="/predict">Gerar outra</a>
    """
    layout("Resultado", body)
end

function page_train(state::InterfaceState)
    quick_checked = state.cfg.max_samples_per_class > 0 ? "checked" : ""
    disabled = state.training_status == "running" ? "disabled" : ""
    body = """
    <h1>Treinar modelo</h1>
    <p class="muted">O treino roda em segundo plano. Esta página atualiza a cada 5 segundos. Em CPU, pode levar bastante tempo.</p>
    <div class="card">
      <div class="status">Status: $(html_escape(state.training_status))</div>
      <p style="margin-top:12px">$(html_escape(state.training_message))</p>
      <form method="post" action="/train" style="margin-top:16px">
        <label><input type="checkbox" name="quick" value="1" $quick_checked> Treino rápido (20 imagens por classe, 5 épocas)</label>
        <button class="btn warn" type="submit" $disabled>Iniciar treinamento</button>
      </form>
    </div>
    <meta http-equiv="refresh" content="5">
    """
    layout("Treinar", body)
end

function page_evaluate(state::InterfaceState, error_msg::String = "")
    alert = isempty(error_msg) ? "" : "<div class=\"status err\">$(html_escape(error_msg))</div>"
    metrics_block = ""
    if state.last_metrics !== nothing
        m = state.last_metrics
        metrics_block = """
        <div class="card status ok">
          <h2>Últimos resultados (teste)</h2>
          <p>Dice: $(round(m.dice, digits=4))</p>
          <p>IoU: $(round(m.iou, digits=4))</p>
          <p>Acurácia: $(round(m.accuracy, digits=4))</p>
        </div>
        """
    end
    body = """
    <h1>Avaliar modelo</h1>
    $alert
    $metrics_block
    <div class="card">
      <form method="post" action="/evaluate">
        <button class="btn" type="submit">Rodar avaliação no conjunto de teste</button>
      </form>
    </div>
    """
    layout("Avaliar", body)
end

function run_predict!(state::InterfaceState, image_rel::String)
    model, cfg = load_model_for_inference(project_relative(state, joinpath(state.cfg.checkpoint_dir, "best_model.bson")))
    image_path = project_relative(state, image_rel)
    saved_path, _, probs = predict_and_save(model, image_path; cfg = cfg)
    pred_rel = rel_path(state, saved_path)

    comparison_rel = nothing
    gt_path = replace(image_path, r"\.png$" => "_mask.png")
    if isfile(gt_path)
        comparison_path = replace(saved_path, "_pred.png" => "_comparison.png")
        save_comparison(image_path, probs, comparison_path; ground_truth_path = gt_path)
        comparison_rel = rel_path(state, comparison_path)
    end
    return image_rel, pred_rel, comparison_rel
end

function start_training!(state::InterfaceState; quick::Bool = false)
    state.training_status == "running" && return
    state.training_status = "running"
    state.training_message = "Treinamento em andamento..."

    cfg = if quick
        Config(
            dataset_path = state.cfg.dataset_path,
            image_size = state.cfg.image_size,
            train_ratio = state.cfg.train_ratio,
            val_ratio = state.cfg.val_ratio,
            test_ratio = state.cfg.test_ratio,
            random_seed = state.cfg.random_seed,
            epochs = 5,
            batch_size = state.cfg.batch_size,
            learning_rate = state.cfg.learning_rate,
            early_stopping_patience = state.cfg.early_stopping_patience,
            use_gpu = state.cfg.use_gpu,
            max_samples_per_class = 20,
            prediction_threshold = state.cfg.prediction_threshold,
            checkpoint_dir = state.cfg.checkpoint_dir,
            predictions_dir = state.cfg.predictions_dir,
        )
    else
        state.cfg
    end

    @async begin
        try
            checkpoint = run_training!(cfg)
            state.training_status = "done"
            state.training_message = "Treinamento concluído. Modelo salvo em $(rel_path(state, checkpoint))."
        catch err
            state.training_status = "error"
            state.training_message = "Erro no treinamento: $(sprint(showerror, err))"
        end
    end
end

function open_browser(port::Int)
    url = "http://127.0.0.1:$port/"
    try
        if Sys.iswindows()
            run(`cmd /c start $url`; wait=false)
        elseif Sys.isapple()
            run(`open $url`; wait=false)
        else
            run(`xdg-open $url`; wait=false)
        end
    catch
        @info "Abra manualmente no navegador: $url"
    end
end

function serve_interface(; project_root::String = pwd(), port::Int = INTERFACE_PORT, open_browser_flag::Bool = true)
    state = InterfaceState(abspath(project_root))

    handler(req) = try
        uri = HTTP.URI(req.target)
        path = uri.path

        if req.method == "GET" && path == "/"
            return HTTP.Response(200, ["Content-Type" => "text/html; charset=utf-8"], page_home(state))
        elseif req.method == "GET" && path == "/predict"
            if !model_ready(state)
                return HTTP.Response(200, ["Content-Type" => "text/html; charset=utf-8"], page_predict_form(state, "Treine o modelo antes de gerar máscaras."))
            end
            return HTTP.Response(200, ["Content-Type" => "text/html; charset=utf-8"], page_predict_form(state))
        elseif req.method == "POST" && path == "/predict"
            if !model_ready(state)
                return HTTP.Response(200, ["Content-Type" => "text/html; charset=utf-8"], page_predict_form(state, "Treine o modelo antes de gerar máscaras."))
            end
            form = HTTP.parseform(String(req.body))
            image_rel = get(form, "image", [""])[1]
            image_rel, pred_rel, comparison_rel = run_predict!(state, image_rel)
            html = page_predict_result(state, image_rel, pred_rel, comparison_rel)
            return HTTP.Response(200, ["Content-Type" => "text/html; charset=utf-8"], html)
        elseif req.method == "GET" && path == "/train"
            return HTTP.Response(200, ["Content-Type" => "text/html; charset=utf-8"], page_train(state))
        elseif req.method == "POST" && path == "/train"
            form = HTTP.parseform(String(req.body))
            quick = haskey(form, "quick")
            start_training!(state; quick = quick)
            return HTTP.Response(302, ["Location" => "/train"], "")
        elseif req.method == "GET" && path == "/evaluate"
            return HTTP.Response(200, ["Content-Type" => "text/html; charset=utf-8"], page_evaluate(state))
        elseif req.method == "POST" && path == "/evaluate"
            try
                state.last_metrics = run_evaluation!(state.cfg)
                return HTTP.Response(200, ["Content-Type" => "text/html; charset=utf-8"], page_evaluate(state))
            catch err
                msg = sprint(showerror, err)
                return HTTP.Response(200, ["Content-Type" => "text/html; charset=utf-8"], page_evaluate(state, msg))
            end
        elseif req.method == "GET" && path == "/file"
            params = URI.queryparams(URI.query(uri))
            rel = get(params, "path", "")
            isempty(rel) && return HTTP.Response(400, "path ausente")
            full = project_relative(state, rel)
            isfile(full) || return HTTP.Response(404, "arquivo não encontrado")
            bytes = read(full)
            return HTTP.Response(200, ["Content-Type" => "image/png"], bytes)
        else
            return HTTP.Response(404, "Página não encontrada")
        end
    catch err
        msg = sprint(showerror, err)
        return HTTP.Response(500, ["Content-Type" => "text/plain; charset=utf-8"], msg)
    end

    println("Interface web em http://127.0.0.1:$port/")
    println("Pressione Ctrl+C para encerrar.")
    open_browser_flag && open_browser(port)
    HTTP.serve(handler, "127.0.0.1", port)
end
