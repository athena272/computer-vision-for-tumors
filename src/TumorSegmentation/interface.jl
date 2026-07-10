using HTTP
using URIs

const INTERFACE_PORT = 8765

mutable struct InterfaceState
    project_root::String
    cfg::Config
    training_status::String
    training_message::String
    training_percent::Float64
    training_epoch::Int
    training_total_epochs::Int
    training_batch::Int
    training_batches_per_epoch::Int
    training_started_at::Float64
    training_eta::String
    training_val_dice::Union{Nothing, Float64}
    training_loss::Union{Nothing, Float64}
    predict_status::String
    predict_message::String
    predict_percent::Float64
    last_predict_result::Union{Nothing, NamedTuple}
    evaluation_status::String
    evaluation_message::String
    evaluation_percent::Float64
    training_checkpoint_rel::String
    last_metrics::Union{Nothing, NamedTuple}
end

function InterfaceState(project_root::String)
    InterfaceState(
        project_root,
        load_config(),
        "idle",
        "",
        0.0,
        0,
        0,
        0,
        0,
        0.0,
        "",
        nothing,
        nothing,
        "idle",
        "",
        0.0,
        nothing,
        "idle",
        "",
        0.0,
        "",
        nothing,
    )
end

function job_status_label(status::String)
    status == "running" && return "Em andamento"
    status == "done" && return "Concluído"
    status == "error" && return "Erro"
    return "Ocioso"
end

const training_status_label = job_status_label

function training_percent(epoch::Int, total_epochs::Int, batch::Int, batches_per_epoch::Int)
    total_epochs <= 0 && return 0.0
    batches_per_epoch <= 0 && return 0.0
    epoch_fraction = (epoch - 1 + batch / batches_per_epoch) / total_epochs
    return clamp(epoch_fraction * 100, 0.0, 100.0)
end

function format_eta_pt(seconds::Real)
    seconds = max(0, Int(round(seconds)))
    seconds == 0 && return "quase lá"
    if seconds < 60
        return "~$(seconds) s restantes"
    elseif seconds < 3600
        minutes = seconds ÷ 60
        return "~$(minutes) min restantes"
    else
        hours = seconds ÷ 3600
        minutes = (seconds % 3600) ÷ 60
        return "~$(hours) h $(minutes) min restantes"
    end
end

function estimate_eta_pt(started_at::Float64, percent::Float64)
    percent <= 0.5 && return "calculando..."
    elapsed = time() - started_at
    remaining = elapsed * (100 - percent) / percent
    return format_eta_pt(remaining)
end

function reset_training_progress!(state::InterfaceState)
    state.training_percent = 0.0
    state.training_epoch = 0
    state.training_total_epochs = 0
    state.training_batch = 0
    state.training_batches_per_epoch = 0
    state.training_started_at = 0.0
    state.training_eta = ""
    state.training_val_dice = nothing
    state.training_loss = nothing
end

function update_training_progress!(state::InterfaceState, progress)
    if progress.phase == :prepare
        state.training_message = get(progress, :message, "Preparando dataset e carregando imagens...")
        state.training_percent = get(progress, :percent, 2.0)
        return
    end

    if progress.phase == :batch
        state.training_epoch = progress.epoch
        state.training_total_epochs = progress.total_epochs
        state.training_batch = progress.batch
        state.training_batches_per_epoch = progress.batches_per_epoch
        state.training_percent = training_percent(
            progress.epoch,
            progress.total_epochs,
            progress.batch,
            progress.batches_per_epoch,
        )
        state.training_message = "Época $(progress.epoch)/$(progress.total_epochs), lote $(progress.batch)/$(progress.batches_per_epoch)"
        if state.training_started_at > 0
            state.training_eta = estimate_eta_pt(state.training_started_at, state.training_percent)
        end
        return
    end

    if progress.phase in (:epoch, :early_stop)
        state.training_epoch = progress.epoch
        state.training_total_epochs = progress.total_epochs
        state.training_batches_per_epoch = progress.batches_per_epoch
        state.training_batch = progress.batches_per_epoch
        state.training_loss = progress.train_loss
        state.training_val_dice = progress.val_dice
        state.training_percent = training_percent(
            progress.epoch,
            progress.total_epochs,
            progress.batches_per_epoch,
            progress.batches_per_epoch,
        )
        loss_txt = round(progress.train_loss, digits = 4)
        if progress.val_dice !== nothing
            dice_txt = round(progress.val_dice, digits = 4)
            state.training_message = "Época $(progress.epoch)/$(progress.total_epochs) concluída · loss $loss_txt · Dice validação $dice_txt"
        else
            state.training_message = "Época $(progress.epoch)/$(progress.total_epochs) concluída · loss $loss_txt"
        end
        if progress.phase == :early_stop
            state.training_message *= " · parada antecipada"
        end
        if state.training_started_at > 0
            state.training_eta = estimate_eta_pt(state.training_started_at, state.training_percent)
        end
        return
    end

    if progress.phase == :finished
        state.training_percent = 100.0
        state.training_eta = ""
    end
end

function job_display_message(status::String, message::String)
    if !isempty(message)
        return message
    end
    return status == "idle" ? "Aguardando ação." :
           status == "running" ? "Processando..." :
           status == "done" ? "Concluído." : ""
end

function run_background!(f)
    if Threads.nthreads() > 1
        Threads.@spawn f()
    else
        @async f()
    end
    return nothing
end

function log_background_error!(task_name::String, err)
    println(stderr, "\n[$task_name] Erro:")
    showerror(stderr, err, catch_backtrace())
    println(stderr)
end

function format_task_error(err)
    if err isa Base.OutOfMemoryError
        return "Memória insuficiente. Tente o treino rápido ou reduza image_size em config/default.toml."
    end
    return sprint(showerror, err)
end

function progress_block_html(percent::Real; eta::String = "", panel_id::String = "")
    pct = round(clamp(percent, 0, 100), digits = 1)
    display_pct = pct <= 0 ? 0.0 : pct
    bar_id = isempty(panel_id) ? "" : " id=\"$(panel_id)-bar\""
    pct_id = isempty(panel_id) ? "" : " id=\"$(panel_id)-pct\""
    eta_id = isempty(panel_id) ? "" : " id=\"$(panel_id)-eta\""
    eta_block = isempty(eta) ? "" : "<span$(eta_id)>$eta</span>"
    return """
    <div class="progress-wrap" aria-label="Progresso">
      <div class="progress-bar"$bar_id style="width: $(display_pct)%"></div>
    </div>
    <div class="progress-meta">
      <span><strong$pct_id>$(pct)%</strong></span>
      $eta_block
    </div>
    """
end

function job_progress_html(status::String, percent::Real, message::String; eta::String = "", panel_id::String = "")
    status_label = job_status_label(status)
    status_class = status == "error" ? "status err" :
                   status == "done" ? "status ok" : "status"
    display_message = job_display_message(status, message)
    wrapper_open = isempty(panel_id) ?
        "<div class=\"job-progress\">" :
        "<div class=\"job-progress\" id=\"$panel_id\">"
    status_id = isempty(panel_id) ? "" : " id=\"$(panel_id)-status\""
    msg_id = isempty(panel_id) ? "" : " id=\"$(panel_id)-message\""
    return """
    $wrapper_open
    <div$status_id class="$status_class">Status: $(html_escape(status_label))</div>
    $(progress_block_html(percent; eta = eta, panel_id = panel_id))
    <p$msg_id style="margin-top:12px">$(html_escape(display_message))</p>
    </div>
    """
end

function auto_refresh_script(seconds::Int)
    seconds <= 0 && return ""
    # Reload após a página carregar (evita cancelar a resposta HTTP em andamento).
    return """
    <script>
    (function () {
      var delay = $seconds * 1000;
      function reload() { location.reload(); }
      if (document.readyState === "complete") setTimeout(reload, delay);
      else window.addEventListener("load", function () { setTimeout(reload, delay); });
    })();
    </script>
    """
end

function _keep_http_log(log)
    log.level <= Info && return true
    msg = log.message
    msg isa String || return true
    if occursin("handle_connection handler error", msg) &&
       (occursin("ECANCELED", msg) || occursin("ECONNRESET", msg) || occursin("ECONNABORTED", msg))
        return false
    end
    return true
end

function html_response(body::String)
    return HTTP.Response(200, [
        "Content-Type" => "text/html; charset=utf-8",
        "Connection" => "close",
        "Cache-Control" => "no-store",
    ], body)
end

function redirect_response(location::String)
    return HTTP.Response(302, ["Location" => location, "Connection" => "close"], "")
end

function file_img_tag(rel_path::String; alt::String = "Imagem")
    src = "/file?path=$(URIs.escapeuri(rel_path))"
    return """<img src="$src" alt="$(html_escape(alt))">"""
end

function predict_result_html(result)
    image_rel = result.image_rel
    pred_rel = result.pred_rel
    comparison_rel = get(result, :comparison_rel, nothing)
    gt_rel = get(result, :gt_rel, nothing)

    comparison_hero = ""
    if comparison_rel !== nothing
        comparison_hero = """
        <div class="card result-hero">
          <h2>Comparação visual</h2>
          <p class="muted">Ultrassom, máscara prevista e máscara real (quando existir)</p>
          $(file_img_tag(comparison_rel; alt = "Comparação"))
        </div>
        """
    end

    gt_panel = ""
    if gt_rel !== nothing
        gt_panel = """
        <div>
          <h3>Máscara real</h3>
          $(file_img_tag(gt_rel; alt = "Máscara real"))
        </div>
        """
    end

    return """
    <h2 class="section-title">Resultado na tela</h2>
    $comparison_hero
    <div class="card images">
      <div>
        <h3>Entrada</h3>
        $(file_img_tag(image_rel; alt = "Entrada"))
      </div>
      <div>
        <h3>Máscara prevista</h3>
        $(file_img_tag(pred_rel; alt = "Predição"))
      </div>
      $gt_panel
    </div>
    """
end

function metrics_explanation_table_html(; dice = nothing, iou = nothing, acc_pct = nothing)
    show_values = dice !== nothing
    value_header = show_values ? "<th>Seu valor</th>" : ""
    dice_value = show_values ? "<td><strong>$dice</strong></td>" : ""
    iou_value = show_values ? "<td><strong>$iou</strong></td>" : ""
    acc_value = show_values ? "<td><strong>$acc_pct%</strong></td>" : ""
    dice_approx = show_values ? round(Float64(dice), digits = 2) : 0.50
    acc_example = show_values ? acc_pct : 94.1
    return """
    <table class="metrics-table">
      <thead>
        <tr><th>Métrica</th>$value_header<th>Significado</th></tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Dice</strong></td>
          $dice_value
          <td>Sobreposição entre máscara prevista e real. Escala 0–1; <strong>quanto mais perto de 1, melhor</strong>. ~$dice_approx indica que o modelo acerta boa parte do tumor, mas ainda erra bordas ou áreas.</td>
        </tr>
        <tr>
          <td><strong>IoU</strong></td>
          $iou_value
          <td>Interseção ÷ união das máscaras. Mais rigorosa que Dice; <strong>costuma ser menor</strong>. ~0.40 é razoável para segmentação médica com dataset pequeno.</td>
        </tr>
        <tr>
          <td><strong>Acurácia</strong></td>
          $acc_value
          <td>% de pixels corretos (tumor + fundo). <strong>Pode enganar</strong>: a maior parte da imagem é fundo, então $acc_example% de acurácia com Dice ~$dice_approx é comum — o modelo acerta o fundo e ainda erra parte do tumor.</td>
        </tr>
      </tbody>
    </table>
    """
end

function metrics_result_html(m)
    acc_pct = round(100 * m.accuracy, digits = 1)
    dice = round(m.dice, digits = 4)
    iou = round(m.iou, digits = 4)
    return """
    <div class="metrics-grid">
      <div class="metric-box">
        <span class="metric-label">Dice</span>
        <strong>$dice</strong>
      </div>
      <div class="metric-box">
        <span class="metric-label">IoU</span>
        <strong>$iou</strong>
      </div>
      <div class="metric-box">
        <span class="metric-label">Acurácia</span>
        <strong>$acc_pct%</strong>
      </div>
    </div>
    <h3 class="section-title">O que significam suas métricas</h3>
    $(metrics_explanation_table_html(; dice, iou, acc_pct))
    <p class="muted" style="margin-top:16px">Médias no conjunto de teste (todas as classes). No treino, o Dice de validação usa só imagens com tumor; aqui entram também imagens <em>normal</em>, o que tende a baixar Dice e IoU. Para julgar a segmentação do tumor, priorize Dice e IoU.</p>
    """
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
        select, input[type=text], input[type=file] { width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #334155; background: #0f172a; color: #e2e8f0; box-sizing: border-box; }
        input[type=file]::file-selector-button { margin-right: 12px; border: 0; border-radius: 6px; padding: 8px 12px; background: #334155; color: #e2e8f0; cursor: pointer; }
        .images { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
        .images img { width: 100%; border-radius: 8px; background: #000; }
        .muted { color: #94a3b8; }
        .status { padding: 10px 12px; border-radius: 8px; background: #334155; }
        .ok { background: #14532d; }
        .err { background: #7f1d1d; }
        nav a { margin-right: 12px; }
        code { background: #334155; padding: 2px 6px; border-radius: 4px; }
        .progress-wrap { margin-top: 16px; height: 16px; border-radius: 999px; background: #334155; overflow: hidden; border: 1px solid #475569; }
        .progress-bar { height: 100%; background: linear-gradient(90deg, #2563eb, #38bdf8); transition: width 0.4s ease; }
        .progress-meta { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; color: #cbd5e1; font-size: 0.95rem; }
        .result-hero { margin: 16px 0; }
        .result-hero img { width: 100%; border-radius: 12px; background: #000; border: 1px solid #334155; }
        .metrics-grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); margin-top: 12px; }
        .metric-box { background: #0f172a; border-radius: 10px; padding: 14px; text-align: center; border: 1px solid #334155; }
        .metric-box strong { display: block; font-size: 1.4rem; color: #f8fafc; margin-top: 4px; }
        .metric-label { font-weight: 600; }
        .metrics-table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.92rem; }
        .metrics-table th, .metrics-table td { border: 1px solid #334155; padding: 10px 12px; text-align: left; vertical-align: top; }
        .metrics-table th { background: #0f172a; color: #e2e8f0; }
        .metrics-table td:first-child { white-space: nowrap; width: 90px; }
        .section-title { margin: 24px 0 8px; font-size: 1.1rem; color: #e2e8f0; }
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

function parse_form_body(body)
    return URIs.queryparams(String(body))
end

function save_uploaded_image!(state::InterfaceState, req)
    content_type = HTTP.header(req, "Content-Type", "")
    occursin("multipart/form-data", content_type) ||
        error("Use o botão de escolher arquivo para enviar uma imagem PNG.")

    parts = HTTP.parse_multipart_form(req)
    (parts === nothing || isempty(parts)) && error("Nenhuma imagem foi enviada.")

    for part in parts
        part.name == "image" || continue
        filename = something(part.filename, "upload.png")
        endswith(lowercase(filename), ".png") || error("Envie um arquivo PNG de ultrassom.")

        upload_dir = project_relative(state, joinpath(state.cfg.predictions_dir, "uploads"))
        mkpath(upload_dir)
        safe_name = replace(basename(filename), r"[^\w\.\- ()]" => "_")
        dest = joinpath(upload_dir, safe_name)
        write(dest, read(part))
        return rel_path(state, dest)
    end

    error("Campo de imagem não encontrado no formulário.")
end

function list_example_images(state::InterfaceState)
    samples = list_samples(state.cfg.dataset_path; max_per_class = 8)
    return [rel_path(state, s.image_path) for s in samples]
end

function model_ready(state::InterfaceState)
    return isfile(project_relative(state, joinpath(state.cfg.checkpoint_dir, "best_model.bson")))
end

function page_home(state::InterfaceState)
    ready = model_ready(state)
    status = ready ? "Modelo treinado encontrado." : "Ainda não há modelo treinado."
    status_class = ready ? "status ok" : "status"

    last_result_block = ""
    if state.last_predict_result !== nothing
        r = state.last_predict_result
        last_result_block = """
        <div class="card">
          <h2>Última predição</h2>
          $(predict_result_html(r))
          <a class="btn secondary" href="/predict" style="margin-top:12px">Gerar outra máscara</a>
        </div>
        """
    end

    body = """
    <h1>Segmentação de Tumores (U-Net)</h1>
    <p class="muted">Treine, gere máscaras e avalie o modelo — tudo no navegador, sem abrir pastas no computador.</p>
    <div class="$status_class">$status</div>
    <div class="grid">
      <div class="card">
        <h2>1. Treinar</h2>
        <p class="muted">Ensina a rede com as imagens do dataset BUSI.</p>
        <a class="btn" href="/train">Abrir treino</a>
      </div>
      <div class="card">
        <h2>2. Gerar máscara</h2>
        <p class="muted">Escolhe uma imagem e vê o resultado aqui mesmo.</p>
        <a class="btn" href="/predict">Abrir predição</a>
      </div>
      <div class="card">
        <h2>3. Avaliar</h2>
        <p class="muted">Métricas Dice, IoU e acurácia no conjunto de teste.</p>
        <a class="btn secondary" href="/evaluate">Abrir avaliação</a>
      </div>
    </div>
    $last_result_block
    """
    layout("U-Net BUSI", body)
end

function page_predict(state::InterfaceState, message::String = "")
    alert = isempty(message) ? "" : "<div class=\"status err\">$(html_escape(message))</div>"
    running = state.predict_status == "running"
    disabled = running ? "disabled" : ""
    predict_status = running ? "running" : state.predict_status

    progress_block = """
    <div class="card">
      $(job_progress_html(predict_status, state.predict_percent, state.predict_message; panel_id = "progress-predict"))
    </div>
    """

    result_block = ""
    if !running && state.last_predict_result !== nothing
        result_block = """
        <div class="card">
          $(predict_result_html(state.last_predict_result))
        </div>
        """
    end

    form_block = running ? "" : """
    <div class="card">
      <form method="post" action="/predict" enctype="multipart/form-data">
        <label for="image">Imagem de ultrassom (PNG)</label>
        <input id="image" type="file" name="image" accept="image/png,.png" required>
        <p class="muted" style="margin-top:16px">Escolha um arquivo do seu computador. O resultado aparece abaixo nesta página.</p>
        <button class="btn" type="submit" style="margin-top:12px" data-disable-while-running $disabled>Gerar máscara</button>
      </form>
    </div>
    """

    refresh_script = auto_refresh_script(running ? 2 : 0)

    body = """
    <h1>Gerar máscara</h1>
    <p class="muted">Escolha uma imagem PNG do seu computador. A barra de progresso atualiza automaticamente durante o processamento.</p>
    $alert
    $progress_block
    $form_block
    $result_block
    $refresh_script
    """
    layout("Gerar máscara", body)
end

function page_train(state::InterfaceState)
    quick_checked = state.cfg.max_samples_per_class > 0 ? "checked" : ""
    disabled = state.training_status == "running" ? "disabled" : ""
    eta = isempty(state.training_eta) && state.training_status == "running" ? "calculando..." : state.training_eta
    cfg = state.cfg

    success_block = ""
    if state.training_status == "done"
        success_block = """
        <div class="card status ok" style="margin-top:16px">
          <h2>Modelo pronto</h2>
          <p>O treinamento terminou. Você já pode gerar máscaras na interface.</p>
          <a class="btn" href="/predict" style="margin-top:12px">Gerar máscara agora</a>
        </div>
        """
    end

    running = state.training_status == "running"
    form_block = running ? "" : """
      <form method="post" action="/train" style="margin-top:16px">
        <label><input type="checkbox" name="quick" value="1" $quick_checked> Treino rápido (40 imagens/classe, 256×256, loss ponderada; GPU se disponível)</label>
        <button class="btn warn" type="submit" $disabled>Iniciar treinamento</button>
      </form>
    """
    running_note = running ? """<p class="muted" style="margin-top:12px">Atualizando a cada 2 segundos...</p>""" : ""
    refresh_script = auto_refresh_script(running ? 2 : 0)

    body = """
    <h1>Treinar modelo</h1>
    <p class="muted">O treino roda em segundo plano com barra de progresso e tempo estimado. Em CPU, pode levar bastante tempo.</p>
    <div class="card" style="margin-bottom:12px">
      <h2 style="margin-top:0;font-size:1rem">O que cada modo faz</h2>
      <ul class="muted" style="margin:0;padding-left:20px">
        <li><strong>Padrão (sem marcar):</strong> dataset completo em $(cfg.image_size)×$(cfg.image_size), até $(cfg.epochs) épocas, loss ponderada, Dice só em imagens com tumor, early stopping paciente (mín. ~8 épocas). Com GPU costuma ser bem mais rápido que em CPU.</li>
        <li><strong>Rápido:</strong> 40 imagens/classe, 20 épocas, 256×256, loss ponderada — bom para testar; qualidade menor que o treino completo.</li>
      </ul>
    </div>
    <div class="card">
      $(job_progress_html(state.training_status, state.training_percent, state.training_message; eta = eta, panel_id = "progress-train"))
      $running_note
      $form_block
    </div>
    $success_block
    $refresh_script
    """
    layout("Treinar", body)
end

function page_evaluate(state::InterfaceState, error_msg::String = "")
    alert = isempty(error_msg) ? "" : "<div class=\"status err\">$(html_escape(error_msg))</div>"

    metrics_block = ""
    if state.last_metrics !== nothing && state.evaluation_status != "running"
        metrics_block = """
        <div class="card status ok">
          <h2>Resultado no conjunto de teste</h2>
          $(metrics_result_html(state.last_metrics))
        </div>
        """
    else
        metrics_block = """
        <div class="card">
          <h2>O que significam as métricas</h2>
          <p class="muted" style="margin-top:0">Após rodar a avaliação, seus valores aparecem na tabela abaixo.</p>
          $(metrics_explanation_table_html())
        </div>
        """
    end

    progress_block = """
    <div class="card">
      $(job_progress_html(state.evaluation_status, state.evaluation_percent, state.evaluation_message; panel_id = "progress-evaluate"))
    </div>
    """

    running = state.evaluation_status == "running"
    disabled = running ? "disabled" : ""
    refresh_script = auto_refresh_script(running ? 2 : 0)

    body = """
    <h1>Avaliar modelo</h1>
    <p class="muted">As métricas aparecem aqui ao terminar — sem precisar abrir arquivos no disco.</p>
    $alert
    $progress_block
    $metrics_block
    <div class="card">
      <form method="post" action="/evaluate">
        <button class="btn" type="submit" $disabled>Rodar avaliação no conjunto de teste</button>
      </form>
    </div>
    $refresh_script
    """
    layout("Avaliar", body)
end

function run_predict!(state::InterfaceState, image_rel::String; on_progress = nothing)
    notify(progress) = on_progress !== nothing && on_progress(progress)

    notify((; phase = :load_model, percent = 15.0, message = "Carregando modelo treinado..."))
    model, cfg = load_model_for_inference(project_relative(state, joinpath(state.cfg.checkpoint_dir, "best_model.bson")))

    notify((; phase = :infer, percent = 45.0, message = "Gerando máscara com a U-Net..."))
    image_path = project_relative(state, image_rel)
    saved_path, _, probs = predict_and_save(model, image_path; cfg = cfg)
    pred_rel = rel_path(state, saved_path)

    comparison_rel = nothing
    gt_rel = nothing
    gt_path = replace(image_path, r"\.png$" => "_mask.png")
    if isfile(gt_path)
        gt_rel = rel_path(state, gt_path)
        notify((; phase = :save, percent = 80.0, message = "Montando comparação visual..."))
        comparison_path = replace(saved_path, "_pred.png" => "_comparison.png")
        save_comparison(
            image_path,
            probs,
            comparison_path;
            ground_truth_path = gt_path,
            target_size = cfg.image_size,
        )
        comparison_rel = rel_path(state, comparison_path)
    else
        notify((; phase = :save, percent = 80.0, message = "Finalizando máscara prevista..."))
    end

    notify((; phase = :done, percent = 100.0, message = "Pronto! Veja o resultado abaixo."))
    return (
        image_rel = image_rel,
        pred_rel = pred_rel,
        comparison_rel = comparison_rel,
        gt_rel = gt_rel,
    )
end

function start_predict!(state::InterfaceState, image_rel::String)
    state.predict_status == "running" && return
    state.predict_status = "running"
    state.predict_message = "Preparando predição..."
    state.predict_percent = 5.0
    state.last_predict_result = nothing

    on_progress = progress -> begin
        state.predict_percent = progress.percent
        state.predict_message = progress.message
    end

    run_background!() do
        try
            state.last_predict_result = run_predict!(state, image_rel; on_progress = on_progress)
            state.predict_status = "done"
            state.predict_percent = 100.0
            state.predict_message = "Máscara gerada com sucesso."
        catch err
            state.predict_status = "error"
            state.predict_message = "Erro ao gerar máscara: $(format_task_error(err))"
            log_background_error!("Predição", err)
        end
    end
end

function start_evaluation!(state::InterfaceState)
    state.evaluation_status == "running" && return
    state.evaluation_status = "running"
    state.evaluation_message = "Preparando avaliação..."
    state.evaluation_percent = 5.0

    on_progress = progress -> begin
        state.evaluation_percent = progress.percent
        state.evaluation_message = progress.message
    end

    run_background!() do
        try
            state.last_metrics = run_evaluation!(state.cfg; on_progress = on_progress)
            state.evaluation_status = "done"
            state.evaluation_percent = 100.0
            state.evaluation_message = "Avaliação concluída."
        catch err
            state.evaluation_status = "error"
            state.evaluation_message = "Erro na avaliação: $(format_task_error(err))"
            log_background_error!("Avaliação", err)
        end
    end
end

function start_training!(state::InterfaceState; quick::Bool = false)
    state.training_status == "running" && return
    state.training_status = "running"
    reset_training_progress!(state)
    state.training_started_at = time()
    state.training_message = "Preparando treinamento..."
    state.training_percent = 3.0

    cfg = if quick
        Config(
            dataset_path = state.cfg.dataset_path,
            image_size = 256,
            train_ratio = state.cfg.train_ratio,
            val_ratio = state.cfg.val_ratio,
            test_ratio = state.cfg.test_ratio,
            random_seed = state.cfg.random_seed,
            epochs = 20,
            batch_size = 4,
            learning_rate = 3e-4,
            early_stopping_patience = 6,
            use_gpu = state.cfg.use_gpu,
            max_samples_per_class = 40,
            prediction_threshold = 0.35,
            checkpoint_dir = state.cfg.checkpoint_dir,
            predictions_dir = state.cfg.predictions_dir,
            tumor_sample_weight = 4.0,
            pos_pixel_weight = 20.0,
        )
    else
        # Treino completo: herda default.toml, mas garante loss ponderada e threshold estáveis
        base = state.cfg
        Config(
            dataset_path = base.dataset_path,
            image_size = base.image_size,
            train_ratio = base.train_ratio,
            val_ratio = base.val_ratio,
            test_ratio = base.test_ratio,
            random_seed = base.random_seed,
            epochs = base.epochs,
            batch_size = base.batch_size,
            learning_rate = base.learning_rate,
            early_stopping_patience = max(base.early_stopping_patience, 10),
            use_gpu = base.use_gpu,
            max_samples_per_class = 0,
            prediction_threshold = min(base.prediction_threshold, 0.35),
            checkpoint_dir = base.checkpoint_dir,
            predictions_dir = base.predictions_dir,
            tumor_sample_weight = max(base.tumor_sample_weight, 4.0),
            pos_pixel_weight = max(base.pos_pixel_weight, 20.0),
        )
    end
    state.training_total_epochs = cfg.epochs

    on_progress = progress -> update_training_progress!(state, progress)

    sample_count = cfg.max_samples_per_class > 0 ? "~$(3 * cfg.max_samples_per_class)" : "dataset completo"
    println("[Treinamento] Iniciando ($sample_count, $(cfg.epochs) épocas, $(cfg.image_size)px)...")

    run_background!() do
        try
            checkpoint = run_training!(cfg; on_progress = on_progress)
            state.training_status = "done"
            state.training_percent = 100.0
            state.training_eta = ""
            state.training_checkpoint_rel = rel_path(state, checkpoint)
            state.training_message = "Treinamento concluído com sucesso."
            println("[Treinamento] Concluído: $(state.training_checkpoint_rel)")
        catch err
            state.training_status = "error"
            state.training_eta = ""
            state.training_message = "Erro no treinamento: $(format_task_error(err))"
            log_background_error!("Treinamento", err)
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
            return html_response(page_home(state))
        elseif req.method == "GET" && path == "/predict"
            if !model_ready(state)
                return html_response(page_predict(state, "Treine o modelo antes de gerar máscaras."))
            end
            return html_response(page_predict(state))
        elseif req.method == "POST" && path == "/predict"
            if !model_ready(state)
                return html_response(page_predict(state, "Treine o modelo antes de gerar máscaras."))
            end
            try
                image_rel = save_uploaded_image!(state, req)
                start_predict!(state, image_rel)
                return redirect_response("/predict")
            catch err
                msg = sprint(showerror, err)
                return html_response(page_predict(state, msg))
            end
        elseif req.method == "GET" && path == "/train"
            return html_response(page_train(state))
        elseif req.method == "POST" && path == "/train"
            form = parse_form_body(req.body)
            quick = haskey(form, "quick")
            start_training!(state; quick = quick)
            return redirect_response("/train")
        elseif req.method == "GET" && path == "/evaluate"
            return html_response(page_evaluate(state))
        elseif req.method == "POST" && path == "/evaluate"
            start_evaluation!(state)
            return redirect_response("/evaluate")
        elseif req.method == "GET" && path == "/file"
            params = URIs.queryparams(uri.query)
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
    if Threads.nthreads() == 1
        println("Aviso: use 'julia --project -t auto scripts/interface.jl' para o treino não bloquear a barra de progresso.")
    else
        println("Threads Julia: $(Threads.nthreads())")
    end
    println("Pressione Ctrl+C para encerrar.")
    open_browser_flag && open_browser(port)
    filtered_logger = ActiveFilteredLogger(_keep_http_log, current_logger())
    with_logger(filtered_logger) do
        HTTP.serve(handler, "127.0.0.1", port; verbose = false)
    end
end
