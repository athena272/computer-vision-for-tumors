# Guia de Julia para este projeto

Se você nunca programou em Julia, este guia ajuda a ler o código deste repositório com mais confiança.

---

## O que é Julia?

Julia é uma linguagem criada para **ciência de dados** e **computação numérica**. É rápida (compilada) e tem sintaxe parecida com Python/MATLAB em alguns pontos.

Neste projeto usamos Julia **1.12+**.

---

## Estrutura de um pacote Julia

```
computer-vision-for-tumors/
├── Project.toml      ← “receita” do projeto (nome, dependências)
├── Manifest.toml     ← versões exatas instaladas
├── src/              ← código-fonte
├── test/             ← testes automatizados
└── scripts/          ← programas executáveis
```

### `Project.toml`

Lista as bibliotecas que o projeto precisa (Flux, Images, etc.).

### Ativar o ambiente

Sempre use `--project` na raiz:

```bash
julia --project scripts/train.jl
```

Ou dentro do REPL:

```julia
using Pkg
Pkg.activate(".")
```

### Instalar dependências

```julia
using Pkg
Pkg.instantiate()
```

---

## Conceitos básicos que você verá no código

### 1. `using` — importar bibliotecas

```julia
using Flux
using Images
```

Equivale a “traga as ferramentas desta biblioteca para eu usar”.

### 2. `module` — organizar código

O arquivo `src/TumorSegmentation.jl` define o módulo `TumorSegmentation`, que agrupa funções relacionadas.

### 3. `struct` — definir um tipo de dados

```julia
struct Sample
    image_path::String
    mask_path::String
    label::String
end
```

Cada `Sample` guarda caminhos de imagem, máscara e a classe (`benign`, `malignant`, `normal`).

### 4. `function` — criar funções

```julia
function preprocess_image(img; target_size::Int = 256)
    # ...
end
```

- `img` é argumento obrigatório
- `target_size` é opcional (valor padrão 256)

### 5. `::Tipo` — dizer o tipo esperado

```julia
function resize_array(arr::AbstractMatrix{<:Real}, target_size::Int)
```

Ajuda o Julia a gerar código eficiente e detectar erros cedo.

### 6. `export` — tornar funções públicas

No módulo principal, exportamos funções para usar assim:

```julia
using TumorSegmentation
samples = list_samples("docs/material/Dataset_BUSI_with_GT")
```

---

## Arrays e imagens

### Formato Flux (importante!)

Redes no Flux esperam tensores 4D:

```
(altura, largura, canais, batch)
```

Exemplo: batch de 4 imagens 256×256 em escala de cinza:

```
(256, 256, 1, 4)
```

O **último** índice é sempre o batch.

### Operação com ponto (`.>`)

```julia
mask = Float32.(probabilities .>= 0.5)
```

O ponto antes do operador aplica a operação **elemento a elemento** em arrays.

---

## Flux.jl em 3 ideias

### 1. Modelo = função composável

```julia
model = build_unet(in_channels=1, out_channels=1)
y = model(x)
```

### 2. Treino com gradientes automáticos

```julia
loss, grads = Flux.withgradient(model) do m
    combined_loss(m(x), y_true)
end
Flux.update!(opt, model, grads[1])
```

O Flux calcula **como ajustar os pesos** para reduzir o erro.

### 3. Otimizador

```julia
opt = Flux.setup(Adam(0.001), model)
```

`Adam` é um algoritmo popular de aprendizado.

---

## Como ler o fluxo do projeto

### Carregar configuração

```julia
cfg = load_config()  # lê config/default.toml
```

### Listar amostras do dataset

```julia
samples = list_samples(cfg.dataset_path)
splits = split_samples(samples; train_ratio=0.7, val_ratio=0.15, seed=42)
```

### Treinar

```julia
loaders = create_dataloaders(splits; image_size=256, batch_size=4)
model = build_unet()
model = train_model!(model, loaders.train, loaders.val; cfg=cfg)
```

### Prever

```julia
model, cfg = load_model_for_inference("outputs/checkpoints/best_model.bson")
predict_and_save(model, "caminho/para/imagem.png"; cfg=cfg)
```

---

## Scripts vs módulo

| Local | Papel |
|-------|------|
| `src/TumorSegmentation/` | lógica reutilizável, testável |
| `scripts/train.jl` | ponto de entrada para treinar |
| `scripts/predict.jl` | ponto de entrada para inferência |
| `test/` | garante que funções continuam corretas |

Boa prática: **regras de negócio no módulo**, **orquestração nos scripts**.

---

## Testes automatizados

```bash
julia --project -e "include(\"test/runtests.jl\")"
```

Usamos `@test` do Julia:

```julia
@test dice_coefficient(perfect, perfect) ≈ 1.0
```

Se algo quebrar, o teste falha e avisa.

---

## Erros comuns e dicas

### “Package not found”

Rode `Pkg.instantiate()` na raiz do projeto.

### Treino muito lento

- Normal em CPU com imagens 256×256.
- Reduza `epochs` ou `max_samples_per_class` em `config/default.toml` para experimentar.

### Resultado da máscara ruim

- Deep learning precisa de treino suficiente.
- Verifique métricas com `scripts/evaluate.jl`.
- Compare visualmente com `*_comparison.png`.

---

## Próximos passos sugeridos

1. Leia [guia-unet.md](guia-unet.md) para entender a arquitetura.
2. Execute `scripts/train.jl` com poucas épocas.
3. Rode `scripts/predict.jl` em uma imagem que você já conhece.
4. Explore o Lab3 manual em `docs/material/Laboratorios_U_Net/` para ver a matemática por trás.

---

## Onde pedir ajuda

- Documentação Julia: https://docs.julialang.org/
- Flux: https://fluxml.ai/Flux.jl/stable/
- Material do curso neste repositório: `docs/material/`
