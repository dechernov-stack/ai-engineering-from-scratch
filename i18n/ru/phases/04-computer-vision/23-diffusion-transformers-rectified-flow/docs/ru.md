# Diffusion Transformers и Rectified Flow

> U-Net — это не секрет диффузии. Замените её трансформером, замените расписание шума на прямолинейный поток — и вдруг у вас получаются SD3, FLUX и все модели text-to-image 2026 года.

**Type:** Learn + Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 10 (Diffusion DDPM), Phase 4 Lesson 14 (ViT), Phase 7 Lesson 02 (Self-Attention)
**Time:** ~75 минут

## Цели обучения

- Проследить эволюцию от U-Net DDPM (урок 10) к Diffusion Transformer (DiT), MMDiT (SD3) и DiT с одним и двумя потоками (FLUX)
- Объяснить rectified flow: почему прямолинейная траектория между шумом и данными позволяет моделям сэмплировать за 20 шагов вместо 1000
- Реализовать небольшой блок DiT и обучающий цикл rectified flow — оба менее чем на 100 строк
- Различать варианты моделей (SD3, FLUX.1-dev, FLUX.1-schnell, Z-Image, Qwen-Image) по архитектуре, числу параметров и лицензированию

## Проблема

В уроке 10 мы построили DDPM с U-Net в качестве денойзера. Этот рецепт доминировал в 2020-2023 годах: U-Net + beta-расписание + функция потерь на предсказание шума. Он дал нам Stable Diffusion 1.5, 2.1 и DALL-E 2.

Каждая современная (2026 года) модель text-to-image уровня state-of-the-art давно ушла от этого подхода. Stable Diffusion 3, FLUX, SD4, Z-Image, Qwen-Image, Hunyuan-Image — ни одна из них не использует U-Net. Они используют Diffusion Transformers (DiT). SD3 и FLUX также заменяют расписание шума DDPM на rectified flow, которое выпрямляет путь от шума к данным и позволяет делать инференс за 1-4 шага с помощью consistency- или дистиллированных вариантов.

Этот сдвиг важен, потому что именно он сделал генерацию изображений на основе диффузии управляемой, точно следующей промпту (SD3/SD4 решили проблему рендеринга текста) и достаточно быстрой для продакшна. Понимание DiT + rectified flow — это понимание стека генерации изображений 2026 года.

## Концепция

### От U-Net к трансформеру

```mermaid
flowchart LR
    subgraph UNET["DDPM U-Net (2020)"]
        U1["Conv encoder"] --> U2["Conv bottleneck"] --> U3["Conv decoder"]
    end
    subgraph DIT["DiT (2023)"]
        D1["Patch embed"] --> D2["Transformer blocks"] --> D3["Unpatchify"]
    end
    subgraph MMDIT["MMDiT (SD3, 2024)"]
        M1["Text stream"] --> M3["Joint attention<br/>(separate weights per modality)"]
        M2["Image stream"] --> M3
    end
    subgraph FLUX["FLUX (2024)"]
        F1["Double-stream blocks<br/>(text + image separate)"] --> F2["Single-stream blocks<br/>(concat + shared weights)"]
    end

    style UNET fill:#e5e7eb,stroke:#6b7280
    style DIT fill:#dbeafe,stroke:#2563eb
    style MMDIT fill:#fef3c7,stroke:#d97706
    style FLUX fill:#dcfce7,stroke:#16a34a
```

- **DiT** (Peebles & Xie, 2023) — замена U-Net на ViT-подобный трансформер, работающий на патчах латентов. Кондиционирование через adaptive layer norm (AdaLN).
- **MMDiT** (SD3, Esser et al., 2024) — два потока с отдельными весами для текстовых и графических токенов, которые разделяют совместное внимание (joint attention).
- **FLUX** (Black Forest Labs, 2024) — первые N блоков двухпотоковые, как в SD3, последующие блоки объединяют потоки и разделяют веса (однопотоковый режим) ради эффективности при большей глубине.
- **Z-Image** (2025) — эффективный однопотоковый DiT с 6 млрд параметров, бросающий вызов идее «масштаб любой ценой».

### Rectified flow в одном абзаце

DDPM определяет прямой процесс как зашумлённый SDE, в котором `x_t` постепенно искажается. Выученный обратный процесс — это второй SDE, решаемый за 1000 маленьких шагов.

Rectified flow определяет **прямолинейную** интерполяцию между чистыми данными и чистым шумом:

```
x_t = (1 - t) * x_0 + t * epsilon,     t in [0, 1]
```

Сеть обучается предсказывать скорость `v_theta(x_t, t) = epsilon - x_0` — направление вдоль прямолинейного пути от чистых данных к шуму (`dx_t/dt`). Во время сэмплирования эту скорость интегрируют в обратном направлении, чтобы шагать от шума к данным. Получающееся ОДУ намного ближе к прямой линии, поэтому для сэмплирования нужно значительно меньше шагов интегрирования.

SD3 называет это **Rectified Flow Matching**. FLUX, Z-Image и большинство моделей 2026 года используют ту же целевую функцию. Типичный инференс: 20-30 шагов Эйлера (детерминированных) против 50+ шагов DDIM в старом режиме DDPM. Дистиллированные варианты / turbo / schnell / LCM сокращают это до 1-4 шагов.

### Кондиционирование через AdaLN

DiT кондиционируются на временной шаг и класс/текст через **adaptive layer norm**: `scale` и `shift` предсказываются из вектора условия и применяются после LayerNorm. Это гораздо чище, чем модуляция в стиле FiLM в U-Net, и является стандартом в каждом современном DiT.

```
cond -> MLP -> (scale, shift, gate)
norm(x) * (1 + scale) + shift, then residual add * gate
```

### Текстовые энкодеры в SD3 и FLUX

- **SD3** использует три текстовых энкодера: две модели CLIP + T5-XXL. Эмбеддинги конкатенируются и подаются в графический поток как текстовое условие.
- **FLUX** использует один CLIP-L + T5-XXL.
- **Qwen-Image / Z-Image** используют собственные текстовые энкодеры, согласованные со своими базовыми LLM.

Текстовый энкодер — одна из главных причин, почему SD3/FLUX настолько лучше понимают промпты, чем SD1.5. Один только T5-XXL содержит 4,7 млрд параметров.

### Classifier-free guidance по-прежнему работает

Rectified flow меняет сэмплер, а не кондиционирование. Classifier-free guidance (отбрасывание текста с вероятностью 10% во время обучения, смешивание условных и безусловных предсказаний при инференсе) работает с rectified flow точно так же. Большинство моделей 2026 года используют масштаб guidance 3,5-5 — ниже, чем 7,5 у SD1.5, потому что модели с rectified flow по умолчанию точнее следуют промпту.

### Consistency, Turbo, Schnell, LCM

Четыре названия для одной и той же идеи: дистиллировать медленную многошаговую модель в быструю малошаговую.

- **LCM (Latent Consistency Model)** — обучение студента предсказывать финальное значение `x_0` из любого промежуточного `x_t` за один шаг.
- **SDXL Turbo / FLUX schnell** — модели на 1-4 шага, обученные с помощью adversarial diffusion distillation.
- **SD Turbo** — Consistency Models в стиле OpenAI, адаптированные для латентной диффузии.

Продакшн-развёртывание любой новой модели обычно включает как чекпойнт «полного качества», так и вариант «turbo / schnell». Schnell («быстрый» по-немецки, конвенция Black Forest Labs) работает за 1-4 шага и подходит для пайплайнов реального времени.

### Ландшафт моделей в 2026 году

| Модель | Размер | Архитектура | Лицензия |
|-------|------|--------------|---------|
| Stable Diffusion 3 Medium | 2B | MMDiT | SAI Community |
| Stable Diffusion 3.5 Large | 8B | MMDiT | SAI Community |
| FLUX.1-dev | 12B | Double + Single Stream DiT | некоммерческая |
| FLUX.1-schnell | 12B | то же, дистиллированная | Apache 2.0 |
| FLUX.2 | — | развитие FLUX.1 | смешанная |
| Z-Image | 6B | S3-DiT (Scalable Single-Stream) | разрешительная |
| Qwen-Image | ~20B | DiT + текстовая башня Qwen | Apache 2.0 |
| Hunyuan-Image-3.0 | ~80B | DiT | исследовательская |
| SD4 Turbo | 3B | DiT + дистилляция | SAI Commercial |

FLUX.1-schnell — стандарт с открытым исходным кодом 2026 года. Z-Image — лидер по эффективности. FLUX.2 и SD4 — текущие ориентиры по качеству.

### Почему этот сдвиг парадигмы важен

DDPM + U-Net работали. DiT + rectified flow работают **лучше, быстрее и масштабируются чище**. Этот переход параллелен переходу от RNN к трансформерам в NLP: обе архитектуры решали одну и ту же задачу, но трансформеры лучше масштабировались и теперь доминируют. Каждая статья 2026 года о генерации изображений, видео или 3D использует денойзер в форме DiT и обычно целевую функцию rectified flow. U-Net DDPM теперь в основном имеет педагогическое значение (урок 10).

```figure
cv3-rectified-flow
```

## Реализация

### Шаг 1: Блок DiT с AdaLN

```python
import torch
import torch.nn as nn


class AdaLNZero(nn.Module):
    """
    Adaptive LayerNorm with a gate. Predicts (scale, shift, gate) from the conditioning.
    Init such that the whole block starts as identity ("zero init").
    """

    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Linear(cond_dim, dim * 3)
        nn.init.zeros_(self.mlp.weight)
        nn.init.zeros_(self.mlp.bias)

    def forward(self, x, cond):
        scale, shift, gate = self.mlp(cond).chunk(3, dim=-1)
        h = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return h, gate.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, dim=192, heads=3, mlp_ratio=4, cond_dim=192):
        super().__init__()
        self.adaln1 = AdaLNZero(dim, cond_dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.adaln2 = AdaLNZero(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, cond):
        h, gate1 = self.adaln1(x, cond)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1 * a
        h, gate2 = self.adaln2(x, cond)
        x = x + gate2 * self.mlp(h)
        return x
```

`AdaLNZero` изначально представляет собой тождественное отображение, поскольку веса его MLP инициализируются нулями. Обучение постепенно сдвигает блок от тождественности; это значительно стабилизирует глубокие трансформерные диффузионные модели.

### Шаг 2: Маленький DiT

```python
def timestep_embedding(t, dim):
    import math
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class TinyDiT(nn.Module):
    def __init__(self, image_size=16, patch_size=2, in_channels=3, dim=96, depth=4, heads=3):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, cond_dim=dim) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.head = nn.Linear(dim, patch_size * patch_size * in_channels)

    def forward(self, x, t):
        n = x.size(0)
        x = self.patch(x)
        x = x.flatten(2).transpose(1, 2) + self.pos
        t_emb = self.time_mlp(timestep_embedding(t, self.pos.size(-1)))
        for blk in self.blocks:
            x = blk(x, t_emb)
        x = self.norm_out(x)
        x = self.head(x)
        return self._unpatchify(x, n)

    def _unpatchify(self, x, n):
        p = self.patch_size
        h = w = int(self.num_patches ** 0.5)
        x = x.view(n, h, w, p, p, -1).permute(0, 5, 1, 3, 2, 4).reshape(n, -1, h * p, w * p)
        return x
```

### Шаг 3: Обучение rectified flow

```python
import torch.nn.functional as F

def rectified_flow_train_step(model, x0, optimizer, device):
    model.train()
    x0 = x0.to(device)
    n = x0.size(0)
    t = torch.rand(n, device=device)
    epsilon = torch.randn_like(x0)
    x_t = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * epsilon

    target_velocity = epsilon - x0
    pred_velocity = model(x_t, t)

    loss = F.mse_loss(pred_velocity, target_velocity)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

Сравните с функцией потерь предсказания шума DDPM (урок 10): та же структура, другая цель. Вместо предсказания шума `epsilon` мы предсказываем **скорость** `epsilon - x_0`, которая направлена от данных к шуму вдоль прямолинейной интерполяции.

### Шаг 4: Сэмплер Эйлера

Rectified flow — это ОДУ. Метод Эйлера — самый простой и для хорошо обученной модели rectified flow почти так же точен, как решатели более высокого порядка, при 20+ шагах.

```python
@torch.no_grad()
def rectified_flow_sample(model, shape, steps=20, device="cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    t = torch.ones(shape[0], device=device)
    for _ in range(steps):
        v = model(x, t)
        x = x - dt * v
        t = t - dt
    return x
```

20 шагов. На обученной модели это даёт сэмплы, сопоставимые с 1000-шаговым DDPM.

### Шаг 5: Сквозной smoke-тест

```python
import numpy as np

def synthetic_blobs(num=200, size=16, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros((num, 3, size, size), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    for i in range(num):
        cx, cy = rng.uniform(4, size - 4, size=2)
        r = rng.uniform(2, 4)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        colour = rng.uniform(-1, 1, size=3)
        for c in range(3):
            out[i, c][mask] = colour[c]
    return torch.from_numpy(out)
```

Обучите `TinyDiT` на этом с rectified flow. После 500 шагов сэмплированные результаты должны выглядеть как блёклые цветные пятна.

## Применение

Для реальной генерации изображений с FLUX / SD3 / Z-Image `diffusers` поставляет все их с единым API:

```python
from diffusers import FluxPipeline, StableDiffusion3Pipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
).to("cuda")

out = pipe(
    prompt="a golden retriever surfing a tsunami, hyperrealistic, studio lighting",
    guidance_scale=0.0,           # schnell was trained without CFG
    num_inference_steps=4,
    max_sequence_length=256,
).images[0]
out.save("surf.png")
```

Три строки. `FLUX.1-schnell` за четыре шага. Замените id модели на `black-forest-labs/FLUX.1-dev` для более высокого качества за 20-30 шагов с CFG.

Для SD3:

```python
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    torch_dtype=torch.bfloat16,
).to("cuda")
out = pipe(prompt, guidance_scale=3.5, num_inference_steps=28).images[0]
```

## Оформление

Этот урок даёт:

- `outputs/prompt-dit-model-picker.md` — выбирает между SD3, FLUX.1-dev, FLUX.1-schnell, Z-Image, SD4 Turbo с учётом ограничений по качеству, задержке и лицензии.
- `outputs/skill-rectified-flow-trainer.md` — пишет полный обучающий цикл для rectified flow с AdaLN DiT и сэмплированием Эйлера.

## Упражнения

1. **(Лёгкое)** Обучите TinyDiT выше на синтетическом датасете пятен в течение 500 шагов. Сравните сэмплы, полученные с 10, 20 и 50 шагами Эйлера.
2. **(Среднее)** Добавьте текстовое кондиционирование, конкатенируя обученный эмбеддинг класса с эмбеддингом времени (10 «классов» пятен по цвету). Сэмплируйте с классами 0, 5 и 9 и проверьте, что цвета совпадают.
3. **(Сложное)** Вычислите расстояние Фреше (прокси для FID) между сгенерированными сэмплами из версий rectified-flow и DDPM сети одинакового размера, обученных на одних и тех же данных за одинаковое число шагов. Сообщите, какая сходится быстрее.

## Ключевые термины

| Термин | Что говорят | Что это на самом деле означает |
|------|----------------|----------------------|
| DiT | «Диффузионный трансформер» | Трансформер, заменяющий U-Net в качестве денойзера диффузии; работает на патчифицированных латентах |
| AdaLN | «Адаптивная layer norm» | Кондиционирование по временному шагу/тексту через обученные масштаб, сдвиг, гейт, применяемые после LayerNorm; стандарт в каждом современном DiT |
| MMDiT | «Мультимодальный DiT (SD3)» | Отдельные потоки весов для текстовых и графических токенов, разделяющие совместное self-attention |
| Single-stream / double-stream | «Трюк FLUX» | Первые N блоков двухпотоковые (отдельные веса на модальность), последующие блоки однопотоковые (конкатенация + общие веса) ради эффективности |
| Rectified flow | «Прямолинейный путь от шума к данным» | Линейная интерполяция между данными и шумом; сеть предсказывает скорость; требуется меньше шагов ОДУ при инференсе |
| Velocity target | «epsilon - x_0» | Целевая переменная регрессии в rectified flow; направлена от чистых данных к шуму |
| CFG guidance | «classifier-free guidance» | Смешивание условных и безусловных предсказаний; по-прежнему используется в моделях с rectified flow |
| Schnell / turbo / LCM | «Дистилляция в 1-4 шага» | Малошаговые варианты, дистиллированные из моделей полного качества; для продакшна в реальном времени |

## Дополнительное чтение

- [Scalable Diffusion Models with Transformers (Peebles & Xie, 2023)](https://arxiv.org/abs/2212.09748) — статья, предложившая DiT
- [Scaling Rectified Flow Transformers (Esser et al., статья SD3)](https://arxiv.org/abs/2403.03206) — MMDiT и rectified flow в большом масштабе
- [FLUX.1 model card and technical report (Black Forest Labs)](https://huggingface.co/black-forest-labs/FLUX.1-dev) — подробности о double + single-stream
- [Z-Image: Efficient Image Generation Foundation Model (2025)](https://arxiv.org/html/2511.22699v1) — однопотоковый DiT на 6 млрд параметров
- [Elucidating the Design Space of Diffusion (Karras et al., 2022)](https://arxiv.org/abs/2206.00364) — справочник по всем компромиссам проектирования диффузии
- [Latent Consistency Models (Luo et al., 2023)](https://arxiv.org/abs/2310.04378) — как LCM-LoRA даёт 4-шаговый инференс