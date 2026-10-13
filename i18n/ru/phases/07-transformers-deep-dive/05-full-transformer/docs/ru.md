# Полный трансформатор  Кодировщик + Декодер

> Внимание - это звезда. Все остальное - остатки, нормализация, переход, перекрестное внимание - это эталон, который позволяет вам глубоко положить его.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (Positional Encoding)
**Time:** ~75 minutes

## Проблема

Один слой внимания - это экстрактор, а не модель. Один матмуль на слой недостаточно для языка.

В статье Vaswani 2017 года было принято шесть дизайнерских решений, которые превратили один слой внимания в стековый блок. Каждый трансформатор, начиная с  только кодирующий (BERT), только декодирующий (GPT), кодирующий-декодирующий (T5)  наследует тот же скелет.

Следующие уроки специализируют его на  06 для кодировщиков, 07 для декодировщиков, 08 для кодировщиков-декодировщиков.

## Концепция

![Encoder and decoder block internals, wired](../assets/full-transformer.svg)

### Шесть кусочков

1. **Embedding + positional signal.**Токены → векторы. Положение, вводимое через RoPE (современный) или синусоидный (классический).
2. **Self-attention.**Каждая позиция сопровождается другой, скрыта в декодерах.
3. **Feed-forward network (FFN).**Двухслойная МЛП по положению: `W_2 · activation(W_1 · x)`. Уровень расширения 4x по умолчанию.
4. **Residual connection.** `x + sublayer(x)`Без этого градиенты исчезают за 6 слоев.
5. **Layer normalization.** `LayerNorm`или `RMSNorm`Стабилизирует остаточный поток.
6. **Cross-attention (decoder only).**Запросы поступают из декодера, ключи и значения из выхода кодера.

Наблюдайте за течением вектора через один блок: внимание смешивается по позициям, остаток переносит его вперед, FFN преобразует его, а норма поддерживает точку потока.

```figure
transformer-block
```

### Блок кодировщика (используется кодировщиком BERT, T5)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

Код в двух направлениях, без маскировки, все позиции видят все позиции.

### Блок декодера (используется GPT, T5 декодер)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

Декодер имеет три подслоя на блок. Средний  перекрестное внимание  является единственным местом, где информация течет от кодера к декодру. В чистой архитектуре только для декодера (GPT), перекрестное внимание исключается, и вы просто маскируете самовнимание + FFN.

### Пренормальный против постнормальный

Оригинальная бумага: `x + sublayer(LN(x))`Вс `LN(x + sublayer(x))`. После нормального потерял популярность около 2019 года  сложнее тренироваться глубоко без тщательного отопления.`LN`* до* подслойка) является дефолтом 2026: Llama, Qwen, GPT-3+, Mistral все используют его.

### Модернизированный блок 2026

В 2017 году Vaswani выпустил LayerNorm + ReLU.

| Component | 2017 | 2026 |
|-----------|------|------|
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU | SwiGLU |
| FFN expansion | 4× | 2.6× (SwiGLU uses three matrices, total params match) |
| Position | Sinusoidal absolute | RoPE |
| Attention | Full MHA | GQA (or MLA) |
| Bias terms | Yes | No |

RMSNorm снижает среднее центрирование LayerNorm (одно меньшее удаление), что экономит вычисления и эмпирически не менее стабильно.`Swish(W1 x) ⊙ W3 x`) последовательно превосходит ReLU/GELU FFN на ~0,5 пункта в документах Llama, PaLM и Qwen.

### Счет параметров

На один квартал с `d_model = d`и расширение FFN `r`В следующем случае:

- МГА: `4 · d²`(проекции Q, K, V, O)
- FFN (SwiGLU): `3 · d · (r · d)`≈ ≈`3rd²`
- Нормы: незначительные

В `d = 4096, r = 2.6, layers = 32`(около Llama 3 8B), общее: `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B`(с добавлением вкладки и головы)

## Создайте его

### Шаг 1: строительные блоки

Используя маленький `Matrix`класс из урока 03 (копированный в данный файл для самостоятельной работы):

- `layer_norm(x, eps=1e-5)` вычитать среднюю, делить на std.
- `rms_norm(x, eps=1e-6)` разделить на RMS. Нет среднего вычитания.
- `gelu(x)`и `silu(x) * W3 x`(Свигл)
- `ffn_swiglu(x, W1, W2, W3)`- Я не знаю .
- `encoder_block(x, params)`и `decoder_block(x, enc_out, params)`- Я не знаю .

Посмотрите .`code/main.py`для полного проводки.

### Шаг 2: провод 2-слойный кодировщик и 2-слойный декодер

Передай выход кодера в каждый декодер, добави окончательный LN перед проекцией выхода.

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### Шаг 3: пробегайте вперед на пример игрушки

Передавайте 6-токенный источник и 5-токенную цель.`(5, vocab)`- Нет, нет, нет, нет. - Это урок о архитектуре, не о потере.

### Шаг 4: обмен в RMSNorm + SwiGLU

Заменить LayerNorm и ReLU-FFN с RMSNorm и SwiGLU. подтвердить формы все еще совпадают. Это 2026 модернизация с одной функцией замена.

## Используйте его

Референц-использования PyTorch/TF: `nn.TransformerEncoderLayer`- Я не знаю .`nn.TransformerDecoderLayer`Но большинство производственных кодов 2026 проводят свой собственный блок , потому что:

- Взрывная внимание призвана внутри внимания, а не через `nn.MultiheadAttention`- Я не знаю .
- ГКА/МЛА не в списке СДДЛИБ.
- RoPE, RMSNorm, SwiGLU не являются дефолтами PyTorch.

HF `transformers`имеет чистые справочные блоки , которые следует прочитать: `modeling_llama.py`Это 500 строк и стоит пройти через один раз.

**Encoder vs decoder vs encoder-decoder — when to pick:**

| Need | Pick | Example |
|------|------|---------|
| Classification, embeddings, QA over text | Encoder-only | BERT, DeBERTa, ModernBERT |
| Text generation, chat, code, reasoning | Decoder-only | GPT, Llama, Claude, Qwen |
| Structured input → structured output (translation, summarization) | Encoder-decoder | T5, BART, Whisper |

Язык, используемый только для декодера, выигрывает, потому что он имеет самый чистый масштаб и справляется как с пониманием, так и с генерацией.

## Погрузи его

Посмотрите .`outputs/skill-transformer-block-reviewer.md`. Умение пересматривает внедрение нового блока трансформаторов по сравнению с дефолтами 2026 года и выявляет отсутствующие части (донормальный, RoPE, RMSNorm, GQA, FFN соотношение расширения).

## Упражнения

1. **Easy.**Считать параметры в блоке кодера на `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`. Подтвердить , внедряя блок и используя `sum(p.numel() for p in block.parameters())`- Я не знаю .
2. **Medium.**Перейти от постнормального к преднормальному. Инициализировать оба и измерить норму активации после 12 слоев на случайном входе.
3. **Hard.**Внедрить 4-слойный кодировщик-декодер на задачу копирования игрушки (копировать `x`Загрузка 100 шагов. Отчет о потерях. Смена в RMSNorm + SwiGLU + RoPE  потеря падает?

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Block | "One transformer layer" | Stack of norm + attention + norm + FFN, wrapped in residual connections. |
| Residual | "Skip connection" | `x + f(x)` output; enables gradient flow through deep stacks. |
| Pre-norm | "Normalize before, not after" | Modern: `x + sublayer(LN(x))`. Trains deeper without warmup gymnastics. |
| RMSNorm | "LayerNorm without the mean" | Divide by RMS; one less op, same empirical stability. |
| SwiGLU | "The FFN everyone switched to" | `Swish(W1 x) ⊙ W3 x → W2`. Beats ReLU/GELU on LM ppl. |
| Cross-attention | "How the decoder sees the encoder" | MHA with Q from decoder, K/V from encoder outputs. |
| FFN expansion | "How wide the middle MLP is" | Ratio of hidden-size to d_model, usually 4 (LayerNorm) or 2.6 (SwiGLU). |
| Bias-free | "Drop the +b terms" | Modern stacks omit biases in linear layers; slight ppl improvement, smaller model. |

## Дальнейшее чтение

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) оригинальная спецификация блока.
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) почему донормальная ситуация сильно ухудшается посленормальной.
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) RMSNorm.
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) бумаги SwiGLU.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) канонический блок только для декодера 2026
