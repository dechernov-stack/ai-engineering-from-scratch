# Позиционное кодирование  Синусоидные, RoPE, ALiBi

> Внимание - это не изменяющаяся пермутация. "Кот сидел на матке" и "мат на спутниковой кошке" производят один и тот же выход без позиционного сигнала.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention)
**Time:** ~45 minutes

## Проблема

Умеренное внимание точка-продукт слепопо в порядке.`softmax(Q K^T / √d) V`Вычислить из паров сходства.`X`Ничто внутри внимания не заботится о положении.

Это не ошибка в модели слов, для языка, кода, аудио, видео  все, что имеет значение для порядка  это фатально.

Решается, как-то внедрять положение в вкладки.

1. **Absolute sinusoidal**(Васвани 2017). Добавить `sin/cos`Простые, без обучения, плохо экстраполируют за рамки обученных длин.
2. **RoPE — Rotary Position Embeddings**(Su 2021). Переключите векторы Q и K угол пропорционально положению.
3. **ALiBi — Attention with Linear Biases**(Пресс 2022). Включение полностью перестать; добавить линейный штраф на голову для баллов внимания на основе расстояния. Отличная экстраполяция длины.

По состоянию на 2026 год, по существу, каждая модель открытого пограничного типа использует RoPE: Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi.

## Концепция

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias](../assets/positional-encoding.svg)

### Абсолютный синусоидный

Предварительно рассчитать фиксированную матрицу `PE`формы `(max_len, d_model)`В следующем случае:

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

Тогда ...`X' = X + PE[:N]`Каждый измерение - это синусоид с другой частотой.`max_len`: ничто не рассказывало модели о том, что происходит на позиции 2048 когда она видела только позиции 02047.

### Ропэ

Переключите векторы Q и K (не встроенные).`(2i, 2i+1)`В следующем случае:

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

Применить такое же вращение к клавишам с положением `pos_k`. Продукция точка `q'_m · k'_n`становится функцией `(m - n)`Это значит:**the attention score depends only on the relative distance**Прекрасный трюк.

Расширение RoPE: `base`Llama 3 расширился с 8K до 128K контекста таким образом.

### АЛИБИ

Пропускайте трюк внедрения.

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

Где ?`m_h`является наклоном, специфичным для головы (например `1 / 2^(8·h/H)`В статье показано, что экстраполяция длины превышает синусоидальную и соответствует RoPE на его первоначальной длине.

### Что выбрать в 2026 году

| Variant | Extrapolation | Training cost | Used by |
|---------|---------------|---------------|---------|
| Absolute sinusoidal | poor | free | original transformer, early BERT |
| Learned absolute | none | tiny | GPT-2, GPT-3 |
| RoPE | good with scaling | free | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | excellent | fine-tune stage | Qwen2-1M, Llama 3.1 128K |
| ALiBi | excellent | free | BLOOM, MPT, Baichuan |

RoPE выиграл, потому что он привлекает внимание без изменения архитектуры, кодирует относительное положение, и его `base`Гиперпараметр дает чистую кнопку для длинного контекста.

```figure
rope-explorer
```

## Создайте его

### Шаг 1: синусоидная кодировка

Посмотрите .`code/main.py`- Четверолинейный вычисление:

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

Добавьте это к матрице вкладки до первого слоя внимания.

### Шаг 2: RoPE, применяемый к Q, K

RoPE работает на месте на Q и K. Для каждой пары размывающих устройств:

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

Ключевое: применить ту же функцию к Q на позиции `m`И К на позиции `n`Их продукт с точки появляется в`cos((m-n)·θ_i)`Учитель на каждой паре координат.

### Шаг 3: наклоны и предвзятость ALiBi

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

Добавить`bias[h]`к `(seq_len, seq_len)`матрица оценки внимания головы `h`, затем Softmax.

### Шаг 4: проверка свойства относительно-отдаленности RoPE

Выберите два случайных вектора .`a, b`- Поворачивайся .`(pos_a, pos_b)`- Тогда ...`(pos_a + k, pos_b + k)`.Обе продукты точка должны совпадать в пределах ошибки плавающей точки.

## Используйте его

PyTorch 2.5+ поставляет коммунальные услуги RoPE в `torch.nn.functional`. Большинство производственных кодов используют `flash_attn`или `xformers`где RoPE применяется внутри ядра внимания.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**Long-context tricks in 2026:**

- **NTK-aware interpolation.**Перерасширить`base`к `base * (scale_factor)^(d/(d-2))`при расширении от 4K до 16K+.
- **YaRN.**Умная интерполяция, которая сохраняет энтропию внимания в длинных контекстах.
- **LongRoPE.**Метод 2024 года, использующий эволюционный поиск для выбора факторов в масштабе измерений.
- **Position interpolation + fine-tuning.**Просто сократите позиции по фактору расширения и настройьте на токены 15B. Удивительно эффективны.

## Погрузи его

Посмотрите .`outputs/skill-positional-encoding-picker.md`. Умение выбирает стратегию кодирования для новой модели с учетом целевой длины контекста, потребностей в экстраполяции и бюджета на обучение.

## Упражнения

1. **Easy.**Запишите синусоидный `PE`матрица как тепловая карта для `max_len=512, d=128`- Подтвердите схему "полосы ширеют по мере увеличения показателя измерений".
2. **Medium.**Используйте масштабирование RoPE, которое осознает NTK. Обучайте крошечную LM на последовательности длины 256, затем проверьте на длине 1024 с масштабированием и без масштабирования. Измерите запутанность.
3. **Hard.**Внедрить ALiBi и RoPE в один и тот же модуль внимания. Обучить 4-слойный трансформатор на копировании с последовательностями длины 512. Экстраполировать до 2048 в момент испытания. Сравнять деградацию.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Positional encoding | "Tells attention about order" | Any signal added to embeddings or attention that encodes position. |
| Sinusoidal | "The original one" | `sin/cos` at geometric frequencies added to embeddings; doesn't extrapolate. |
| RoPE | "Rotary embeddings" | Rotate Q, K by position-dependent angle; dot product encodes relative distance. |
| ALiBi | "Linear bias trick" | Add `-m·\|i-j\|` to attention scores; no embedding needed, great extrapolation. |
| base | "RoPE's knob" | The frequency scaler in RoPE; increase to extend context at inference. |
| NTK-aware | "A RoPE scaling trick" | Rescale `base` so high-frequency dims aren't squeezed when context expands. |
| YaRN | "The fancy one" | Per-dimension interpolation+extrapolation that preserves attention entropy. |
| Extrapolation | "Works beyond trained length" | Can the position scheme serve correct output past `max_len` seen in training? |

## Дальнейшее чтение

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) оригинальный синусоидный.
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) Ропэ-папер.
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) ALiBi.
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) современное масштабирование RoPE.
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) Долгоконтекстовая статья "Ллама 2" Меты.
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) метод Microsoft, используемый Phi-3-Long и указанный в разделе "Используйте его".
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) реализация на уровне производства каждой схемы масштабирования RoPE (по умолчанию, линейная, динамическая, YaRN, LongRoPE, Llama-3).
