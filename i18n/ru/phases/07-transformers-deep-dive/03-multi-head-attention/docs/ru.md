# Уделяемое внимание

> Один глаз внимания учится одному отношению за раз, восемь голов учится восьми, головы свободны, возьмите их больше.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention from Scratch)
**Time:** ~75 minutes

## Проблема

Если у ваших данных есть согласие субъекта-слова, соотношение, длиннометражный дискурс и синтаксический сбор, все это запутанное вместе, одна голова размазывает их в одно мягкое распределение максимума и теряет половину сигнала.

Исправление из статьи Васвани 2017 года: выполняйте несколько функций внимания параллельно, каждая с собственными проекциями Q, K, V, и конкацейте выходы.`d_model / n_heads`Общие параметры остаются теми же.

Многоголовное внимание является дефолтом каждого трансформатора на 2026 кораблях. Единственный аргумент заключается в том, сколько голов и разделяют ли ключи и значения проекции (Группированное внимание, многоголовное внимание, многоголовное латентное внимание).

## Концепция

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**Split.**Возьми .`X`формы `(N, d_model)`. Проекты до Q, K, V каждая формы `(N, d_model)`- Переоборудовать .`(N, n_heads, d_head)`где`d_head = d_model / n_heads`Перевести в `(n_heads, N, d_head)`- Я не знаю .

**Attend in parallel.**Запустите масштабированный продукт точка внимания внутри каждой головы.`(N, d_head)`Главы работают на разных подпространствах встроения и никогда не говорят во время вычисления внимания.

**Concatenate and project.**Складки вернут к `(N, d_model)`и умножить на вычисленную матрицу выхода `W_o`формы `(d_model, d_model)`- Я не знаю .`W_o`Это место, где головы смешиваются.

**Why it works.**Каждый глава может специализироваться без конкуренции с другими за представительный бюджет. Исследования с 20192024 показывают различные роли голов: позиционные головы, голова, которая придерживается предыдущего токен, головы копирования, головы именных существ, головы индукции (которые лежат в основе обучения в контексте).

**The 2026 lineage of variations:**

| Variant | Q heads | K/V heads | Used by |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (e.g. N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | compressed to low-rank | DeepSeek-V2, V3 |

GQA является современным дефолтом , потому что он сокращает память KV-каше на фактор `N/G`MLA идет дальше, сжимая K/V в скрытом пространстве, а затем проецируя обратно в вычислительное время  затрачивает FLOPs, экономит гораздо больше памяти.

```figure
multihead-split
```

## Создайте его

### Шаг 1: разделить головы от внимания одинокой головы у нас уже есть

Возьми `SelfAttention`В случае, если вы не можете получить информацию о том, что происходит в течение дня, то вы можете получить информацию о том, что происходит в течение дня.`code/main.py`для выполнения непрямых задач; логика заключается в следующем:

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

Один переобразует и один перерабатывает.`nn.MultiheadAttention`- Я не знаю .

### Шаг 2: запустить масштабированный пункт-продукт внимание на голову

Каждая голова получает свой собственный кусок Q, K, V.

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

На настоящем оборудовании .`Qh @ Kh.transpose(...)`- Это один .`bmm`ГПУ видит одну партию маммула формы .`(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`Добавление голов - это бесплатно.

### Шаг 3: Группированный запрос Вариант внимания

Только ключевые и ценные прогнозы меняются.`n_heads`группы; К и V получают `n_kv_heads < n_heads`группы и повторяются, чтобы соответствовать:

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

По сути , это сберегает память , потому что только`n_kv_heads`копии живут в кеше KV, не `n_heads`Llama 3 70B использует 64 заданных заданных с 8 KV заданных  8x кеширующий сократитель.

### Шаг 4: проверьте , что выучили каждый глава

Запустите MHA на короткое предложение с четырьмя головами.`(N, N)`Вы увидите, что разные головы выбирают разную структуру даже при случайной инициализации, что частично является сигналом, частично является вращающейся симметрией в подпространстве.

## Используйте его

В PyTorch, однострочная версия:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

GQA по PyTorch 2.5+:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**How many heads?**Правила производства в 2026 году:

| Model size | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head`Это единица того, сколько головы могут "видеть".`sqrt(d_head)`Если вы переходите на 256 и потеряете льготу "многих малых специалистов".

## Погрузи его

Посмотрите .`outputs/skill-mha-configurator.md`. Умение рекомендует количество голов, количество кв-головок и стратегию проекции для нового трансформатора с учетом параметров бюджета, длины последовательности и цели развертывания.

## Упражнения

1. **Easy.**Возьмите МХА из `code/main.py`и изменение`n_heads`от 1 до 16 с `d_model=64`Уменьшение головы помогает, поражает или вредит?
2. **Medium.**Внедрить MQA (один KV-головок, разделенный на все заданные заданные). Измерить, сколько параметров отсчета падает против полного MHA. Вычислить, насколько размер KV-каши сокращается при выводе для N = 2048.
3. **Hard.**Внедрить крошечную версию многоглавного латентного внимания: сжать K, V на ранговую`r`- Задерживать запятнавшееся в КВ-каше, разжимать в момент внимания.`r`Каш-память переходит ниже 1/8 полной MHA, а качество остается в пределах 1 бит от проверки?

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Head | "A single attention circuit" | One Q/K/V projection of dimension `d_head = d_model / n_heads` with its own attention matrix. |
| d_head | "Head dimension" | Per-head hidden width; almost always 64 or 128 in production. |
| Split / combine | "Reshape tricks" | `(N, d_model) ↔ (n_heads, N, d_head)` reshape+transpose around attention. |
| W_o | "Output projection" | `(d_model, d_model)` matrix applied after concatenating heads; where heads mix. |
| MQA | "One KV head" | Multi-Query Attention: single shared K/V projection. Smallest KV cache, some quality loss. |
| GQA | "The default since Llama 2" | Grouped-Query Attention with `n_kv_heads < n_heads`; repeats to match Q. |
| MLA | "DeepSeek's trick" | Multi-head Latent Attention: K,V compressed to low-rank latent, decompressed at attend time. |
| Induction head | "The circuit behind in-context learning" | A pair of heads that detect previous occurrences and copy what followed them. |

## Дальнейшее чтение

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) оригинальная спецификация многоглавной машины.
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) МКВА.
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) как преобразовать MHA в GQA после обучения.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) MLA и почему он лучше MHA / GQA на памяти кеша.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) механистический взгляд на то, что на самом деле делают головы.
