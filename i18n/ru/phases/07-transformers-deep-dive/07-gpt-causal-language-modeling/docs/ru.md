# GPT  Моделирование причинного языка

> BERT видит обе стороны, GPT видит только прошлое, триугольная маска - самая значительная единая строка кода в современном ИИ.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT)
**Time:** ~75 minutes

## Проблема

Языковая модель отвечает на один вопрос: учитывая первый `t-1`Токены, каково распределение вероятности по токену `t`На тренировке на этот сигнал  предсказание следующего токана  и вы получаете модель, которая может генерировать произвольный текст один токен за раз.

Чтобы тренировать его от конца до конца на целой последовательности параллельно, вам нужно, чтобы предсказание каждой позиции зависело только от предыдущих позиций.

Это единственная верхняя треугольная матрица`-inf`После softmax эти позиции становятся 0. Каждая позиция может присутствовать только на себе и на предыдущих позициях. И потому, что вы применяете его один раз на всю последовательность, вы получаете N параллельных предсказаний следующего токана в одном проходе вперед.

GPT-1 (2018), GPT-2 (2019), GPT-3 (2020), GPT-4 (2023), GPT-5 (2025), Claude, Llama, Qwen, Mistral, DeepSeek, Kimi  все они являются декодерными причинно-следственными трансформаторами с одной и той же основной петлей.

## Концепция

![Causal mask creates a triangular attention matrix](../assets/causal-attention.svg)

### Маска

Учитывая последовательность длины `N`, построить `N × N`матрица:

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

Добавить`M`до расточительных результатов внимания до softmax. `exp(-inf) = 0`Каждый ряд матрицы внимания является распределением вероятности по предыдущим позициям только.

Стоимость реализации: 1 `torch.tril()`Время для вычисления: наносекунды.

### Откуда взялся треугольник

Маска обычно представляется как пласт, завязанный на внимание. Запустите вывод в другом направлении и он перестанет быть загадочным: внимание - это третье уточнение префикса среднего, а треугольник - это петляные границы этого среднего, написанные как матрица.

**Stage 1 — prefix average.**Самый глупый причинный резюме последовательности: положение`i`становится средней позиции `0…i`- Как цикл, это`out[i] = X[:i+1].mean(0)`Примите нижнюю треугольную матрицу из единиц, делите каждый ряд по его числу, умножьте:

```python
import numpy as np

A = np.tril(np.ones((n, n)))
A = A / A.sum(axis=1, keepdims=True)
out = A @ X
```

Рынок `i``A`- Это ...`[1/(i+1), …, 1/(i+1), 0, …, 0]`Ничто о будущем не было скрыто, будущее никогда не было в сумме.

**Stage 2 — learned weights.**Единый средний относится к каждому прошедшему токену как к равной важности.`S`Теперь строки больше не суммируют на одну по конструкции, поэтому нормализуйте каждый ряд с softmax вместо разделения на подсчет. Softmax никогда не выпускает точную нуль, что нарушает причинность , если будущие оценки не входят в `-inf`, потому что`exp(-inf) = 0`В следующем случае:

```python
def softmax(x, axis):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

S = S + np.triu(np.full((n, n), -np.inf), k=1)
A = softmax(S, axis=1)
out = A @ X
```

Тот же треугольник, та же строковая матрица, та же мамуль.`-inf`Маска - это не новая машина. Это нулевые записи стадии 1, переведенные в домен ввода Softmax.

**Stage 3 — content-dependent weights.**На 2-й стадии.`S`Позиция 7 всегда весит позицию 3 одинаково, независимо от того, что говорят токены.`S = Q @ K.T / sqrt(d_k)`Маска, мягкая маска, мамуль - все одинаково.

Три этапа, одна инвариантная: нижняя треугольная строковая-стохастическая матрица умножается на последовательность.

```figure
mask-derivation
```

### Параллельное обучение, серийное выводы

Обучение: переход вперед `(N, d_model)`В процессе выполнения секвенции один раз вычисляете N потери кросс-энтропии (один на одну позицию), сумму, обратную опору. Параллельно вдоль секвенции.

Инфереция: вы генерируете токен по токену.`[t1, t2, t3]`- Я не могу .`t4`- Корм .`[t1, t2, t3, t4]`- Я не могу .`t5`- Корм .`[t1, t2, t3, t4, t5]`- Я не могу .`t6`. Кэширование КВ (урок 12) сохраняет скрытые состояния `t1…tn`Это авторегрессивный налог и почему декодирование является узлом задержки каждого LLM.

### Потеря  смещение по одному

Указанные токены `[t1, t2, t3, t4]`В следующем случае:

- Ввод: `[t1, t2, t3]`
- Цели: `[t2, t3, t4]`

Для каждой позиции`i`, вычислить`-log P(target_i | inputs[:i+1])`Это кросс-энтропия всей последовательности.

Каждый трансформатор LM, о котором вы слышали, загружает поезда на эту потерю.

### Стратегии декодирования

После тренировки выбор образцов имеет большее значение, чем думают люди.

| Method | What it does | When to use |
|--------|--------------|-------------|
| Greedy | Argmax every step | Deterministic tasks, code completion |
| Temperature | Divide logits by T, sample | Creative tasks, higher T = more diversity |
| Top-k | Sample from top-k tokens only | Kills low-probability tails |
| Top-p (nucleus) | Sample from smallest set with cumulative prob ≥ p | 2020+ default; adapts to distribution shape |
| Min-p | Keep tokens with `p > min_p * max_p` | 2024+; better at rejecting long tails than top-p |
| Speculative decoding | Draft model proposes N tokens, big model verifies | 2–3× latency reduction at same quality |

В 2026 году, температура min-p + 0,7 является разумным дефолтом для моделей с открытыми весями.

### Что заставило "рецепт GPT" работать

1. **Decoder-only.**Нет кодировки, один проход внимания + FFN на слой.
2. **Scaling.**124M → 1.5B → 175B → триллионы. Законы масштабирования Чинчилла (урок 13) рассказывают вам, как тратить вычислительные средства.
3. **In-context learning.**Появился около 6B13B. Модель может следовать нескольким примерам без тонкой настройки.
4. **RLHF.**После обучения человеческим предпочтениям преобразовано сырое предварительно подготовленное текстовое сообщение в помощников чат.
5. **Pre-norm + RoPE + SwiGLU.**Устойчивое обучение в масштабах.

Основная архитектура не сильно изменилась с момента GPT-2. Все интересное произошло в данных, масштабе и послетренировании.

```figure
causal-mask
```

## Создайте его

### Шаг 1: причинная маска

Посмотрите .`code/main.py`- Однолойник:

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Добавьте его к оценкам внимания до мягкого максимума.

### Шаг 2: 2-слойная модель GPT-ish

Складывайте два блока декодера (маскированное самовнимание + FFN, без перекрестного внимания). Добавьте встроенное токено, позиционное кодирование и не встроенное (привязанное к матрице встроенного токено  стандартный трюк с момента GPT-2).

### Шаг 3: предсказание следующего токана, от конца до конца

На 20-токенной игрушечной лексике, выделяйте логиты на каждой позиции. Вычислите потери кросс-энтропии против цели сдвига-по-одному.

### Шаг 4: выборка образцов

Используйте жадность, температуру, верхний k, верхний p, min-p. Запустите каждый на фиксированном запросе и сравните выходы.

## Используйте его

Питорч, 2026 г.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

Под капутой.`generate()`Затем он пропускает пропуск вперед, вытаскивает логиты окончательной позиции, пробивает следующий токен, добавляет его и повторяет. Каждый производственный набор выводов LLM (vLLM, TensorRT-LLM, llama.cpp, Ollama, MLX) реализует ту же петлю с тяжелой оптимизацией  предварительная заполняемость в партиях, непрерывное партийное загрузка, KV кеширование страниц, спекулятивное декодирование.

**GPT vs BERT, one line each:**Предсказания GPT `P(x_t | x_{<t})`- ПРЕДЕГИТЕЛЬНО .`P(x_masked | x_unmasked)`- Потеря определяет, может ли модель генерировать.

## Погрузи его

Посмотрите .`outputs/skill-sampling-tuner.md`. Умение выбирает параметры выборки образцов для задачи нового поколения и указывает, когда требуется детерминистическое декодирование.

## Упражнения

1. **Easy.**Беги .`code/main.py`и проверять, что матрица причинного внимания является нижней треугольной после softmax.
2. **Medium.**Сравните запутанность пучка 4 с жадным на 10 коротких запросов.
3. **Hard.**Используйте спекулятивный декодирование: используйте крошечную 2-слойную модель в качестве чертежа и 6-слойную модель в качестве проверщика. Измерить стенно-часы скорость на 100 завершениях длины 64.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Causal mask | "The triangle" | Upper-triangular `-inf` matrix added to attention scores so position `i` only sees positions `≤ i`. |
| Next-token prediction | "The loss" | Cross-entropy of the model's distribution against the true next token at every position. |
| Autoregressive | "Generate one at a time" | Feed output back as input; parallelism only during training, not during generation. |
| Logits | "Pre-softmax scores" | Raw output of the LM head before softmax; sampling happens on these. |
| Temperature | "Creativity knob" | Divide logits by T; T→0 = greedy, T→∞ = uniform. |
| Top-p | "Nucleus sampling" | Truncate distribution to smallest set summing to ≥p; sample from what remains. |
| Min-p | "Better than top-p" | Keep tokens where `p ≥ min_p × max_p`; adapts cutoff to sharpness of distribution. |
| Speculative decoding | "Draft + verify" | Cheap model proposes N tokens; big model verifies in parallel. |
| Teacher forcing | "Training trick" | During training, feed the true previous token, not the model's prediction. Standard for every seq2seq LM. |

## Дальнейшее чтение

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) GPT-1.
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) GPT-2.
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) GPT-3 и обучение в контексте.
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) спецификация декодирующей бумаги.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) канонический код причинно-следственной связи.
