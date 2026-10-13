# Внимание Варианты  Сдвижные окна, Спарс, Дифференциальный

> Всякое внимание - это круг. Каждый знак видит каждый знак, а память платит цену. Четыре варианта сгибают форму круга и восстанавливают половину стоимости.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head), Phase 7 · 12 (KV Cache / Flash Attention)
**Time:** ~60 minutes

## Проблема

Стоимость полной внимания `O(N²)`память и `O(N²)`Для 128K-контекста Llama 3 70B, что составляет 16 миллиардов записей внимания на слой, умножается на 80 слоев.`O(N²)`активационная память, но не изменяет арифметические затраты  каждый токен все еще обслуживает каждый другой токен.

Три класса вариантов меняют топологию самой матрицы внимания:

1. **Sliding window attention (SWA).**Каждый токен обслуживает фиксированное окно соседей, а не полный префикс.`O(N · W)`где`W`Gemma 2/3, первые слои Мистрала 7Б, Phi-3-Long.
2. **Sparse / block attention.**Только выбранные пары `(i, j)`- Длинный формер, Большой птица, открытый формер.
3. **Differential attention.**Вычислить две карты внимания с отдельными проекциями Q / K, вычитать один из других. Убивает "отварительное растворение", которое кровоточит вес в первые несколько токенов.

Эти сосуществовают. Граничная модель 2026 часто смешивает их: большинство слоев являются SWA-1024, каждый пятый является глобальным полным вниманием, а несколько дифференциальных голов, которые очищают отзыв. Соотношение SWA-to-global GEMA 3 5:1 является текущим стандартным учебником.

## Концепция

### Сдвигающее окно (SWA)

Каждый запрос на позиции `i`занимает только должности в `[i - W, i]`(причинная СВА) или `[i - W/2, i + W/2]`Токены за окном получают`-inf`в матрице счетов.

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

Для `N = 8192`и `W = 1024`, матрица счетов имеет 1024 × 8192 не нулевые строки в ожидании  8x уменьшение.

**KV cache shrinks with SWA.**Только последнее .`W`Для конфигурации Gemma-3-ish (1024 окна, контекст 128K), кеширование KV падает 128x.

**Quality cost.**Трансформаторы, использующие только SWA, борются с удалением с дальнего радиуса действия. Решаем: перемещайте слои SWA с полной внимательностью. Gemma 3 использует 5:1 SWA:global. Mistral 7B использовал станок причинно-следственной SWA, где информация "перетекает вперед" через перекрывающиеся окна  каждый слой расширяет эффективное восприимчивое поле на `W`, и после`L`слои , на которые может присутствовать модель `L × W`Токены обратно.

### Сортовое / блокирующее внимание

Выберите `N × N`Спарситетный образ вперед.

- **Local + strided (OpenAI sparse transformer).**Прислушайся к последнему .`W`токены плюс каждый `stride`- Это и местный, и дальнего радиуса действия.`O(N · sqrt(N))`вычислить.
- **Longformer / BigBird.**Местное окно + небольшой набор глобальных токенов (например `[CLS]`В этом случае, в случае, если вы не хотите, чтобы это было возможно, вы можете получить информацию о том, что происходит в процессе использования данных.
- **Native Sparse Attention (DeepSeek, 2025).**Узнайте , какие блоки `(Q, K)`- Я не могу сказать, что это не так.

Sparse attention - это история по разработке ядра. Математика проста (маскируйте матрицу счета); выигрыш происходит от того, что никогда не загружают нулевые записи в SRAM. FlashAttention-3 и API 2026 FlexAttention делают пользовательские редкие шаблоны первого класса в PyTorch.

### Дифференциальное внимание (DIFF Transformer, 2024)

Регулярная внимание имеет проблему "отвод внимания": softmax заставляет каждый ряд суммировать до 1, поэтому токены, которые не хотят обращаться к чему-либо конкретному, сбросают вес на первый токен (или первые несколько).

Дифференциальное внимание исправляет это с помощью вычислений .**two**карты внимания и вычитание:

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

где`λ`A1 захватывает реальные весы содержания; A2 захватывает раковину.

Отчетные результаты (Microsoft 2024): 510% меньше путаницы, 1,52x более длинный эффективный контекст при той же тренированной длине, более резкий извлечение иглы в стеке сена.

### Сравнение вариантов

| Variant | Compute | KV cache | Quality vs full | Production use |
|---------|---------|----------|-----------------|----------------|
| Full attention | O(N²) | O(N) per layer | baseline | every model's default layer |
| SWA (window 1024) | O(N·W) | O(W) per layer | -0.1 ppl, good with global layers | Gemma 2/3, Phi-3-Long |
| Local + strided sparse | O(N·√N) | mixed | similar to SWA | OpenAI sparse transformer, Longformer |
| BigBird (local + global + random) | O(N) approx | mixed | matches full at 2× context | early long-context BERT |
| Native Sparse (DeepSeek-V3.2) | O(N · active fraction) | O(N) | within 0.05 ppl | DeepSeek-V3.2, 2025 |
| Differential | O(2·N²) | O(2N) | -5 to -10% ppl | DIFF Transformer, early 2026 models |

```figure
gqa-kv-sharing
```

## Создайте его

Посмотрите .`code/main.py`Мы используем сравнительный метод причинной маски, который показывает полное, SWA, локальное+погруженное и дифференциальное внимание бок о бок на последовательности игрушек.

### Шаг 1: полная причинно-следственная маска (базовый показатель)

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Исходный показатель из урока 07. Нижняя треугольная; нулевая масса над диагональю.

### Шаг 2: скользящая оконная причинно-следственная маска

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

Один параметр  `window`- Для .`window >= n`В результате вы восстанавливаете полное причинное внимание.`window = 1`, каждый знак обслуживает только себя.

### Шаг 3: локальная + шагчатая маска

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

Плотный местный окно плюс каждый `stride`Поле восприятия растет в лог-этап с дополнительными слоями.

### Шаг 4: дифференциальное внимание

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

В коде мы сравниваем тепловую карту внимания-погружения единого против дифференциального и наблюдаем, как погружение.

### Шаг 5: Размеры кеша KV

Печать размер кеша на слой в `N = 131072`СВА и редкие варианты снижаются на 10×100×. Дифференциальные двойные.

## Используйте его

2026 моделей производства:

```python
from transformers import AutoModelForCausalLM
# Gemma 3 mixes SWA (window=1024) and global layers at 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

FlexAttention в PyTorch 2.5+ принимает функцию маски:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

Это составляет на пользовательский яд Тритона. В пределах 10% скорости FlashAttention-3 для обычных шаблонов, и функция маски является Python.

**When to pick each:**

- **Pure full attention** каждый слой до ~ 16K контекста, или когда качество извлечения является важнейшим.
- **SWA + global mix** длинный контекст (> 32K), обучение и вывод памяти.
- **Sparse block attention** пользовательский яд, пользовательский шаблон.
- **Differential attention** любая рабочая нагрузка, при которой повреждается загрязнение водоем (RAG в длинном контексте, игла в стекле сена).

## Погрузи его

Посмотрите .`outputs/skill-attention-variant-picker.md`. Умение выбирает топологию внимания для новой модели, учитывая длину целевого контекста, требования к извлечению и профиль вычисления обучения/инференции.

## Упражнения

1. **Easy.**Беги .`code/main.py`. Проверьте SWA на `window=4`- Нулевая все за пределами последних 4 токенов в строке.`window=n`воспроизводит полное причинное внимание по bits-идентично.
2. **Medium.**Внедрение причинно-следственной СВА с `window=1024`Наверху урока 07 - тренировка на 1000 шагов на Tinyshakespeare.
3. **Hard.**Внедрить смесь 5:1 слоев в стиле Gemma-3 (5 SWA, 1 глобальный) в верхушке модели. Сравните качество потерь, памяти и генерации с чистыми SWA и чистыми глобальными базовыми линиями при сопоставленных параметрах.
4. **Hard.**Реализуйте дифференцированное внимание с обученным `λ`Например, в случае, если вы не можете получить информацию о результатах поиска, то вы можете получить информацию о результатах поиска, если вы не можете получить информацию о результатах поиска.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Sliding window attention (SWA) | "Local attention" | Each query attends to its last `W` tokens; KV cache shrinks to `O(W)`. |
| Effective receptive field | "How far back the model sees" | In an `L`-layer SWA stack with window `W`, up to `L × W` tokens. |
| Longformer / BigBird | "Local + global + random" | Sparse patterns with a few always-attending global tokens; early long-context approach. |
| Native Sparse Attention | "DeepSeek's kernel trick" | Learn block-level sparsity; skip zero blocks at the kernel level while keeping quality. |
| Differential attention | "Two maps, one subtracts" | DIFF Transformer: subtract a learned `λ` times a second attention map from the first to cancel attention sinks. |
| Attention sink | "Weight bleeds to token 0" | Softmax normalization forces rows to sum to 1; uninformative queries dump weight on position 0. |
| FlexAttention | "Mask-as-Python" | PyTorch 2.5+ API that compiles arbitrary mask functions into FlashAttention-shape kernels. |
| Layer type mix | "5:1 SWA-to-global" | Interleave sparse and full attention layers in a stack to keep quality at lower memory. |

## Дальнейшее чтение

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) канонический скользящий окно + глобальный токен бумаги.
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) локальные + глобальные + случайные.
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) Локальный+поступный шаблон OpenAI.
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) 1:1 SWA: глобальная смесь.
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) смесь 5:1 с окном=1024 это теперь стандартный учебник.
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) Бумага-трансформатор DIFF.
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) Учился внимания DeepSeek-V3.2 спарсности.
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) Ссылка API для модели использования маски в использовании.
