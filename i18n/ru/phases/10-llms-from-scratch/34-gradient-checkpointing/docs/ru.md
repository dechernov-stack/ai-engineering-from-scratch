# Gradient Checkpointing and Activation Recomputation (Чекпоинтинг градиентов и пересчёт активаций)

> Backprop хранит каждую промежуточную активацию. При 70B параметров и контексте 128K это 3 TB активаций на ранг. Чекпоинтинг меняет FLOPs на память: пересчитывать вместо того, чтобы сохранять. Вопрос в том, какие сегменты отбрасывать, и ответ не «все».

**Type:** Build
**Languages:** Python (с numpy, опционально torch)
**Prerequisites:** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**Time:** ~70 минут

## Проблема

При обучении трансформера для каждого слоя сохраняются входы всех операций, которые дифференцируются при обратном проходе: входы внимания, проекции Q/K/V, выход softmax, входы FFN, выходы нормализации и residual-поток. Для слоя со скрытой размерностью `d`, длиной последовательности `L`, батчем `B` это порядка `12 * B * L * d` чисел с плавающей точкой на слой.

Для `d=8192, L=8192, B=1` это 800 МБ/слой в BF16. 64-слойная модель — это 51 ГБ активаций, и это до умножения на размер микробатча, до добавления промежуточных значений softmax внимания (`L^2` на голову), и до учёта частичных копий при тензорном параллелизме.

Счёт с двух сторон: веса в BF16 плюс состояние оптимизатора могут уместиться в 80 ГБ, но активации выталкивают вас за этот предел. Чекпоинтинг градиентов (он же пересчёт активаций) — стандартное решение. Отбросить большую часть активаций; пересчитать их во время обратного прохода. Цена: дополнительные FLOPs. Выгода: память снижается пропорционально отношению числа checkpoint-сегментов к общему числу слоёв.

Сделанный наивно, чекпоинтинг обходится примерно в 33% дополнительных FLOPs прямого прохода на шаг. Сделанный правильно — с избирательным (selective) чекпоинтингом по методу «умного отбора» Korthikanti et al. — вы экономите в 5 раз больше памяти при накладных расходах на FLOPs менее 5%. А с учётом матричных умножений в FP8, выгрузки FSDP и MoE с экспертным параллелизмом это действительно важно: вы не можете позволить себе ни лишнюю память, ни впустую потраченные вычисления.

## Концепция

### Что на самом деле нужно обратному проходу

`output = layer(input)`. Обратному проходу нужны `grad_input` и `grad_params`. Чтобы их вычислить, требуется:

- `input` (для вычисления `grad_params = input.T @ grad_output` для линейных слоёв)
- некоторые промежуточные значения производных активации (производная ReLU/GELU/softmax зависит от значения активации)

Прямой проход автоматически сохраняет их в графе autograd. Каждый `tensor.retain_grad()` и каждая операция, которой нужен её вход, сохраняет на него ссылку.

### Наивный полный чекпоинтинг

Разбейте сеть на `N` сегментов. Во время прямого прохода сохраняйте только *вход* каждого сегмента. Когда обратному проходу нужны промежуточные значения, заново прогоните прямой проход сегмента, чтобы их материализовать, затем дифференцируйте.

Пример: 32-слойный трансформер, разбитый на 32 сегмента по 1 слою каждый.

- Память: 32 входа слоёв (малый объём) против 32 * (объём активаций на слой) (огромный объём).
- Дополнительные вычисления: 1 дополнительный прямой проход на сегмент, то есть ~33% больше FLOPs прямого прохода в целом (поскольку обратный проход вдвое затратнее прямого, полный шаг становится 1 + 1 + 2 = 4 единицы вместо 1 + 2 = 3).

Это исходный рецепт Chen et al. 2016: одна точка checkpoint на каждые `sqrt(L)` слоёв для баланса памяти и вычислений. Для L=64 это 8 checkpoint'ов.

### Избирательный чекпоинтинг (Korthikanti 2022)

Не все активации стоят одинаково. Выход softmax внимания имеет размер `B*L*L*heads` и растёт *квадратично* с длиной последовательности. Скрытая активация FFN имеет размер `B*L*4d` и растёт линейно. Для длинных последовательностей softmax доминирует.

Избирательный чекпоинтинг сохраняет дешёвые для хранения активации (линейные проекции, residual) и пересчитывает только дорогие (внимание). Вы платите минимальными FLOPs за пересчёт, но экономите O(L^2) памяти.

Megatron-Core реализует это как «selective» (избирательный) пересчёт активаций. Используется в большинстве передовых обучающих запусков 2024+ года.

### Offload (выгрузка)

Альтернатива пересчёту: передавать активации в оперативную память CPU между прямым и обратным проходом. Требует пропускной способности PCIe; выгодна, когда свободная пропускная способность превышает стоимость рематериализации. Смешанные стратегии распространены: часть слоёв чекпоинтится, часть — выгружается.

FSDP2 поставляет offload как полноценную опцию. Offload особенно хорош, когда GPU упирается в память, а пропускная способность CPU-GPU имеет запас.

### Модель стоимости пересчёта

FLOPs на шаг при наивном чекпоинтинге каждые `k` слоёв из `L`:

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

При избирательном чекпоинтинге вы пересчитываете только ядро внимания, а не весь слой:

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### Модель экономии памяти

Объём активаций на слой: `A`. Для `L` слоёв общий объём памяти активаций: `L * A`.

Полный чекпоинт (размер сегмента 1): храним только `L * input_volume` (~`L * 1/10 A` для стандартного трансформера). Экономия ~`9 * L * A * 1/10`.

Чекпоинт через каждые `k` слоёв: храним `L/k * A` плюс объём для `k-1` слоёв внутри активного сегмента.

При `k = sqrt(L)` и память, и стоимость пересчёта масштабируются как `sqrt(L)` — оптимальный компромисс для слоёв с одинаковой стоимостью.

### Когда не стоит использовать чекпоинтинг

- Для самых внутренних слоёв этапа конвейера, уже выполняющегося. Им в любом случае придётся завершиться.
- Для первого и последнего слоёв, если они доминируют в вычислениях этапа (редкий случай для трансформеров).
- Для ядер внимания, уже использующих FlashAttention — Flash уже быстро пересчитывает softmax, поэтому дополнительный чекпоинтинг на уровне слоя даёт мало выгоды сверху.

### Паттерны реализации

1. **Обёртка функции:** оберните сегмент в `torch.utils.checkpoint.checkpoint(fn, input)`. PyTorch сохраняет только `input`, пересчитывает всё остальное при обратном проходе.

2. **На основе декораторов:** помечайте слои как поддающиеся чекпоинтингу; тренер решает на этапе конфигурации, какие сегменты оборачивать.

3. **Явный ручной пересчёт:** напишите обратный проход сами, вызывая пользовательскую функцию `recompute_forward`, которая дублирует прямой проход с сохранённым входом.

Все три подхода дают одинаковый функциональный результат. Обёртки — стандартная идиома.

### Взаимодействие с TP / PP / FP8

- **Тензорный параллелизм:** входы checkpoint должны быть собраны или заново распределены при пересчёте; учитывайте стоимость коммуникации.
- **Конвейерный параллелизм:** типичный паттерн — чекпоинтить прямой проход каждого этапа конвейера, чтобы микробатчи в обратном порядке могли повторно использовать память активаций.
- **Пересчёт FP8:** истории amax, обновляемые во время пересчёта, должны совпадать с историями исходного прямого прохода, иначе масштаб FP8 "уплывёт". Большинство фреймворков делают снимок масштаба.

```figure
activation-recompute
```

## Постройте это

### Шаг 1: Игрушечная модель с сегментами

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### Шаг 2: Наивный обратный проход, требующий всех активаций

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### Шаг 3: Память при чекпоинтинге через каждые k слоёв

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### Шаг 4: Модель стоимости

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### Шаг 5: Оценщик памяти

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### Шаг 6: Оптимальный размер сегмента

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### Шаг 7: Решение об избирательном чекпоинтинге

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## Применение на практике

- **torch.utils.checkpoint**: `from torch.utils.checkpoint import checkpoint` — каноническая обёртка в PyTorch. Оборачивает функцию; сохраняет только входы, пересчитывает при обратном проходе.
- **Megatron-Core activation recomputation**: поддерживает режимы `selective`, `full` и `block`. Стандарт в передовом обучении 2024+ года.
- **FSDP2 offload**: `module.to_empty(device="cpu")` с `offload_policy` в FSDP2 выгружает активации на CPU вместо пересчёта.
- **DeepSpeed ZeRO-Offload**: выгрузка на CPU состояний оптимизатора и активаций, дополняющая чекпоинтинг.

## Довести до результата

Этот урок создаёт `outputs/prompt-activation-recompute-policy.md` — промпт, который принимает конфигурацию вашей модели (слои, скрытая размерность, длина последовательности, батч) и доступную память GPU и выдаёт политику пересчёта по слоям (none / selective / full / offload).

## Упражнения

1. Проверьте корректность. Запустите `model_forward` + `model_backward` (полные активации) против `model_forward_checkpointed` + `model_backward_checkpointed` (сегменты). Градиенты параметров должны совпадать с точностью до машинной.

2. Переберите размер сегмента `k` от 1 до `L`. Постройте график накладных расходов FLOPs и памяти. Найдите точку перегиба кривой.

3. Реализуйте избирательный чекпоинтинг: сохраняйте вход модуля внимания, но не его промежуточные значения. Измерьте накладные расходы FLOPs по сравнению с полным чекпоинтингом слоя для 32-слойной модели при seq=8192.

4. Добавьте offload. Сохраняйте входы сегментов в симулированный «буфер CPU» (отдельный список). Измерьте «пропускную способность PCIe» как байты/время и найдите точку безубыточности между offload и пересчётом.

5. Проведите бенчмарк реального трансформера PyTorch с `torch.utils.checkpoint` и без него. Измерьте память (через `torch.cuda.max_memory_allocated`) и время шага.

## Ключевые термины

| Термин | Что говорят люди | Что это означает на самом деле |
|------|----------------|----------------------|
| Gradient checkpointing | «Экономия памяти за счёт повторного прямого прохода» | Сохранение только входов сегментов; пересчёт промежуточных значений при обратном проходе для получения тензоров, нужных для градиента |
| Activation recomputation | «То же, что и чекпоинтинг» | Название той же техники в стиле HPC |
| Размер сегмента (k) | «Сколько слоёв на один checkpoint» | Число слоёв, чьи промежуточные значения отбрасываются и рематериализуются вместе |
| Selective checkpointing | «Трюк Korthikanti» | Пересчёт только дорогих для хранения активаций (softmax внимания); сохранение дешёвых |
| Full checkpointing | «Наивная версия» | Пересчёт промежуточных значений каждого слоя в каждом сегменте |
| Block checkpointing | «Крупнозернистый» | Чекпоинтинг целых блоков трансформера; наибольшая гранулярность |
| FLOP overhead | «Налог на вычисления» | Дополнительные FLOPs на шаг = (FLOPs пересчёта) / (FLOPs прямого + обратного прохода); 33% наивно, 5% избирательно |
| Activation offload | «Отправить на CPU» | Перенос активаций в оперативную память CPU между прямым и обратным проходом; альтернатива пересчёту |
| sqrt-L rule | «Классический оптимум» | Для слоёв с равной стоимостью оптимальный интервал чекпоинтов — sqrt(L) слоёв |
| Attention-softmax volume | «Проблема O(L^2)» | L^2 * число голов * батч чисел с плавающей точкой; доминирует в памяти активаций при длинных контекстах |

## Дополнительное чтение

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) — исходная статья, формализовавшая чекпоинтинг градиентов
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) — избирательный пересчёт активаций и формальный анализ стоимости
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) — альтернативный подход с постоянным объёмом памяти через обратную рематериализацию
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) — выгрузка активаций в масштабе
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) — стандартный API
- [Megatron Bridge activation recomputation documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/activation-recomputation.html) — режимы selective, full и block