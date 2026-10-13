# Отладка и профилирование

> Худшие баги в ИИ не падают с ошибкой. Они молча обучаются на мусоре и выдают красивую кривую потерь.

**Type:** Build
**Language:** Python
**Prerequisites:** Урок 1 (Dev Environment), базовое знакомство с PyTorch
**Time:** ~60 минут

## Цели обучения

- Использовать условные `breakpoint()` и `debug_print` для проверки форм тензоров, типов данных и значений NaN в процессе обучения
- Профилировать циклы обучения с помощью `cProfile`, `line_profiler` и `tracemalloc` для поиска узких мест
- Обнаруживать типичные баги в ИИ: несовпадение форм, NaN в потере, утечку данных и тензоры не на том устройстве
- Настроить TensorBoard для визуализации кривых потерь, гистограмм весов и распределений градиентов

## Проблема

Код для ИИ ломается иначе, чем обычный код. Веб-приложение падает со стеком вызовов. Неправильно настроенный цикл обучения работает 8 часов, сжигает $200 на GPU-времени и выдаёт модель, которая предсказывает среднее значение по всем входным данным. Код ни разу не выдал ошибку. Баг заключался в тензоре на неправильном устройстве, забытом `.detach()` или метках, просочившихся в признаки.

Вам нужны инструменты отладки, которые ловят такие тихие сбои до того, как они потратят впустую ваше время и вычислительные ресурсы.

## Концепция

Отладка ИИ работает на трёх уровнях:

```mermaid
graph TD
    L3["3. Training Dynamics<br/>Loss curves, gradient norms, activations"] --> L2
    L2["2. Tensor Operations<br/>Shapes, dtypes, devices, NaN/Inf values"] --> L1
    L1["1. Standard Python<br/>Breakpoints, logging, profiling, memory"]
```

Большинство людей сразу переходят к уровню 3 (пялятся в TensorBoard). Но 80% багов в ИИ живут на уровнях 1 и 2.

```figure
s0-flame-hot
```

## Реализация

### Часть 1: отладка через print (да, это работает)

Отладку через print принято списывать со счетов. Напрасно. Для тензорного кода точечный вызов print бьёт пошаговый проход в отладчике, потому что вам нужно видеть формы, типы данных и диапазоны значений одновременно.

```python
def debug_print(name, tensor):
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
          f"device={tensor.device}, "
          f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
          f"mean={tensor.mean().item():.4f}, "
          f"has_nan={tensor.isnan().any().item()}")
```

Вызывайте это после каждой подозрительной операции. Когда баг найден, уберите print'ы. Просто.

### Часть 2: отладчик Python (pdb и breakpoint)

Встроенный отладчик недооценён для работы с ИИ. Вставьте `breakpoint()` в цикл обучения и исследуйте тензоры интерактивно.

```python
def training_step(model, batch, criterion, optimizer):
    inputs, labels = batch
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    if loss.item() > 100 or torch.isnan(loss):
        breakpoint()

    loss.backward()
    optimizer.step()
```

Когда отладчик останавливает выполнение, полезные команды:

- `p outputs.shape` для проверки форм
- `p loss.item()` для просмотра значения потерь
- `p torch.isnan(outputs).sum()` для подсчёта NaN
- `p model.fc1.weight.grad` для проверки градиентов
- `c` чтобы продолжить, `q` чтобы выйти

Это условная отладка. Вы останавливаетесь только тогда, когда что-то выглядит не так. Для прогона обучения на 10 000 шагов это имеет значение.

### Часть 3: логирование Python

Замените print-ы на логирование, когда отладка выходит за рамки быстрой проверки.

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting training: lr=%.4f, batch_size=%d", lr, batch_size)
logger.warning("Loss spike detected: %.4f at step %d", loss.item(), step)
logger.error("NaN loss at step %d, stopping", step)
```

Логирование даёт вам временные метки, уровни важности и вывод в файл. Когда прогон обучения падает в 3 часа ночи, вам нужен файл лога, а не вывод терминала, который уже прокрутился за пределы экрана.

### Часть 4: замер времени по секциям кода

Знание того, куда уходит время, — первый шаг к оптимизации.

```python
import time

class Timer:
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        print(f"[{self.name}] {elapsed:.4f}s")

with Timer("data loading"):
    batch = next(dataloader_iter)

with Timer("forward pass"):
    outputs = model(batch)

with Timer("backward pass"):
    loss.backward()
```

Типичная находка: загрузка данных занимает 60% времени обучения. Решение — `num_workers > 0` в вашем DataLoader, а не более быстрый GPU.

### Часть 5: cProfile и line_profiler

Когда ручных таймеров недостаточно:

```bash
python -m cProfile -s cumtime train.py
```

Это показывает каждый вызов функции, отсортированный по совокупному времени. Для построчного профилирования:

```bash
pip install line_profiler
```

```python
@profile
def train_step(model, data, target):
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    return loss

# Run with: kernprof -l -v train.py
```

### Часть 6: профилирование памяти

#### Память CPU с tracemalloc

```python
import tracemalloc

tracemalloc.start()

# your code here
model = build_model()
data = load_dataset()

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics("lineno")
for stat in top_stats[:10]:
    print(stat)
```

#### Память CPU с memory_profiler

```bash
pip install memory_profiler
```

```python
from memory_profiler import profile

@profile
def load_data():
    raw = read_csv("data.csv")       # watch memory jump here
    processed = preprocess(raw)       # and here
    return processed
```

Запустите с `python -m memory_profiler your_script.py`, чтобы увидеть построчное использование памяти.

#### Память GPU с PyTorch

```python
import torch

if torch.cuda.is_available():
    print(torch.cuda.memory_summary())

    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

При достижении OOM (нехватки памяти):

1. Уменьшите размер батча (первое, что стоит попробовать, всегда)
2. Используйте `torch.cuda.empty_cache()` для освобождения кэшированной памяти
3. Используйте `del tensor`, за которым следует `torch.cuda.empty_cache()`, для больших промежуточных значений
4. Используйте смешанную точность (`torch.cuda.amp`), чтобы сократить использование памяти вдвое
5. Используйте checkpoint градиентов для очень глубоких моделей

### Часть 7: типичные баги в ИИ и как их ловить

#### Несовпадение форм

Самый частый баг. Тензор имеет форму `[batch, features]`, когда модель ожидает `[batch, channels, height, width]`.

```python
def check_shapes(model, sample_input):
    print(f"Input: {sample_input.shape}")
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            in_shape = inp[0].shape if isinstance(inp, tuple) else inp.shape
            out_shape = out.shape if hasattr(out, "shape") else type(out)
            print(f"  {name}: {in_shape} -> {out_shape}")
        return hook

    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(sample_input)

    for h in hooks:
        h.remove()
```

Запустите это один раз на выборочном батче. Это отобразит каждое преобразование формы в вашей модели.

#### NaN в потере

NaN в потере означает, что что-то «взорвалось». Типичные причины:

- Слишком высокая скорость обучения
- Деление на ноль в кастомной функции потерь
- Логарифм нуля или отрицательного числа
- Взрывающиеся градиенты в RNN

```python
def detect_nan(model, loss, step):
    if torch.isnan(loss):
        print(f"NaN loss at step {step}")
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"  NaN gradient in {name}")
                if torch.isinf(param.grad).any():
                    print(f"  Inf gradient in {name}")
        return True
    return False
```

#### Утечка данных

Ваша модель получает 99% точности на тестовом наборе. Звучит отлично. Это баг.

```python
def check_data_leakage(train_set, test_set, id_column="id"):
    train_ids = set(train_set[id_column].tolist())
    test_ids = set(test_set[id_column].tolist())
    overlap = train_ids & test_ids
    if overlap:
        print(f"DATA LEAKAGE: {len(overlap)} samples in both train and test")
        return True
    return False
```

Также проверяйте временную утечку: использование будущих данных для предсказания прошлого. Сортируйте по временной метке перед разбиением.

#### Неправильное устройство

Тензоры на разных устройствах (CPU vs GPU) вызывают ошибки времени выполнения. Но иногда тензор молча остаётся на CPU, пока всё остальное на GPU, и обучение просто идёт медленно.

```python
def check_devices(model, *tensors):
    model_device = next(model.parameters()).device
    print(f"Model device: {model_device}")
    for i, t in enumerate(tensors):
        if t.device != model_device:
            print(f"  WARNING: tensor {i} on {t.device}, model on {model_device}")
```

### Часть 8: основы TensorBoard

TensorBoard показывает, что происходит внутри обучения во времени.

```bash
pip install tensorboard
```

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/experiment_1")

for step in range(num_steps):
    loss = train_step(model, batch)

    writer.add_scalar("loss/train", loss.item(), step)
    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], step)

    if step % 100 == 0:
        for name, param in model.named_parameters():
            writer.add_histogram(f"weights/{name}", param, step)
            if param.grad is not None:
                writer.add_histogram(f"grads/{name}", param.grad, step)

writer.close()
```

Запустите его:

```bash
tensorboard --logdir=runs
```

На что обращать внимание:

- **Потеря не снижается**: слишком низкая скорость обучения или проблема в архитектуре модели
- **Потеря сильно колеблется**: слишком высокая скорость обучения
- **Потеря уходит в NaN**: численная нестабильность (см. раздел о NaN выше)
- **Потеря на обучении снижается, на валидации растёт**: переобучение
- **Гистограммы весов схлопываются к нулю**: затухающие градиенты
- **Гистограммы градиентов взрываются**: нужна обрезка градиентов

### Часть 9: отладчик VS Code

Для интерактивной отладки настройте VS Code с `launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Training",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```

Устанавливайте точки останова, кликая по полю слева от кода. Используйте панель Variables для проверки свойств тензоров. Debug Console позволяет выполнять произвольные выражения Python прямо посреди выполнения.

Полезно для пошагового прохода через конвейеры предобработки данных, где вы хотите видеть каждое преобразование.

## Применение

Вот рабочий процесс отладки, который ловит большинство багов в ИИ:

1. **Перед обучением**: запустите `check_shapes` на выборочном батче. Убедитесь, что размерности входа и выхода соответствуют ожиданиям.
2. **Первые 10 шагов**: используйте `debug_print` на потере, выходах и градиентах. Убедитесь, что нет NaN и значения находятся в разумных диапазонах.
3. **Во время обучения**: логируйте потерю, скорость обучения и нормы градиентов. Используйте TensorBoard для визуализации.
4. **Когда что-то ломается**: вставьте `breakpoint()` в точке сбоя. Исследуйте тензоры интерактивно.
5. **Для производительности**: замерьте время загрузки данных против прямого и обратного прохода. Профилируйте память, если вы близки к OOM.

## Сдача работы

Запустите скрипт набора инструментов для отладки:

```bash
python phases/00-setup-and-tooling/12-debugging-and-profiling/code/debug_tools.py
```

См. `outputs/prompt-debug-ai-code.md` для промпта, который помогает диагностировать баги, специфичные для ИИ.

## Упражнения

1. Запустите `debug_tools.py` и прочитайте вывод каждого раздела. Измените фиктивную модель, чтобы ввести NaN (подсказка: деление на ноль в прямом проходе), и посмотрите, как детектор его ловит.
2. Профилируйте цикл обучения с помощью `cProfile` и определите самую медленную функцию.
3. Используйте `tracemalloc`, чтобы найти, какая строка в вашем конвейере загрузки данных выделяет больше всего памяти.
4. Настройте TensorBoard для простого прогона обучения и определите, переобучается ли модель.
5. Используйте `breakpoint()` внутри цикла обучения. Потренируйтесь исследовать формы тензоров, устройства и значения градиентов из приглашения отладчика.