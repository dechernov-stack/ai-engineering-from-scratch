# Введение в PyTorch

> Вы собрали двигатель из поршней и коленвала. Теперь изучите тот, на котором все реально ездят.

**Type:** Build
**Languages:** Python
**Prerequisites:** Урок 03.10 (Создайте свой мини-фреймворк)
**Time:** ~75 минут

## Цели обучения

- Создавать и обучать нейронные сети с помощью nn.Module, nn.Sequential и autograd в PyTorch
- Использовать тензоры PyTorch, ускорение на GPU и стандартный цикл обучения (zero_grad, forward, loss, backward, step)
- Преобразовать компоненты вашего самодельного мини-фреймворка в их эквиваленты в PyTorch
- Профилировать и сравнить скорость обучения вашего фреймворка на чистом Python и PyTorch на одной и той же задаче

## Проблема

У вас есть рабочий мини-фреймворк. Линейные слои, ReLU, dropout, batch norm, Adam, DataLoader, цикл обучения. Он обучает 4-слойную сеть задаче классификации точек по кругу на чистом Python.

Он также в 500 раз медленнее PyTorch на той же задаче.

Ваш мини-фреймворк обрабатывает по одному образцу за раз с помощью вложенных циклов Python. PyTorch отправляет те же операции в оптимизированные ядра C++/CUDA, которые работают на GPU. На одном NVIDIA A100 PyTorch обучает ResNet-50 (25,6 млн параметров) на ImageNet (1,28 млн изображений) примерно за 6 часов. Вашему фреймворку на этой же задаче потребовалось бы около 3000 часов — если бы он не закончил память раньше.

Скорость — не единственный разрыв. В вашем фреймворке нет поддержки GPU. Нет автоматического дифференцирования — вы вручную писали backward() для каждого модуля. Нет сериализации. Нет распределённого обучения. Нет смешанной точности. Нет способа отладить поток градиентов без print-инструкций.

PyTorch закрывает каждый из этих пробелов. И делает это, сохраняя ту же самую ментальную модель, которую вы уже построили: Module, forward(), parameters(), backward(), optimizer.step(). Концепции переносятся один в один. Синтаксис почти идентичен. Разница в том, что PyTorch оборачивает десятилетие инженерной работы над системами за тем же интерфейсом, который вы спроектировали с нуля.

## Концепция

### Почему победил PyTorch

В 2015 году TensorFlow требовал определить статический граф вычислений перед запуском чего-либо. Вы строили граф, компилировали его, затем пропускали через него данные. Отладка означала разглядывание визуализаций графа. Изменение архитектуры означало перестройку графа с нуля.

PyTorch вышел в 2017 году с другой философией: eager execution (немедленное выполнение). Вы пишете Python-код. Он выполняется немедленно. `y = model(x)` действительно вычисляет y прямо сейчас, а не «добавляет узел в граф, который вычислит y позже». Это означало, что стандартные инструменты отладки Python работали. print() работал. pdb работал. if/else в вашем forward-проходе работал.

К 2020 году рынок высказал своё мнение. Доля PyTorch в научных статьях по ML выросла с 7% (2017) до более чем 75% (2022). Meta, Google DeepMind, OpenAI, Anthropic и Hugging Face — все используют PyTorch как основной фреймворк. TensorFlow 2.x в ответ принял eager execution — молчаливое признание того, что дизайн PyTorch был верным.

Урок: удобство для разработчика накапливается. Фреймворк, который на 10% медленнее, но на 50% быстрее отлаживается, побеждает каждый раз.

### Тензоры

Тензор — это многомерный массив с тремя критическими свойствами: shape (форма), dtype (тип данных) и device (устройство).

```python
import torch

x = torch.zeros(3, 4)           # shape: (3, 4), dtype: float32, device: cpu
x = torch.randn(2, 3, 224, 224) # batch of 2 RGB images, 224x224
x = torch.tensor([1, 2, 3])     # from a Python list
```

**Shape** — это размерность. Скаляр имеет форму (), вектор — (n,), матрица — (m, n), пакет изображений — (batch, channels, height, width).

**Dtype** управляет точностью и памятью.

| dtype | Биты | Диапазон | Применение |
|-------|------|----------|------------|
| float32 | 32 | ~7 десятичных разрядов | Обучение по умолчанию |
| float16 | 16 | ~3.3 десятичных разряда | Смешанная точность |
| bfloat16 | 16 | Тот же диапазон, что у float32, но меньше точность | Обучение LLM |
| int8 | 8 | от -128 до 127 | Квантованный инференс |

**Device** определяет, где происходят вычисления.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(3, 4, device=device)
x = x.to("cuda")
x = x.cpu()
```

Каждая операция требует, чтобы все тензоры находились на одном устройстве. Это ошибка №1, с которой сталкиваются новички в PyTorch: `RuntimeError: Expected all tensors to be on the same device`. Исправляется переносом всех тензоров на одно устройство перед вычислением.

**Изменение формы (reshaping)** выполняется за константное время — меняются метаданные, а не сами данные.

```python
x = torch.randn(2, 3, 4)
x.view(2, 12)      # reshape to (2, 12) -- must be contiguous
x.reshape(6, 4)    # reshape to (6, 4) -- works always
x.permute(2, 0, 1) # reorder dimensions
x.unsqueeze(0)     # add dimension: (1, 2, 3, 4)
x.squeeze()        # remove size-1 dimensions
```

### Autograd

Ваш мини-фреймворк требовал от вас реализации backward() для каждого модуля. PyTorch — нет. Он записывает каждую операцию над тензорами в направленный ациклический граф (граф вычислений), а затем обходит этот граф в обратном порядке, чтобы автоматически вычислить градиенты.

```mermaid
graph LR
    x["x (leaf)"] --> mul["*"]
    w["w (leaf, requires_grad)"] --> mul
    mul --> add["+"]
    b["b (leaf, requires_grad)"] --> add
    add --> loss["loss"]
    loss --> |".backward()"| add
    add --> |"grad"| b
    add --> |"grad"| mul
    mul --> |"grad"| w
```

Ключевое отличие от вашего фреймворка: PyTorch использует автодифференцирование на основе «ленты» (tape-based autodiff). Каждая операция добавляется на «ленту» во время прямого прохода. Вызов `.backward()` воспроизводит ленту в обратном порядке.

```python
x = torch.randn(3, requires_grad=True)
y = x ** 2 + 3 * x
z = y.sum()
z.backward()
print(x.grad)  # dz/dx = 2x + 3
```

Три правила autograd:

1. Градиенты накапливают только листовые тензоры с `requires_grad=True`
2. Градиенты накапливаются по умолчанию — вызывайте `optimizer.zero_grad()` перед каждым обратным проходом
3. `torch.no_grad()` отключает отслеживание градиентов (используется при оценке)

### nn.Module

`nn.Module` — базовый класс для каждого компонента нейронной сети в PyTorch. Вы уже строили эту абстракцию в Уроке 10. Версия PyTorch добавляет автоматическую регистрацию параметров, рекурсивное обнаружение модулей, управление устройствами и сериализацию через state dict.

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
```

Когда вы присваиваете `nn.Module` или `nn.Parameter` в качестве атрибута в `__init__`, PyTorch автоматически регистрирует его. `model.parameters()` рекурсивно собирает каждый зарегистрированный параметр. Именно поэтому вам никогда не придётся вручную собирать веса, как вы делали это в мини-фреймворке.

Ключевые строительные блоки:

| Модуль | Что делает | Параметры |
|--------|-------------|------------|
| nn.Linear(in, out) | Wx + b | in*out + out |
| nn.Conv2d(in_ch, out_ch, k) | 2D-свёртка | in_ch*out_ch*k*k + out_ch |
| nn.BatchNorm1d(features) | Нормализация активаций | 2 * features |
| nn.Dropout(p) | Случайное обнуление | 0 |
| nn.ReLU() | max(0, x) | 0 |
| nn.GELU() | Гауссова линейная ошибка | 0 |
| nn.Embedding(vocab, dim) | Таблица поиска | vocab * dim |
| nn.LayerNorm(dim) | Нормализация по образцу | 2 * dim |

### Функции потерь и оптимизаторы

PyTorch поставляет готовые к production версии всего, что вы построили.

**Функции потерь** (из `torch.nn`):

| Потеря | Задача | Вход |
|------|------|-------|
| nn.MSELoss() | Регрессия | Любая форма |
| nn.CrossEntropyLoss() | Многоклассовая классификация | Логиты (не softmax) |
| nn.BCEWithLogitsLoss() | Бинарная классификация | Логиты (не sigmoid) |
| nn.L1Loss() | Регрессия (устойчивая) | Любая форма |
| nn.CTCLoss() | Выравнивание последовательностей | Логарифмы вероятностей |

Примечание: `CrossEntropyLoss` внутренне объединяет `LogSoftmax` + `NLLLoss`. Передавайте сырые логиты, а не выходы softmax. Это распространённая ошибка, которая незаметно приводит к неверным градиентам.

**Оптимизаторы** (из `torch.optim`):

| Оптимизатор | Когда использовать | Типичный LR |
|-----------|-------------|-----------|
| SGD(params, lr, momentum) | CNN, хорошо настроенные пайплайны | 0.01--0.1 |
| Adam(params, lr) | Стартовая точка по умолчанию | 1e-3 |
| AdamW(params, lr, weight_decay) | Трансформеры, fine-tuning | 1e-4--1e-3 |
| LBFGS(params) | Небольшой масштаб, второй порядок | 1.0 |

### Цикл обучения

Каждый цикл обучения в PyTorch следует одному и тому же шаблону из 5 шагов. Вы уже знаете это из Урока 10.

```mermaid
sequenceDiagram
    participant D as DataLoader
    participant M as Model
    participant L as Loss fn
    participant O as Optimizer

    loop Each Epoch
        D->>M: batch = next(dataloader)
        M->>L: predictions = model(batch)
        L->>L: loss = criterion(predictions, targets)
        L->>M: loss.backward()
        O->>M: optimizer.step()
        O->>O: optimizer.zero_grad()
    end
```

Канонический шаблон:

```python
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

Пять строк внутри цикла по батчам. Пять строк, которые обучили GPT-4, Stable Diffusion и LLaMA. Архитектура меняется. Данные меняются. Эти пять строк — нет.

### Dataset и DataLoader

`Dataset` в PyTorch — это абстрактный класс с двумя методами: `__len__` и `__getitem__`. `DataLoader` оборачивает его пакетированием, перемешиванием и многопроцессной загрузкой данных.

```python
from torch.utils.data import Dataset, DataLoader

class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
```

`num_workers=4` порождает 4 процесса для параллельной загрузки данных, пока GPU обучается на текущем батче. На задачах, ограниченных скоростью диска (большие изображения, аудио), это само по себе может удвоить скорость обучения.

### Обучение на GPU

Перенос модели на GPU:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

Это рекурсивно переносит каждый параметр и буфер на GPU. Затем переносите каждый батч во время обучения:

```python
inputs, targets = inputs.to(device), targets.to(device)
```

**Смешанная точность** вдвое уменьшает использование памяти и вдвое увеличивает пропускную способность на современных GPU (A100, H100, RTX 4090), выполняя прямой/обратный проход в float16, сохраняя мастер-веса в float32:

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()
for inputs, targets in loader:
    with autocast(device_type="cuda"):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### Сравнение: мини-фреймворк vs PyTorch vs JAX

| Функция | Мини-фреймворк (Урок 10) | PyTorch | JAX |
|---------|---------------------|---------|-----|
| Автодифференцирование | Ручной backward() | Autograd на основе ленты | Функциональные трансформации |
| Выполнение | Eager (циклы Python) | Eager (ядра C++) | Трассировка + JIT-компиляция |
| Поддержка GPU | Нет | Да (CUDA, ROCm, MPS) | Да (CUDA, TPU) |
| Скорость (MNIST MLP) | ~300с/эпоха | ~0.5с/эпоха | ~0.3с/эпоха |
| Система модулей | Собственный класс Module | nn.Module | Безсостояние-функции (Flax/Equinox) |
| Отладка | print() | print(), pdb, breakpoint() | Сложнее (JIT-трассировка ломает print) |
| Экосистема | Нет | Hugging Face, Lightning, timm | Flax, Optax, Orbax |
| Кривая обучения | Вы построили его сами | Умеренная | Крутая (функциональная парадигма) |
| Применение в production | Учебные задачи | Meta, OpenAI, Anthropic, HF | Google DeepMind, Midjourney |

```figure
dropout-mask
```

## Создайте это

3-слойный MLP, обученный на MNIST с использованием только примитивов PyTorch. Без высокоуровневых обёрток. Без `torchvision.datasets`. Мы скачиваем и разбираем сырые данные сами.

### Шаг 1: Загрузка MNIST из сырых файлов

MNIST поставляется в виде 4 gzip-файлов: обучающие изображения (60 000 x 28 x 28), обучающие метки, тестовые изображения (10 000 x 28 x 28), тестовые метки. Мы скачиваем их и разбираем бинарный формат.

```python
import torch
import torch.nn as nn
import struct
import gzip
import urllib.request
import os

def download_mnist(path="./mnist_data"):
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    os.makedirs(path, exist_ok=True)
    for f in files:
        filepath = os.path.join(path, f)
        if not os.path.exists(filepath):
            urllib.request.urlretrieve(base_url + f, filepath)

def load_images(filepath):
    with gzip.open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = f.read()
        images = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        images = images.reshape(num, rows * cols).float() / 255.0
    return images

def load_labels(filepath):
    with gzip.open(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = f.read()
        labels = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
    return labels
```

### Шаг 2: Определение модели

3-слойный MLP: 784 -> 256 -> 128 -> 10. Активации ReLU. Dropout для регуляризации. Без batch norm — для простоты.

```python
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)
```

Выходной слой выдаёт 10 сырых логитов (по одному на цифру). Без softmax — `CrossEntropyLoss` обрабатывает это внутри себя.

Количество параметров: 784*256 + 256 + 256*128 + 128 + 128*10 + 10 = 235 146. По современным меркам — крошечное. У GPT-2 small — 124 млн. Это обучается за секунды.

### Шаг 3: Цикл обучения

Канонический шаблон forward-loss-backward-step.

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total
```

Обратите внимание на `torch.no_grad()` во время оценки. Это отключает autograd, уменьшая использование памяти и ускоряя инференс. Без этого PyTorch строит граф вычислений, который вы никогда не используете.

### Шаг 4: Соединяем всё вместе

```python
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    download_mnist()
    train_images = load_images("./mnist_data/train-images-idx3-ubyte.gz")
    train_labels = load_labels("./mnist_data/train-labels-idx1-ubyte.gz")
    test_images = load_images("./mnist_data/t10k-images-idx3-ubyte.gz")
    test_labels = load_labels("./mnist_data/t10k-labels-idx1-ubyte.gz")

    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )

    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Parameters: {num_params:,}")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print()

    for epoch in range(10):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        print(
            f"Epoch {epoch+1:2d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

    torch.save(model.state_dict(), "mnist_mlp.pt")
    print(f"\nModel saved to mnist_mlp.pt")
    print(f"Final test accuracy: {test_acc:.4f}")
```

Ожидаемый результат после 10 эпох: ~97,8% точности на тесте. Время обучения на CPU: ~30 секунд. На GPU: ~5 секунд. На вашем мини-фреймворке с той же архитектурой: ~45 минут.

## Используйте это

### Краткое сравнение: мини-фреймворк vs PyTorch

| Мини-фреймворк (Урок 10) | PyTorch |
|---------------------------|---------|
| `model = Sequential(Linear(784, 256), ReLU(), ...)` | `model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), ...)` |
| `pred = model.forward(x)` | `pred = model(x)` |
| `optimizer.zero_grad()` | `optimizer.zero_grad()` |
| `grad = criterion.backward()` затем `model.backward(grad)` | `loss.backward()` |
| `optimizer.step()` | `optimizer.step()` |
| Без GPU | `model.to("cuda")` |
| Ручной backward для каждого модуля | Autograd делает всё сам |

Интерфейс почти идентичен. Разница — во всём, что находится под капотом.

### Сохранение и загрузка моделей

```python
torch.save(model.state_dict(), "model.pt")

model = MNISTModel()
model.load_state_dict(torch.load("model.pt", weights_only=True))
model.eval()
```

Всегда сохраняйте `state_dict()` (словарь параметров), а не сам объект модели. Сохранение объекта модели использует pickle, что ломается при рефакторинге кода. State dict'ы переносимы.

### Планирование скорости обучения

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10
)
for epoch in range(10):
    train_one_epoch(model, train_loader, criterion, optimizer, device)
    scheduler.step()
```

PyTorch поставляет более 15 планировщиков: StepLR, ExponentialLR, CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau. Все они подключаются к одному и тому же интерфейсу оптимизатора.

## Поставка

Этот урок производит два артефакта:

- `outputs/prompt-pytorch-debugger.md` — промпт для диагностики распространённых сбоев обучения в PyTorch
- `outputs/skill-pytorch-patterns.md` — справочник навыков по паттернам обучения в PyTorch

## Упражнения

1. **Добавьте batch normalization.** Вставьте `nn.BatchNorm1d` после каждого линейного слоя (перед активацией). Сравните точность на тесте и скорость обучения с версией, использующей только dropout. Batch norm должна достигать 98%+ за меньшее число эпох.

2. **Реализуйте поиск learning rate.** Обучайте одну эпоху с экспоненциально возрастающим learning rate (от 1e-7 до 1.0). Постройте график loss от LR. Оптимальный LR находится чуть перед тем, как loss начинает расти. Используйте это, чтобы подобрать лучший LR для модели MNIST.

3. **Перенесите на GPU со смешанной точностью.** Добавьте `torch.amp.autocast` и `GradScaler` в цикл обучения. Измерьте пропускную способность (образцов в секунду) со смешанной точностью и без неё на GPU. На A100 ожидайте ~2-кратного ускорения.

4. **Создайте собственный Dataset.** Скачайте Fashion-MNIST (тот же формат, что и MNIST, но с предметами одежды). Реализуйте класс `FashionMNISTDataset(Dataset)` с `__getitem__` и `__len__`. Обучите ту же MLP и сравните точность. Fashion-MNIST сложнее — ожидайте ~88% против ~98%.

5. **Замените Adam на SGD + momentum.** Обучайте с `SGD(params, lr=0.01, momentum=0.9)`. Сравните кривые сходимости. Затем добавьте планировщик `CosineAnnealingLR` и посмотрите, догонит ли SGD Adam к 10-й эпохе.

## Ключевые термины

| Термин | Что говорят люди | Что это на самом деле означает |
|------|----------------|----------------------|
| Tensor | «Многомерный массив» | Типизированный, учитывающий устройство массив с поддержкой автоматического дифференцирования, встроенной в каждую операцию |
| Autograd | «Автоматический backprop» | Система на основе «ленты», которая записывает операции во время прямого прохода, а затем воспроизводит их в обратном порядке для вычисления точных градиентов |
| nn.Module | «Слой» | Базовый класс для любого дифференцируемого вычислительного блока — регистрирует параметры, поддерживает вложенность, обрабатывает режимы train/eval |
| state_dict | «Веса модели» | OrderedDict, отображающий имена параметров на тензоры — переносимое, сериализуемое представление обученной модели |
| .backward() | «Вычислить градиенты» | Обойти граф вычислений в обратном порядке, вычисляя и накапливая градиенты для каждого листового тензора с requires_grad=True |
| .to(device) | «Перенести на GPU» | Рекурсивно перенести все параметры и буферы на указанное устройство (CPU, CUDA, MPS) |
| DataLoader | «Конвейер данных» | Итератор, который пакетирует, перемешивает и опционально параллелизует загрузку данных из Dataset |
| Mixed precision | «Использовать float16» | Обучение с float16 в прямом/обратном проходе ради скорости при сохранении мастер-весов в float32 для численной устойчивости |
| Eager execution | «Выполнить это сейчас» | Операции выполняются немедленно при вызове, а не откладываются до последующего этапа компиляции — ключевое проектное решение, отличающее PyTorch от TF 1.x |
| zero_grad | «Сбросить градиенты» | Обнулить градиенты всех параметров перед следующим обратным проходом, поскольку PyTorch по умолчанию накапливает градиенты |

## Дополнительное чтение

- Paszke et al., «PyTorch: An Imperative Style, High-Performance Deep Learning Library» (2019) — оригинальная статья, объясняющая компромиссы в дизайне PyTorch
- PyTorch Tutorials: «Learning PyTorch with Examples» (https://pytorch.org/tutorials/beginner/pytorch_with_examples.html) — официальный путь от тензоров до nn.Module
- PyTorch Performance Tuning Guide (https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) — смешанная точность, воркеры DataLoader, закреплённая память и другие production-оптимизации
- Horace He, «Making Deep Learning Go Brrrr» (https://horace.io/brrr_intro.html) — почему обучение на GPU быстрое, со стратегиями оптимизации, специфичными для PyTorch