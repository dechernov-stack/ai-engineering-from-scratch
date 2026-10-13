# Классификация изображений

> Классификатор — это функция, отображающая пиксели в распределение вероятностей по классам. Всё остальное — сантехника.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 Lesson 09 (Model Evaluation), Phase 3 Lesson 10 (Mini Framework), Phase 4 Lesson 03 (CNNs)
**Time:** ~75 минут

## Цели обучения

- Построить сквозной пайплайн классификации изображений на CIFAR-10: датасет, аугментация, модель, цикл обучения, оценка
- Объяснить роль каждого компонента (загрузчик данных, функция потерь, оптимизатор, планировщик скорости обучения, аугментация) и предсказать, как поломка каждого из них проявляется на кривой потерь
- Реализовать с нуля mixup, cutout и label smoothing и обосновать, когда каждый из них стоит добавлять
- Читать матрицу ошибок (confusion matrix) и таблицу precision/recall по классам, чтобы диагностировать проблемы датасета и модели сверх агрегированной точности

## Проблема

Любая продакшн-задача компьютерного зрения на каком-то уровне сводится к классификации изображений. Детекция классифицирует области. Сегментация классифицирует пиксели. Поиск по сходству ранжирует по близости к центроидам классов. Умение правильно делать классификацию — цикл работы с датасетом, политика аугментации, функция потерь, оценка — это навык, переносимый на любую другую задачу фазы.

Большинство багов в классификации — не в модели. Они живут в пайплайне: сломанная нормализация, неперемешанный обучающий набор, аугментация, искажающая метки, валидационная выборка, загрязнённая обучающими данными, скорость обучения, которая незаметно расходится после 30-й эпохи. CNN, которая при правильной настройке достигла бы 93% на CIFAR-10, при сломанной набирает обычно 70-75%, и при этом кривая потерь всё это время выглядит правдоподобно.

В этом уроке весь пайплайн собирается вручную, чтобы каждая часть была доступна для инспекции. Вы не будете использовать ничего из `torchvision.datasets`, что могло бы скрыть баг.

## Концепция

### Пайплайн классификации

```mermaid
flowchart LR
    A["Dataset<br/>(images + labels)"] --> B["Augment<br/>(random transforms)"]
    B --> C["Normalise<br/>(mean/std)"]
    C --> D["DataLoader<br/>(batch + shuffle)"]
    D --> E["Model<br/>(CNN)"]
    E --> F["Logits<br/>(N, C)"]
    F --> G["Cross-entropy loss"]
    F --> H["Argmax<br/>at eval"]
    G --> I["Backward"]
    I --> J["Optimizer step"]
    J --> K["Scheduler step"]
    K --> E

    style A fill:#dbeafe,stroke:#2563eb
    style E fill:#fef3c7,stroke:#d97706
    style G fill:#fecaca,stroke:#dc2626
    style H fill:#dcfce7,stroke:#16a34a
```

Каждая строка в этом цикле — место, где может жить баг. Cross-entropy принимает сырые логиты, а не выходы softmax, поэтому любой `model(x).softmax()` перед функцией потерь незаметно вычисляет неверный градиент. Аугментации применяются только ко входам, не к меткам — за исключением mixup, который смешивает и то, и другое. `optimizer.zero_grad()` должно происходить ровно один раз за шаг; пропуск этого накапливает градиенты и выглядит как дико нестабильная скорость обучения. Каждый из этих багов сглаживает кривую обучения, не выбрасывая ошибку.

### Cross-entropy, логиты и softmax

Классификатор выдаёт `C` чисел на изображение, называемых логитами. Применение softmax превращает их в распределение вероятностей:

```
softmax(z)_i = exp(z_i) / sum_j exp(z_j)
```

Cross-entropy измеряет отрицательную логарифмическую вероятность правильного класса:

```
CE(z, y) = -log( softmax(z)_y )
        = -z_y + log( sum_j exp(z_j) )
```

Правая форма — численно устойчивая (log-sum-exp). `nn.CrossEntropyLoss` в PyTorch объединяет softmax + NLL в одну операцию и принимает сырые логиты напрямую. Применение softmax вручную перед этим — почти всегда баг: вы вычисляете log(softmax(softmax(z))) — бессмысленную величину.

### Почему работает аугментация

У CNN есть индуктивное смещение в пользу трансляционной инвариантности (благодаря разделяемым весам), но нет встроенной инвариантности к кропам, отражениям, цветовому джиттеру или окклюзиям. Единственный способ научить её этим инвариантностям — показать ей пиксели, которые эти инвариантности задействуют. Каждое случайное преобразование во время обучения — это способ сказать: «эти два изображения имеют одну и ту же метку; выучи признаки, игнорирующие различие».

```
Original crop:  "dog facing left"
Flip:           "dog facing right"       <- same label, different pixels
Rotate(+15):    "dog, slight tilt"
Colour jitter:  "dog in warmer light"
RandomErasing:  "dog with patch missing"
```

Правило: аугментация должна сохранять метку. Cutout и поворот на цифре могут превратить «6» в «9»; для такого датасета используют меньшие диапазоны поворота и выбирают аугментации, уважающие специфичные для цифр инвариантности.

### Mixup и cutmix

Обычная аугментация преобразует пиксели, но оставляет метки one-hot. **Mixup** и **cutmix** нарушают это правило, интерполируя и то, и другое.

```
Mixup:
  lambda ~ Beta(a, a)
  x = lambda * x_i + (1 - lambda) * x_j
  y = lambda * y_i + (1 - lambda) * y_j

Cutmix:
  paste a random rectangle of x_j into x_i
  y = area-weighted mix of y_i and y_j
```

Почему это помогает: модель перестаёт заучивать острые one-hot цели и учится интерполировать между классами. Потери на обучении растут, точность на тесте растёт. Это самое дешёвое обновление робастности для любого классификатора.

### Label smoothing

Родственник mixup. Вместо обучения против `[0, 0, 1, 0, 0]` обучайте против `[eps/C, eps/C, 1-eps, eps/C, eps/C]` для небольшого `eps`, например 0.1. Не даёт модели производить произвольно резкие логиты и улучшает калибровку почти без затрат. Встроено в `nn.CrossEntropyLoss(label_smoothing=0.1)` начиная с PyTorch 1.10.

### Оценка сверх точности

Агрегированная точность скрывает дисбаланс. Бинарный классификатор 90-10, всегда предсказывающий класс большинства, набирает 90%. Инструменты, которые реально показывают, что происходит:

- **Точность по классам** — одно число на класс; сразу выявляет недообучающиеся категории.
- **Матрица ошибок (confusion matrix)** — сетка C x C, где строка i столбец j = количество примеров истинного класса i, предсказанных как класс j; диагональ — верные ответы, недиагональные элементы — там, где живёт ваша модель.
- **Top-1 / Top-5** — находится ли правильный класс среди 1 или 5 топ-предсказаний; Top-5 важна для ImageNet, потому что классы вроде «норвичский терьер» и «норфолкский терьер» по-настоящему неоднозначны.
- **Калибровка (ECE)** — предсказание с уверенностью 0.8 действительно оказывается верным в 80% случаев? Современные сети систематически переуверены; исправляется temperature scaling или label smoothing.

```figure
receptive-field
```

## Постройте это

### Шаг 1: Детерминированный синтетический датасет

CIFAR-10 живёт на диске. Чтобы сделать этот урок воспроизводимым и быстрым, построим синтетический датасет, похожий на CIFAR — RGB-изображения 32x32 со специфичной для класса структурой, которую модель должна выучить. Точно такой же пайплайн работает без изменений на реальном CIFAR-10.

```python
import numpy as np
import torch
from torch.utils.data import Dataset


def synthetic_cifar(num_per_class=1000, num_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    X = []
    Y = []
    for c in range(num_classes):
        centre = rng.uniform(0, 1, (3,))
        freq = 2 + c
        for _ in range(num_per_class):
            yy, xx = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij")
            r = np.sin(xx * freq) * 0.5 + centre[0]
            g = np.cos(yy * freq) * 0.5 + centre[1]
            b = (xx + yy) * 0.5 * centre[2]
            img = np.stack([r, g, b], axis=-1)
            img += rng.normal(0, 0.08, img.shape)
            img = np.clip(img, 0, 1)
            X.append(img.astype(np.float32))
            Y.append(c)
    X = np.stack(X)
    Y = np.array(Y)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


class ArrayDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = self.X[i]
        if self.transform is not None:
            img = self.transform(img)
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, int(self.Y[i])
```

Каждый класс получает собственную цветовую палитру и частотный паттерн, плюс гауссов шум, вынуждающий модель учить сигнал, а не заучивать пиксели. Десять классов, по тысяче изображений каждый, перемешанных.

### Шаг 2: Нормализация и аугментация

Два преобразования, присутствующие в каждом пайплайне компьютерного зрения.

```python
def standardize(mean, std):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    def _fn(img):
        return (img - mean) / std
    return _fn


def random_hflip(p=0.5):
    def _fn(img):
        if np.random.random() < p:
            return img[:, ::-1, :].copy()
        return img
    return _fn


def random_crop(pad=4):
    def _fn(img):
        h, w = img.shape[:2]
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        y = np.random.randint(0, 2 * pad)
        x = np.random.randint(0, 2 * pad)
        return padded[y:y + h, x:x + w, :]
    return _fn


def compose(*fns):
    def _fn(img):
        for fn in fns:
            img = fn(img)
        return img
    return _fn
```

Отражающий паддинг (reflect-pad) перед кропом, а не нулевой, потому что чёрные границы — это сигнал, который модель научилась бы бесполезно игнорировать.

### Шаг 3: Mixup

Смешивает два изображения и две метки внутри шага обучения. Реализовано как преобразование над батчем, чтобы оно жило рядом с прямым проходом, а не внутри датасета.

```python
def mixup_batch(x, y, num_classes, alpha=0.2):
    if alpha <= 0:
        return x, torch.nn.functional.one_hot(y, num_classes).float()
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[idx]
    y_onehot = torch.nn.functional.one_hot(y, num_classes).float()
    y_mixed = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return x_mixed, y_mixed


def soft_cross_entropy(logits, soft_targets):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()
```

`soft_cross_entropy` — это cross-entropy против распределения мягких меток. Он сводится к обычному one-hot случаю, когда цель — точно one-hot.

### Шаг 4: Цикл обучения

Полный рецепт: один проход по данным, градиенты ровно один раз за батч, планировщик шагает ровно один раз за эпоху.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_one_epoch(model, loader, optimizer, device, num_classes, use_mixup=True):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if use_mixup:
            x_m, y_soft = mixup_batch(x, y, num_classes)
            logits = model(x_m)
            loss = soft_cross_entropy(logits, y_soft)
        else:
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        # Training accuracy vs the un-mixed labels `y` is only an approximation
        # when mixup is on (the model saw soft targets, not y). Treat it as a
        # rough progress signal; rely on val accuracy for real performance.
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)
        for t, p in zip(y.cpu(), pred.cpu()):
            cm[t, p] += 1
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        correct += (pred == y).sum().item()
    return loss_sum / total, correct / total, cm
```

Пять инвариантов, которые проверяются каждый раз при написании цикла обучения:

1. `model.train()` перед обучением, `model.eval()` перед оценкой — переключает поведение dropout и batchnorm.
2. `.zero_grad()` перед `.backward()`.
3. `.item()` при накоплении метрик, чтобы ничего не держало вычислительный граф живым.
4. `@torch.no_grad()` во время оценки — экономит память и время, предотвращает тонкие ошибки.
5. Argmax по сырым логитам, не по softmax — тот же результат, на одну операцию меньше.

### Шаг 5: Соберите всё вместе

Используйте `TinyResNet` из предыдущего урока, обучите несколько эпох, оцените.

```python
from main import synthetic_cifar, ArrayDataset
from main import standardize, random_hflip, random_crop, compose
from main import mixup_batch, soft_cross_entropy
from main import train_one_epoch, evaluate
# TinyResNet comes from the previous lesson (03-cnns-lenet-to-resnet).
# Adjust the import path to wherever you stored the previous lesson's code.
from cnns_lenet_to_resnet import TinyResNet  # example placeholder

X, Y = synthetic_cifar(num_per_class=500)
split = int(0.9 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_val, Y_val = X[split:], Y[split:]

mean = [0.5, 0.5, 0.5]
std = [0.25, 0.25, 0.25]
train_tf = compose(random_hflip(), random_crop(pad=4), standardize(mean, std))
eval_tf = standardize(mean, std)

train_ds = ArrayDataset(X_train, Y_train, transform=train_tf)
val_ds = ArrayDataset(X_val, Y_val, transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyResNet(num_classes=10).to(device)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, 10, use_mixup=True)
    va_loss, va_acc, _ = evaluate(model, val_loader, device, 10)
    scheduler.step()
    print(f"epoch {epoch:2d}  lr {scheduler.get_last_lr()[0]:.4f}  "
          f"train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}")
```

На синтетическом датасете это достигает почти идеальной точности на валидации в пределах пяти эпох, и в этом суть: пайплайн корректен, модель способна выучить то, что можно выучить. Замените датасет на реальный CIFAR-10, и тот же цикл обучится до ~90% без изменений.

### Шаг 6: Прочитайте матрицу ошибок

Одна только точность никогда не говорит вам, где модель ошибается. Матрица ошибок говорит.

```python
def print_confusion(cm, labels=None):
    c = cm.shape[0]
    labels = labels or [str(i) for i in range(c)]
    print(f"{'':>6}" + "".join(f"{l:>5}" for l in labels))
    for i in range(c):
        row = cm[i].tolist()
        print(f"{labels[i]:>6}" + "".join(f"{v:>5}" for v in row))
    print()
    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    prec = tp / (tp + fp).clamp_min(1)
    rec = tp / (tp + fn).clamp_min(1)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-9)
    for i in range(c):
        print(f"{labels[i]:>6}  prec {prec[i]:.3f}  rec {rec[i]:.3f}  f1 {f1[i]:.3f}")

_, _, cm = evaluate(model, val_loader, device, 10)
print_confusion(cm)
```

Строки — истинные классы, столбцы — предсказания. Кластер недиагональных значений между классами 3 и 5 означает, что модель путает эти два класса, и даёт отправную точку для целевого сбора данных или специфичной для класса аугментации.

## Используйте это

`torchvision` оборачивает всё вышеперечисленное в идиоматические компоненты. Для реального CIFAR-10 полный пайплайн — это четыре строки плюс цикл обучения.

```python
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
train_tf = Compose([
    RandomCrop(32, padding=4, padding_mode="reflect"),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean, std),
])
eval_tf = Compose([ToTensor(), Normalize(mean, std)])

train_ds = CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
val_ds   = CIFAR10(root="./data", train=False, download=True, transform=eval_tf)
```

Две вещи, на которые стоит обратить внимание: среднее/стандартное отклонение **специфичны для датасета** — вычислены на обучающем наборе CIFAR-10, а не на ImageNet — а отражающий паддинг — стандартная в сообществе политика кропа. Копипаста статистик ImageNet сюда — это утечка точности примерно в 1%, которую никто не замечает, пока кто-нибудь не профилирует модель.

## Сдайте это

Этот урок производит:

- `outputs/prompt-classifier-pipeline-auditor.md` — промпт, который проверяет обучающий скрипт на пять инвариантов выше и выявляет первое нарушение.
- `outputs/skill-classification-diagnostics.md` — навык, который, получив матрицу ошибок и список имён классов, резюмирует ошибки по классам и предлагает самое эффективное единственное исправление.

## Упражнения

1. **(Лёгкое)** Обучите одну и ту же модель с mixup и без него в течение пяти эпох на синтетическом датасете. Постройте графики потерь на обучении и валидации для обоих случаев. Объясните, почему потери на обучении с mixup выше, но точность на валидации схожая или лучше.
2. **(Среднее)** Реализуйте Cutout — обнуление случайного квадрата 8x8 в каждом обучающем изображении — и проведите абляцию: без аугментации, hflip+crop, hflip+crop+cutout, hflip+crop+mixup. Сообщите точность на валидации для каждого варианта.
3. **(Сложное)** Постройте пайплайн для CIFAR-100 (100 классов, тот же размер входа) и воспроизведите обучение ResNet-34 в пределах 1% от опубликованной точности. Дополнительно: переберите три скорости обучения и два значения weight decay, логируйте в локальный CSV, постройте итоговую таблицу топ-путаниц по матрице ошибок.

## Ключевые термины

| Термин | Что говорят люди | Что это на самом деле означает |
|------|----------------|----------------------|
| Логиты | «Сырые выходы» | Вектор из C чисел на изображение до softmax; cross-entropy ожидает именно их, а не значения после softmax |
| Cross-entropy | «Функция потерь» | Отрицательная логарифмическая вероятность правильного класса; объединяет log-softmax и NLL в одной устойчивой операции |
| DataLoader | «Батчер» | Оборачивает датасет перемешиванием, батчингом и (опционально) многопоточной загрузкой; на него списывают половину багов обучения |
| Аугментация | «Случайные преобразования» | Любое попиксельное преобразование во время обучения, сохраняющее метку; обучает инвариантностям, которых у CNN нет от природы |
| Mixup / Cutmix | «Смешать два изображения» | Смешивание и входов, и меток, чтобы классификатор учил плавные интерполяции вместо жёстких границ |
| Label smoothing | «Более мягкие цели» | Замена one-hot на (1-eps, eps/(C-1), ...); улучшает калибровку и немного повышает точность |
| Top-k точность | «Top-5» | Правильный класс входит в k предсказаний с наибольшей вероятностью; используется на датасетах с по-настоящему неоднозначными классами |
| Матрица ошибок | «Где живут ошибки» | Таблица C x C, где элемент (i, j) считает изображения истинного класса i, предсказанные как j; диагональ верна, недиагональ говорит, что исправлять |

## Дополнительное чтение

- [CS231n: Training Neural Networks](https://cs231n.github.io/neural-networks-3/) — по-прежнему самый ясный обзор пайплайна обучения на одной странице
- [Bag of Tricks for Image Classification (He et al., 2019)](https://arxiv.org/abs/1812.01187) — все небольшие приёмы, которые вместе добавляют 3-4% к точности ResNet на ImageNet
- [mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)](https://arxiv.org/abs/1710.09412) — оригинальная статья про mixup; три страницы теории плюс убедительные эксперименты
- [Why temperature scaling matters (Guo et al., 2017)](https://arxiv.org/abs/1706.04599) — статья, доказавшая, что современные сети плохо откалиброваны, и исправившая это одним скалярным параметром