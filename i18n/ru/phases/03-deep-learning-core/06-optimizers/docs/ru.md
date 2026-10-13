# Оптимизаторы

> Градиентный спуск говорит вам, в каком направлении двигаться. Он ничего не говорит о том, насколько далеко и как быстро. SGD — это компас. Adam — это GPS с данными о трафике.

**Type:** Build
**Languages:** Python
**Prerequisites:** Lesson 03.05 (Loss Functions)
**Time:** ~75 минут

## Learning Objectives

- Реализовать с нуля на Python оптимизаторы SGD, SGD с моментом, Adam и AdamW
- Объяснить, как коррекция смещения (bias correction) в Adam компенсирует нулевую инициализацию оценок моментов на начальных шагах обучения
- Продемонстрировать, почему AdamW даёт лучшую обобщающую способность, чем Adam с L2-регуляризацией, на одной и той же задаче
- Выбрать подходящий оптимизатор и гиперпараметры по умолчанию для трансформеров, CNN, GAN и дообучения (fine-tuning)

## The Problem

Вы вычислили градиенты. Вы знаете, что вес #4721 нужно уменьшить на 0.003, чтобы снизить loss. Но 0.003 в каких единицах? Масштабированных чем? И нужно ли двигаться на столько же на шаге 1, сколько на шаге 1000?

Ванильный градиентный спуск применяет одну и ту же скорость обучения ко всем параметрам на каждом шаге: w = w - lr * gradient. Это создаёт три проблемы, которые на практике делают обучение нейронных сетей мучительным.

Первая — осцилляция. Ландшафт функции потерь редко имеет форму гладкой чаши. Чаще это длинная узкая долина. Градиент указывает поперёк долины (крутое направление), а не вдоль неё (пологое направление). Градиентный спуск мечется поперёк узкого измерения, делая крошечный прогресс вдоль полезного. Вы это видели: loss падает быстро, затем выходит на плато — не потому что модель сошлась, а потому что она осциллирует.

Вторая — одна скорость обучения для всех параметров ошибочна. Одним весам нужны большие обновления (они на ранней, недообученной стадии). Другим — крошечные (они близки к оптимальному значению). Скорость обучения, подходящая для первых, разрушает вторые, и наоборот.

Третья — седловые точки. В высоких размерностях ландшафт функции потерь имеет обширные плоские области, где градиент близок к нулю. Ванильный SGD ползёт через них со скоростью градиента, которая фактически равна нулю. Модель выглядит застрявшей. На самом деле она не застряла — она находится в плоской области, за которой есть полезный спуск. Но у SGD нет механизма, чтобы протолкнуться через неё.

Adam решает все три проблемы. Он поддерживает два скользящих среднего на параметр — среднее значение градиента (момент, решает осцилляцию) и среднее значение квадрата градиента (адаптивная скорость, решает проблему разных масштабов). В сочетании с коррекцией смещения для первых нескольких шагов это даёт единый оптимизатор, который работает на 80% задач с гиперпараметрами по умолчанию. Этот урок строит его с нуля, чтобы вы точно понимали, когда и почему он не справляется с оставшимися 20%.

## The Concept

### Стохастический градиентный спуск (SGD)

Простейший оптимизатор. Вычислите градиент на мини-батче и сделайте шаг в противоположном направлении.

```
w = w - lr * gradient
```

«Стохастический» означает, что вы используете случайное подмножество (мини-батч) данных для оценки градиента, а не весь датасет. Этот шум на самом деле полезен — он помогает избегать острых локальных минимумов. Но этот же шум вызывает осцилляцию.

Скорость обучения — единственный регулятор. Слишком высокая: loss расходится. Слишком низкая: обучение занимает вечность. Оптимальное значение зависит от архитектуры, данных, размера батча и текущей стадии обучения. Для ванильного SGD на современных сетях типичные значения варьируются от 0.01 до 0.1. Но даже в рамках одного запуска обучения идеальная скорость обучения меняется.

### Момент (Momentum)

Аналогия с катящимся вниз по склону шаром затёрта, но точна. Вместо того чтобы делать шаг только по градиенту, вы поддерживаете скорость (velocity), которая накапливает прошлые градиенты.

```
m_t = beta * m_{t-1} + gradient
w = w - lr * m_t
```

Beta (обычно 0.9) контролирует, сколько истории сохранять. При beta = 0.9 момент приблизительно равен среднему последних 10 градиентов (1 / (1 - 0.9) = 10).

Почему это решает осцилляцию: градиенты, указывающие в одном направлении, накапливаются. Градиенты, меняющие направление, взаимно гасятся. В той узкой долине компонента «поперёк» меняет знак на каждом шаге и подавляется. Компонента «вдоль» остаётся согласованной и усиливается. Результат — плавное ускорение в полезном направлении.

Реальные цифры: один SGD на плохо обусловленном ландшафте функции потерь может потребовать 10 000 шагов. SGD с моментом (beta=0.9) обычно занимает 3000–5000 шагов на той же задаче. Ускорение далеко не маргинальное.

### RMSProp

Первый метод адаптивной скорости обучения на параметр, который реально заработал. Предложен Хинтоном в лекции на Coursera (никогда не был формально опубликован).

```
s_t = beta * s_{t-1} + (1 - beta) * gradient^2
w = w - lr * gradient / (sqrt(s_t) + epsilon)
```

s_t отслеживает скользящее среднее квадратов градиентов. Параметры с устойчиво большими градиентами делятся на большое число (меньшая эффективная скорость обучения). Параметры с маленькими градиентами делятся на маленькое число (большая эффективная скорость обучения).

Это решает проблему «одной скорости обучения для всех параметров». Вес, который уже получал крупные обновления, вероятно близок к целевому значению — его нужно замедлить. Вес, получавший крошечные обновления, возможно, недообучен — его нужно ускорить.

Epsilon (обычно 1e-8) предотвращает деление на ноль, когда параметр ещё не обновлялся.

### Adam: момент + RMSProp

Adam объединяет обе идеи. Он поддерживает два экспоненциальных скользящих средних на параметр:

```
m_t = beta1 * m_{t-1} + (1 - beta1) * gradient        (first moment: mean)
v_t = beta2 * v_{t-1} + (1 - beta2) * gradient^2       (second moment: variance)
```

**Коррекция смещения** — ключевая деталь, которую пропускает большинство объяснений. На шаге 1, m_1 = (1 - beta1) * gradient. При beta1 = 0.9 это 0.1 * gradient — в десять раз меньше, чем нужно. Скользящее среднее ещё не "разогрелось". Коррекция смещения компенсирует это:

```
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
```

На шаге 1 при beta1 = 0.9: m_hat = m_1 / (1 - 0.9) = m_1 / 0.1 = фактический градиент. На шаге 100: (1 - 0.9^100) приблизительно равно 1.0, так что коррекция исчезает. Коррекция смещения имеет значение для первых ~10 шагов и становится неактуальной после ~50.

Обновление:

```
w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
```

Значения Adam по умолчанию: lr = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8. Эти значения по умолчанию работают на 80% задач. Если нет, сначала меняйте lr. Затем beta2. Почти никогда не меняйте beta1 или epsilon.

### AdamW: правильный weight decay

L2-регуляризация добавляет lambda * w^2 к loss. В ванильном SGD это эквивалентно weight decay (вычитанию lambda * w из веса на каждом шаге). В Adam эта эквивалентность нарушается.

Инсайт Лошчилова и Хуттера (Loshchilov & Hutter): когда вы добавляете L2 к loss, а затем Adam обрабатывает градиент, адаптивная скорость обучения масштабирует и член регуляризации тоже. Параметры с большой дисперсией градиента получают меньше регуляризации. Параметры с маленькой дисперсией — больше. Это не то, что вам нужно — вам нужна равномерная регуляризация независимо от статистики градиента.

AdamW исправляет это, применяя weight decay напрямую к весам, после обновления Adam:

```
w = w - lr * m_hat / (sqrt(v_hat) + epsilon) - lr * lambda * w
```

Член weight decay (lr * lambda * w) не масштабируется адаптивным фактором Adam. Каждый параметр получает одинаковое пропорциональное уменьшение.

Это кажется мелкой деталью. Это не так. AdamW сходится к лучшим решениям, чем Adam + L2-регуляризация, практически на любой задаче. Это оптимизатор по умолчанию в PyTorch для обучения трансформеров, диффузионных моделей и большинства современных архитектур. BERT, GPT, LLaMA, Stable Diffusion — все обучены с AdamW.

### Скорость обучения: самый важный гиперпараметр

```mermaid
graph TD
    LR["Learning Rate"] --> TooHigh["Too high (lr > 0.01)"]
    LR --> JustRight["Just right"]
    LR --> TooLow["Too low (lr < 0.00001)"]

    TooHigh --> Diverge["Loss explodes<br/>NaN weights<br/>Training crashes"]
    JustRight --> Converge["Loss decreases steadily<br/>Reaches good minimum<br/>Generalizes well"]
    TooLow --> Stall["Loss decreases slowly<br/>Gets stuck in suboptimal minimum<br/>Wastes compute"]

    JustRight --> Schedule["Usually needs scheduling"]
    Schedule --> Warmup["Warmup: ramp from 0 to max<br/>First 1-10% of training"]
    Schedule --> Decay["Decay: reduce over time<br/>Cosine or linear"]
```

Если вы настраиваете только один гиперпараметр, настраивайте скорость обучения. Изменение скорости обучения в 10 раз имеет большее значение, чем любое архитектурное решение, которое вы примете. Распространённые значения по умолчанию:

- SGD: lr = от 0.01 до 0.1
- Adam/AdamW: lr = от 1e-4 до 3e-4
- Дообучение (fine-tuning) предобученных моделей: lr = от 1e-5 до 5e-5
- Разогрев (warmup) скорости обучения: линейное нарастание в течение первых 1–10% шагов

### Сравнение оптимизаторов

```mermaid
flowchart LR
    subgraph "Optimization Path"
        SGD_P["SGD<br/>Oscillates across valley<br/>Slow but finds flat minima"]
        Mom_P["SGD + Momentum<br/>Smoother path<br/>3x faster than SGD"]
        Adam_P["Adam<br/>Adapts per-parameter<br/>Fast convergence"]
        AdamW_P["AdamW<br/>Adam + proper decay<br/>Best generalization"]
    end
    SGD_P --> Mom_P --> Adam_P --> AdamW_P
```

### Когда какой оптимизатор побеждает

```mermaid
flowchart TD
    Task["What are you training?"] --> Type{"Model type?"}

    Type -->|"Transformer / LLM"| AdamW["AdamW<br/>lr=1e-4, wd=0.01-0.1"]
    Type -->|"CNN / ResNet"| SGD_M["SGD + Momentum<br/>lr=0.1, momentum=0.9"]
    Type -->|"GAN"| Adam2["Adam<br/>lr=2e-4, beta1=0.5"]
    Type -->|"Fine-tuning"| AdamW2["AdamW<br/>lr=2e-5, wd=0.01"]
    Type -->|"Don't know yet"| Default["Start with AdamW<br/>lr=3e-4, wd=0.01"]
```

```figure
optimizer-trajectory
```

## Build It

### Шаг 1: Ванильный SGD

```python
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
```

### Шаг 2: SGD с моментом

```python
class SGDMomentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocities = None

    def step(self, params, grads):
        if self.velocities is None:
            self.velocities = [0.0] * len(params)
        for i in range(len(params)):
            self.velocities[i] = self.beta * self.velocities[i] + grads[i]
            params[i] -= self.lr * self.velocities[i]
```

### Шаг 3: Adam

```python
import math

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
```

### Шаг 4: AdamW

```python
class AdamW:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
            params[i] -= self.lr * self.weight_decay * params[i]
```

### Шаг 5: Сравнение обучения

Обучите одну и ту же двухслойную сеть на датасете с кругом (circle dataset) из урока 05 со всеми четырьмя оптимизаторами. Сравните сходимость.

```python
import random

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class OptimizerTestNetwork:
    def __init__(self, optimizer, hidden_size=8):
        random.seed(0)
        self.hidden_size = hidden_size
        self.optimizer = optimizer

        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def get_params(self):
        params = []
        for row in self.w1:
            params.extend(row)
        params.extend(self.b1)
        params.extend(self.w2)
        params.append(self.b2)
        return params

    def set_params(self, params):
        idx = 0
        for i in range(self.hidden_size):
            for j in range(2):
                self.w1[i][j] = params[idx]
                idx += 1
        for i in range(self.hidden_size):
            self.b1[i] = params[idx]
            idx += 1
        for i in range(self.hidden_size):
            self.w2[i] = params[idx]
            idx += 1
        self.b2 = params[idx]

    def forward(self, x):
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(max(0.0, z))

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def compute_grads(self, target):
        eps = 1e-15
        p = max(eps, min(1 - eps, self.out))
        d_loss = -(target / p) + (1 - target) / (1 - p)
        d_sigmoid = self.out * (1 - self.out)
        d_out = d_loss * d_sigmoid

        grads = [0.0] * (self.hidden_size * 2 + self.hidden_size + self.hidden_size + 1)
        idx = 0
        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            d_h = d_out * self.w2[i] * d_relu
            grads[idx] = d_h * self.x[0]
            grads[idx + 1] = d_h * self.x[1]
            idx += 2

        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            grads[idx] = d_out * self.w2[i] * d_relu
            idx += 1

        for i in range(self.hidden_size):
            grads[idx] = d_out * self.h[i]
            idx += 1

        grads[idx] = d_out
        return grads

    def train(self, data, epochs=300):
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for x, y in data:
                pred = self.forward(x)
                grads = self.compute_grads(y)
                params = self.get_params()
                self.optimizer.step(params, grads)
                self.set_params(params)

                eps = 1e-15
                p = max(eps, min(1 - eps, pred))
                total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            losses.append((avg_loss, accuracy))
            if epoch % 75 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")
        return losses
```

## Use It

Оптимизаторы PyTorch обрабатывают группы параметров, обрезку градиента (gradient clipping) и расписания скорости обучения (learning rate scheduling):

```python
import torch
import torch.optim as optim

model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10),
)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(torch.randn(32, 784))
    loss = torch.nn.functional.cross_entropy(output, torch.randint(0, 10, (32,)))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
```

Схема всегда одна: zero_grad, forward, loss, backward, (clip), step, (schedule). Запомните этот порядок. Ошибка в нём (например, вызов scheduler.step() до optimizer.step()) — распространённый источник трудноуловимых багов.

Для CNN многие практики по-прежнему предпочитают SGD + момент (lr=0.1, momentum=0.9, weight_decay=1e-4) со ступенчатым или косинусным расписанием. SGD находит более пологие минимумы, которые часто лучше обобщаются. Для трансформеров и LLM AdamW с warmup + косинусным затуханием — универсальный выбор по умолчанию. Не идите против консенсуса без измеримой причины.

## Ship It

Этот урок производит:
- `outputs/prompt-optimizer-selector.md` — подсказку для принятия решения при выборе подходящего оптимизатора и скорости обучения для любой архитектуры

## Exercises

1. Реализуйте момент Нестерова (Nesterov momentum), при котором вы вычисляете градиент в «опережающей» позиции (w - lr * beta * v), а не в текущей. Сравните сходимость со стандартным моментом на датасете с кругом.

2. Реализуйте расписание разогрева скорости обучения (warmup): линейное нарастание от 0 до max_lr в течение первых 10% шагов обучения, затем косинусное затухание до 0. Обучите с Adam + warmup и с Adam без warmup. Измерьте, сколько эпох требуется для достижения 90% точности на датасете с кругом.

3. Отслеживайте эффективную скорость обучения для каждого параметра во время обучения Adam. Эффективная скорость равна lr * m_hat / (sqrt(v_hat) + eps). Постройте распределение эффективных скоростей после 10, 50 и 200 шагов. Все ли параметры обновляются с одинаковой скоростью?

4. Реализуйте обрезку градиента (gradient clipping) по глобальной норме. Установите максимальную норму градиента равной 1.0. Обучите с обрезкой и без неё, используя высокую скорость обучения (lr=0.01 для Adam). Посчитайте, сколько запусков расходится (loss уходит в NaN) с обрезкой и без неё на 10 случайных сидах (seeds).

5. Сравните Adam и AdamW на сети с большими весами. Инициализируйте все веса случайными значениями в диапазоне [-5, 5] (намного больше обычного). Обучайте 200 эпох с weight_decay=0.1. Постройте график L2-нормы весов в процессе обучения для обоих оптимизаторов. AdamW должен показать более быстрое уменьшение весов.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Скорость обучения (Learning rate) | «Размер шага» | Скалярный множитель на обновлении градиента; единственный наиболее влиятельный гиперпараметр при обучении |
| SGD | «Базовый градиентный спуск» | Стохастический градиентный спуск: обновление весов путём вычитания lr * gradient, вычисленного на мини-батче |
| Момент (Momentum) | «Аналогия с катящимся шаром» | Экспоненциальное скользящее среднее прошлых градиентов; подавляет осцилляцию и ускоряет движение в согласованных направлениях |
| RMSProp | «Адаптивная скорость обучения» | Делит градиент каждого параметра на скользящий RMS его недавних градиентов; выравнивает скорости обучения |
| Adam | «Оптимизатор по умолчанию» | Объединяет момент (первый момент) и RMSProp (второй момент) с коррекцией смещения для начальных шагов |
| AdamW | «Adam, сделанный правильно» | Adam с развязанным (decoupled) weight decay; применяет регуляризацию напрямую к весам, а не через градиент |
| Коррекция смещения (Bias correction) | «Разогрев для скользящих средних» | Деление на (1 - beta^t) для компенсации нулевой инициализации оценок моментов Adam |
| Weight decay | «Уменьшение весов» | Вычитание доли значения веса на каждом шаге; регуляризатор, штрафующий большие веса |
| Расписание скорости обучения (Learning rate schedule) | «Изменение lr во времени» | Функция, которая регулирует скорость обучения в процессе обучения; warmup + косинусное затухание — современный выбор по умолчанию |
| Обрезка градиента (Gradient clipping) | «Ограничение нормы градиента» | Уменьшение масштаба вектора градиента, когда его норма превышает порог; предотвращает взрывной рост обновлений градиента |

## Further Reading

- Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014) — оригинальная статья об Adam с анализом сходимости и выводом коррекции смещения
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2017) — доказали, что L2-регуляризация и weight decay не эквивалентны в Adam, и предложили AdamW
- Smith, "Cyclical Learning Rates for Training Neural Networks" (2017) — представил LR range test и циклические расписания, устраняющие необходимость настройки фиксированной скорости обучения
- Ruder, "An Overview of Gradient Descent Optimization Algorithms" (2016) — лучший единый обзор всех вариантов оптимизаторов с чёткими сравнениями и интуитивными объяснениями