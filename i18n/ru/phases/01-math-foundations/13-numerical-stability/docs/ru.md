# Численная стабильность

> Floating point is a leaky abstraction. It will bite you during training, and you will not see it coming.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~120 minutes

## Learning Objectives

- Implement numerically stable softmax and log-sum-exp using the max-subtraction trick
- Identify overflow, underflow, and catastrophic cancellation in floating-point computations
- Verify analytical gradients against numerical gradients using centered finite differences
- Explain why bfloat16 is preferred over float16 for training and how loss scaling prevents gradient underflow

## The Problem

Ваша модель обучается три часа, а затем loss становится NaN. Вы добавляете print-оператор. Логиты в порядке на шаге 9000. На шаге 9001 они равны `inf`. На шаге 9002 каждый градиент равен `nan` и обучение мертво.

Или: ваша модель обучается до конца, но точность на 2% ниже, чем утверждает статья. Вы проверяете всё. Архитектура совпадает. Гиперпараметры совпадают. Данные совпадают. Проблема в том, что в статье использовался float32, а вы использовали float16 без правильного масштабирования. Тридцать два бита накопленной ошибки округления тихо съели вашу точность.

Или: вы реализуете cross-entropy loss с нуля. Он работает на малых логитах. Когда логиты превышают 100, он возвращает `inf`. Softmax переполнился, потому что `exp(100)` больше, чем может представить float32. Каждый фреймворк ML обрабатывает это двухстрочным трюком. Вы не знали, что трюк существует.

Численная стабильность — это не теоретическое беспокойство. Это разница между удачным и молчаливо неудавшимся этапом обучения. Каждая серьёзная ML-ошибка, которую вы будете отлаживать, в конечном итоге сводится к floating point.

## The Concept

### IEEE 754: How Computers Store Real Numbers

Компьютеры хранят вещественные числа в виде значений с плавающей точкой в соответствии со стандартом IEEE 754. Float состоит из трёх частей: бита знака, порядка (экспоненты) и мантиссы (значащей части).

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

Мантисса определяет точность (сколько значащих цифр). Экспонента определяет диапазон (насколько большие или малые числа могут быть).

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 даёт примерно 7 десятичных цифр точности. Это означает, что он может различить 1.0000001 и 1.0000002, но не 1.00000001 и 1.00000002. После 7 цифр всё становится шумом округления.

float16 даёт примерно 3 цифры. Наибольшее число, которое он может представить, — 65 504. Это угрожающе мало для ML, где логиты, градиенты и активации регулярно превышают это значение.

bfloat16 — ответ Google на проблему диапазона float16. У него одинаковая 8-битная экспонента с float32 (тот же диапазон, вплоть до 3.4e38), но только 7 битов мантиссы (меньше точности, чем float16). Для обучения нейронных сетей диапазон важнее точности, поэтому bfloat16 обычно выигрывает.

### Why 0.1 + 0.2 != 0.3

Число 0.1 не может быть представлено точно в двоичной плавающей точке. В базе 2 это периодическая дробь:

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32 обрезает это до 23 битов мантиссы. Хранимое значение приблизительно 0.100000001490116. Аналогично, 0.2 хранится примерно как 0.200000002980232. Их сумма — 0.300000004470348, а не 0.3.

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

Это важно для ML, потому что:

1. Сравнения loss типа `if loss < threshold` могут дать неправильные результаты
2. Накопление многих малых значений (обновления градиентов над тысячами шагов) отклоняется от истинной суммы
3. Контрольные суммы и тесты воспроизводимости не срабатывают, если вы сравниваете floats с `==`

Исправление: никогда не сравнивайте floats с `==`. Используйте `abs(a - b) < epsilon` или `math.isclose()`.

### Catastrophic Cancellation

Когда вы вычитаете две почти равные числа с плавающей точкой, значащие цифры сокращаются и вы остаются с шумом округления, преобразованным в ведущие цифры.

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

Это 19% относительной ошибки от одного вычитания. В ML это случается, когда вы:

- Вычисляете дисперсию данных с большим средним: `E[x^2] - E[x]^2` когда E[x] велико
- Вычитаете почти равные log-вероятности
- Вычисляете градиенты конечных разностей с слишком малым epsilon

Исправление: переформулируйте формулы, чтобы избежать вычитания больших, почти равных чисел. Для дисперсии используйте алгоритм Велфорда или сначала отцентрируйте данные. Для log-вероятностей работайте в log-пространстве на протяжении всего времени.

### Overflow and Underflow

Overflow происходит, когда результат слишком большой для представления. Underflow происходит, когда он слишком мал (ближе к нулю, чем наименьшее представимое положительное число).

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

Функция `exp()` — основной источник переполнения в ML:

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

Функция `log()` опасна в другом направлении:

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

В ML `exp()` появляется в softmax, sigmoid и вычислениях вероятности. `log()` появляется в cross-entropy, log-likelihood и KL-дивергенции. Комбинация `log(exp(x))` — минное поле без нужных трюков.

### The Log-Sum-Exp Trick

Прямое вычисление `log(sum(exp(x_i)))` численно опасно. Если какой-либо `x_i` большой, `exp(x_i)` переполняется. Если все `x_i` очень отрицательные, каждый `exp(x_i)` underflow в ноль и `log(0)` равен `-inf`.

Трюк: вычтите максимальное значение перед возведением в степень.

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

Почему это работает: после вычитания `max(x)`, наибольшая экспонента — это `exp(0) = 1`. Переполнение невозможно. По крайней мере один член суммы равен 1, поэтому сумма по крайней мере 1, и `log(1) = 0`. Underflow в `-inf` невозможен.

Доказательство:

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

Установите `c = max(x)` и переполнение исключено.

Этот трюк появляется везде в ML:
- Normalization softmax
- Вычисление cross-entropy loss
- Суммирование log-вероятностей в последовательностных моделях
- Mixture of Gaussians
- Вариационный вывод

### Why Softmax Needs the Max-Subtraction Trick

Softmax преобразует логиты в вероятности:

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

Без трюка логиты [100, 101, 102] вызывают переполнение:

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

С трюком вычтите max(x) = 102:

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

Вероятности одинаковы. Вычисление безопасно. Это не оптимизация. Это требование для корректности.

### NaN and Inf: Detection and Prevention

`nan` (Not a Number) и `inf` (бесконечность) распространяются вирально через вычисление. Один `nan` в обновлении градиента делает вес `nan`, что делает каждый последующий выход `nan`. Обучение мертво за один шаг.

Как появляется `inf`:
- `exp()` большого положительного числа
- Деление на ноль: `1.0 / 0.0`
- `float32` переполнение в накоплениях

Как появляется `nan`:
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- `sqrt()` отрицательного числа
- `log()` отрицательного числа
- Любая арифметика, включающая существующий `nan`

Обнаружение:

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

Стратегии профилактики:

1. Ограничьте входные значения `exp()`: `exp(clamp(x, -80, 80))`
2. Добавьте epsilon в знаменатели: `x / (y + 1e-8)`
3. Добавьте epsilon внутри `log()`: `log(x + 1e-8)`
4. Используйте стабильные реализации (log-sum-exp, stable softmax)
5. Gradient clipping для предотвращения взрыва веса
6. Проверяйте `nan`/`inf` после каждого forward pass во время отладки

### Numerical Gradient Checking

Аналитические градиенты (от обратного распространения) могут иметь ошибки. Численная проверка градиентов проверяет их путём вычисления градиентов с конечными разностями.

Формула центральной разности:

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

Это O(h^2) точность, намного лучше, чем прямая разность `(f(x+h) - f(x)) / h`, которая только O(h).

Выбор h: слишком большой и аппроксимация неправильна. Слишком малый и катастрофическое сокращение разрушает ответ. `h = 1e-5` к `1e-7` типично.

Проверка: вычислите относительную разницу между аналитическими и численными градиентами.

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

Правила большого пальца:
- relative_error < 1e-7: идеально, градиент верен
- relative_error < 1e-5: приемлемо, вероятно верно
- relative_error > 1e-3: что-то не так
- relative_error > 1: градиент совершенно неправильный

Всегда проверяйте градиенты при реализации нового слоя или функции потерь. PyTorch предоставляет `torch.autograd.gradcheck()` для этого.

### Mixed Precision Training

Современные GPU имеют специализированное оборудование (Tensor Cores), которое вычисляет матричные умножения float16 в 2-8 раз быстрее, чем float32. Обучение со смешанной точностью использует это:

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

Проблема чистого обучения float16: градиенты часто очень малы (1e-8 или меньше). Float16 underflow что-либо меньше ~6e-8 в ноль. Ваша модель перестаёт обучаться, потому что все обновления градиентов нулевые.

Исправление — масштабирование потерь:

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

Динамическое масштабирование потерь автоматически регулирует коэффициент масштабирования. Начните с большого значения (65536). Если градиенты переполняются в `inf`, уменьшите в два раза. Если N шагов проходят без переполнения, удвойте.

### bfloat16 vs float16: Why bfloat16 Wins for Training

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 имеет большую точность (10 битов мантиссы против 7), но ограниченный диапазон (макс ~65 504). bfloat16 имеет меньше точности, но тот же диапазон, что и float32 (макс ~3.4e38).

Для обучения нейронных сетей:

- Активации и логиты регулярно превышают 65 504 во время пиков обучения. float16 переполняется; bfloat16 это обрабатывает.
- Масштабирование потерь требуется для float16, но обычно не требуется для bfloat16, потому что его диапазон охватывает спектр величин градиентов.
- bfloat16 — это простое усечение float32: удалите нижние 16 битов мантиссы. Преобразование тривиально и без потерь по экспоненте.

float16 предпочтителен для вывода, где значения ограничены и точность важнее. bfloat16 предпочтителен для обучения, где диапазон важнее. Поэтому TPU и современные GPU NVIDIA (A100, H100) имеют встроенную поддержку bfloat16.

### Gradient Clipping

Взрывающиеся градиенты происходят, когда градиенты растут экспоненциально через много слоёв (частое явление в RNN, глубоких сетях и трансформерах). Один большой градиент может испортить все веса за один шаг.

Два типа clipping:

**Clip по значению:** ограничьте каждый элемент градиента независимо.

```
grad = clamp(grad, -max_val, max_val)
```

Простой, но может изменить направление вектора градиента.

**Clip по норме:** масштабируйте весь вектор градиента так, чтобы его норма не превышала порог.

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

Сохраняет направление градиента. Это то, что делает `torch.nn.utils.clip_grad_norm_()`. Это стандартный выбор.

Типичные значения: `max_norm=1.0` для трансформеров, `max_norm=0.5` для RL, `max_norm=5.0` для более простых сетей.

Gradient clipping — это не хак. Это механизм безопасности. Без него один выбросный batch может произвести градиент достаточно большой, чтобы испортить недели обучения.

### Normalization Layers as Numerical Stabilizers

Batch normalization, layer normalization и RMS normalization обычно представляются как регуляризаторы, которые помогают обучению сходиться. Они также являются числовыми стабилизаторами.

Без нормализации активации могут расти или сжиматься экспоненциально через слои:

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

Нормализация переценивает и перемасштабирует активации на каждом слое:

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon` (обычно 1e-5) предотвращает деление на ноль, когда все активации идентичны. Изученные параметры `gamma` и `beta` позволяют сети восстановить любой масштаб, который ей нужен.

Это держит значения в числовом безопасном диапазоне на протяжении всей сети, предотвращая как переполнение в forward pass, так и взрыв градиентов в backward pass.

### Common ML Numerical Bugs

**Ошибка: Loss — NaN после нескольких эпох.**
Причина: логиты выросли слишком большими, softmax переполнился. Или learning rate слишком высокий и веса разошлись.
Исправление: используйте стабильный softmax (вычитание максимума), уменьшите learning rate, добавьте gradient clipping.

**Ошибка: Loss застрял на log(num_classes).**
Причина: выходы модели — почти однородные вероятности. Часто означает исчезающие градиенты или модель вообще не учится.
Исправление: проверьте, что метки данных правильные, проверьте функцию потерь, проверьте на мертвые ReLU.

**Ошибка: Validation accuracy на 1-3% ниже ожидаемой.**
Причина: смешанная точность без правильного масштабирования потерь. Underflow градиента тихо обнуляет малые обновления.
Исправление: включите динамическое масштабирование потерь или переключитесь на bfloat16.

**Ошибка: Нормы градиентов 0.0 для некоторых слоёв.**
Причина: мертвые ReLU-нейроны (все входы отрицательные) или float16 underflow.
Исправление: используйте LeakyReLU или GELU, используйте масштабирование градиентов, проверьте инициализацию веса.

**Ошибка: Модель работает на одном GPU, но даёт разные результаты на другом.**
Причина: недетерминированный порядок накопления floating point. Параллельные редукции GPU суммируют в разных порядках на разном оборудовании, и сложение floating point не ассоциативно.
Исправление: примите малые разницы (1e-6) или установите `torch.use_deterministic_algorithms(True)` и примите штраф по скорости.

**Ошибка: `exp()` возвращает `inf` при вычислении потерь.**
Причина: сырые логиты переданы `exp()` без трюка вычитания максимума.
Исправление: используйте `torch.nn.functional.log_softmax()`, которая реализует log-sum-exp внутри.

**Ошибка: Обучение расходится после переключения с float32 на float16.**
Причина: float16 не может представить величины градиентов ниже 6e-8 или активации выше 65 504.
Исправление: используйте смешанную точность с масштабированием потерь (AMP) или используйте bfloat16 вместо этого.

```figure
logsumexp-stability
```

## Build It

### Step 1: Demonstrate floating point precision limits

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### Step 2: Implement naive vs stable softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### Step 3: Implement stable log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### Step 4: Implement stable cross-entropy

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### Step 5: Gradient checking

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## Use It

### Mixed precision simulation

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### Gradient clipping

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf detection

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

Смотрите `code/numerical.py` для полных реализаций со всеми продемонстрированными граничными случаями.

## Ship It

Этот урок производит:
- `code/numerical.py` со стабильным softmax, log-sum-exp, cross-entropy, проверкой градиентов и симуляцией смешанной точности
- `outputs/prompt-numerical-debugger.md` для диагностики NaN/Inf и численных проблем при обучении

Эти стабильные реализации перепоявляются в Phase 3 при создании цикла обучения и в Phase 4 при реализации механизмов внимания.

## Exercises

1. **Catastrophic cancellation.** Вычислите дисперсию [1000000.0, 1000001.0, 1000002.0], используя наивную формулу `E[x^2] - E[x]^2` в float32. Затем вычислите её, используя онлайн-алгоритм Велфорда. Сравните ошибки против истинной дисперсии (0.6667).

2. **Precision hunt.** Найдите наименьшее положительное значение float32 `x` такое, что `1.0 + x == 1.0` в Python. Это машинный epsilon. Проверьте, что он совпадает с `numpy.finfo(numpy.float32).eps`.

3. **Log-sum-exp edge cases.** Протестируйте вашу функцию `logsumexp_stable` с: (a) все значения равны, (b) одно значение намного больше остального, (c) все значения очень отрицательные (-1000). Убедитесь, что она даёт правильные результаты, когда наивная версия терпит неудачу.

4. **Gradient checking a neural network layer.** Реализуйте один слой linear `y = Wx + b` и его аналитический backward pass. Используйте `numerical_gradient` для проверки корректности матрицы весов 3x2.

5. **Loss scaling experiment.** Симулируйте обучение с float16: создайте случайные градиенты в диапазоне [1e-9, 1e-3], преобразуйте в float16 и измерьте, какая доля становится нулевой. Затем примените масштабирование потерь (умножьте на 1024), преобразуйте в float16, масштабируйте назад и снова измерьте нулевую долю.

## Key Terms

| Термин | Что говорят люди | Что это на самом деле означает |
|--------|------------------|-------------------------------|
| IEEE 754 | "Стандарт float" | Международный стандарт, определяющий форматы двоичной плавающей точки, правила округления и специальные значения (inf, nan). Каждый современный CPU и GPU реализует его. |
| Machine epsilon | "Предел точности" | Наименьшее значение e такое, что 1.0 + e != 1.0 в данном формате float. Для float32 это примерно 1.19e-7. |
| Catastrophic cancellation | "Потеря точности от вычитания" | Когда вычитаются почти равные числа с плавающей точкой, значащие цифры сокращаются и шум округления доминирует в результате. |
| Overflow | "Число слишком большое" | Результат превышает максимальное представимое значение и становится inf. exp(89) переполняет float32. |
| Underflow | "Число слишком малое" | Результат ближе к нулю, чем наименьшее представимое положительное число, и становится 0.0. exp(-104) underflow float32. |
| Log-sum-exp trick | "Сначала вычтите максимум" | Вычисление log(sum(exp(x))) путём факторизации exp(max(x)) для предотвращения переполнения и underflow. Используется в softmax, cross-entropy и логарифм-вероятностной математике. |
| Stable softmax | "Softmax, который не взрывается" | Вычитание max(logits) перед возведением в степень. Численно идентичный результат, переполнение невозможно. |
| Gradient checking | "Проверьте вашу обратную пропаганду" | Сравнение аналитических градиентов из обратного распространения с численными градиентами из конечных разностей для обнаружения ошибок реализации. |
| Mixed precision | "Float16 forward, float32 backward" | Использование lower-precision floats для критических по скорости операций и higher-precision floats для численно чувствительных операций. Типичное ускорение 2-3x. |
| Loss scaling | "Предотвратить underflow градиента" | Умножение потерь на большую константу перед backprop, чтобы градиенты оставались в представимом диапазоне float16, а затем деление на ту же константу перед обновлениями веса. |
| bfloat16 | "Brain floating point" | 16-битный формат Google с 8 битами экспоненты (тот же диапазон, что float32) и 7 битами мантиссы (меньше точности, чем float16). Предпочтителен для обучения. |
| Gradient clipping | "Ограничьте норму градиента" | Масштабирование вектора градиента так, чтобы его норма не превышала порог. Предотвращает взрывающиеся градиенты от испорчения весов. |
| NaN | "Not a Number" | Специальное значение float из неопределённых операций (0/0, inf-inf, sqrt(-1)). Распространяется через всю последующую арифметику. |
| Inf | "Бесконечность" | Специальное значение float из переполнения или деления на ноль. Может комбинироваться для создания NaN (inf - inf, inf * 0). |
| Numerical gradient | "Грубая сила производная" | Аппроксимация производной путём вычисления f(x+h) и f(x-h) и деления на 2h. Медленно, но надёжно для проверки. |

## Further Reading

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- определяющий справочник, плотный, но полный
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- статья NVIDIA, которая представила масштабирование потерь для обучения float16
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- практическое руководство для смешанной точности в PyTorch
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- почему Google выбрал этот формат для TPU
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- алгоритм уменьшения ошибки округления в суммах floating point