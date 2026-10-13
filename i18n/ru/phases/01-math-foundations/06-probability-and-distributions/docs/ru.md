# Вероятность и распределения

> Вероятность — это язык, который ИИ использует для выражения неопределённости.

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~75 minutes

## Learning Objectives

- Implement PMFs and PDFs from scratch for Bernoulli, categorical, Poisson, uniform, and normal distributions
- Compute expected value, variance, and use the Central Limit Theorem to explain why Gaussians dominate
- Build softmax and log-softmax functions with the numerical stability trick (subtract max logit)
- Calculate cross-entropy loss from logits and connect it to negative log-likelihood

## The Problem

Классификатор выводит `[0.03, 0.91, 0.06]`. Языковая модель выбирает следующее слово из 50 000 кандидатов. Модель диффузии генерирует изображения путём выборки из изученных распределений. Всё это — вероятность в действии.

Каждое предсказание модели — это вероятностное распределение. Каждая функция потерь измеряет, насколько далеко предсказанное распределение от истинного. Каждый шаг обучения регулирует параметры так, чтобы одно распределение больше походило на другое. Без вероятности вы не сможете прочитать ни одну статью по ML, отладить ни одну модель и понять, почему ваши потери обучения равны NaN.

## The Concept

### Events, Sample Spaces, and Probability

Пространство выборки S — это множество всех возможных исходов. Событие — это подмножество пространства выборки. Вероятность отображает события в числа от 0 до 1.

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

Три аксиомы определяют всю вероятность:
1. P(A) >= 0 для любого события A
2. P(S) = 1 (что-то всегда происходит)
3. P(A или B) = P(A) + P(B), когда A и B не могут произойти одновременно

Всё остальное (теорема Байеса, ожидания, распределения) следует из этих трёх правил.

### Conditional Probability and Independence

P(A|B) — это вероятность A при условии, что произошло B.

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

Два события независимы, когда знание об одном ничего не говорит вам о другом:

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

Бросания монеты независимы. Извлечение карт без замены — нет.

### Probability Mass Functions vs Probability Density Functions

Дискретные случайные величины имеют функцию вероятности масс (PMF). Каждый исход имеет конкретную вероятность, которую вы можете прочитать непосредственно.

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

Непрерывные случайные величины имеют функцию плотности вероятности (PDF). Плотность в отдельной точке — это не вероятность. Вероятность получается путём интегрирования плотности по интервалу.

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

Это различие важно в ML. Выходы классификации — это PMF (дискретные выборы). Скрытые пространства VAE используют PDF (непрерывные).

### Common Distributions

**Bernoulli:** один опыт, два исхода. Моделирует бинарную классификацию.

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**Categorical:** один опыт, k исходов. Моделирует многоклассовую классификацию (вывод softmax).

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Uniform:** все исходы равновероятны. Используется для случайной инициализации.

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**Normal (Gaussian):** кривая колокола. Параметризуется средним значением (mu) и дисперсией (sigma^2).

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson:** количество редких событий в фиксированном интервале. Моделирует темпы событий.

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### Expected Value and Variance

Ожидаемое значение — это взвешенный по вероятности средний исход.

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

Дисперсия измеряет разброс вокруг среднего.

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

В ML ожидаемое значение появляется как функция потерь (средняя потеря по распределению данных). Дисперсия говорит вам о стабильности модели. Высокая дисперсия в градиентах означает шумное обучение.

### Joint and Marginal Distributions

Совместное распределение P(X, Y) описывает две случайные величины вместе.

Пример совместного PMF (X = погода, Y = зонтик):

| | Y=0 (без зонтика) | Y=1 (с зонтиком) | Маргинальное P(X) |
|---|---|---|---|
| X=0 (солнце) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (дождь) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **Маргинальное P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

Маргинальное распределение суммирует другую переменную:

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

Суммы строк и столбцов в таблице выше — это маргиналы.

### Why the Normal Distribution Shows Up Everywhere

Центральная предельная теорема: сумма (или среднее) многих независимых случайных величин сходится к нормальному распределению, независимо от исходного распределения.

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

Вот почему:
- Ошибки измерения приблизительно нормальны (множество малых независимых источников)
- Инициализация весов в нейронных сетях использует нормальные распределения
- Шум градиента в SGD приблизительно нормален (сумма многих градиентов выборки)
- Нормальное распределение — это распределение с максимальной энтропией для данного среднего и дисперсии

### Log Probabilities

Сырые вероятности вызывают численные проблемы. Умножение многих малых вероятностей вместе быстро приводит к исчезновению в ноль.

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

Логарифмические вероятности решают эту проблему. Умножения становятся сложениями.

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

Правила:
- log(a * b) = log(a) + log(b)
- логарифмические вероятности всегда <= 0 (поскольку 0 < P <= 1)
- более отрицательное = менее вероятно
- потеря перекрёстной энтропии — это отрицательный логарифм вероятности правильного класса

### Softmax as a Probability Distribution

Нейронные сети выводят сырые оценки (logits). Softmax преобразует их в действительное вероятностное распределение.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

Трюк softmax: вычтите максимальный logit перед экспоненцированием, чтобы предотвратить переполнение.

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax объединяет softmax и log для численной стабильности. PyTorch использует это внутренне для потери перекрёстной энтропии.

### Sampling

Выборка означает рисование случайных значений из распределения. В ML:
- Dropout случайно выбирает, какие нейроны обнулить
- Увеличение данных выбирает случайные преобразования
- Языковые модели выбирают следующий токен из предсказанного распределения
- Модели диффузии выбирают шум и прогрессивно его уменьшают

Выборка из произвольных распределений требует техник, таких как выборка обратного преобразования, выборка отклонения или трюк репараметризации (используется в VAE).

```figure
gaussian-pdf
```

## Build It

### Step 1: Probability basics

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### Step 2: PMF and PDF from scratch

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### Step 3: Expected value and variance

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### Step 4: Sampling from distributions

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### Step 5: Softmax and log probabilities

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### Step 6: Central Limit Theorem demonstration

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### Step 7: Visualization

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

Полные реализации со всеми визуализациями находятся в `code/probability.py`.

## Use It

С NumPy и SciPy всё выше — это однострочные вызовы:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

Вы построили это с нуля. Теперь вы знаете, что делают библиотечные вызовы.

## Exercises

1. Реализуйте выборку обратного преобразования для экспоненциального распределения. Проверьте, отобрав 10 000 значений и сравнив гистограмму с истинным PDF.

2. Постройте таблицу совместного распределения для двух нечестных кубиков. Вычислите маргинальные распределения и проверьте, независимы ли кубики.

3. Вычислите потерю перекрёстной энтропии для классификатора 5 классов, который выводит logits `[2.0, 0.5, -1.0, 3.0, 0.1]` при правильном классе индекс 3. Затем проверьте ваш ответ с помощью PyTorch `nn.CrossEntropyLoss`.

4. Напишите функцию, которая принимает список логарифмических вероятностей и возвращает наиболее вероятную последовательность, общую логарифмическую вероятность и эквивалентную сырую вероятность. Протестируйте её предложением из 50 слов, где каждое слово имеет вероятность 0.01.

## Key Terms

| Термин | Что люди говорят | Что это на самом деле означает |
|---|---|---|
| Sample space | "Все возможности" | Множество S каждого возможного исхода опыта |
| PMF | "Функция вероятности" | Функция, которая даёт точную вероятность каждого дискретного исхода, суммируясь в 1 |
| PDF | "Кривая вероятности" | Функция плотности для непрерывных переменных. Интегрируйте её по интервалу, чтобы получить вероятность |
| Conditional probability | "Вероятность при условии чего-то" | P(A\|B) = P(A и B) / P(B). Основа байесовского мышления и теоремы Байеса |
| Independence | "Они не влияют друг на друга" | P(A и B) = P(A) * P(B). Знание об одном событии ничего не говорит о другом |
| Expected value | "Среднее значение" | Взвешенная по вероятности сумма всех исходов. Функция потерь — это ожидаемое значение |
| Variance | "Насколько разброс" | Ожидаемое квадратичное отклонение от среднего. Высокая дисперсия = шумные, нестабильные оценки |
| Normal distribution | "Кривая колокола" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2)). Появляется везде благодаря CLT |
| Central Limit Theorem | "Средние становятся нормальными" | Среднее значение многих независимых выборок сходится к нормальному распределению независимо от источника |
| Joint distribution | "Две переменные вместе" | P(X, Y) описывает вероятность каждой комбинации исходов X и Y |
| Marginal distribution | "Просуммировать другую переменную" | P(X) = sum_y P(X, Y). Восстанавливает распределение одной переменной из совместного |
| Log probability | "Логарифм вероятности" | log P(x). Превращает произведения в суммы, предотвращая численное исчезновение в длинных последовательностях |
| Softmax | "Превратить оценки в вероятности" | softmax(z_i) = exp(z_i) / sum(exp(z_j)). Отображает вещественнозначные logits в действительное вероятностное распределение |
| Cross-entropy | "Функция потерь" | -sum(p_true * log(p_predicted)). Измеряет, насколько различны два распределения. Меньше — лучше |
| Logits | "Сырые выходы модели" | Ненормализованные оценки перед softmax. Названы в честь логистической функции |
| Sampling | "Рисование случайных значений" | Генерирование значений в соответствии с вероятностным распределением. Как модели генерируют выход |

## Further Reading

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - визуальное доказательство, почему средние становятся нормальными
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - краткая справка, охватывающая всё здесь и более того
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - почему численная стабильность важна и как её достичь