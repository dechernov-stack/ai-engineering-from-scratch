# Логистическая регрессия

> Логистическая регрессия изгибает прямую линию в S-образную кривую, чтобы отвечать на вопросы «да или нет» с помощью вероятностей.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 Lesson 1-2 (What Is ML, Linear Regression)
**Time:** ~90 минут

## Цели обучения

- Реализовать логистическую регрессию с нуля, используя sigmoid-функцию и бинарную кросс-энтропию
- Вычислить и интерпретировать precision, recall, F1-меру и матрицу ошибок для бинарной классификации
- Объяснить, почему MSE не подходит для классификации и почему бинарная кросс-энтропия даёт выпуклую поверхность стоимости
- Построить модель softmax-регрессии для многоклассовой классификации и оценить компромиссы при настройке порога

## Проблема

Вы хотите предсказать, является ли опухоль злокачественной или доброкачественной, зная её размер. Вы пробуете линейную регрессию. Она выдаёт числа вроде 0.3, 1.7 или -0.5. Что они означают? Значит ли 1.7 «очень злокачественная»? А -0.5 — «очень доброкачественная»? Линейная регрессия выдаёт неограниченные числа. Классификации же нужны ограниченные вероятности в диапазоне от 0 до 1 и чёткое решение: да или нет.

Логистическая регрессия решает эту проблему. Она берёт ту же линейную комбинацию (wx + b) и пропускает её через sigmoid-функцию, которая сжимает любое число в диапазон (0, 1). На выходе получается вероятность. Вы задаёте порог (обычно 0.5) и принимаете решение.

Это один из самых широко используемых алгоритмов на практике. Несмотря на название, логистическая регрессия — это алгоритм классификации, а не регрессии. Название происходит от логистической (sigmoid) функции, которую она использует.

## Концепция

### Почему линейная регрессия не подходит для классификации

Представьте, что вы предсказываете сдал/не сдал (1/0) в зависимости от часов учёбы. Линейная регрессия проводит прямую через данные:

```
hours:  1   2   3   4   5   6   7   8   9   10
actual: 0   0   0   0   1   1   1   1   1   1
```

Линейная аппроксимация может дать предсказания вроде -0.2 при 1 часе и 1.3 при 10 часах. Эти значения не являются вероятностями. Они выходят за пределы 0 и 1. Хуже того, один выброс (человек, который учился 50 часов) потянет за собой всю линию, изменив предсказания для всех.

Классификации нужна функция, которая:
- Выдаёт значения от 0 до 1 (вероятности)
- Создаёт резкий переход (границу решения)
- Не искажается выбросами, далёкими от границы

### Sigmoid-функция

Sigmoid-функция делает именно это:

```
sigmoid(z) = 1 / (1 + e^(-z))
```

Свойства:
- Когда z велико и положительно, sigmoid(z) стремится к 1
- Когда z велико и отрицательно, sigmoid(z) стремится к 0
- Когда z = 0, sigmoid(z) = 0.5
- Выход всегда находится между 0 и 1
- Функция гладкая и всюду дифференцируемая

Производная имеет удобную форму: sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z)). Это делает вычисление градиента эффективным.

### Логистическая регрессия = линейная модель + sigmoid

Модель вычисляет z = wx + b (так же, как линейная регрессия), а затем применяет sigmoid:

```mermaid
flowchart LR
    X[Input features x] --> L["Linear: z = wx + b"]
    L --> S["Sigmoid: p = 1/(1+e^-z)"]
    S --> D{"p >= 0.5?"}
    D -->|Yes| P[Predict 1]
    D -->|No| N[Predict 0]
```

Выход p интерпретируется как P(y=1 | x) — вероятность того, что вход принадлежит классу 1. Граница решения находится там, где wx + b = 0, что даёт на выходе sigmoid ровно 0.5.

### Бинарная кросс-энтропия

Для логистической регрессии нельзя использовать MSE. MSE вместе с sigmoid создаёт невыпуклую поверхность стоимости с множеством локальных минимумов. Вместо этого используйте бинарную кросс-энтропию (log loss):

```
Loss = -(1/n) * sum(y * log(p) + (1-y) * log(1-p))
```

Почему это работает:
- Когда y=1 и p близко к 1: log(1) = 0, поэтому потеря близка к 0 (правильно, низкая стоимость)
- Когда y=1 и p близко к 0: log(0) стремится к минус бесконечности, поэтому потеря огромна (неправильно, высокая стоимость)
- Когда y=0 и p близко к 0: log(1) = 0, поэтому потеря близка к 0 (правильно, низкая стоимость)
- Когда y=0 и p близко к 1: log(0) стремится к минус бесконечности, поэтому потеря огромна (неправильно, высокая стоимость)

Эта функция потерь выпукла для логистической регрессии, что гарантирует единственный глобальный минимум.

### Градиентный спуск для логистической регрессии

Градиенты бинарной кросс-энтропии с sigmoid имеют простую форму:

```
dL/dw = (1/n) * sum((p - y) * x)
dL/db = (1/n) * sum(p - y)
```

Они выглядят идентично градиентам линейной регрессии. Разница в том, что p = sigmoid(wx + b), а не p = wx + b. Sigmoid вносит нелинейность, но правило обновления градиента остаётся тем же.

```mermaid
flowchart TD
    A[Initialize w=0, b=0] --> B[Forward pass: z = wx+b, p = sigmoid z]
    B --> C[Compute loss: binary cross-entropy]
    C --> D["Compute gradients: dw = (1/n) * sum((p-y)*x)"]
    D --> E[Update: w = w - lr*dw, b = b - lr*db]
    E --> F{Converged?}
    F -->|No| B
    F -->|Yes| G[Model trained]
```

### Граница решения

Для двумерного входа (два признака) граница решения — это линия, где:

```
w1*x1 + w2*x2 + b = 0
```

Точки по одну сторону классифицируются как 1, по другую — как 0. Логистическая регрессия всегда создаёт линейную границу решения. Если вам нужна изогнутая граница, добавьте полиномиальные признаки или используйте нелинейную модель.

### Многоклассовая классификация с softmax

Бинарная логистическая регрессия работает с двумя классами. Для k классов используйте softmax-функцию:

```
softmax(z_i) = e^(z_i) / sum(e^(z_j) for all j)
```

У каждого класса есть свой вектор весов. Модель вычисляет оценку z_i для каждого класса, затем softmax преобразует оценки в вероятности, сумма которых равна 1. Предсказанный класс — тот, у которого наибольшая вероятность.

Функция потерь становится категориальной кросс-энтропией:

```
Loss = -(1/n) * sum(sum(y_k * log(p_k)))
```

где y_k равно 1 для истинного класса и 0 для всех остальных (one-hot-кодирование).

### Метрики оценки

Одной точности (accuracy) недостаточно. Для набора данных с 95% отрицательных и 5% положительных примеров модель, которая всегда предсказывает отрицательный класс, получает точность 95%, но бесполезна.

**Матрица ошибок**:

| | Предсказан положительный | Предсказан отрицательный |
|---|---|---|
| Фактически положительный | Истинно положительный (TP) | Ложноотрицательный (FN) |
| Фактически отрицательный | Ложноположительный (FP) | Истинно отрицательный (TN) |

**Precision (точность)**: Из всех предсказанных положительных, сколько действительно положительных?
```
Precision = TP / (TP + FP)
```

**Recall (полнота)** (чувствительность): Из всех фактически положительных, сколько мы поймали?
```
Recall = TP / (TP + FN)
```

**F1-мера**: Среднее гармоническое precision и recall. Балансирует обе метрики.
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

Когда что приоритизировать:
- **Precision**: когда ложные срабатывания дорого обходятся (спам-фильтр, вы не хотите блокировать легитимную почту)
- **Recall**: когда пропуски дорого обходятся (скрининг на рак, вы не хотите пропустить опухоль)
- **F1**: когда нужна единая сбалансированная метрика

```figure
logistic-sigmoid
```

## Реализация

### Шаг 1: Sigmoid-функция и генерация данных

```python
import random
import math

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


random.seed(42)
N = 200
X = []
y = []

for _ in range(N // 2):
    X.append([random.gauss(2, 1), random.gauss(2, 1)])
    y.append(0)

for _ in range(N // 2):
    X.append([random.gauss(5, 1), random.gauss(5, 1)])
    y.append(1)

combined = list(zip(X, y))
random.shuffle(combined)
X, y = zip(*combined)
X = list(X)
y = list(y)

print(f"Generated {N} samples (2 classes, 2 features)")
print(f"Class 0 center: (2, 2), Class 1 center: (5, 5)")
print(f"First 5 samples:")
for i in range(5):
    print(f"  Features: [{X[i][0]:.2f}, {X[i][1]:.2f}], Label: {y[i]}")
```

### Шаг 2: Логистическая регрессия с нуля

```python
class LogisticRegression:
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.loss_history = []

    def predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return sigmoid(z)

    def predict(self, x, threshold=0.5):
        return 1 if self.predict_proba(x) >= threshold else 0

    def compute_loss(self, X, y):
        n = len(y)
        total = 0.0
        for i in range(n):
            p = self.predict_proba(X[i])
            p = max(1e-15, min(1 - 1e-15, p))
            total += y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p)
        return -total / n

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            dw = [0.0] * n_features
            db = 0.0
            for i in range(n):
                p = self.predict_proba(X[i])
                error = p - y[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error
            for j in range(n_features):
                self.weights[j] -= self.lr * (dw[j] / n)
            self.bias -= self.lr * (db / n)
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Loss: {loss:.4f} | w: [{self.weights[0]:.3f}, {self.weights[1]:.3f}] | b: {self.bias:.3f}")
        return self

    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(y)) if self.predict(X[i]) == y[i])
        return correct / len(y)


split = int(0.8 * N)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("\n=== Training Logistic Regression ===")
model = LogisticRegression(n_features=2, learning_rate=0.1)
model.fit(X_train, y_train, epochs=1000, print_every=200)

print(f"\nTrain accuracy: {model.accuracy(X_train, y_train):.4f}")
print(f"Test accuracy:  {model.accuracy(X_test, y_test):.4f}")
print(f"Weights: [{model.weights[0]:.4f}, {model.weights[1]:.4f}]")
print(f"Bias: {model.bias:.4f}")
```

### Шаг 3: Матрица ошибок и метрики с нуля

```python
class ClassificationMetrics:
    def __init__(self, y_true, y_pred):
        self.tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        self.tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        self.fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        self.fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    def accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0

    def precision(self):
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0

    def recall(self):
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0

    def f1(self):
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    def print_confusion_matrix(self):
        print(f"\n  Confusion Matrix:")
        print(f"                  Predicted")
        print(f"                  Pos   Neg")
        print(f"  Actual Pos     {self.tp:4d}  {self.fn:4d}")
        print(f"  Actual Neg     {self.fp:4d}  {self.tn:4d}")

    def print_report(self):
        self.print_confusion_matrix()
        print(f"\n  Accuracy:  {self.accuracy():.4f}")
        print(f"  Precision: {self.precision():.4f}")
        print(f"  Recall:    {self.recall():.4f}")
        print(f"  F1 Score:  {self.f1():.4f}")


y_pred_test = [model.predict(x) for x in X_test]
print("\n=== Classification Report (Test Set) ===")
metrics = ClassificationMetrics(y_test, y_pred_test)
metrics.print_report()
```

### Шаг 4: Анализ границы решения

```python
print("\n=== Decision Boundary ===")
w1, w2 = model.weights
b = model.bias
print(f"Decision boundary: {w1:.4f}*x1 + {w2:.4f}*x2 + {b:.4f} = 0")
if abs(w2) > 1e-10:
    print(f"Solved for x2:     x2 = {-w1/w2:.4f}*x1 + {-b/w2:.4f}")

print("\nSample predictions near the boundary:")
test_points = [
    [3.0, 3.0],
    [3.5, 3.5],
    [4.0, 4.0],
    [2.5, 2.5],
    [5.0, 5.0],
]
for point in test_points:
    prob = model.predict_proba(point)
    pred = model.predict(point)
    print(f"  [{point[0]}, {point[1]}] -> prob={prob:.4f}, class={pred}")
```

### Шаг 5: Многоклассовая классификация с softmax

```python
class SoftmaxRegression:
    def __init__(self, n_features, n_classes, learning_rate=0.01):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = learning_rate
        self.weights = [[0.0] * n_features for _ in range(n_classes)]
        self.biases = [0.0] * n_classes

    def softmax(self, scores):
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        return [e / total for e in exp_scores]

    def predict_proba(self, x):
        scores = [
            sum(self.weights[k][j] * x[j] for j in range(self.n_features)) + self.biases[k]
            for k in range(self.n_classes)
        ]
        return self.softmax(scores)

    def predict(self, x):
        probs = self.predict_proba(x)
        return probs.index(max(probs))

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        for epoch in range(epochs):
            grad_w = [[0.0] * self.n_features for _ in range(self.n_classes)]
            grad_b = [0.0] * self.n_classes
            total_loss = 0.0
            for i in range(n):
                probs = self.predict_proba(X[i])
                for k in range(self.n_classes):
                    target = 1.0 if y[i] == k else 0.0
                    error = probs[k] - target
                    for j in range(self.n_features):
                        grad_w[k][j] += error * X[i][j]
                    grad_b[k] += error
                true_prob = max(probs[y[i]], 1e-15)
                total_loss -= math.log(true_prob)
            for k in range(self.n_classes):
                for j in range(self.n_features):
                    self.weights[k][j] -= self.lr * (grad_w[k][j] / n)
                self.biases[k] -= self.lr * (grad_b[k] / n)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Loss: {total_loss / n:.4f}")
        return self

    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(y)) if self.predict(X[i]) == y[i])
        return correct / len(y)


random.seed(42)
X_3class = []
y_3class = []

centers = [(1, 1), (5, 1), (3, 5)]
for label, (cx, cy) in enumerate(centers):
    for _ in range(50):
        X_3class.append([random.gauss(cx, 0.8), random.gauss(cy, 0.8)])
        y_3class.append(label)

combined = list(zip(X_3class, y_3class))
random.shuffle(combined)
X_3class, y_3class = zip(*combined)
X_3class = list(X_3class)
y_3class = list(y_3class)

split_3 = int(0.8 * len(X_3class))
X_train_3 = X_3class[:split_3]
y_train_3 = y_3class[:split_3]
X_test_3 = X_3class[split_3:]
y_test_3 = y_3class[split_3:]

print("\n=== Multi-class Softmax Regression (3 classes) ===")
softmax_model = SoftmaxRegression(n_features=2, n_classes=3, learning_rate=0.1)
softmax_model.fit(X_train_3, y_train_3, epochs=1000, print_every=200)
print(f"\nTrain accuracy: {softmax_model.accuracy(X_train_3, y_train_3):.4f}")
print(f"Test accuracy:  {softmax_model.accuracy(X_test_3, y_test_3):.4f}")

print("\nSample predictions:")
for i in range(5):
    probs = softmax_model.predict_proba(X_test_3[i])
    pred = softmax_model.predict(X_test_3[i])
    print(f"  True: {y_test_3[i]}, Predicted: {pred}, Probs: [{', '.join(f'{p:.3f}' for p in probs)}]")
```

### Шаг 6: Настройка порога

```python
print("\n=== Threshold Tuning ===")
print("Default threshold: 0.5. Adjusting the threshold trades precision for recall.\n")

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print(f"{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 52)

for t in thresholds:
    y_pred_t = [1 if model.predict_proba(x) >= t else 0 for x in X_test]
    m = ClassificationMetrics(y_test, y_pred_t)
    print(f"{t:>10.1f} {m.accuracy():>10.4f} {m.precision():>10.4f} {m.recall():>10.4f} {m.f1():>10.4f}")
```

## Применение

Теперь то же самое с scikit-learn.

```python
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(42)
X_0 = np.random.randn(100, 2) + [2, 2]
X_1 = np.random.randn(100, 2) + [5, 5]
X_sk = np.vstack([X_0, X_1])
y_sk = np.array([0] * 100 + [1] * 100)

X_tr, X_te, y_tr, y_te = train_test_split(X_sk, y_sk, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

lr = SklearnLR()
lr.fit(X_tr_sc, y_tr)
y_pred = lr.predict(X_te_sc)

print("=== Scikit-learn Logistic Regression ===")
print(f"Accuracy:  {accuracy_score(y_te, y_pred):.4f}")
print(f"Precision: {precision_score(y_te, y_pred):.4f}")
print(f"Recall:    {recall_score(y_te, y_pred):.4f}")
print(f"F1:        {f1_score(y_te, y_pred):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_te, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_te, y_pred)}")
```

Ваша реализация с нуля даёт ту же границу решения и те же метрики. Scikit-learn добавляет варианты решателей (liblinear, lbfgs, saga), автоматическую регуляризацию, стратегии многоклассовой классификации (one-vs-rest, multinomial) и оптимизации численной устойчивости.

## Результат

Этот урок даёт:
- `code/logistic_regression.py` — логистическая регрессия с нуля с метриками

## Упражнения

1. Сгенерируйте набор данных, который НЕ является линейно разделимым (например, две концентрические окружности). Обучите логистическую регрессию и понаблюдайте за её провалом. Затем добавьте полиномиальные признаки (x1^2, x2^2, x1*x2) и обучите снова. Покажите, что точность улучшается.
2. Реализуйте многоклассовую матрицу ошибок для 3-классовой softmax-модели. Вычислите precision и recall для каждого класса. Какой класс сложнее всего классифицировать?
3. Постройте ROC-кривую с нуля. Для 100 значений порога от 0 до 1 вычислите истинно положительную и ложноположительную частоты. Вычислите AUC (площадь под кривой), используя метод трапеций.

## Ключевые термины

| Термин | Что говорят люди | Что это на самом деле означает |
|------|----------------|----------------------|
| Логистическая регрессия | «Регрессия для классификации» | Линейная модель, за которой следует sigmoid-функция, выдающая вероятности классов |
| Sigmoid-функция | «S-образная кривая» | Функция 1/(1+e^(-z)), которая отображает любое вещественное число в диапазон (0, 1) |
| Бинарная кросс-энтропия | «Log loss» | Функция потерь -[y*log(p) + (1-y)*log(1-p)], которая сильно штрафует уверенные неправильные предсказания |
| Граница решения | «Разделяющая линия» | Поверхность, где выходная вероятность модели равна 0.5, разделяющая предсказанные классы |
| Softmax | «Многоклассовый sigmoid» | Функция, преобразующая вектор оценок в вероятности, сумма которых равна 1 |
| Precision | «Сколько из выбранных релевантны» | TP / (TP + FP), доля положительных предсказаний, которые действительно положительны |
| Recall | «Сколько из релевантных выбрано» | TP / (TP + FN), доля фактически положительных примеров, которые модель правильно определила |
| F1-мера | «Сбалансированная точность» | Среднее гармоническое precision и recall: 2*P*R / (P+R) |
| Матрица ошибок | «Разбивка ошибок» | Таблица, показывающая количество TP, TN, FP, FN для каждой пары классов |
| Порог | «Точка отсечения» | Значение вероятности, выше которого модель предсказывает класс 1 (по умолчанию 0.5, настраивается) |
| One-hot-кодирование | «Бинарные столбцы для категорий» | Представление класса k в виде вектора нулей с 1 в позиции k |
| Категориальная кросс-энтропия | «Многоклассовый log loss» | Расширение бинарной кросс-энтропии на k классов с использованием one-hot-кодированных меток |