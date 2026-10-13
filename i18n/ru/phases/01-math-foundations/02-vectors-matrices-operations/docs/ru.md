# Векторы, матрицы и операции

> Любая нейронная сеть — это просто умножение матриц с дополнительными шагами.

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Linear Algebra Intuition)
**Time:** ~60 минут

## Цели обучения

- Построить класс Matrix с поэлементными операциями, умножением матриц, транспонированием, определителем и обратной матрицей
- Отличать поэлементное умножение от матричного и объяснять, когда применяется каждое
- Реализовать один плотный слой нейронной сети (`relu(W @ x + b)`), используя только написанный с нуля класс Matrix
- Объяснить правила broadcasting и то, как работает сложение с вектором смещения во фреймворках для нейронных сетей

## Проблема

Вы хотите построить нейронную сеть. Вы читаете код и видите вот это:

```
output = activation(weights @ input + bias)
```

Этот `@` — это умножение матриц. `weights` — это матрица. `input` — это вектор. Если вы не знаете, что делают эти операции, эта строка — магия. Если знаете — это весь прямой проход слоя в трёх операциях.

Каждое изображение, которое обрабатывает ваша модель, — это матрица значений пикселей. Каждый эмбеддинг слова — это вектор. Каждый слой каждой нейронной сети — это матричное преобразование. Нельзя строить системы ИИ, не владея матричными операциями, точно так же, как нельзя писать код, не понимая переменных.

Этот урок формирует это владение с нуля.

## Концепция

### Векторы: упорядоченные списки чисел

Вектор — это список чисел, обладающий направлением и величиной. В ИИ векторы представляют точки данных, признаки или параметры.

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

2D-вектор `[3, 4]` указывает на координаты (3, 4) на плоскости. Его длина (величина) равна 5 (треугольник 3-4-5).

### Матрицы: сетки чисел

Матрица — это 2D-сетка. Строки и столбцы. Матрица размером m x n имеет m строк и n столбцов.

```
A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |
```

В нейронных сетях весовые матрицы преобразуют входные векторы в выходные. Слой с 784 входами и 128 выходами использует весовую матрицу размером 128x784.

### Почему важны размерности

Умножение матриц подчиняется строгому правилу: `(m x n) @ (n x p) = (m x p)`. Внутренние размерности должны совпадать.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

Если вы получаете ошибку несовпадения размерностей в PyTorch — вот причина.

### Карта операций

| Операция | Что делает | Применение в нейронных сетях |
|-----------|-------------|-------------------|
| Сложение | Поэлементное объединение | Добавление смещения к выходу |
| Умножение на скаляр | Масштабирование каждого элемента | Скорость обучения * градиенты |
| Умножение матриц | Преобразование векторов | Прямой проход слоя |
| Транспонирование | Переворот строк и столбцов | Обратное распространение ошибки |
| Определитель | Сводка в одном числе | Проверка обратимости |
| Обратная матрица | Отмена преобразования | Решение линейных систем |
| Единичная матрица | Матрица «ничего не делай» | Инициализация, остаточные связи |

### Поэлементное умножение vs умножение матриц

Это различие постоянно сбивает новичков с толку.

Поэлементное: умножение соответствующих позиций. Обе матрицы должны иметь одинаковую форму.

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

Умножение матриц: скалярные произведения строк и столбцов. Внутренние размерности должны совпадать.

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

Разные операции, разные результаты, разные правила.

### Broadcasting

Когда вы добавляете вектор смещения к матрице выходов, их формы не совпадают. Broadcasting растягивает меньший массив, чтобы он подошёл.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

Каждый современный фреймворк делает это автоматически. Понимание этого механизма избавляет от путаницы, когда формы кажутся неправильными, но код всё равно работает.

```figure
vector-projection
```

## Построение

### Шаг 1: Класс Vector

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### Шаг 2: Класс Matrix с основными операциями

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### Шаг 3: Проверка в деле

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### Шаг 4: Связь с нейронными сетями

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

Это один плотный слой: `output = relu(W @ x + b)`. Каждый плотный слой в каждой нейронной сети делает именно это.

## Применение

NumPy делает всё вышеперечисленное за меньшее число строк и на порядки быстрее.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Оператор `@` в Python вызывает `__matmul__`. NumPy реализует его с помощью оптимизированных процедур BLAS, написанных на C и Fortran. Та же математика, в 100 раз быстрее.

Broadcasting в NumPy:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy автоматически распространяет 1D-вектор смещения на обе строки. Именно так работает сложение со смещением в каждом фреймворке для нейронных сетей.

## Поставка

Этот урок формирует промпт для обучения матричным операциям через геометрическую интуицию. См. `outputs/prompt-matrix-operations.md`.

Класс Matrix, построенный здесь, служит основой для мини-фреймворка нейронных сетей, который мы строим в Phase 3, Lesson 10.

## Упражнения

1. **Проверьте обратную матрицу.** Перемножьте `A @ A.inverse_2x2()` и убедитесь, что получаете единичную матрицу. Попробуйте это с тремя разными матрицами 2x2. Что происходит, когда определитель равен нулю?

2. **Реализуйте обращение матрицы 3x3.** Расширьте класс Matrix, чтобы вычислять обратные матрицы для матриц 3x3 методом присоединённой матрицы (adjugate). Проверьте результат относительно `np.linalg.inv` из NumPy.

3. **Постройте двухслойную сеть.** Используя только ваш класс Matrix (без NumPy), создайте двухслойную нейронную сеть: вход (3) -> скрытый слой (4) -> выход (2). Инициализируйте случайные веса, выполните прямой проход и убедитесь, что все размерности верны.

## Ключевые термины

| Термин | Что говорят люди | Что это на самом деле означает |
|------|----------------|----------------------|
| Вектор | «Стрелка» | Упорядоченный список чисел. В ИИ: точка в пространстве высокой размерности. |
| Матрица | «Таблица чисел» | Линейное преобразование. Отображает векторы из одного пространства в другое. |
| Умножение матриц | «Просто перемножить числа» | Скалярные произведения между каждой строкой первой матрицы и каждым столбцом второй. Порядок важен. |
| Транспонирование | «Перевернуть её» | Поменять местами строки и столбцы. Превращает матрицу m x n в n x m. Критически важно при обратном распространении ошибки. |
| Определитель | «Какое-то число из матрицы» | Измеряет, во сколько раз матрица масштабирует площадь (2D) или объём (3D). Ноль означает, что преобразование схлопывает измерение. |
| Обратная матрица | «Отменить матрицу» | Матрица, которая обращает преобразование. Существует только тогда, когда определитель не равен нулю. |
| Единичная матрица | «Скучная матрица» | Матричный эквивалент умножения на 1. Используется в остаточных связях (ResNets). |
| Broadcasting | «Волшебное исправление формы» | Растягивание меньшего массива для соответствия большему путём повторения по отсутствующим измерениям. |
| Поэлементное | «Обычное умножение» | Умножение соответствующих позиций. Оба массива должны иметь одинаковую форму (или быть совместимы для broadcasting). |

## Дополнительные материалы

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) — визуальная интуиция для каждой операции, рассмотренной здесь
- [Документация NumPy по broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) — точные правила, которым следует NumPy
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) — краткий справочник по линейной алгебре применительно к машинному обучению