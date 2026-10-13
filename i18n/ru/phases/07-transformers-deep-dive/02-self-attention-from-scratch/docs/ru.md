# Self-Attention с нуля

> Attention — это таблица поиска, в которой каждое слово спрашивает «кто для меня важен?» — и учится находить ответ.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 (Deep Learning Core), Phase 5 Lesson 10 (Sequence-to-Sequence)
**Time:** ~90 минут

## Цели обучения

- Реализовать scaled dot-product self-attention с нуля, используя только NumPy, включая проекции query/key/value и взвешенную сумму с softmax
- Построить слой multi-head attention, который разбивает на головы, вычисляет параллельное внимание и объединяет результаты
- Проследить, как матрица внимания отражает связи между токенами, и объяснить, почему масштабирование на sqrt(d_k) предотвращает насыщение softmax
- Применить каузальное маскирование, чтобы превратить двунаправленное внимание в авторегрессионное (decoder-style) внимание

## Проблема

RNN обрабатывают последовательности по одному токену за раз. К моменту, когда вы доходите до 50-го токена, информация из первого токена уже прошла через 50 шагов сжатия. Дальние зависимости сминаются в вектор скрытого состояния фиксированного размера — узкое место, которое не решается никаким количеством LSTM-гейтов.

Статья Bahdanau об attention 2014 года показала решение: пусть декодер оглядывается на каждую позицию энкодера и решает, какие из них важны для текущего шага. Но это всё ещё было надстройкой над RNN. Статья "Attention Is All You Need" 2017 года задала более острый вопрос: что если attention — *единственный* механизм? Без рекуррентности. Без свёртки. Только attention.

Self-attention позволяет каждой позиции в последовательности обращать внимание на каждую другую позицию за один параллельный шаг. Именно это делает трансформеры быстрыми, масштабируемыми и доминирующими.

## Концепция

### Аналогия с поиском по базе данных

Представьте attention как мягкий поиск по базе данных:

```
Traditional database:
  Query: "capital of France"  -->  exact match  -->  "Paris"

Attention:
  Query: "capital of France"  -->  similarity to ALL keys  -->  weighted blend of ALL values
```

Каждый токен генерирует три вектора:
- **Query (Q)**: «Что я ищу?»
- **Key (K)**: «Что я содержу?»
- **Value (V)**: «Какую информацию я предоставлю, если буду выбран?»

Скалярное произведение между query и всеми key даёт оценки внимания (attention scores). Высокая оценка означает «этот key соответствует моему query». Эти оценки взвешивают values. Выход — это взвешенная сумма values.

### Вычисление Q, K, V

Каждое вложение токена проецируется через три обучаемые матрицы весов:

```
Input embeddings (sequence of n tokens, each d-dimensional):

  X = [x1, x2, x3, ..., xn]       shape: (n, d)

Three weight matrices:

  Wq  shape: (d, dk)
  Wk  shape: (d, dk)
  Wv  shape: (d, dv)

Projections:

  Q = X @ Wq    shape: (n, dk)      each token's query
  K = X @ Wk    shape: (n, dk)      each token's key
  V = X @ Wv    shape: (n, dv)      each token's value
```

Наглядно, для одного токена:

```
             Wq
  x_i ------[*]------> q_i    "What am I looking for?"
       |
       |     Wk
       +----[*]------> k_i    "What do I contain?"
       |
       |     Wv
       +----[*]------> v_i    "What do I offer?"
```

### Матрица внимания

Как только у вас есть Q, K, V для всех токенов, оценки внимания образуют матрицу:

```
Scores = Q @ K^T    shape: (n, n)

              k1    k2    k3    k4    k5
        +-----+-----+-----+-----+-----+
   q1   | 2.1 | 0.3 | 0.1 | 0.8 | 0.2 |   <- how much q1 attends to each key
        +-----+-----+-----+-----+-----+
   q2   | 0.4 | 1.9 | 0.7 | 0.1 | 0.3 |
        +-----+-----+-----+-----+-----+
   q3   | 0.2 | 0.6 | 2.3 | 0.5 | 0.1 |
        +-----+-----+-----+-----+-----+
   q4   | 0.9 | 0.1 | 0.4 | 1.7 | 0.6 |
        +-----+-----+-----+-----+-----+
   q5   | 0.1 | 0.3 | 0.2 | 0.5 | 2.0 |
        +-----+-----+-----+-----+-----+

Each row: one token's attention over the entire sequence
```

Представьте, как один query за раз проходит по всем key: каждая строка оценивает каждый токен, softmax превращает оценки в веса, а контекстный вектор — это взвешенная смесь values.

```figure
attention-matrix
```

### Зачем масштабировать?

Скалярные произведения растут вместе с размерностью dk. Если dk = 64, скалярные произведения могут достигать десятков, что толкает softmax в область, где градиенты затухают. Решение: разделить на sqrt(dk).

```
Scaled scores = (Q @ K^T) / sqrt(dk)
```

Это удерживает значения в диапазоне, где softmax даёт полезные градиенты.

### Softmax превращает оценки в веса

Softmax преобразует необработанные оценки в распределение вероятностей по каждой строке:

```
Raw scores for q1:   [2.1, 0.3, 0.1, 0.8, 0.2]
                            |
                         softmax
                            |
Attention weights:   [0.52, 0.09, 0.07, 0.14, 0.08]   (sums to ~1.0)
```

Теперь у каждого токена есть набор весов, указывающих, насколько сильно обращать внимание на каждый другой токен.

### Взвешенная сумма values

Итоговый выход для каждого токена — это взвешенная сумма всех векторов values:

```
output_i = sum( attention_weight[i][j] * v_j  for all j )

For token 1:
  output_1 = 0.52 * v1 + 0.09 * v2 + 0.07 * v3 + 0.14 * v4 + 0.08 * v5
```

### Полный пайплайн

```mermaid
flowchart LR
  X["X (input)"] --> Q["Q = X · Wq"]
  X --> K["K = X · Wk"]
  X --> V["V = X · Wv"]
  Q --> S["Q · Kᵀ / √dk"]
  K --> S
  S --> SM["softmax"]
  SM --> WS["weighted sum"]
  V --> WS
  WS --> O["output"]
```

Формула в одну строку:

```
Attention(Q, K, V) = softmax( Q @ K^T / sqrt(dk) ) @ V
```

```figure
softmax-attention-scaling
```

## Постройте это

### Шаг 1: Softmax с нуля

Softmax преобразует необработанные логиты в вероятности. Для численной устойчивости вычитается максимум.

```python
import numpy as np

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
print(f"logits:  {logits}")
print(f"softmax: {softmax(logits)}")
print(f"sum:     {softmax(logits).sum():.4f}")
```

### Шаг 2: Scaled dot-product attention

Ключевая функция. Принимает матрицы Q, K, V и возвращает выход внимания плюс матрицу весов.

```python
def scaled_dot_product_attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    weights = softmax(scores)
    output = weights @ V
    return output, weights
```

### Шаг 3: Класс self-attention с обучаемыми проекциями

Полноценный модуль self-attention с матрицами весов Wq, Wk, Wv, инициализированными масштабированием в духе Xavier.

```python
class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights
```

### Шаг 4: Запустите на предложении

Создайте фиктивные вложения для предложения и посмотрите на веса внимания.

```python
sentence = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(sentence)
d_model = 8
dk = 4
dv = 4

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (n_tokens, d_model))

attn = SelfAttention(d_model, dk, dv, seed=42)
output, weights = attn.forward(X)

print("Attention weights (each row: where that token looks):\n")
print(f"{'':>6}", end="")
for token in sentence:
    print(f"{token:>6}", end="")
print()

for i, token in enumerate(sentence):
    print(f"{token:>6}", end="")
    for j in range(n_tokens):
        w = weights[i][j]
        print(f"{w:6.3f}", end="")
    print()
```

### Шаг 5: Визуализируйте внимание с помощью ASCII-тепловой карты

Отобразите веса внимания в символы для быстрой визуализации.

```python
def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    n = len(tokens)
    print(f"\n{'':>6}", end="")
    for t in tokens:
        print(f"{t:>6}", end="")
    print()

    for i in range(n):
        print(f"{tokens[i]:>6}", end="")
        for j in range(n):
            level = int(weights[i][j] * (len(chars) - 1) / weights.max())
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '   '}", end="")
        print()

ascii_heatmap(weights, sentence)
```

## Используйте это

PyTorch's `nn.MultiheadAttention` делает ровно то же, что мы построили, плюс разбиение на несколько голов и выходную проекцию:

```python
import torch
import torch.nn as nn

d_model = 8
n_heads = 2
seq_len = 6

mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

X_torch = torch.randn(1, seq_len, d_model)

output, attn_weights = mha(X_torch, X_torch, X_torch)

print(f"Input shape:            {X_torch.shape}")
print(f"Output shape:           {output.shape}")
print(f"Attention weight shape: {attn_weights.shape}")
print(f"\nAttn weights (averaged over heads):")
print(attn_weights[0].detach().numpy().round(3))
```

Ключевое отличие: multi-head attention запускает несколько функций внимания параллельно, каждая со своими проекциями Q, K, V размера dk = d_model / n_heads, а затем объединяет результаты. Это позволяет модели одновременно улавливать разные типы связей.

## Ship It

Этот урок производит:
- `outputs/prompt-attention-explainer.md` - промпт для объяснения attention через аналогию с поиском по базе данных

## Упражнения

1. Модифицируйте `scaled_dot_product_attention`, чтобы она принимала опциональную матрицу маски, которая устанавливает определённые позиции в минус бесконечность перед softmax (именно так работает каузальное/декодерное маскирование)
2. Реализуйте multi-head attention с нуля: разбейте Q, K, V на `n_heads` частей, запустите attention на каждой, объедините и спроецируйте через финальную матрицу весов Wo
3. Возьмите два разных предложения одинаковой длины, пропустите их через один и тот же экземпляр SelfAttention и сравните их паттерны внимания. Что меняется? Что остаётся прежним?

## Ключевые термины

| Термин | Что говорят люди | Что это на самом деле означает |
|------|----------------|----------------------|
| Query (Q) | «Вектор вопроса» | Обучаемая проекция входа, представляющая, какую информацию ищет данный токен |
| Key (K) | «Вектор метки» | Обучаемая проекция, представляющая, какую информацию содержит токен, сопоставляемая с query |
| Value (V) | «Вектор содержимого» | Обучаемая проекция, несущая фактическую информацию, которая агрегируется на основе оценок внимания |
| Scaled dot-product attention | «Формула attention» | softmax(QK^T / sqrt(dk)) @ V — масштабирование предотвращает насыщение softmax в высоких размерностях |
| Self-attention | «Токен смотрит на себя и на других» | Attention, где Q, K, V все берутся из одной и той же последовательности, позволяя каждой позиции обращать внимание на каждую другую |
| Attention weights | «Насколько сильно фокусироваться» | Распределение вероятностей по позициям, получаемое через softmax над масштабированными скалярными произведениями |
| Multi-head attention | «Параллельное внимание» | Запуск нескольких функций внимания с разными проекциями с последующим объединением результатов для более богатых представлений |

## Дополнительное чтение

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) - оригинальная статья о трансформерах
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) - лучший визуальный разбор полной архитектуры
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) - построчная реализация на PyTorch с пояснениями