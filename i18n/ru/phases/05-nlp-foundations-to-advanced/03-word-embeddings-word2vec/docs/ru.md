# Word Embeddings — Word2Vec с нуля

> Слово — это компания, в которой оно появляется. Обучите неглубокую сеть этой идее — и геометрия проявится сама собой.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75 минут

## Проблема

TF-IDF знает, что `dog` и `puppy` — разные слова. Он не знает, что они означают почти одно и то же. Классификатор, обученный на `dog`, не сможет обобщиться на отзыв про `puppy`. Можно замаскировать это, составив список синонимов, но это не работает на редких терминах, профессиональном жаргоне и в любом языке, который вы не предусмотрели.

Вам нужно представление, в котором `dog` и `puppy` оказываются рядом в пространстве. Где `king - man + woman` оказывается рядом с `queen`. Где модель, обученная на `dog`, бесплатно передаёт часть сигнала на `puppy`.

Word2Vec дал нам такое пространство. Двухслойная нейросеть, прогоны обучения на триллионах токенов, опубликовано в 2013 году. Архитектура почти неприлично простая. Результаты переформатировали NLP на десятилетие вперёд.

## Концепция

**Дистрибутивная гипотеза** (Фёрс, 1957): «Узнаёшь слово по компании, в которой оно встречается». Если два слова появляются в похожих контекстах, они, вероятно, означают похожие вещи.

Word2Vec существует в двух вариантах, оба эксплуатируют эту идею.

- **Skip-gram.** По центральному слову предсказываем окружающие слова. `cat -> (the, sat, on)` с окном размера 2.
- **CBOW (continuous bag of words).** По окружающим словам предсказываем центральное. `(the, sat, on) -> cat`.

Skip-gram обучается медленнее, но лучше справляется с редкими словами. Он стал вариантом по умолчанию.

У сети один скрытый слой без нелинейности. Вход — one-hot вектор по словарю. Выход — softmax по словарю. После обучения выходной слой выбрасывается. Веса скрытого слоя и есть эмбеддинги.

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

Хитрость в том, что softmax по 100 тыс. слов непомерно дорог. Word2Vec использует **negative sampling (отрицательное сэмплирование)**, превращая задачу в бинарную классификацию. Предсказываем «появилось ли это контекстное слово рядом с этим центральным словом — да или нет». Вместо вычисления softmax по всему словарю для каждой обучающей пары сэмплируется горстка отрицательных (не встречающихся вместе) слов.

```figure
word-vector-arithmetic
```

## Построение

### Шаг 1: обучающие пары из корпуса

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

Каждая пара (центр, контекст) в пределах окна — это положительный обучающий пример.

### Шаг 2: таблицы эмбеддингов

Две матрицы. `W` — таблица эмбеддингов центральных слов (та, которую вы оставляете). `W'` — таблица контекстных слов (часто отбрасывается, иногда усредняется с `W`).

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

Небольшая случайная инициализация. Размер словаря 10 тыс. и размерность 100 — реалистично; для обучающих целей достаточно 50 слов x 16 измерений, чтобы увидеть геометрию.

### Шаг 3: целевая функция с negative sampling

Для каждой положительной пары `(center, context)` сэмплируем `k` случайных слов из словаря в качестве отрицательных. Обучаем модель так, чтобы скалярное произведение `W[center] · W'[context]` было высоким для положительных пар и низким для отрицательных.

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

Волшебная формула: логистическая функция потерь на положительной паре (хотим sigmoid близко к 1) плюс логистическая функция потерь на отрицательных парах (хотим sigmoid близко к 0). Градиенты текут в обе таблицы. Полный вывод есть в оригинальной статье; пройдите его один раз с карандашом и бумагой, если хотите, чтобы это закрепилось.

### Шаг 4: обучение на игрушечном корпусе

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

После достаточного числа эпох на большом корпусе слова, делящие контексты, получают похожие центральные эмбеддинги. На игрушечном корпусе эффект заметен слабо. На миллиардах токенов он проявляется ярко.

### Шаг 5: трюк с аналогиями

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

На предобученных 300-мерных векторах Google News:

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`. Не потому что модель знает, что такое королевская власть. А потому что вектор `(king - man)` улавливает нечто вроде «царственности», и его добавление к `woman` приземляется рядом с регионом «царственная женщина».

## Применение

Написание Word2Vec с нуля — это учебная практика. Продакшн-NLP использует `gensim`.

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

Для реальной работы вы почти никогда не обучаете Word2Vec самостоятельно. Вы скачиваете предобученные векторы.

- **GloVe** — подход Стэнфорда на основе факторизации матрицы совстречаемости. Чекпоинты 50d, 100d, 200d, 300d. Хорошее общее покрытие. Урок 04 посвящён GloVe отдельно.
- **fastText** — расширение Word2Vec от Facebook, встраивающее символьные n-граммы. Справляется со словами вне словаря, составляя их из подслов. Урок 04.
- **Предобученный Word2Vec на Google News** — 300d, словарь из 3 млн слов, опубликован в 2013 году. Его до сих пор скачивают ежедневно.

### Когда Word2Vec всё ещё выигрывает в 2026 году

- Лёгкий домен-специфичный поиск. Обучите на медицинских аннотациях за час на ноутбуке, получите специализированные векторы, которых не улавливает ни одна общая модель.
- Инженерия признаков в стиле аналогий. `gender_vector = mean(man - woman pairs)`. Вычтите его из других слов, чтобы получить гендерно-нейтральную ось. Всё ещё используется в исследованиях справедливости (fairness).
- Интерпретируемость. 100 измерений — достаточно мало, чтобы отобразить через PCA или t-SNE и действительно увидеть формирование кластеров.
- Везде, где инференс должен работать на устройстве без GPU. Поиск в Word2Vec — это выборка одной строки.

### Где Word2Vec отказывает

Стена полисемии. У `bank` один вектор. `river bank` и `financial bank` делят его. `table` (таблица в спредшите vs. предмет мебели) делит его. Классификатор ниже по цепочке не может различить значения по вектору.

Контекстные эмбеддинги (ELMo, BERT, каждый трансформер с тех пор) решили это, выдавая разный вектор для каждого вхождения слова в зависимости от окружающего контекста. Это и есть скачок от Word2Vec к BERT: от статичного к контекстному. Phase 7 охватывает трансформерную половину.

Проблема слов вне словаря (OOV) — другой отказ. Word2Vec никогда не видел `Zoomer-approved`, если его не было в обучающих данных. Запасного варианта нет. fastText исправляет это композицией подслов (урок 04).

## Поставка

Сохраните как `outputs/skill-embedding-probe.md`:

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## Упражнения

1. **Лёгкое.** Запустите цикл обучения на крошечном корпусе (20 предложений про кошек и собак). После 200 эпох убедитесь, что `nearest(vocab, W, W[vocab["cat"]])` возвращает `dog` в топ-3. Если нет — увеличьте число эпох или словарь.
2. **Среднее.** Добавьте субсэмплирование частых слов. Слова с частотой выше `10^-5` исключаются из обучающих пар с вероятностью, пропорциональной их частоте. Измерьте эффект на сходство редких слов.
3. **Сложное.** Обучите модель на корпусе 20 Newsgroups. Вычислите две оси смещения: `he - she` и `doctor - nurse`. Спроецируйте слова-профессии на обе оси. Сообщите, у каких профессий наибольший разрыв смещения. Это тот вид зондирования, который используют исследователи fairness.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|-----------------|-----------------------|
| Word embedding | Слово как вектор | Плотное, низкоразмерное (обычно 100-300) представление, выученное из контекста. |
| Skip-gram | Трюк Word2Vec | Предсказать контекстные слова по центральному слову. Медленнее, чем CBOW, лучше для редких слов. |
| Negative sampling | Ускоряющий приём обучения | Замена softmax по всему словарю бинарной классификацией против `k` случайных слов. |
| Static embedding | Один вектор на слово | Один и тот же вектор независимо от контекста. Отказывает на полисемии. |
| Contextual embedding | Контекстно-зависимый вектор | Разный вектор для каждого вхождения в зависимости от окружающих слов. То, что выдают трансформеры. |
| OOV | Вне словаря (Out of vocabulary) | Слово, не встреченное при обучении. Word2Vec не может построить для них вектор. |

## Дополнительное чтение

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — статья про negative sampling. Короткая и читабельная.
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — самый ясный вывод градиентов, если математика оригинальной статьи кажется слишком плотной.
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — настройки обучения для продакшна, которые реально работают.