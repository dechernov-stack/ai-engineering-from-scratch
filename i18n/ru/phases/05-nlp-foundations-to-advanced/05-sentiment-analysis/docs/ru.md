# Анализ тональности

> Каноническая задача NLP. Здесь проявляется почти всё, что нужно знать о классической классификации текста.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 2 · 14 (Naive Bayes)
**Time:** ~75 минут

## Проблема

«Еда была не очень хорошей». Позитив или негатив?

Тональность звучит просто. Рецензент сказал, что ему понравилось или не понравилось что-то. Присвойте предложению метку. Причина, по которой эта задача стала канонической в NLP, в том, что за каждым простым на вид случаем скрывается сложный. Отрицание переворачивает смысл. Сарказм инвертирует его. «Not bad at all» («совсем неплохо») — позитивно, несмотря на два слова с негативной окраской. Эмодзи несут больше сигнала, чем окружающий текст. Важна доменная лексика (`tight` в музыкальном обзоре против `tight` в обзоре на моду).

Тональность — это рабочая лаборатория для классического NLP. Если вы понимаете, почему у каждого наивного базового решения есть свой специфический сценарий отказа, вы понимаете, зачем были изобретены все более сложные модели. В этом уроке мы построим базовую модель на Naive Bayes с нуля, добавим логистическую регрессию и назовём ловушки, из-за которых промышленный анализ тональности превращается в задачу уровня комплаенса.

## Концепция

Классическая тональность — это рецепт из двух шагов.

1. **Представление.** Превратить текст в вектор признаков. BoW, TF-IDF или n-граммы.
2. **Классификация.** Обучить линейную модель (Naive Bayes, логистическая регрессия, SVM) на размеченных примерах.

Naive Bayes — самая примитивная модель, которая при этом работает. Предполагается, что все признаки независимы при условии метки. Оцените `P(word | positive)` и `P(word | negative)` по счётчикам. На этапе инференса перемножьте вероятности. Наивное предположение о независимости смехотворно неверно, и тем не менее результаты поразительно сильны. Причина: при разреженных текстовых признаках и умеренном объёме данных классификатору важнее, в какую сторону склоняется каждое слово, чем насколько сильно.

Логистическая регрессия устраняет предположение о независимости. Она обучает вес для каждого признака, включая отрицательные веса. `not good` как биграммный признак получает отрицательный вес. Naive Bayes не может сделать это для биграмм, которые никогда не встречались с меткой.

```figure
sentiment-logits
```

## Постройте это

### Шаг 1: реальный мини-датасет

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

Маленький — намеренно. В реальной работе используются десятки тысяч примеров (IMDb, SST-2, Yelp polarity). Математика та же самая.

### Шаг 2: мультиномиальный Naive Bayes с нуля

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

Аддитивное сглаживание (alpha=1.0) — это сглаживание Лапласа. Без него слово, не встреченное в классе, получает нулевую вероятность, и логарифм "взрывается". `alpha=0.01` часто используется на практике. `alpha=1.0` — значение по умолчанию для обучения.

### Шаг 3: логистическая регрессия с нуля

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

Здесь важна L2-регуляризация. Текстовые признаки разрежены; без L2 модель заучивает обучающие примеры наизусть. Начните с `0.01` и подбирайте значение.

### Шаг 4: обработка отрицаний (сценарий отказа)

Рассмотрим «not good» и «not bad». BoW-классификатор видит `{not, good}` и `{not, bad}` и обучается на том, что чаще встречалось в обучающей выборке. Биграммный классификатор видит `not_good` и `not_bad` и учит их как разные признаки. Обычно этого достаточно.

Более грубое решение, работающее, когда биграмм нет: **разметка области действия отрицания (negation scoping)**. Добавляйте префикс `NOT_` к токенам, следующим за словом-отрицанием, вплоть до ближайшего знака препинания.

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

Теперь `good` и `NOT_good` — разные признаки. Классификатор может присвоить им противоположные веса. Три строки предобработки — и заметный прирост точности на бенчмарках тональности.

### Шаг 5: метрики оценки, которые имеют значение

Одна лишь точность (accuracy) может вводить в заблуждение при несбалансированных классах. Реальные корпуса тональности обычно на 70-80% позитивны или на 70-80% негативны; классификатор, всегда предсказывающий мажоритарный класс, получает 80% точности и при этом бесполезен. Указывайте каждую из следующих метрик:

- **Precision и recall по классам.** Одна пара на класс. Усредните их макро-методом, чтобы получить одно число, учитывающее баланс классов.
- **Macro-F1 (основная метрика для несбалансированных данных).** Среднее значение F1 по классам с равным весом. Используйте эту метрику вместо accuracy при несбалансированных классах.
- **Weighted-F1 (альтернатива).** То же самое, что macro, но с весом по частоте класса. Указывайте вместе с macro-F1, когда сам дисбаланс имеет значение для бизнеса.
- **Матрица ошибок (confusion matrix).** Сырые счётчики. Всегда проверяйте её, прежде чем доверять любой скалярной метрике; она показывает, какую пару классов модель путает.
- **Примеры ошибок по классам.** Возьмите 5 неверных предсказаний на класс. Прочитайте их. Ничто не заменит чтение реальных ошибок.

Для сильно несбалансированных данных (соотношение > 95-5) указывайте **AUROC** и **AUPRC** вместо accuracy. AUPRC более чувствительна к миноритарному классу, который обычно и представляет интерес (спам, мошенничество, редкая тональность).

**Типичная ошибка, которой следует избегать.** Указание micro-F1 вместо macro-F1 на несбалансированных данных даёт число, которое выглядит высоким, потому что в нём доминирует мажоритарный класс. Macro-F1 заставляет вас увидеть производительность на миноритарном классе.

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## Используйте это

scikit-learn делает это за шесть строк, причём правильно.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

Обратите внимание на три вещи. `stop_words=None` сохраняет отрицания. `ngram_range=(1, 2)` добавляет биграммы, так что `not_good` становится признаком. `sublinear_tf=True` сглаживает влияние повторяющихся слов. Эти три флага — разница между базовой моделью с точностью 75% и моделью с точностью 85% на SST-2.

### Когда стоит обращаться к трансформеру

- Обнаружение сарказма. Классические модели здесь терпят неудачу. Точка.
- Длинные отзывы, где тональность меняется посреди документа.
- Аспектно-ориентированная тональность. «Камера отличная, но батарея ужасная». Нужно приписывать тональность конкретным аспектам. Только трансформеры или модели со структурированным выводом.
- Неанглийские языки с малым объёмом ресурсов. Многоязычный BERT даёт бесплатную базовую модель в режиме zero-shot.

Если вам нужно что-то из вышеперечисленного, переходите сразу к фазе 7 (глубокое погружение в трансформеры). В остальных случаях Naive Bayes или логистическая регрессия на TF-IDF плюс биграммы плюс обработка отрицаний — это ваша производственная база на 2026 год.

### Ловушка воспроизводимости (снова)

Переобучение моделей тональности — рутинная операция. Переоценка — нет. Показатели точности, приведённые в статьях, получены на конкретных разбиениях данных, с конкретной предобработкой, конкретными токенизаторами. Если вы сравниваете свою новую модель с базовой без использования идентичного пайплайна, вы получите вводящие в заблуждение дельты. Всегда пересчитывайте базовую модель на своём пайплайне, а не берите число из статьи.

## Отправьте в продакшен

Сохраните как `outputs/prompt-sentiment-baseline.md`:

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## Упражнения

1. **Лёгкое.** Добавьте `apply_negation` как шаг предобработки в пайплайн scikit-learn и измерьте изменение F1 на небольшом датасете тональности.
2. **Среднее.** Реализуйте логистическую регрессию с весами классов (передайте `class_weight="balanced"` в scikit-learn или выведите градиент самостоятельно). Измерьте эффект на синтетическом дисбалансе классов 90-10.
3. **Сложное.** Постройте детектор сарказма, обучив второй классификатор на остатках (residuals) модели тональности. Задокументируйте свою экспериментальную постановку. Предупредите читателя, если ваша точность ниже уровня случайности (случайный уровень для бинарного сарказма составляет ~50%, и большинство первых попыток оказываются именно там).

## Ключевые термины

| Термин | Что говорят люди | Что это на самом деле означает |
|------|-----------------|-----------------------|
| Полярность (Polarity) | Позитив или негатив | Бинарная метка; иногда расширяется до нейтральной или тонкой градации (5-звёздочная шкала). |
| Аспектно-ориентированная тональность (Aspect-based sentiment) | Тональность по аспектам | Приписывание тональности конкретным сущностям или атрибутам, упомянутым в тексте. |
| Разметка области отрицания (Negation scoping) | Инвертирование соседних токенов | Добавление префикса `NOT_` к токенам после «not» до знака препинания. |
| Сглаживание Лапласа (Laplace smoothing) | Прибавление 1 к счётчикам | Предотвращает нулевую вероятность признаков в Naive Bayes. |
| L2-регуляризация | Уменьшение весов | Добавляет `lambda * sum(w^2)` к функции потерь. Необходима для разреженных текстовых признаков. |

## Дополнительное чтение

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — основополагающий обзор. Длинный, но первые четыре раздела охватывают всю классику.
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — статья, показавшая, что биграммы + Naive Bayes сложно превзойти на коротком тексте.
- [Документация scikit-learn по извлечению текстовых признаков](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — справочник по `CountVectorizer`, `TfidfVectorizer` и всем параметрам, которые вы будете настраивать.