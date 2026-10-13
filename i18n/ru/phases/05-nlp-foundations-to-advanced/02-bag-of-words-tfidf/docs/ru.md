# Мешок слов, TF-IDF и представление текста

> Сначала считай, потом думай. В 2026 году TF-IDF всё ещё превосходит эмбеддинги на чётко определённых задачах.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 02 (Linear Regression from Scratch)
**Time:** ~75 минут

## Проблема

Модели нужны числа. У вас есть строки.

Каждый NLP-пайплайн должен ответить на один и тот же вопрос. Как превратить поток токенов переменной длины в вектор фиксированного размера, который сможет потребить классификатор. Первый ответ, к которому пришла область, был самым тупым из работающих. Посчитай слова. Сделай вектор.

Этот вектор обслужил больше промышленного NLP, чем любая модель эмбеддингов. Спам-фильтры, классификаторы тем, обнаружение аномалий в логах, ранжирование в поиске (до BM25), первая волна анализа тональности, первое десятилетие академических бенчмарков NLP. Практики в 2026 году по-прежнему первым делом тянутся к нему на узких задачах классификации. Он быстрый, интерпретируемый и часто неотличим по качеству от модели эмбеддингов с 400 миллионами параметров на задачах, где важно само наличие слова.

Этот урок строит мешок слов, затем TF-IDF, с нуля. Потом показывает, как scikit-learn делает то же самое в три строки. Потом называет режим отказа, из-за которого приходится переходить на эмбеддинги.

## Концепция

**Мешок слов (Bag of Words, BoW)** отбрасывает порядок. Для каждого документа считается, сколько раз встречается каждое слово словаря. Длина вектора равна размеру словаря. Позиция `i` — это количество вхождений слова `i`.

**TF-IDF** переназначает веса BoW. Слово, встречающееся в каждом документе, неинформативно, поэтому его вес уменьшают. Слово, редкое по всему корпусу, но частое в одном конкретном документе, — это сигнал, поэтому его вес увеличивают.

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

Где `TF` — частота термина в документе, `df` — документная частота (в скольких документах встречается слово), `N` — общее число документов. `log` ограничивает вес для повсеместных слов.

Ключевое свойство: оба метода дают разреженные векторы с интерпретируемыми осями. Можно посмотреть на веса обученного классификатора и прочитать, какие слова толкают документ к тому или иному классу. С 768-мерным эмбеддингом BERT так сделать нельзя.

```figure
bow-tfidf
```

## Строим

### Шаг 1: построение словаря

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

Вход: список токенизированных документов (подойдёт любой пословный токенизатор; `code/main.py` в этом уроке использует упрощённый вариант с приведением к нижнему регистру). Выход: словарь `{word: index}`. Стабильный порядок вставки означает, что слово с индексом 0 — это первое слово, встреченное в первом документе. Соглашения разнятся; scikit-learn сортирует по алфавиту.

### Шаг 2: мешок слов

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

Строки — это документы. Столбцы — индексы словаря. Элемент `[i][j]` означает «сколько раз слово `j` встречается в документе `i`». В документе 1 слово `cat` встречается дважды, потому что так оно и есть. В документе 0 слово `ran` встречается ноль раз, потому что его там нет.

### Шаг 3: частота термина и документная частота

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

Стоит назвать два приёма сглаживания. `(n+1)/(d+1)` позволяет избежать `log(x/0)`. Завершающее `+1` гарантирует, что слово, встречающееся в каждом документе, всё равно получит IDF, равный 1 (а не 0), что совпадает со значением по умолчанию в scikit-learn. В других реализациях используется необработанный `log(N/df)`. Оба варианта работают; сглаженный вариант удобнее.

### Шаг 4: TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

Три документа, пять слов словаря (`the`, `cat`, `sat`, `dog`, `ran`). `the` встречается во всех трёх, поэтому его IDF низкий. `dog` встречается в одном, поэтому его IDF высокий. Векторы разреженные (большинство значений малы), и дискриминативные слова выделяются.

### Шаг 5: L2-нормализация строк

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

Без нормализации у более длинного документа получается больший вектор, и он доминирует в оценках схожести. L2-нормализация помещает каждый документ на единичную гиперсферу. Косинусное сходство между строками теперь — это просто скалярное произведение.

## Используем

scikit-learn поставляет промышленную версию.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` выполняет токенизацию, построение словаря и BoW за один вызов. `TfidfVectorizer` добавляет взвешивание IDF и L2-нормализацию. Оба возвращают разреженные матрицы. Для 100 тысяч документов плотная версия не помещается в память; оставайтесь разреженными, пока классификатор не потребует плотный формат.

Параметры, которые меняют всё:

| Аргумент | Эффект |
|-----|--------|
| `ngram_range=(1, 2)` | Включает биграммы. Обычно повышает качество классификации. |
| `min_df=2` | Отбрасывает слова, встречающиеся менее чем в 2 документах. Сокращает словарь на зашумлённых данных. |
| `max_df=0.95` | Отбрасывает слова, встречающиеся более чем в 95% документов. Приближённо имитирует удаление стоп-слов без жёстко заданного списка. |
| `stop_words="english"` | Встроенный список стоп-слов scikit-learn. Зависит от задачи — при анализе тональности *не* следует отбрасывать отрицания. |
| `sublinear_tf=True` | Использовать `1 + log(tf)` вместо необработанного `tf`. Помогает, когда термин многократно повторяется в одном документе. |

### Когда TF-IDF всё ещё побеждает (по состоянию на 2026 год)

- Обнаружение спама, разметка тем, флагирование аномалий в логах. Важно наличие слова, а не смысловые нюансы.
- Режимы с малым объёмом данных (сотни размеченных примеров). TF-IDF плюс логистическая регрессия не требуют затрат на предобучение.
- Везде, где важна задержка. TF-IDF плюс линейная модель отвечают за микросекунды. Получение эмбеддинга документа через трансформер занимает 10-100 мс.
- Системы, которые должны объяснять свои предсказания. Посмотрите на коэффициенты классификатора. Топ положительных слов — это и есть причина.

### Когда TF-IDF терпит неудачу

Отказ из-за семантической слепоты. Рассмотрим два документа:

- «Фильм был совсем не хорош.»
- «Фильм был превосходен.»

Один — отрицательный отзыв. Другой — положительный. Их пересечение по TF-IDF — ровно `{the, movie, was}`. Классификатору на основе мешка слов приходится запоминать, что слово `not` рядом с `good` переворачивает метку. Он может выучить это на достаточном объёме данных, но никогда так же изящно, как модель, понимающая синтаксис.

Другой отказ: слова вне словаря на этапе инференса. Модель BoW, обученная на отзывах IMDb, понятия не имеет, что делать с `Zoomer-approved`, если этот токен ни разу не встречался при обучении. Субсловные эмбеддинги (урок 04) справляются с этим. TF-IDF — нет.

### Гибрид: эмбеддинги, взвешенные TF-IDF

Прагматичный выбор по умолчанию в 2026 году для классификации на данных среднего объёма: использовать веса TF-IDF как внимание над эмбеддингами слов.

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

Вы получаете семантическую емкость от эмбеддингов и акцент на редких словах от TF-IDF. Классификатор обучается на объединённом (пулированном) векторе. Это превосходит оба подхода по отдельности для классификации тональности, тем и намерений при объёме размеченных данных менее примерно 50 тысяч примеров.

## Отправляем в прод

Сохраните как `outputs/prompt-vectorization-picker.md`:

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## Упражнения

1. **Лёгкое.** Реализуйте `cosine_similarity(doc_vec_a, doc_vec_b)` на выходных данных TF-IDF с L2-нормализацией. Убедитесь, что идентичные документы получают оценку 1.0, а документы с непересекающимися словарями — 0.0.
2. **Среднее.** Добавьте поддержку `n-gram` в `bag_of_words`. Параметр `n` должен производить подсчёты по `n`-граммам. Проверьте, что `n=2` на `["the", "cat", "sat"]` даёт подсчёты биграмм для `["the cat", "cat sat"]`.
3. **Сложное.** Постройте описанный выше гибрид TF-IDF-взвешенных эмбеддингов, используя векторы GloVe размерности 100 (скачайте один раз, кешируйте). Сравните точность классификации с обычным TF-IDF и с обычными усреднёнными (mean-pooled) эмбеддингами на датасете 20 Newsgroups. Укажите, что где побеждает.

## Ключевые термины

| Термин | Что говорят люди | Что это на самом деле означает |
|------|-----------------|-----------------------|
| BoW | Вектор частот слов | Подсчёты слов словаря в одном документе. Отбрасывает порядок. |
| TF | Частота термина | Количество вхождений слова в документе, опционально нормированное на длину документа. |
| DF | Документная частота | Количество документов, содержащих слово хотя бы один раз. |
| IDF | Обратная документная частота | `log(N / df)`, сглаженное. Понижает вес слов, встречающихся повсюду. |
| Разреженный вектор | В основном нули | Словарь обычно содержит 10-100 тысяч слов; большинство из них отсутствует в любом конкретном документе. |
| Косинусное сходство | Угол между векторами | Скалярное произведение L2-нормализованных векторов. 1 — идентичны, 0 — ортогональны. |

## Дополнительное чтение

- [scikit-learn — извлечение признаков из текста](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — каноническая справка по API, плюс заметки о каждом параметре.
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — статья, сделавшая TF-IDF стандартом по умолчанию на целое десятилетие.
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — взгляд из 2026 года на то, когда старый метод побеждает и почему.