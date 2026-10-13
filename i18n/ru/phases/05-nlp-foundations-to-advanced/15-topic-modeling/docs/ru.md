# Тематическое моделирование — LDA и BERTopic

> LDA: документы — это смеси тем, темы — это распределения по словам. BERTopic: документы образуют кластеры в пространстве эмбеддингов, кластеры — это темы. Одна цель, разные разложения.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word2Vec)
**Time:** ~45 минут

## Проблема

У вас 10 000 обращений в службу поддержки, 50 000 новостных статей или 200 000 твитов. Вам нужно понять, о чём коллекция, не читая её целиком. У вас нет размеченных категорий. Вы даже не знаете, сколько категорий существует.

Тематическое моделирование решает эту задачу без учителя. Дайте ему корпус — получите небольшой набор согласованных тем и для каждого документа распределение по этим темам.

В этой области доминируют два семейства алгоритмов. LDA (2003) трактует каждый документ как смесь скрытых тем, а каждую тему — как распределение по словам. Вывод байесовский. Алгоритм до сих пор используется в продакшене там, где нужны смешанные назначения тем и объяснимые распределения вероятностей на уровне слов.

BERTopic (2020) кодирует документы с помощью BERT, снижает размерность с помощью UMAP, кластеризует с помощью HDBSCAN и извлекает тематические слова через class-based TF-IDF. Он выигрывает на коротких текстах, в соцсетях и везде, где семантическое сходство важнее пересечения слов. Один документ получает одну тему, что является ограничением для длинных текстов.

Этот урок формирует интуицию для обоих подходов и определяет, какой выбрать для конкретного корпуса.

## Концепция

![LDA mixture model vs BERTopic clustering](../assets/topic-modeling.svg)

**Генеративная история LDA.** Каждая тема — это распределение по словам. Каждый документ — смесь тем. Чтобы сгенерировать слово в документе, нужно выбрать тему из смеси документа, затем выбрать слово из распределения этой темы. Вывод делает это в обратную сторону: по наблюдаемым словам восстанавливается распределение тем по документу и распределение слов по теме. Вычисления выполняются с помощью свёрнутого сэмплирования Гиббса или вариационного байесовского вывода.

Ключевой результат работы LDA:

- `doc_topic`: матрица `(n_docs, n_topics)`, каждая строка суммируется в 1 (смесь тем документа).
- `topic_word`: матрица `(n_topics, vocab_size)`, каждая строка суммируется в 1 (распределение слов темы).

**Пайплайн BERTopic.**

1. Закодировать каждый документ с помощью sentence transformer (например, `all-MiniLM-L6-v2`). Векторы размерности 384.
2. Снизить размерность с помощью UMAP до ~5 измерений. Эмбеддинги BERT слишком многомерны для кластеризации.
3. Кластеризовать с помощью HDBSCAN. Плотностной метод, даёт кластеры переменного размера и метку «выброс».
4. Для каждого кластера вычислить class-based TF-IDF по документам кластера, чтобы извлечь топ-слова.

Результат — одна тема на документ (плюс метка выброса -1). Опционально — мягкое членство через вектор вероятностей HDBSCAN.

```figure
topic-drift
```

## Построение

### Шаг 1: LDA через scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

Обратите внимание: стоп-слова удалены, min_df и max_df отфильтровывают редкие и повсеместные термины, используется CountVectorizer (а не TfidfVectorizer), потому что LDA ожидает сырые счётчики.

### Шаг 2: BERTopic (продакшен)

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

Фильтр по `Topic != -1` отбрасывает корзину выбросов BERTopic (документы, которые HDBSCAN не смог кластеризовать). `min_topic_size` контролирует минимальный размер кластера в HDBSCAN; значение по умолчанию в библиотеке BERTopic — 10. В этом примере оно явно установлено в 15 для масштаба урока. Для корпусов свыше 10 000 документов увеличивайте до 50 или 100.

### Шаг 3: оценка

Оба метода выдают тематические слова. Вопрос в том, согласованы ли эти слова между собой.

- **Когерентность темы (c_v).** Сочетает NPMI (нормализованную поточечную взаимную информацию) пар топ-слов в контексте скользящего окна, агрегирует оценки в векторы тем и сравнивает эти векторы через косинусное сходство. Чем выше, тем лучше. Используйте `gensim.models.CoherenceModel` с `coherence="c_v"`.
- **Разнообразие тем.** Доля уникальных слов среди топ-слов всех тем. Чем выше, тем лучше (темы не пересекаются).
- **Качественная проверка.** Прочитайте топ-слова каждой темы. Называют ли они что-то реальное? Человеческое суждение по-прежнему остаётся последней линией защиты.

## Что выбирать и когда

| Ситуация | Выбор |
|-----------|------|
| Короткий текст (твиты, отзывы, заголовки) | BERTopic |
| Длинные документы со смесью тем | LDA |
| Нет GPU / ограниченные вычисления | LDA или NMF |
| Нужны мультитематические распределения на уровне документа | LDA |
| Интеграция с LLM для присвоения названий темам | BERTopic (прямая поддержка) |
| Развёртывание на устройствах с ограниченными ресурсами | LDA |
| Максимальная семантическая когерентность | BERTopic |

Главное практическое соображение — длина документа. Эмбеддинги BERT усекаются; счётчики LDA работают с текстом любой длины. Для документов длиннее контекста модели эмбеддингов либо разбивайте на части и агрегируйте результат, либо используйте LDA.

## Применение

Стек 2026 года:

- **BERTopic.** Выбор по умолчанию для коротких текстов и всего, где важна семантика.
- **`gensim.models.LdaModel`.** Классический LDA для продакшена, зрелый, проверенный временем.
- **`sklearn.decomposition.LatentDirichletAllocation`.** Простой LDA для экспериментов.
- **NMF.** Неотрицательное матричное разложение. Быстрая альтернатива LDA, сравнимое качество на коротких текстах.
- **Top2Vec.** Схож по устройству с BERTopic. Меньшее сообщество, но хорошие результаты на некоторых бенчмарках.
- **FASTopic.** Более новый, быстрее BERTopic на очень больших корпусах.
- **Разметка через LLM.** Запустите любую кластеризацию, затем попросите модель назвать каждый кластер.

## Внедрение

Сохраните как `outputs/skill-topic-picker.md`:

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## Упражнения

1. **Лёгкое.** Обучите LDA с 5 темами на датасете 20 Newsgroups. Выведите топ-10 слов для каждой темы. Назовите каждую тему вручную. Нашёл ли алгоритм реальные категории?
2. **Среднее.** Обучите BERTopic на том же подмножестве 20 Newsgroups. Сравните количество найденных тем, топ-слова и качественную согласованность с LDA. Какой метод точнее выявляет реальные категории?
3. **Сложное.** Вычислите когерентность c_v для LDA и BERTopic на вашем корпусе. Запустите каждый с 5, 10, 20, 50 темами. Постройте график когерентности в зависимости от количества тем. Сделайте вывод, какой метод более стабилен при разном количестве тем.

## Ключевые термины

| Термин | Что говорят люди | Что это на самом деле означает |
|------|-----------------|-----------------------|
| Тема | То, о чём корпус | Распределение вероятностей по словам (LDA) или кластер похожих документов (BERTopic). |
| Смешанное членство | Документ относится к нескольким темам | LDA присваивает каждому документу распределение по всем темам. |
| UMAP | Снижение размерности | Метод многообразного обучения, сохраняющий локальную структуру; используется в BERTopic. |
| HDBSCAN | Плотностная кластеризация | Находит кластеры переменного размера; присваивает метку «шум» (-1) выбросам. |
| Когерентность c_v | Метрика качества темы | Средняя поточечная взаимная информация топ-слов темы в скользящих окнах. |

## Дополнительное чтение

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — статья, представившая LDA.
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — статья по BERTopic.
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — статья, в которой введены c_v и другие метрики.
- [Документация BERTopic](https://maartengr.github.io/BERTopic/) — справочник для продакшена. Отличные примеры.