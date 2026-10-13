# Информационный поиск и поиск по данным

> BM25 точен, но хрупок. Dense охватывает широко, но упускает ключевые слова. Hybrid — стандарт 2026 года. Всё остальное — настройка.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 минут

## Проблема

Пользователь вводит «что будет, если кто-то обманом получил деньги» и ожидает найти статью, которая реально это покрывает: «Статья 420 УК». Поиск по ключевым словам полностью промахивается (нет общей лексики). Семантический поиск промахивается, если эмбеддинги не обучались на юридических текстах. Реальный поиск должен справляться с обоими случаями.

IR — это конвейер, лежащий под каждой RAG-системой, каждой строкой поиска, каждым нечётким поиском в документации. Архитектура 2026 года, работающая в продакшене, — это не единый метод. Это цепочка взаимодополняющих методов, каждый из которых перехватывает ошибки предыдущего.

В этом уроке мы соберём каждую часть и назовём, какие ошибки перехватывает каждая из них.

## Концепция

![Hybrid retrieval: BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

Четыре уровня. Выбирайте те, что нужны.

1. **Разреженный поиск (BM25).** Быстрый, точный на точных совпадениях, ужасен в семантике. Работает поверх инвертированного индекса. Менее 10 мс на запрос для миллионов документов. Даёт правильные результаты для ссылок на статьи, кодов продуктов, сообщений об ошибках, именованных сущностей.
2. **Плотный поиск (Dense retrieval).** Кодирует запрос и документы в векторы. Поиск ближайших соседей. Улавливает перефразирования и семантическое сходство. Упускает точные совпадения ключевых слов, отличающиеся на один символ. 50-200 мс на запрос с FAISS или векторной БД.
3. **Слияние (Fusion).** Объединяет ранжированные списки из разреженного и плотного поиска. Reciprocal Rank Fusion (RRF) — простой вариант по умолчанию, поскольку он игнорирует необработанные оценки (которые находятся в разных масштабах) и использует только позиции в ранге. Взвешенное слияние — вариант, когда вы знаете, что один сигнал доминирует в вашей области.
4. **Переранжирование кросс-энкодером.** Берём топ-30 после слияния. Запускаем кросс-энкодер (запрос + документ вместе, оценка каждой пары). Оставляем топ-5. Кросс-энкодеры медленнее на пару, чем би-энкодеры, но намного точнее. Затраты амортизируются тем, что они запускаются только на топ-30.

Трёхсторонний поиск (BM25 + dense + обученный разреженный, например SPLADE) превосходит двусторонний в бенчмарках 2026 года, но требует инфраструктуры для обученных разреженных индексов. Для большинства команд двусторонний поиск плюс переранжирование кросс-энкодером — оптимальная точка.

```figure
gx-hybrid-retrieval
```

## Собираем

### Шаг 1: BM25 с нуля

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

Два параметра, которые стоит знать. `k1=1.5` управляет насыщением по частоте термина; чем выше значение, тем больше веса придаётся повторению термина. `b=0.75` управляет нормализацией по длине; 0 игнорирует длину документа, 1 полностью нормализует. Значения по умолчанию — рекомендации Робертсона из оригинальной статьи, их редко нужно настраивать.

### Шаг 2: плотный поиск с би-энкодером

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

L2-нормализуйте эмбеддинги, чтобы скалярное произведение равнялось косинусу. `all-MiniLM-L6-v2` имеет 384 измерения, быстрый и достаточно сильный для большинства задач поиска на английском. Для мультиязычной работы используйте `paraphrase-multilingual-MiniLM-L12-v2`. Для максимальной точности — `bge-large-en-v1.5` или `e5-large-v2`.

### Шаг 3: Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

Константа `k=60` взята из оригинальной статьи по RRF. Более высокое значение `k` сглаживает вклад разницы в рангах; более низкое `k` заставляет доминировать верхние ранги. 60 — опубликованное значение по умолчанию, его редко нужно настраивать.

### Шаг 4: гибридный поиск + переранжирование

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

Три этапа, скомпонованные вместе. BM25 находит лексические совпадения. Dense находит семантические совпадения. RRF объединяет два ранжирования без необходимости калибровки оценок. Кросс-энкодер переоценивает топ-30, используя пары запрос-документ вместе, что улавливает тонкую релевантность, упущенную би-энкодером. Оставляем топ-5.

### Шаг 5: оценка

| Метрика | Значение |
|--------|---------|
| Recall@k | Из запросов, для которых правильный документ существует, как часто он оказывается в топ-k? |
| MRR (Mean Reciprocal Rank) | Среднее значение 1/ранг первого релевантного документа. |
| nDCG@k | Учитывает градации релевантности, а не только бинарное релевантно/нерелевантно. |

Конкретно для RAG **Recall@k** ретривера — самый важный показатель. Ваш reader не может ответить, если нужный фрагмент не попал в извлечённый набор.

Совет по отладке: для неудачных запросов сравните разреженное и плотное ранжирования. Если один находит правильный документ, а другой нет, у вас либо несовпадение лексики (решение: добавить недостающую половину), либо семантическая неоднозначность (решение: лучшие эмбеддинги или реранкер).

## Применяем

Стек 2026 года:

| Масштаб | Стек |
|-------|-------|
| 1k-100k документов | BM25 в памяти + эмбеддинги `all-MiniLM-L6-v2` + RRF. Отдельная БД не нужна. |
| 100k-10M документов | FAISS или pgvector для dense + Elasticsearch / OpenSearch для BM25. Запускаются параллельно. |
| 10M+ документов | Qdrant / Weaviate / Vespa / Milvus с поддержкой гибридного поиска. Переранжирование кросс-энкодером на топ-30. |
| Передовой край качества | Трёхсторонний поиск (BM25 + dense + SPLADE) + переранжирование с поздним взаимодействием ColBERT |

Что бы вы ни выбрали, заложите бюджет на оценку. Бенчмаркайте recall ретривера до того, как бенчмаркать точность RAG целиком. Reader не может исправить то, что упустил ретривер.

### Трудно добытые уроки из продакшен-RAG 2026 года

- **80% сбоев RAG связаны с загрузкой данных и разбиением на чанки, а не с моделью.** Команды тратят недели на замену LLM и настройку промптов, пока поиск молча возвращает неверный контекст в каждом третьем запросе. Сначала исправьте chunking.
- **Стратегия разбиения на чанки важнее размера чанка.** Разбиение фиксированного размера ломает таблицы, код и вложенные заголовки. Разбиение с учётом предложений — вариант по умолчанию; семантическое или на основе LLM разбиение окупается для технической документации и руководств по продуктам.
- **Шаблон родитель-потомок (parent-doc).** Извлекайте маленькие «дочерние» чанки для точности. Когда появляется несколько дочерних чанков из одного родительского раздела, подставляйте родительский блок, чтобы сохранить контекст. Это стабильно повышает качество ответов без переобучения.
- **k_rerank=3 обычно оптимально.** Каждый дополнительный чанк сверх этого добавляет расход токенов и задержку генерации, не повышая качество ответа. Если k=8 всё ещё лучше, чем k=3, значит реранкер работает хуже, чем нужно.
- **HyDE / расширение запроса.** Сгенерируйте гипотетический ответ на основе запроса, закодируйте его, извлекайте по нему. Устраняет разрыв между формулировкой коротких вопросов и длинными документами. Бесплатный прирост точности без обучения.
- **Бюджет контекста менее 8K токенов.** Стабильное достижение этого предела означает, что порог реранкера слишком мягкий.
- **Версионируйте всё.** Промпты, правила разбиения на чанки, модель эмбеддингов, реранкер. Любой дрейф незаметно портит качество ответов. CI-проверки по достоверности (faithfulness), точности контекста и доле неотвеченных вопросов блокируют регрессии до того, как их увидят пользователи.
- **Трёхсторонний поиск (BM25 + dense + обученный разреженный, например SPLADE) превосходит двусторонний** в бенчмарках 2026 года, особенно для запросов, смешивающих имена собственные с семантикой. Внедряйте его, когда инфраструктура поддерживает индексы SPLADE.

Правильное проектирование ретривера снижает галлюцинации на 70-90% по данным отраслевых измерений 2026 года. Большая часть прироста производительности RAG происходит за счёт улучшения поиска, а не дообучения модели.

## Выпускаем в продакшен

Сохраните как `outputs/skill-retrieval-picker.md`:

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## Упражнения

1. **Лёгкое.** Реализуйте `hybrid_search` выше на корпусе из 500 документов. Протестируйте на 20 запросах. Сравните recall@5 для BM25-only, dense-only и hybrid.
2. **Среднее.** Добавьте вычисление MRR. Для каждого тестового запроса с известным правильным документом найдите ранг правильного документа в ранжированиях BM25, dense и hybrid. Сообщите значение MRR для каждого.
3. **Сложное.** Дообучите плотный энкодер на вашей предметной области, используя MultipleNegativesRankingLoss (Sentence Transformers). Постройте обучающий набор из 500 пар запрос-документ. Сравните recall до и после дообучения.

## Ключевые термины

| Термин | Что обычно говорят | Что это на самом деле означает |
|------|-----------------|-----------------------|
| BM25 | Поиск по ключевым словам | Okapi BM25. Оценивает документы по частоте терминов, IDF и длине. |
| Dense retrieval | Векторный поиск | Кодирует запрос + документ в векторы, находит ближайших соседей. |
| Bi-encoder | Модель эмбеддингов | Кодирует запрос и документ независимо друг от друга. Быстр во время запроса. |
| Cross-encoder | Модель-реранкер | Кодирует запрос + документ вместе. Медленная, но точная. |
| RRF | Слияние ранжирований | Объединяет два ранжирования суммированием `1/(k + rank)`. |
| Recall@k | Метрика ретривера | Доля запросов, для которых релевантный документ есть в топ-k. |

## Дополнительное чтение

- [Robertson и Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — исчерпывающее изложение BM25.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR, канонический би-энкодер.
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — обученный разреженный ретривер, сокращающий разрыв с dense.
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — статья по RRF.
- [Khattab и Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — поиск с поздним взаимодействием.