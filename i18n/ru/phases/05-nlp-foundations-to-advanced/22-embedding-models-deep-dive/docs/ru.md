# Модели эмбеддингов — глубокое погружение, 2026

> Word2Vec давал вектор для каждого слова. Современные модели эмбеддингов дают вектор для каждого отрывка текста, кросс-языковой, с разреженным, плотным и мульти-векторным представлением, подогнанным под размер вашего индекса. Выберете неправильно — и ваш RAG будет извлекать не то, что нужно.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec), Phase 5 · 14 (Information Retrieval)
**Time:** ~60 минут

## Проблема

Ваша RAG-система в 40% случаев извлекает не тот отрывок. Виновата редко векторная база данных или промпт. Виновата модель эмбеддингов.

Выбор эмбеддинга в 2026 году означает выбор по пяти осям:

1. **Плотные vs разреженные vs мульти-векторные.** Один вектор на отрывок, или один на токен, или разреженный взвешенный мешок слов.
2. **Языковое покрытие.** Одноязычные английские модели всё ещё выигрывают на задачах только на английском. Многоязычные модели выигрывают, когда корпуса смешанные.
3. **Длина контекста.** 512 токенов vs 8 192 vs 32 768 — и реальная эффективная ёмкость часто составляет 60-70% от заявленного максимума.
4. **Бюджет размерности.** 3 072 числа с плавающей точкой полной точности = 12 КБ на вектор. При 100 млн векторов хранение стоит $1 300/месяц. Усечение Matryoshka сокращает это в 4 раза.
5. **Открытые vs хостинговые.** Открытые веса означают, что вы контролируете стек и данные. Хостинговые означают, что вы меняете контроль на постоянную актуальность.

В этом уроке названы компромиссы, чтобы вы могли выбирать на основе фактов, а не того, что было популярно в прошлом квартале.

## Концепция

![Dense, sparse, and multi-vector embeddings](../assets/embedding-modes.svg)

**Плотные эмбеддинги.** Один вектор на отрывок (обычно 384-3 072 измерений). Косинусное сходство ранжирует отрывки по семантической близости. OpenAI `text-embedding-3-large`, BGE-M3 в плотном режиме, Voyage-3. Выбор по умолчанию.

**Разреженные эмбеддинги.** В стиле SPLADE. Трансформер предсказывает вес для каждого токена словаря, затем обнуляет большинство из них. Результат — разреженный вектор размером |словарь|. Отражает лексическое соответствие (как BM25), но с обученными весами терминов. Силён на запросах, насыщенных ключевыми словами.

**Мульти-векторное представление (позднее взаимодействие).** ColBERTv2, Jina-ColBERT. Один вектор на токен. Оценка через MaxSim: для каждого токена запроса находится наиболее похожий токен документа, оценки суммируются. Дороже в хранении и вычислении, но выигрывает на длинных запросах и доменно-специфичных корпусах.

**BGE-M3: все три сразу.** Единая модель выдаёт плотное, разреженное и мульти-векторное представления одновременно. Каждое можно запрашивать независимо; оценки объединяются взвешенной суммой. Выбор по умолчанию в 2026 году, когда нужна гибкость одной контрольной точки.

**Matryoshka Representation Learning.** Обучена так, что первые N измерений вектора образуют полезный самостоятельный эмбеддинг. Усеките 1 536-мерный вектор до 256 измерений и заплатите ~1% точности за 6-кратную экономию хранения. Поддерживается OpenAI text-3, Cohere v4, Voyage-4, Jina v5, Gemini Embedding 2, Nomic v1.5+.

### Лидерборд MTEB рассказывает лишь часть истории

Massive Text Embedding Benchmark — 56 задач 8 типов на момент запуска (2022), расширен до 100+ задач в MTEB v2. В начале 2026 года Gemini Embedding 2 лидирует в retrieval (67,71 MTEB-R). Cohere embed-v4 лидирует в общем зачёте (65,2 MTEB). BGE-M3 лидирует среди открытых многоязычных моделей (63,0). Лидерборд необходим, но недостаточен — всегда проверяйте на своём домене.

### Трёхуровневый паттерн

| Сценарий использования | Паттерн |
|----------|---------|
| Быстрый первый проход | Плотный bi-encoder (BGE-M3, text-3-small) |
| Повышение полноты (recall) | Разреженный (SPLADE, BGE-M3 sparse) + слияние RRF |
| Точность на топ-50 | Мульти-векторный (ColBERTv2) или cross-encoder реранкер |

Большинство production-стеков используют все три.

```figure
gx-matryoshka
```

## Реализация

### Шаг 1: базовый уровень — плотные эмбеддинги с Sentence-BERT

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` делает скалярное произведение равным косинусному сходству. Всегда устанавливайте это.

### Шаг 2: усечение Matryoshka

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

Пере-нормализуйте после усечения. Nomic v1.5, OpenAI text-3 и Voyage-4 обучены так, что это безошибочно для первых нескольких уровней. Модели без Matryoshka (оригинальный Sentence-BERT) резко деградируют при усечении.

### Шаг 3: мульти-функциональность BGE-M3

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

Три индекса, один вызов инференса. Слияние оценок:

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

Настройте веса под свой домен.

### Шаг 4: оценка MTEB на пользовательской задаче

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

Запустите кандидатные модели на *репрезентативном* подмножестве. Не доверяйте только рангу в лидерборде — важен ваш домен.

### Шаг 5: ручная реализация косинусного сходства с нуля

См. `code/main.py`. Усреднённые эмбеддинги Hashing Trick (только стандартная библиотека). Не конкурентоспособны с трансформерными эмбеддингами, но показывают форму: токенизация → вектор → нормализация → скалярное произведение.

## Подводные камни

- **Одна и та же модель для запроса и документа.** Некоторые модели (Voyage, Jina-ColBERT) используют асимметричное кодирование — запрос и документ проходят через разные пути. Всегда проверяйте карточку модели.
- **Отсутствующий префикс.** Модели `bge-*` требуют добавления `"Represent this sentence for searching relevant passages: "` к запросам. Разрыв в recall на 3-5 пунктов, если забыть.
- **Чрезмерное усечение Matryoshka.** 1 536 → 256 обычно безопасно. 1 536 → 64 — нет. Проверяйте на своём наборе для оценки.
- **Усечение контекста.** Большинство моделей молча обрезают входные данные, превышающие максимальную длину. Длинные документы требуют разбиения на части (см. урок 23).
- **Игнорирование хвоста задержек.** Оценки MTEB скрывают задержку p99. Модель на 600M параметров может обойти модель на 335M на 2 пункта, но стоить в 3 раза дороже за запрос.

## Применение

Стек 2026 года:

| Ситуация | Выбор |
|-----------|------|
| Только английский, быстро, API | `text-embedding-3-large` или `voyage-3-large` |
| Открытые веса, английский | `BAAI/bge-large-en-v1.5` |
| Открытые веса, многоязычный | `BAAI/bge-m3` или `Qwen3-Embedding-8B` |
| Длинный контекст (32k+) | Voyage-3-large, Cohere embed-v4, Qwen3-Embedding-8B |
| Развёртывание только на CPU | Nomic Embed v2 (137M параметров, MoE) |
| Ограничения по хранению | Усечение Matryoshka + int8-квантизация |
| Запросы, насыщенные ключевыми словами | Добавьте разреженный SPLADE, слияние RRF с плотным |

Паттерн 2026 года: начните с BGE-M3 или text-3-large, оцените на своём домене с помощью MTEB, замените, если доменно-специфичная модель выигрывает более чем на 3 пункта.

## Готово к работе

Сохраните как `outputs/skill-embedding-picker.md`:

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## Упражнения

1. **Лёгкое.** Закодируйте 100 предложений с помощью `bge-small-en-v1.5` в полной размерности (384), затем в Matryoshka-128. Измерьте падение MRR на 10 запросах.
2. **Среднее.** Сравните BGE-M3 плотный, разреженный и colbert-режимы на 500 отрывках из вашего домена. Что выигрывает по recall@10? Превосходит ли слияние RRF лучший отдельный режим?
3. **Сложное.** Запустите MTEB на трёх моделях-кандидатах на двух ваших главных доменных задачах. Сообщите оценку MTEB, задержку p99 на пакете из 100 запросов и $/1 млн запросов. Выберите Парето-оптимальную модель.

## Ключевые термины

| Термин | Что говорят люди | Что это на самом деле означает |
|------|-----------------|-----------------------|
| Плотный эмбеддинг | Вектор | Один вектор фиксированного размера на текст. Косинусное сходство для ранжирования. |
| Разреженный эмбеддинг | Обученный BM25 | Один вес на токен словаря; в основном нули; обучается сквозным образом. |
| Мульти-векторный | В стиле ColBERT | Один вектор на токен; оценка MaxSim; больший индекс, лучше recall. |
| Matryoshka | Трюк русской матрёшки | Первые N измерений сами по себе образуют корректный меньший эмбеддинг. |
| MTEB | Бенчмарк | Massive Text Embedding Benchmark — 56 задач на момент запуска, 100+ в v2. |
| BEIR | Бенчмарк retrieval | 18 задач retrieval в режиме zero-shot; часто упоминается для оценки устойчивости между доменами. |
| Асимметричное кодирование | Путь запроса ≠ путь документа | Модель использует разные проекции для запросов и документов. |

## Дополнительное чтение

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — статья о bi-encoder.
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — статья о лидерборде.
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — унифицированная трёхрежимная модель.
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — целевая функция обучения с лестницей измерений.
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — позднее взаимодействие в продакшене.
- [Лидерборд MTEB на Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — актуальные рейтинги.