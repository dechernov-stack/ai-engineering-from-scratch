# Стратегии чанкинга для RAG

> Конфигурация чанкинга влияет на качество ретрива не меньше, чем выбор модели эмбеддингов (Vectara, NAACL 2025). Если ошибиться с чанкингом, никакой реранкинг не спасёт.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 14 (Information Retrieval), Phase 5 · 22 (Embedding Models)
**Time:** ~60 минут

## Проблема

Вы загружаете 50-страничный контракт в RAG-систему. Пользователь спрашивает: «Какие условия расторжения договора?». Ретривер возвращает титульную страницу. Почему? Потому что модель была обучена на чанках по 512 токенов, а пункт о расторжении находится на 20-й странице, разорван разрывом страницы, и не содержит локальных ключевых слов, связывающих его с запросом.

Решение — не «купить модель эмбеддингов получше». Решение — это чанкинг. Насколько большие чанки? Нужен ли нахлёст (overlap)? Где делить текст? С каким окружающим контекстом?

Бенчмарки февраля 2026 года показывают неожиданные результаты:

- Исследование Vectara 2026 года: рекурсивный чанкинг по 512 токенов обошёл семантический чанкинг с результатом 69% против 54% по точности.
- SPLADE + Mistral-8B на Natural Questions: нахлёст не дал измеримого выигрыша.
- «Обрыв контекста»: качество ответов резко падает примерно на отметке 2500 токенов контекста.

«Очевидный» ответ (семантический чанкинг, 20% нахлёста, 1000 токенов) часто оказывается неверным. Этот урок формирует интуицию для шести стратегий и объясняет, когда к какой из них обращаться.

## Концепция

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**Фиксированный чанкинг.** Разбиение через каждые N символов или токенов. Простейший базовый вариант. Разрывает текст посреди предложения. Хорошее сжатие, плохая связность.

**Рекурсивный.** `RecursiveCharacterTextSplitter` от LangChain. Сначала пробуем делить по `\n\n`, затем по `\n`, затем по `.`, затем по пробелу. Корректно откатывается на более простой разделитель. Вариант по умолчанию в 2026 году.

**Семантический.** Эмбеддинг каждого предложения. Вычисление косинусного сходства между соседними предложениями. Разделение там, где сходство падает ниже порога. Сохраняет тематическую связность. Медленнее; иногда создаёт крошечные фрагменты по 40 токенов, которые вредят ретриву.

**Пословный (по предложениям).** Разбиение по границам предложений. Одно предложение на чанк или окно из N предложений. Соответствует по качеству семантическому чанкингу вплоть до ~5 тыс. токенов, но при гораздо меньших затратах.

**Parent-document (родитель-потомок).** Хранение небольших дочерних чанков для ретрива *и* более крупного родительского чанка для контекста. Поиск ведётся по потомку; возвращается родитель. Деградирует плавно: даже неудачные дочерние чанки возвращают приемлемых родителей.

**Late chunking (поздний чанкинг, 2024).** Сначала эмбеддинг всего документа на уровне токенов, затем пулинг токенных эмбеддингов в эмбеддинги чанков. Сохраняет контекст между чанками. Работает с эмбеддерами, поддерживающими длинный контекст (BGE-M3, Jina v3). Требует больше вычислений.

**Contextual retrieval (контекстный ретрив, Anthropic, 2024).** К каждому чанку добавляется сгенерированное LLM краткое описание его положения в документе («Этот чанк — раздел 3.2 пунктов о расторжении договора...»). По собственному бенчмарку Anthropic улучшение ретрива составляет 35-50%. Дорого при индексации.

### Правило, которое превосходит любые настройки по умолчанию

Подбирайте размер чанка под тип запроса:

| Тип запроса | Размер чанка |
|------------|-----------|
| Фактоидный («как зовут CEO?») | 256-512 токенов |
| Аналитический / многошаговый | 512-1024 токена |
| Понимание целого раздела | 1024-2048 токена |

Бенчмарк NVIDIA 2026 года. Чанк должен быть достаточно большим, чтобы вмещать ответ вместе с локальным контекстом, и достаточно маленьким, чтобы топ-K результатов ретривера фокусировались на ответе, а не на контекстном шуме.

```figure
n5-chunk-cuts
```

## Реализация

### Шаг 1: фиксированный и рекурсивный чанкинг

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### Шаг 2: семантический чанкинг

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

Настройте `threshold` под свою предметную область. Слишком высокое значение → фрагментация. Слишком низкое → один гигантский чанк.

### Шаг 3: parent-document

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

Ключевой момент: дедуплицируйте родителей. Несколько потомков могут указывать на одного и того же родителя; возврат всех был бы пустой тратой контекста.

### Шаг 4: contextual retrieval (паттерн Anthropic)

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

Индексируйте контекстуализированные чанки. На этапе запроса ретрив выигрывает за счёт дополнительного окружающего сигнала.

### Шаг 5: оценка

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

Всегда проводите бенчмаркинг. «Лучшая» стратегия для вашего корпуса может не совпасть ни с одной статьёй в блоге.

## Подводные камни

- **Оценка чанкинга только на фактоидных запросах.** Многошаговые запросы могут выявить совершенно других победителей. Используйте оценочный набор, стратифицированный по типам запросов.
- **Семантический чанкинг без минимального размера.** Приводит к фрагментам по 40 токенов, которые вредят ретриву. Всегда соблюдайте `min_tokens`.
- **Нахлёст как карго-культ.** Исследования 2026 года показывают, что нахлёст часто не даёт никакого выигрыша, но удваивает стоимость индексации. Измеряйте, а не предполагайте.
- **Отсутствие контроля min/max.** Чанки по 5 токенов или по 5000 токенов одинаково ломают ретрив. Ограничивайте диапазон.
- **Чанкинг между документами.** Никогда не допускайте, чтобы чанк охватывал два документа. Всегда делите текст по документам отдельно, а затем объединяйте.

## Применение

Стек 2026 года:

| Ситуация | Стратегия |
|-----------|----------|
| Первая сборка, неизвестный корпус | Рекурсивный, 512 токенов, без нахлёста |
| Фактоидные вопросы | Рекурсивный, 256-512 токенов |
| Аналитические / многошаговые запросы | Рекурсивный, 512-1024 токена + parent-document |
| Интенсивные перекрёстные ссылки (контракты, статьи) | Late chunking или contextual retrieval |
| Диалоговый / разговорный корпус | Чанки на уровне реплик + метаданные говорящего |
| Короткие высказывания (твиты, отзывы) | Один документ = один чанк |

Начните с рекурсивного чанкинга по 512 токенов. Измерьте recall@5 на наборе из 50 оценочных запросов. Настраивайте дальше, отталкиваясь от этого.

## Готовый результат

Сохраните как `outputs/skill-chunker.md`:

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## Упражнения

1. **Лёгкое.** Разбейте один 20-страничный документ с помощью fixed(512, 0), recursive(512, 0) и recursive(512, 100). Сравните количество чанков и качество границ.
2. **Среднее.** Постройте оценочный набор из 30 запросов по 5 документам. Измерьте recall@5 для рекурсивного, семантического и parent-document подходов. Кто побеждает? Совпадает ли это с результатами из статей в блогах?
3. **Сложное.** Реализуйте contextual retrieval. Измерьте улучшение MRR относительно базового рекурсивного подхода. Сравните стоимость индексации (вызовы LLM) с приростом точности.

## Ключевые термины

| Термин | Что говорят люди | Что это означает на самом деле |
|------|-----------------|-----------------------|
| Чанк (Chunk) | Кусок документа | Субдокументная единица, которая эмбеддится, индексируется и извлекается. |
| Нахлёст (Overlap) | Запас надёжности | N токенов, общих для соседних чанков; часто бесполезен по бенчмаркам 2026 года. |
| Семантический чанкинг | Умный чанкинг | Разделение там, где падает сходство эмбеддингов соседних предложений. |
| Parent-document | Двухуровневый ретрив | Поиск по мелким потомкам, возврат более крупных родителей. |
| Late chunking | Чанкинг после эмбеддинга | Эмбеддинг всего документа на уровне токенов с последующим пулингом в векторы чанков. |
| Contextual retrieval | Трюк Anthropic | Сгенерированное LLM краткое описание, добавляемое к каждому чанку перед индексацией. |
| Обрыв контекста (Context cliff) | «Стена» в 2500 токенов | Падение качества, наблюдаемое около отметки 2,5 тыс. токенов контекста в RAG (январь 2026). |

## Дополнительные материалы

- [Yepes et al. / LangChain — документация по рекурсивному разбиению по символам](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — вариант по умолчанию в продакшене.
- [Vectara (2024, NAACL 2025). Анализ конфигураций чанкинга](https://arxiv.org/abs/2410.13070) — чанкинг важен не меньше, чем выбор эмбеддингов.
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — статья о позднем чанкинге.
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — улучшение ретрива на 35-50% за счёт сгенерированных LLM префиксов с контекстом.
- [Бенчмарк NVIDIA 2026 года по размеру чанков — сводка Premai](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — размер чанка в зависимости от типа запроса.