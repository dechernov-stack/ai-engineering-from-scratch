# Entity Linking & Disambiguation (Связывание и разрешение сущностей)

> NER нашёл "Paris". Entity linking решает: Париж, Франция? Пэрис Хилтон? Париж, Техас? Парис (троянский принц)? Без связывания ваш граф знаний остаётся неоднозначным.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 24 (Coreference Resolution)
**Time:** ~60 минут

## Проблема

Предложение гласит: "Jordan beat the press." Ваш NER помечает "Jordan" как PERSON. Хорошо. Но *какой именно* Jordan?

- Майкл Джордан (баскетболист)?
- Майкл Б. Джордан (актёр)?
- Майкл И. Джордан (профессор ML из Беркли — да, эта путаница реальна в ML-статьях)?
- Иордания (страна)?
- Джордан (еврейское личное имя)?

Entity linking (EL) сопоставляет каждое упоминание с уникальной записью в базе знаний: Wikidata, Wikipedia, DBpedia или вашей предметной БЗ. Две подзадачи:

1. **Генерация кандидатов.** Дано "Jordan" — какие записи БЗ правдоподобны?
2. **Разрешение неоднозначности.** Учитывая контекст, какой кандидат правильный?

Оба шага поддаются обучению. Оба измеряются бенчмарками. Объединённый пайплайн стабилен уже десятилетие — меняется лишь качество дизамбигуатора.

## Концепция

![Entity linking pipeline: mention → candidates → disambiguated entity](../assets/entity-linking.svg)

**Генерация кандидатов.** По поверхностной форме упоминания ("Jordan") ищутся кандидаты в индексе псевдонимов (alias index). Словари псевдонимов Wikipedia покрывают большинство именованных сущностей: "JFK" → Джон Ф. Кеннеди, Жаклин Кеннеди, аэропорт JFK, фильм "JFK". Типичный индекс возвращает 10-30 кандидатов на упоминание.

**Разрешение неоднозначности: три подхода.**

1. **Приоритет + контекст (Milne & Witten, 2008).** `P(entity | mention) × context-similarity(entity, text)`. Работает хорошо, быстро, без обучения.
2. **На основе эмбеддингов (ESS / REL / Blink).** Кодируем упоминание + контекст. Кодируем описание каждого кандидата. Выбираем максимум косинусного сходства. Подход по умолчанию в 2020-2024 годах.
3. **Генеративный (GENRE, 2021; на базе LLM, 2023+).** Декодируем каноническое имя сущности токен за токеном. Ограничение по trie допустимых имён сущностей гарантирует, что результат будет валидным id из БЗ.

**Сквозной подход против пайплайна.** Современные модели (ELQ, BLINK, ExtEnD, GENRE) выполняют NER + генерацию кандидатов + разрешение неоднозначности за один проход. Пайплайн-системы всё ещё доминируют в продакшене, потому что компоненты можно менять местами.

### Два измерения

- **Mention recall (генерация кандидатов).** Доля эталонных упоминаний, для которых правильная запись БЗ присутствует в списке кандидатов. Нижняя граница для всего пайплайна.
- **Точность/F1 разрешения неоднозначности.** При наличии правильных кандидатов — как часто топ-1 оказывается верным.

Всегда отчитывайтесь по обоим показателям. Система с 99% точностью разрешения при 80% candidate recall — это пайплайн с эффективностью 80%.

```figure
gx-entity-linking
```

## Построение

### Шаг 1: построение индекса псевдонимов из редиректов Wikipedia

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Данные псевдонимов Wikipedia: ~18 млн пар (псевдоним, сущность). Загружаются из дампов Wikidata. Хранятся как инвертированный индекс.

### Шаг 2: разрешение неоднозначности на основе контекста

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Пересечение по Жаккару — это игрушечный пример. Замените его на косинусное сходство эмбеддингов (см. `code/main.py` шаг 2 для версии на трансформерах).

### Шаг 3: подход на основе эмбеддингов (в стиле BLINK)

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

На этапе индексации каждая сущность БЗ кодируется один раз. На этапе запроса упоминание + контекст кодируются один раз, вычисляется скалярное произведение с пулом кандидатов, выбирается максимум.

### Шаг 4: генеративное связывание сущностей (концепция)

GENRE декодирует заголовок страницы Wikipedia для сущности посимвольно. Ограниченное декодирование (см. урок 20) гарантирует, что на выходе будут только валидные заголовки. Тесная интеграция с trie на основе БЗ. Современный наследник — REL-GEN и EL на базе промптов к LLM со структурированным выводом.

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

В сочетании с белым списком (Outlines `choice`) это простейший пайплайн EL для продакшена в 2026 году.

### Шаг 5: оценка на AIDA-CoNLL

AIDA-CoNLL — стандартный бенчмарк для EL: 1393 статьи Reuters, 34 тыс. упоминаний, сущности Wikipedia. Отчитывайтесь по точности внутри БЗ (in-KB accuracy, `P@1`) и по доле обнаружения NIL (вне БЗ).

## Подводные камни

- **Обработка NIL.** Некоторые упоминания отсутствуют в БЗ (новые сущности, малоизвестные люди). Системы должны предсказывать NIL, а не угадывать неверную сущность. Измеряется отдельно.
- **Ошибки границ упоминания.** Вышестоящий NER пропускает частичные спаны ("Bank of America" помечен просто как "Bank"). Recall для EL падает.
- **Смещение к популярности.** Обученные системы чрезмерно часто предсказывают частотные сущности. Упоминание "Michael I. Jordan" в ML-статье часто связывается с баскетболистом Джорданом.
- **Межъязыковой EL.** Сопоставление упоминаний в китайском тексте с английскими сущностями Wikipedia. Требует многоязычного энкодера или этапа перевода.
- **Устаревание БЗ.** Новые компании, события, люди отсутствуют в прошлогоднем дампе Wikipedia. Продакшен-пайплайны нуждаются в цикле обновления.

## Применение

Стек 2026 года:

| Ситуация | Выбор |
|-----------|------|
| Общий английский + Wikipedia | BLINK или REL |
| Межъязыковой, БЗ = Wikipedia | mGENRE |
| LLM-дружественный, мало упоминаний в день | Промпт Claude/GPT-4 со списком кандидатов + ограниченный JSON |
| Предметная БЗ (медицина, право) | Кастомный BERT с поиском на основе БЗ + дообучение на предметном наборе в стиле AIDA |
| Крайне низкая задержка | Только точное совпадение по приоритету (базовая линия Milne-Witten) |
| SOTA для исследований | GENRE / ExtEnD / генеративный LLM-EL |

Продакшен-паттерн 2026 года: NER → coref → EL для каждого упоминания → схлопывание кластеров в одну каноническую сущность на кластер. Результат: один id БЗ на сущность в документе, а не на упоминание.

## Публикация

Сохраните как `outputs/skill-entity-linker.md`:

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## Упражнения

1. **Лёгкое.** Реализуйте дизамбигуатор на основе приоритета + контекста в `code/main.py` на 10 неоднозначных упоминаниях (Paris, Jordan, Apple). Вручную разметьте правильную сущность. Измерьте точность.
2. **Среднее.** Закодируйте 50 неоднозначных упоминаний с помощью sentence transformer. Закодируйте описание каждого кандидата. Сравните разрешение неоднозначности на основе эмбеддингов с пересечением по Жаккару на контексте.
3. **Сложное.** Постройте предметную БЗ на 1000 сущностей (например, сотрудники + продукты вашей компании). Реализуйте сквозной NER + EL. Измерьте точность и полноту на 100 отложенных предложениях.

## Ключевые термины

| Термин | Что говорят люди | Что это на самом деле означает |
|------|-----------------|-----------------------|
| Entity linking (EL) | Связать с Wikipedia | Сопоставить упоминание с уникальной записью БЗ. |
| Генерация кандидатов | Кем это может быть? | Вернуть короткий список правдоподобных записей БЗ для упоминания. |
| Разрешение неоднозначности | Выбрать правильный вариант | Оценить кандидатов по контексту, выбрать победителя. |
| Индекс псевдонимов | Таблица поиска | Отображение поверхностной формы → кандидаты-сущности. |
| NIL | Нет в БЗ | Явное предсказание того, что ни одна запись БЗ не подходит. |
| KB (БЗ) | База знаний | Wikidata, Wikipedia, DBpedia или ваша предметная БЗ. |
| AIDA-CoNLL | Бенчмарк | 1393 статьи Reuters с эталонными связями сущностей. |

## Дополнительное чтение

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://researchcommons.waikato.ac.nz/entities/publication/b9a0b520-abc5-47c5-a86a-da6c579893ab) — основополагающий подход "приоритет + контекст".
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — рабочая лошадка на основе эмбеддингов.
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — генеративный EL с ограниченным декодированием.
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — статья с бенчмарком.
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — открытый продакшен-стек.