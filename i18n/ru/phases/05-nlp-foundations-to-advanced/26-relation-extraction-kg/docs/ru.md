# Извлечение отношений и построение графов знаний

> NER нашёл сущности. Entity linking привязал их. Извлечение отношений находит рёбра между ними. Граф знаний — это сумма узлов, рёбер и их происхождения (provenance).

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 25 (Entity Linking)
**Time:** ~60 минут

## Проблема

Аналитик читает: "Tim Cook became CEO of Apple in 2011." Четыре факта:

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

Извлечение отношений (Relation Extraction, RE) превращает свободный текст в структурированные тройки `(subject, relation, object)`. Агрегируйте по корпусу — и у вас граф знаний. Агрегируйте и запрашивайте — и у вас есть основа для рассуждений в RAG, аналитике или аудите соответствия.

Проблема 2026 года: LLM извлекают отношения с энтузиазмом. Слишком большим энтузиазмом. Они галлюцинируют тройки, которые исходный текст не подтверждает. Без provenance вы не сможете отличить реальные тройки от правдоподобного вымысла. Ответ 2026 года — пайплайны в стиле AEVS: anchor-and-verify (закрепление и верификация).

## Концепция

![Text → triples → knowledge graph](../assets/relation-extraction.svg)

**Форма тройки.** `(subject_entity, relation_type, object_entity)`. Отношения берутся либо из закрытой онтологии (свойства Wikidata, FIBO, UMLS), либо из открытого набора (в стиле OpenIE, что угодно).

**Три подхода к извлечению.**

1. **На основе правил/шаблонов.** Шаблоны Хёрста: "X such as Y" → `(Y, isA, X)`. Плюс написанные вручную регулярные выражения. Хрупко, точно, объяснимо.
2. **Обучаемый классификатор (supervised).** Имея два упоминания сущностей в предложении, предсказываем отношение из фиксированного набора. Обучается на TACRED, ACE, KBP. Стандарт 2015–2022 годов.
3. **Генеративная LLM.** Просим модель выдать тройки. Работает "из коробки". Требует provenance, иначе галлюцинирует правдоподобно выглядящий мусор.

**AEVS (Anchor-Extraction-Verification-Supplement, 2026).** Текущий фреймворк для снижения галлюцинаций:

- **Anchor (закрепление).** Определить каждый спан сущности и спан фразы-отношения с точными позициями.
- **Extract (извлечение).** Сгенерировать тройки, привязанные к закреплённым спанам.
- **Verify (верификация).** Сопоставить каждый элемент тройки обратно с исходным текстом; отклонить всё неподтверждённое.
- **Supplement (дополнение).** Проход на полноту покрытия гарантирует, что ни один закреплённый спан не потерян.

Количество галлюцинаций резко снижается. Требует больше вычислений, но поддаётся аудиту.

**Компромисс между открытым и закрытым подходами.**

- **Закрытая онтология.** Фиксированный список свойств (например, 11 000+ свойств Wikidata). Предсказуемо. Запрашиваемо. Трудно что-то выдумать.
- **Open IE.** Любая глагольная фраза становится отношением. Высокая полнота (recall). Низкая точность (precision). Запросы сложны.

Производственные графы знаний обычно смешивают оба подхода: open IE для обнаружения, затем канонизация отношений в закрытую онтологию перед слиянием в основной граф.

```figure
relation-triples
```

## Реализация

### Шаг 1: извлечение на основе шаблонов

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

См. `code/main.py` для полного игрушечного экстрактора. Шаблоны Хёрста до сих пор используются в доменных пайплайнах, потому что их легко отлаживать.

### Шаг 2: обучаемая классификация отношений

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL — это seq2seq-экстрактор отношений: текст на входе, тройки на выходе, уже в виде идентификаторов свойств Wikidata. Дообучен на данных distant supervision. Стандартный базовый вариант с открытыми весами.

### Шаг 3: извлечение с помощью LLM-промптов и закреплением (anchoring)

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

Проверяйте каждый возвращённый спан относительно исходного текста. Отклоняйте всё, где `text[start:end] != triple_entity`. Это минимальная форма шага "verify" из AEVS.

### Шаг 4: канонизация в закрытую онтологию

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

Канонизация часто составляет 60–80% всей инженерной работы. Закладывайте на неё время.

### Шаг 5: строим небольшой граф и делаем запросы

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

Это атомарный элемент любой системы RAG-over-KG. Масштабируйте его с помощью RDF-хранилищ троек (Blazegraph, Virtuoso), графов свойств (Neo4j) или векторно-дополненных графовых хранилищ.

## Подводные камни

- **Coreference перед RE.** "He founded Apple" — RE нужно знать, кто такой "he". Сначала запустите coreference resolution (урок 24).
- **Канонизация сущностей.** "Apple Inc" и "Apple" должны разрешаться в один и тот же узел. Сначала entity linking (урок 25).
- **Галлюцинированные тройки.** LLM выдают тройки, которые текст не подтверждает. Обеспечьте верификацию спанов.
- **Дрейф канонизации отношений.** Отношения из Open IE непоследовательны ("was born in," "came from," "is a native of"). Сворачивайте их к каноническим идентификаторам, иначе граф не будет пригоден для запросов.
- **Временные ошибки.** "Tim Cook is CEO of Apple" — верно сейчас, неверно в 2005 году. Многие отношения ограничены во времени. Используйте квалификаторы (`P580` время начала, `P582` время окончания в Wikidata).
- **Несоответствие домена.** REBEL обучен на Википедии. Юридические, медицинские и научные тексты часто требуют доменно-дообученных моделей RE.

## Применение

Стек 2026 года:

| Ситуация | Выбор |
|-----------|------|
| Быстрое production-решение, общий домен | REBEL или LlamaPred с канонизацией в Wikidata |
| Доменно-специфичный (биомед, юриспруденция) | Дообучение в стиле SciREX + пользовательская онтология |
| LLM-промптинг, аудируемый вывод | Пайплайн AEVS: anchor → extract → verify → supplement |
| Высокообъёмное IE для новостей | Гибрид шаблонов и обучаемого классификатора |
| Построение графа знаний с нуля | Open IE + ручной проход канонизации |
| Временной граф знаний | Извлечение с квалификаторами (время начала/окончания, момент времени) |

Паттерн интеграции: NER → coreference → entity linking → извлечение отношений → отображение на онтологию → загрузка в граф. Каждый этап — потенциальный контроль качества.

## Готовый код

Сохраните как `outputs/skill-re-designer.md`:

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## Упражнения

1. **Лёгкое.** Запустите экстрактор на шаблонах из `code/main.py` на 5 предложениях из новостных статей. Проверьте точность вручную.
2. **Среднее.** Примените REBEL (или небольшую LLM) к тем же предложениям. Сравните тройки. У какого экстрактора точность выше? А полнота?
3. **Сложное.** Постройте пайплайн AEVS: извлечение с помощью LLM + верификация спанов относительно источника. Измерьте долю галлюцинаций до и после шага верификации на 50 предложениях в стиле Википедии.

## Ключевые термины

| Термин | Что говорят | Что это означает на самом деле |
|------|-----------------|-----------------------|
| Тройка (Triple) | Subject-relation-object | `(s, r, o)` кортеж, являющийся атомарной единицей графа знаний. |
| Open IE | Извлечь всё что угодно | Фразы отношений из открытого словаря; высокая полнота, низкая точность. |
| Закрытая онтология | Фиксированная схема | Ограниченный набор типов отношений (Wikidata, UMLS, FIBO). |
| Канонизация | Нормализовать всё | Отображение поверхностных имён/отношений в канонические идентификаторы. |
| AEVS | Обоснованное извлечение | Пайплайн Anchor-Extraction-Verification-Supplement (2026). |
| Provenance | Связь с источником истины | Каждая тройка несёт id документа + символьный спан до своего источника. |
| Distant supervision | Дешёвые метки | Сопоставление текста с существующим графом знаний для создания обучающих данных. |

## Дополнительное чтение

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — статья о distant supervision.
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — рабочая лошадка seq2seq RE.
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — совместное IE.
- [AEVS — фреймворк Anchor-Extraction-Verification-Supplement](https://www.mdpi.com/2073-431X/15/3/178) — дизайн 2026 года для снижения галлюцинаций.
- [Учебник по Wikidata SPARQL](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — канонические запросы к графу.