# Структурированные выводы и ограниченная генерация (constrained decoding)

> Попросите LLM выдать JSON. В большинстве случаев получите JSON. В продакшене «в большинстве случаев» — это проблема. Ограниченная генерация превращает «в большинстве случаев» в «всегда», редактируя логиты перед сэмплированием.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 минут

## Проблема

Классификатор обращается к LLM с запросом: «Верни одно из {positive, negative, neutral}». Модель возвращает «The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ...». Ваш парсер падает. F1 вашего классификатора равен 0.0.

Свободная генерация — это не контракт. Это пожелание. Продакшен-системе нужен контракт.

В 2026 году существует три уровня.

1. **Промптинг.** Вежливо попросить. «Верни только JSON-объект». Работает ~80% на топовых моделях, хуже — на небольших.
2. **Нативные API структурированного вывода.** OpenAI `response_format`, Anthropic tool use, Gemini JSON mode. Надёжны для поддерживаемых схем. Привязка к вендору.
3. **Ограниченная генерация.** Изменение логитов на каждом шаге генерации так, чтобы модель *не могла* выдать недопустимые токены. 100% валидности по построению. Работает с любой локальной моделью.

В этом уроке формируется интуиция по всем трём подходам и объясняется, когда какой из них использовать.

## Концепция

![Constrained decoding masking invalid tokens at each step](../assets/constrained-decoding.svg)

**Как работает ограниченная генерация.** На каждом шаге генерации LLM выдаёт вектор логитов по всему словарю (~100 тыс. токенов). *Процессор логитов* стоит между моделью и семплером. Он вычисляет, какие токены допустимы с учётом текущей позиции в целевой грамматике — JSON Schema, регулярном выражении, контекстно-свободной грамматике — и устанавливает логиты всех недопустимых токенов равными минус бесконечности. Softmax по оставшимся логитам распределяет вероятностную массу только на допустимые продолжения.

Реализации в 2026 году:

- **Outlines.** Компилирует JSON Schema или regex в конечный автомат (FSM). Для каждого токена — проверка допустимости следующего токена за O(1). Основан на FSM, поэтому рекурсивные схемы требуют «уплощения».
- **XGrammar / llguidance.** Движки на основе контекстно-свободных грамматик. Поддерживают рекурсивные JSON Schema. Почти нулевые накладные расходы при декодировании. OpenAI упомянула llguidance в своей реализации структурированного вывода 2025 года.
- **vLLM guided decoding.** Встроенная поддержка `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar` через бэкенды Outlines, XGrammar или lm-format-enforcer.
- **Instructor.** Обёртка на базе Pydantic над любой LLM. Повторяет попытку при ошибке валидации. Кросс-провайдерная, но не изменяет логиты — полагается на повторные попытки и промпты, учитывающие структурированный вывод.

### Контринтуитивный результат

Ограниченная генерация часто *быстрее* неограниченной. Две причины. Во-первых, она сужает пространство поиска следующего токена. Во-вторых, продуманные реализации вообще пропускают генерацию токенов для принудительных токенов (шаблонная часть вроде `{"name": "` — каждый байт заранее определён).

### Ловушка, которая обходится дорого

Порядок полей имеет значение. Поставьте `answer` перед `reasoning` — и модель зафиксирует ответ прежде, чем начнёт рассуждать. JSON валиден. Ответ неверен. Ни одна валидация этого не поймает.

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Порядок полей схемы — это логика, а не форматирование.

```figure
constrained-decoder
```

## Реализация

### Шаг 1: ограниченная regex-генерация с нуля

См. `code/main.py` — отдельную реализацию FSM. Основная идея в 30 строках:

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM отслеживает, какие части грамматики уже удовлетворены. `valid_tokens(state, tokenizer)` вычисляет, какие токены словаря могут продвинуть FSM, не покидая принимающий путь.

### Шаг 2: Outlines для JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

Ноль ошибок валидации. Никогда. FSM делает недопустимый вывод недостижимым.

### Шаг 3: Instructor для провайдеронезависимого Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

Другой механизм. Instructor не трогает логиты. Он встраивает схему в промпт, парсит вывод и повторяет попытку при ошибке валидации (по умолчанию 3 раза). Работает с любым провайдером. Повторные попытки увеличивают задержку и стоимость. Кросс-провайдерная переносимость — основное преимущество.

### Шаг 4: нативные API вендоров

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

Ограниченная генерация на стороне сервера. По надёжности сопоставима с Outlines для поддерживаемых схем. Не требует управления локальной моделью. Привязывает вас к вендору.

## Подводные камни

- **Рекурсивные схемы.** Outlines «уплощает» рекурсию до фиксированной глубины. Древовидный вывод (вложенные комментарии, AST) требует XGrammar или llguidance (на основе CFG).
- **Огромные перечисления (enum).** Перечисление с 10 000 вариантов компилируется медленно или вызывает таймаут. Переключитесь на ретривер: сначала предсказывайте top-k кандидатов, затем ограничивайтесь ими.
- **Слишком строгая грамматика.** Задайте regex для `date: "YYYY-MM-DD"` — и модель не сможет вывести `"unknown"` для отсутствующих дат. Модель компенсирует это, выдумывая дату. Разрешите `null` или используйте sentinel-значение.
- **Преждевременная фиксация.** См. ловушку с порядком полей выше. Всегда ставьте рассуждение перед ответом.
- **Режим JSON вендора без схемы.** Чистый JSON-режим гарантирует только валидный JSON, но не валидность *для вашей задачи*. Всегда предоставляйте полную схему.

## Применение

Стек 2026 года:

| Ситуация | Выбор |
|-----------|------|
| Модель OpenAI/Anthropic/Google, простая схема | Нативный структурированный вывод вендора |
| Любой провайдер, рабочий процесс на Pydantic, допустимы повторные попытки | Instructor |
| Локальная модель, нужна 100% валидность, плоская схема | Outlines (FSM) |
| Локальная модель, рекурсивная схема | XGrammar или llguidance |
| Самостоятельно размещённый inference-сервер | vLLM guided decoding |
| Пакетная обработка с допустимыми повторными попытками | Instructor + самая дешёвая модель |

## Готовое решение

Сохраните как `outputs/skill-structured-output-picker.md`:

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## Упражнения

1. **Лёгкое.** Попросите небольшую модель с открытыми весами (например, Llama-3.2-3B) без ограниченной генерации выполнить `Review(sentiment, confidence, evidence_span)`. Измерьте долю результатов, парсящихся как валидный JSON, на 100 отзывах.
2. **Среднее.** Тот же корпус с JSON-режимом Outlines. Сравните долю соответствия, задержку и смысловую точность.
3. **Сложное.** Реализуйте с нуля ограниченный регулярным выражением декодер для номеров телефонов (`\d{3}-\d{3}-\d{4}`). Проверьте 0 недопустимых результатов на 1000 образцах.

## Ключевые термины

| Термин | Что обычно говорят | Что это означает на самом деле |
|------|-----------------|-----------------------|
| Ограниченная генерация (constrained decoding) | Принудительно получить валидный вывод | Маскирование логитов недопустимых токенов на каждом шаге генерации. |
| Процессор логитов | То, что ограничивает | Функция: `(logits, state) -> masked_logits`. |
| FSM | Конечный автомат | Скомпилированное представление грамматики; проверка допустимости следующего токена за O(1). |
| CFG | Контекстно-свободная грамматика | Грамматика, поддерживающая рекурсию; медленнее, но выразительнее, чем FSM. |
| Порядок полей схемы | Имеет ли значение? | Да — первое поле фиксирует результат; всегда ставьте рассуждение перед ответом. |
| Guided decoding | Так это называется в vLLM | То же понятие, встроенное в inference-сервер. |
| JSON mode | Ранняя версия от OpenAI | Гарантирует синтаксис JSON; НЕ гарантирует соответствие схеме. |

## Дополнительное чтение

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — статья об Outlines.
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — быстрая ограниченная генерация на основе CFG.
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — интеграция с inference-сервером.
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — справочник по API и подводные камни.
- [Instructor library](https://python.useinstructor.com/) — Pydantic + повторные попытки для разных провайдеров.
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — бенчмаркинг 6 фреймворков ограниченной генерации.