# Длинный контекст: оценка — NIAH, RULER, LongBench, MRCR

> Gemini 3 Pro заявляет контекст в 10 млн токенов. На 1 млн токенов 8-needle MRCR падает до 26,3%. Заявленное ≠ пригодное к использованию. Оценка длинного контекста показывает реальную ёмкость модели, которую вы поставляете в продакшн.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 23 (Chunking Strategies)
**Time:** ~60 минут

## Проблема

У вас есть контракт на 200 страниц. Модель заявляет контекст в 1 млн токенов. Вы вставляете контракт целиком и спрашиваете: «Какое условие о расторжении договора?» Модель отвечает — но отвечает, опираясь на титульную страницу, потому что пункт о расторжении находится на глубине 120 тыс. токенов, за пределами того, куда модель на самом деле «смотрит».

Это и есть разрыв в ёмкости контекста 2026 года. В спецификациях указано 1 млн или 10 млн. Реальность же такова, что пригодны к использованию лишь 60–70% от этого объёма, и «пригодность» зависит от задачи.

- **Поиск (single needle in haystack):** почти идеальный результат вплоть до заявленного максимума на передовых моделях.
- **Многошаговый поиск / агрегация:** резко деградирует после ~128k токенов на большинстве моделей.
- **Рассуждение по разрозненным фактам:** первая задача, которая проваливается.

Оценка длинного контекста измеряет именно эти оси. В этом уроке названы бенчмарки, то, что каждый из них на самом деле измеряет, и как построить кастомный needle-тест для вашей предметной области.

## Концепция

![NIAH baseline, RULER multi-task, LongBench holistic](../assets/long-context-eval.svg)

**Needle-in-a-Haystack (NIAH, 2023).** Поместите факт («волшебное слово — ананас») на контролируемую глубину в длинном контексте. Попросите модель найти его. Варьируйте глубину × длину. Оригинальный бенчмарк длинного контекста. Передовые модели сейчас достигают насыщения на нём; это необходимый, но недостаточный базовый тест.

**RULER (Nvidia, 2024).** 13 типов задач в 4 категориях: поиск (single / multi-key / multi-value), многошаговая трассировка (отслеживание переменных), агрегация (частота общих слов), QA. Настраиваемая длина контекста (от 4k до 128k+). Выявляет модели, которые достигают насыщения на NIAH, но проваливаются на многошаговых задачах. В релизе 2024 года только половина из 17 моделей, заявлявших контекст 32k+, сохраняла качество на 32k.

**LongBench v2 (2024).** 503 вопроса с множественным выбором, контексты от 8k до 2M слов, шесть категорий задач: QA по одному документу, QA по нескольким документам, длинное обучение в контексте, длинный диалог, репозиторий кода, длинные структурированные данные. Продакшн-бенчмарк для реального поведения при работе с длинным контекстом.

**MRCR (Multi-Round Coreference Resolution).** Многоходовое разрешение кореференции в большом масштабе. Варианты с 8, 24 и 100 needle. Показывает, сколько фактов модель способна удерживать одновременно, прежде чем внимание начнёт деградировать.

**NoLiMa.** «Нелексическая needle». Needle и запрос не имеют буквального пересечения; поиск требует одного шага семантического рассуждения. Сложнее, чем NIAH.

**HELMET.** Объединяет множество документов, задаёт вопрос по любому из них. Проверяет избирательное внимание.

**BABILong.** Встраивает цепочки рассуждений bAbI внутрь нерелевантного «сена». Проверяет рассуждение внутри «стога сена», а не только поиск.

### Что на самом деле нужно указывать в отчёте

- **Заявленное окно контекста.** Число из спецификации.
- **Эффективная длина поиска.** Прохождение NIAH при определённом пороге (например, 90%).
- **Эффективная длина рассуждения.** Прохождение многошаговой задачи или агрегации при том же пороге.
- **Кривая деградации.** Точность в зависимости от длины контекста, построенная для каждого типа задачи.

Два числа для вашей спецификации: эффективность поиска и эффективность рассуждения. Как правило, эффективность рассуждения составляет 25–50% от заявленного окна.

```figure
gx-niah-decay
```

## Реализация

### Шаг 1: кастомный NIAH для вашей предметной области

См. `code/main.py`. Каркас:

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

Переберите `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}. Постройте тепловую карту. Это и есть карта NIAH для вашей целевой модели.

### Шаг 2: вариант с несколькими needle

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

Вопросы вроде «Какие три волшебных слова?» требуют поиска всех трёх. Успех на одной needle не предсказывает успех на нескольких.

### Шаг 3: многошаговая трассировка переменных (в стиле RULER)

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

Ответ требует цепочки из трёх присваиваний. Передовые модели на 128k часто падают здесь до 50–70% точности.

### Шаг 4: LongBench v2 на вашем стеке

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

Указывайте точность по каждой категории отдельно. Агрегированные оценки скрывают большие различия на уровне отдельных задач.

## Подводные камни

- **Оценка только по NIAH.** Прохождение NIAH на 1 млн токенов ничего не говорит о многошаговых задачах. Всегда запускайте RULER или кастомный многошаговый тест.
- **Равномерная выборка по глубине.** Во многих реализациях тестируется только depth=0.5. Тестируйте depth=0, 0.25, 0.5, 0.75, 1.0 — эффект «потери в середине» реален.
- **Лексическое пересечение с наполнителем.** Если needle содержит те же ключевые слова, что и наполнитель, поиск становится тривиальным. Используйте needle без пересечения, в стиле NoLiMa.
- **Игнорирование задержки.** Промпты на 1 млн токенов обрабатываются 30–120 секунд на этапе prefill. Измеряйте время до первого токена наряду с точностью.
- **Цифры, заявленные вендорами.** OpenAI, Google, Anthropic — все публикуют собственные оценки. Всегда перепроверяйте независимо на своей задаче.

## Применение

Стек 2026 года:

| Ситуация | Бенчмарк |
|-----------|-----------|
| Быстрая проверка на адекватность | Кастомный NIAH на 3 глубинах × 3 длинах |
| Выбор модели для продакшна | RULER (13 задач) на целевой длине |
| Качество QA в реальных условиях | Подмножество single-doc-QA из LongBench v2 |
| Многошаговое рассуждение | BABILong или кастомная трассировка переменных |
| Диалог / разговорный режим | MRCR 8-needle на целевой длине |
| Регрессия при обновлении модели | Фиксированный внутренний набор тестов NIAH + RULER, запускаемый на каждой новой модели |

Эмпирическое правило для продакшна: никогда не доверяйте заявленному окну контекста, пока не прогнали NIAH + хотя бы одну задачу на рассуждение на предполагаемой длине.

## Готовим к продакшну

Сохраните как `outputs/skill-long-context-eval.md`:

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## Упражнения

1. **Лёгкое.** Постройте NIAH с 3 глубинами (0.25, 0.5, 0.75) × 3 длинами (1k, 4k, 16k). Запустите на любой модели. Постройте тепловую карту прохождения 3×3.
2. **Среднее.** Добавьте вариант с 3 needle. Измерьте поиск всех трёх на каждой длине. Сравните с показателем прохождения для одной needle на той же длине.
3. **Сложное.** Постройте задачу трассировки переменных (X1 → X2 → X3, 3 шага), встроенную в 64k наполнителя. Измерьте точность на 3 передовых моделях. Укажите эффективную длину рассуждения для каждой модели.

## Ключевые термины

| Термин | Что говорят | Что это на самом деле означает |
|------|-----------------|-----------------------|
| NIAH | Needle in haystack (игла в стоге сена) | Поместить факт в наполнитель и попросить модель найти его. |
| RULER | NIAH на стероидах | 13 типов задач в категориях поиска / многошаговости / агрегации / QA. |
| Эффективный контекст | Реальная ёмкость | Длина, при которой точность всё ещё держится выше порога. |
| Lost in the middle (потеря в середине) | Смещение по глубине | Модели недостаточно «внимательны» к содержимому в середине длинных входных данных. |
| Multi-needle (несколько needle) | Много фактов одновременно | Несколько вставок; проверяет жонглирование вниманием, а не только поиск. |
| MRCR | Многоходовая кореференция | Разрешение кореференции с 8, 24 или 100 needle; выявляет насыщение внимания. |
| NoLiMa | Нелексическая needle | Needle и запрос не имеют общих буквальных токенов; требует рассуждения. |

## Дополнительные материалы

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — оригинальный репозиторий NIAH.
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — многозадачный бенчмарк.
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — оценка длинного контекста в реальных условиях.
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — более сложные needle.
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — рассуждение внутри «стога сена».
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — статья о смещении по глубине.