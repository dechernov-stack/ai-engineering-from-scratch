# Оценка LLM — RAGAS, DeepEval, G-Eval

> Точное совпадение и F1 не улавливают семантическую эквивалентность. Человеческая проверка не масштабируется. LLM-судья — промышленное решение, при достаточной калибровке, чтобы доверять цифре.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75 минут

## Проблема

Ваша RAG-система отвечает: "June 29th, 2007."
Эталонный ответ: "June 29, 2007."
Exact Match даёт 0. F1 даёт ~75%. Человек поставил бы 100%.

Теперь умножьте это на 10 000 тестовых случаев. Умножьте снова на каждое изменение ретривера, чанкинга, промпта или модели. Нужен оценщик, который понимает смысл, работает дёшево в масштабе, не врёт о регрессиях и выявляет нужные типы сбоев.

В 2026 году есть три фреймворка, владеющих этой проблемой.

- **RAGAS.** Retrieval-Augmented Generation ASsessment. Четыре метрики RAG (faithfulness, answer-relevance, context-precision, context-recall) с бэкендами NLI + LLM-судья. Подкреплён исследованиями, лёгкий.
- **DeepEval.** Pytest для LLM. Метрики G-Eval, task-completion, hallucination, bias. CI/CD-ориентированный.
- **G-Eval.** Метод (и метрика DeepEval): LLM-судья с цепочкой рассуждений (chain-of-thought), пользовательскими критериями, оценкой 0-1.

Все три опираются на LLM-as-judge. Этот урок формирует интуицию для метода и слоя доверия вокруг него.

## Концепция

![Four evaluation dimensions, LLM-as-judge architecture](../assets/llm-evaluation.svg)

**LLM-as-judge.** Заменить статическую метрику на LLM, которая оценивает выходные данные по заданному рубрику. Имея `(query, context, answer)`, промптируем LLM-судью: "Оцени от 0 до 1 по faithfulness." Получаем оценку.

Почему это работает: LLM аппроксимируют человеческую оценку за крошечную долю стоимости. GPT-4o-mini примерно $0.003 per scored case enables 1000-sample regression eval runs for under $5.

Почему это ломается незаметно:

1. **Смещение судьи (judge bias).** Судьи предпочитают более длинные ответы, ответы от моделей той же семьи, ответы, соответствующие стилю промпта.
2. **Ошибки парсинга JSON.** Плохой JSON → оценка NaN → тихо исключается из агрегата. Пользователи RAGAS это знают на собственном опыте. Защищайтесь через try/except с явным режимом отказа.
3. **Дрейф по версиям моделей.** Обновление судьи меняет каждую метрику. Фиксируйте модель судьи + версию.

**Четвёрка RAG.**

| Метрика | Вопрос | Бэкенд |
|--------|----------|---------|
| Faithfulness | Исходит ли каждое утверждение в ответе из извлечённого контекста? | NLI-основанное следование (entailment) |
| Answer relevance | Отвечает ли ответ на вопрос? | Генерация гипотетических вопросов из ответа; сравнение с реальным вопросом |
| Context precision | Какая доля извлечённых чанков была релевантна? | LLM-судья |
| Context recall | Вернул ли ретривер всё необходимое? | LLM-судья относительно золотого ответа |

**G-Eval.** Определите пользовательский критерий: "Указал ли ответ правильный источник?" Фреймворк автоматически разворачивает его в шаги оценки по цепочке рассуждений, затем выставляет оценку 0-1. Хорошо подходит для доменно-специфичных измерений качества, которые не покрывает RAGAS.

**Калибровка.** Никогда не доверяйте сырой оценке судьи, пока не проверите корреляцию с человеческими метками. Проведите 100 размеченных вручную примеров. Постройте график судья против человека. Вычислите коэффициент корреляции Спирмена. Если rho < 0.7, рубрику судьи нужно дорабатывать.

```figure
n5-judge-gauge
```

## Построение

### Шаг 1: faithfulness с NLI (в стиле RAGAS)

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

Разложите ответ на атомарные утверждения. Проверьте каждое утверждение через NLI против извлечённого контекста. Faithfulness = доля подтверждённых утверждений.

### Шаг 2: answer relevance

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

Если ответ подразумевает вопросы, отличные от заданного, релевантность падает.

### Шаг 3: пользовательская метрика G-Eval

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

Шаги оценки и есть рубрика. Явные шаги более стабильны, чем неявные промпты "оцени от 0 до 1".

### Шаг 4: CI-шлюз

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

Оформите как файл pytest. Запускайте на каждый PR. Блокируйте слияния при регрессиях.

### Шаг 5: игрушечная оценка с нуля

См. `code/main.py`. Аппроксимации на stdlib для faithfulness (пересечение утверждений ответа с контекстом) и relevance (пересечение токенов ответа с токенами вопроса). Не для продакшена. Показывает общую форму.

## Подводные камни

- **Отсутствие калибровки.** Судья с корреляцией 0.3 с человеческими метками — это шум. Требуйте калибровочный прогон перед выпуском.
- **Самооценка.** Использование одной и той же LLM для генерации и оценки завышает оценки на 10-20%. Используйте для судьи другую модельную семью.
- **Позиционное смещение в парном судействе.** Судьи предпочитают первый представленный вариант. Всегда рандомизируйте порядок и прогоняйте оба варианта.
- **Сырой агрегат скрывает сбои.** Средняя оценка 0.85 часто скрывает 5% катастрофических сбоев. Всегда проверяйте нижний квантиль.
- **Деградация золотого датасета.** Неверсионированные наборы для оценки, дрейфующие со временем, ломают продольное (лонгитюдное) сравнение. Привязывайте тег датасета к каждому изменению.
- **Стоимость LLM.** В масштабе вызовы судьи доминируют в стоимости. Используйте самую дешёвую модель, удовлетворяющую порогу калибровки. GPT-4o-mini, Claude Haiku, Mistral-small.

## Применение

Стек 2026 года:

| Сценарий использования | Фреймворк |
|---------|-----------|
| Мониторинг качества RAG | RAGAS (4 метрики) |
| Шлюзы регрессии CI/CD | DeepEval + pytest |
| Пользовательские доменные критерии | G-Eval внутри DeepEval |
| Онлайн-мониторинг живого трафика | RAGAS в режиме без эталона (reference-free) |
| Выборочные проверки с участием человека | LangSmith или Phoenix с UI для аннотации |
| Red-teaming / оценка безопасности | Promptfoo + DeepEval |

Типичный стек: RAGAS для мониторинга, DeepEval для CI, G-Eval для новых измерений. Запускайте все три; их расхождения полезны.

## Поставка

Сохраните как `outputs/skill-eval-architect.md`:

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## Упражнения

1. **Лёгкое.** Примените RAGAS к 10 примерам RAG с известными галлюцинациями. Убедитесь, что метрика faithfulness улавливает каждую из них.
2. **Среднее.** Вручную разметьте 50 ответов QA от 0 до 1 по правильности. Оцените с помощью G-Eval. Измерьте коэффициент корреляции Спирмена между судьёй и человеком.
3. **Сложное.** Постройте CI-шлюз pytest с DeepEval. Намеренно ухудшите ретривер (регрессия). Убедитесь, что шлюз срабатывает. Добавьте оповещение по нижнему квантилю через проверку порога на нижних 10%.

## Ключевые термины

| Термин | Что говорят люди | Что это на самом деле означает |
|------|-----------------|-----------------------|
| LLM-as-judge | Оценка с помощью LLM | Промптирование модели-судьи для оценки выходных данных от 0 до 1 по заданной рубрике. |
| RAGAS | Библиотека метрик RAG | Фреймворк оценки с открытым исходным кодом с 4 метриками RAG без эталона. |
| Faithfulness | Обоснован ли ответ? | Доля утверждений ответа, подтверждённых извлечённым контекстом. |
| Context precision | Были ли извлечённые чанки релевантны? | Доля топ-K чанков, которые действительно имели значение. |
| Context recall | Нашёл ли ретривер всё? | Доля утверждений золотого ответа, подтверждённых извлечёнными чанками. |
| G-Eval | Пользовательский LLM-судья | Рубрика + шаги оценки по цепочке рассуждений + оценка 0-1. |
| Calibration | Доверяй, но проверяй | Корреляция Спирмена между оценкой судьи и оценкой человека. |

## Дополнительное чтение

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — статья RAGAS.
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — статья G-Eval.
- [Документация DeepEval](https://deepeval.com/docs/metrics-introduction) — открытый промышленный стек.
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — смещения, калибровка, ограничения.
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — объединяющий фреймворк, интегрирующий RAGAS, DeepEval, Phoenix.