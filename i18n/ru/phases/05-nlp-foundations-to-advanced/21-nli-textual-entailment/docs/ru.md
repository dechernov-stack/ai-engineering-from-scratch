# Natural Language Inference — текстовое следование

> «t влечёт h» означает, что человек, прочитавший t, заключит, что h истинно. NLI — это задача предсказания отношения: следование / противоречие / нейтральность. На поверхности скучно, в продакшене — критично важно.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment Analysis), Phase 5 · 13 (Question Answering)
**Time:** ~60 минут

## Проблема

Вы построили суммаризатор. Он выдал резюме. Как узнать, что резюме не содержит галлюцинаций?

Вы построили чат-бота. Он ответил «да». Как узнать, подтверждается ли ответ найденным фрагментом текста?

Вам нужно классифицировать 10 000 новостных статей по темам. У вас нет размеченных обучающих данных. Можно ли переиспользовать модель?

Все три задачи сводятся к Natural Language Inference. NLI задаёт вопрос: дана посылка (premise) `t` и гипотеза (hypothesis) `h` — следует ли `h` из `t`, противоречит ли ей, или они нейтральны (не связаны)?

- **Проверка галлюцинаций:** `t` = исходный документ, `h` = утверждение из резюме. Не следование = галлюцинация.
- **QA с опорой на источник:** `t` = найденный фрагмент, `h` = сгенерированный ответ. Не следование = выдумка.
- **Zero-shot классификация:** `t` = документ, `h` = вербализованная метка («Это о спорте»). Следование = предсказанная метка.

Одна задача, три применения в продакшене. Именно поэтому каждый фреймворк для оценки RAG под капотом использует модель NLI.

## Концепция

![NLI: three-way classification, premise vs hypothesis](../assets/nli.svg)

**Три метки.**

- **Следование (entailment).** `t` → `h`. «Кот на коврике» влечёт «Есть кот».
- **Противоречие (contradiction).** `t` → ¬`h`. «Кот на коврике» противоречит «Кота нет».
- **Нейтрально.** Никакого вывода в ту или иную сторону. «Кот на коврике» нейтрально по отношению к «Кот голоден».

**Не логическое следование.** NLI — это *естественно*-языковой вывод: то, что обычный читатель-человек вывел бы, а не строгая логика. «Джон выгулял свою собаку» влечёт «У Джона есть собака» в терминах NLI, но строгая логика первого порядка допустила бы это только при аксиоматизации владения.

**Датасеты.**

- **SNLI** (2015). 570 тыс. размеченных людьми пар, подписи к изображениям в качестве посылок. Узкая предметная область.
- **MultiNLI** (2017). 433 тыс. пар из 10 жанров. Стандартный обучающий корпус в 2026 году.
- **ANLI** (2019). Adversarial NLI. Люди писали примеры, специально рассчитанные на то, чтобы сломать существующие модели. Сложнее.
- **DocNLI, ConTRoL** (2020–21). Посылки длиной с документ. Проверяют многошаговый и дальний вывод.

**Архитектура.** Трансформер-энкодер (BERT, RoBERTa, DeBERTa) читает `[CLS] premise [SEP] hypothesis [SEP]`. Представление `[CLS]` подаётся в 3-классовый softmax. Обучение на MNLI, оценка на отложенных бенчмарках даёт точность 90%+ на парах из того же распределения.

**Zero-shot через NLI.** Дан документ и кандидатные метки — превратите каждую метку в гипотезу («Этот текст о спорте»). Вычислите вероятность следования для каждой. Выберите максимум. Это механизм, лежащий в основе пайплайна `zero-shot-classification` от Hugging Face.

```figure
nli-router
```

## Реализация

### Шаг 1: запустить предобученную модель NLI

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

Для продакшен-NLI `facebook/bart-large-mnli` и `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` — открытые модели по умолчанию. DeBERTa-v3 возглавляет лидерборды.

### Шаг 2: zero-shot классификация

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

По умолчанию шаблон — «This example is about {label}.» («Этот пример о {label}»). Настраивается через `hypothesis_template`. Обучающие данные не требуются. Дообучение не требуется. Работает «из коробки».

### Шаг 3: проверка достоверности для RAG

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

Это ядро проверки faithfulness в RAGAS. Разбейте сгенерированный ответ на атомарные утверждения. Проверьте каждое утверждение относительно найденного контекста. Выведите долю тех, что следуют из контекста.

### Шаг 4: самодельный классификатор NLI (концептуально)

См. `code/main.py` — игрушечный пример только на stdlib: посылка и гипотеза сравниваются через лексическое пересечение + обнаружение отрицаний. Не конкурирует с трансформерными моделями — но показывает форму задачи: два текста на входе, 3-классовая метка на выходе, функция потерь = кросс-энтропия по `{entail, contradict, neutral}`.

## Подводные камни

- **Трюки по одной гипотезе.** Модели могут предсказывать метку только по гипотезе с точностью ~60% на SNLI, потому что «not», «nobody», «never» коррелируют с противоречием. Сильный базовый тест для обнаружения утечки меток.
- **Эвристика лексического пересечения.** Эвристика подпоследовательности («любая подпоследовательность влечётся») проходит SNLI, но проваливается на HANS/ANLI. Используйте состязательные бенчмарки.
- **Деградация на длинных документах.** Модели NLI для отдельных предложений теряют 20+ пунктов F1 на посылках длиной с документ. Используйте модели, обученные на DocNLI, для длинного контекста.
- **Чувствительность к шаблону в zero-shot.** «This example is about {label}» против «{label}» против «The topic is {label}» может менять точность на 10+ пунктов. Настраивайте шаблон.
- **Несоответствие домена.** MNLI обучается на общеязыковом английском. Юридические, медицинские и научные тексты требуют доменно-специфичных моделей NLI (например, SciNLI, MedNLI).

## Применение

Стек 2026 года:

| Сценарий использования | Модель |
|---------|-------|
| Универсальный NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| Быстрый / edge | `cross-encoder/nli-deberta-v3-base` |
| Zero-shot классификация (облегчённая) | `facebook/bart-large-mnli` |
| NLI на уровне документа | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| Многоязычный | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| Обнаружение галлюцинаций в RAG | слой NLI внутри RAGAS / DeepEval |

Мета-паттерн 2026 года: NLI — это универсальный скотч для понимания текста. Всякий раз, когда нужно «поддерживает ли A утверждение B?» или «противоречит ли A утверждению B?» — тянитесь к NLI прежде, чем делать ещё один вызов LLM.

## Поставка

Сохраните как `outputs/skill-nli-picker.md`:

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## Упражнения

1. **Лёгкое.** Запустите `facebook/bart-large-mnli` на 20 вручную составленных тройках (посылка, гипотеза, метка), охватывающих все три класса. Измерьте точность. Добавьте состязательные ловушки на «эвристику подпоследовательности» («Я не ел торт» против «Я ел торт») и проверьте, ломается ли модель.
2. **Среднее.** Сравните zero-shot шаблон `"This text is about {label}"` с `"The topic is {label}"` и `"{label}"` на 100 заголовках AG News. Сообщите разброс точности.
3. **Сложное.** Постройте проверку достоверности RAG: декомпозиция на атомарные утверждения + NLI для каждого утверждения. Оцените на 50 сгенерированных RAG-ответах с эталонным контекстом. Измерьте долю ложноположительных и ложноотрицательных срабатываний относительно ручной разметки.

## Ключевые термины

| Термин | Что говорят | Что это на самом деле означает |
|------|-----------------|-----------------------|
| NLI | Natural Language Inference | 3-классовая классификация отношения посылка-гипотеза. |
| RTE | Recognizing Textual Entailment | Более старое название для NLI; та же задача. |
| Следование (Entailment) | «t подразумевает h» | Типичный читатель заключил бы, что h истинно, учитывая t. |
| Противоречие (Contradiction) | «t исключает h» | Типичный читатель заключил бы, что h ложно, учитывая t. |
| Нейтрально (Neutral) | «неопределённо» | Никакого вывода от t к h в ту или иную сторону. |
| Zero-shot классификация | NLI как классификатор | Вербализация меток в виде гипотез, выбор максимального следования. |
| Достоверность (Faithfulness) | Подтверждён ли ответ? | NLI над парой (найденный контекст, сгенерированный ответ). |

## Дополнительное чтение

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI.
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI.
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — бенчмарк ANLI.
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI как классификатор.
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — рабочая лошадка NLI в 2026 году.