# Распознавание именованных сущностей

> Извлеките имена. Звучит просто, пока не столкнёшься с неоднозначными границами, вложенными сущностями и доменным жаргоном.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word Embeddings)
**Time:** ~75 минут

## Проблема

«Apple sued Google over its iPhone search deal in the US.» Пять сущностей: Apple (ORG), Google (ORG), iPhone (PRODUCT), search deal (возможно), US (GPE). Хорошая система NER извлекает их все с правильными типами. Плохая — пропускает iPhone, путает Apple-компанию с apple-фруктом и помечает «US» как PERSON.

NER — это рабочая лошадка, лежащая в основе любого пайплайна структурированного извлечения данных. Парсинг резюме, сканирование журналов комплаенса, анонимизация медицинских записей, понимание поисковых запросов, обоснование ответов чат-ботов, извлечение данных из юридических договоров. Вы почти никогда не видите эту технологию напрямую, но всегда зависите от неё.

Этот урок проходит классический путь (rule-based, HMM, CRF) к современному (BiLSTM-CRF, затем трансформеры). Каждый шаг решает конкретное ограничение предыдущего. Этот паттерн и есть суть урока.

## Концепция

**BIO-разметка** (или BILOU) превращает извлечение сущностей в задачу разметки последовательностей. Каждому токену присваивается метка `B-TYPE` (начало сущности), `I-TYPE` (внутри сущности) или `O` (вне какой-либо сущности).

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

Многотокенные сущности образуют цепочку: `New B-GPE`, `York I-GPE`, `City I-GPE`. Модель, понимающая BIO, может извлекать произвольные спаны.

Прогрессия архитектур:

- **Rule-based.** Регулярные выражения + поиск по газеттирам. Высокая точность на известных сущностях, нулевое покрытие новых.
- **HMM.** Скрытая марковская модель (Hidden Markov Model). Вероятность эмиссии токена при данном теге, вероятность перехода между тегами. Декодирование по алгоритму Витерби. Обучается на размеченных данных.
- **CRF.** Условное случайное поле (Conditional Random Field). Похоже на HMM, но дискриминативная модель, поэтому можно смешивать произвольные признаки (форма слова, заглавные буквы, соседние слова). В 2026 году по-прежнему остаётся классической рабочей лошадкой для продакшена в условиях ограниченных ресурсов.
- **BiLSTM-CRF.** Нейросетевые признаки вместо ручных. LSTM читает предложение в обоих направлениях, слой CRF сверху обеспечивает согласованность последовательности тегов.
- **На основе трансформеров.** Дообучение BERT с головой токен-классификации. Лучшая точность. Больше всего вычислений.

```figure
ner-bio-tagging
```

## Реализация

### Шаг 1: вспомогательные функции для BIO-разметки

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### Шаг 2: ручные признаки

Для классического (не нейросетевого) NER признаки решают всё. Полезные из них:

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` возвращает `xXxxxx`. `word_shape("USA-2024")` возвращает `XXX-dddd`. Паттерны использования заглавных букв — сильный сигнал для имён собственных.

### Шаг 3: простой baseline на правилах + словаре

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

Продакшен-газеттиры содержат миллионы записей, собранных из Wikipedia и DBpedia. Покрытие хорошее. Разрешение неоднозначности (`Apple` компания или фрукт) — ужасное. Именно поэтому статистические модели победили.

### Шаг 4: этап CRF (набросок, не полная реализация)

Полная реализация CRF с нуля на 50 строках не даст понимания без базовых основ теории вероятностей. Используйте вместо этого `sklearn-crfsuite`:

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` и `c2` — это регуляризация L1 и L2. `all_possible_transitions=True` позволяет модели выучить, что недопустимые последовательности (например, `I-ORG` после `O`) маловероятны — именно так CRF обеспечивает согласованность BIO без явного написания ограничений.

### Шаг 5: что добавляет BiLSTM-CRF

Признаки становятся выученными. Входы: эмбеддинги токенов (GloVe или fastText). LSTM читает слева направо и справа налево. Конкатенированные скрытые состояния проходят через выходной слой CRF. CRF по-прежнему обеспечивает согласованность последовательности тегов; LSTM заменяет ручные признаки на выученные.

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

Для слоя CRF используйте `torchcrf.CRF` (pip install pytorch-crf). Прирост качества по сравнению с ручным CRF заметен, но меньше, чем можно ожидать, если у вас нет десятков тысяч размеченных предложений.

## Применение

spaCy предоставляет готовый к продакшену NER из коробки.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

Обратите внимание, что `iPhone` помечено как `ORG`, а не `PRODUCT` — у маленькой модели spaCy слабое покрытие для сущностей-продуктов. Большая модель (`en_core_web_lg`) справляется лучше. Модель на основе трансформера (`en_core_web_trf`) справляется ещё лучше.

Hugging Face для NER на основе BERT:

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` объединяет смежные токены B-X, I-X в один спан. Без этого вы получаете метки на уровне токенов и вам придётся объединять их самостоятельно.

### NER на основе LLM (вариант 2026 года)

Zero-shot и few-shot NER на LLM сейчас конкурентоспособны с дообученными моделями во многих доменах, а при дефиците размеченных данных — значительно лучше.

- **Zero-shot промптинг.** Дайте LLM список типов сущностей и пример схемы. Попросите вывод в JSON. Работает из коробки; точность умеренная на новых доменах.
- **Промптинг в стиле ZeroTuneBio.** Разбейте задачу на этапы: извлечение кандидатов → объяснение смысла → суждение → перепроверка. Многоэтапный промпт (а не одношаговый) существенно повышает точность на биомедицинском NER. Та же схема работает для юридического, финансового и научного доменов.
- **Динамический промптинг с RAG.** Извлекайте наиболее похожие размеченные примеры из небольшого аннотированного seed-набора для каждого вызова инференса; собирайте few-shot промпт на лету. По бенчмаркам 2026 года это повышает F1 биомедицинского NER у GPT-4 на 11-12% по сравнению со статическим промптингом.
- **Декомпозиция по типам сущностей.** Для длинных документов один вызов, извлекающий все типы сущностей сразу, теряет полноту (recall) по мере роста длины. Запускайте отдельный проход извлечения для каждого типа сущности. Выше стоимость инференса, существенно выше точность. Это стандартный паттерн для клинических заметок и юридических договоров.

Рекомендация для продакшена по состоянию на 2026 год: начинайте с zero-shot baseline на LLM, прежде чем собирать обучающие данные. Часто F1 оказывается достаточно хорошим, чтобы дообучение вообще не понадобилось.

### Где классический NER по-прежнему выигрывает

Даже при наличии LLM классический NER выигрывает, когда:

- Бюджет задержки меньше 50 мс.
- У вас есть тысячи размеченных примеров и нужен F1 98%+.
- Домен имеет стабильную онтологию, на которой хорошо переносится предобученный CRF или BiLSTM.
- Регуляторные ограничения требуют локальную (on-prem), не генеративную модель.

### Где всё разваливается

- **Сдвиг домена.** NER, обученный на CoNLL, показывает на юридических договорах результат хуже, чем газеттир. Дообучайте на своём домене.
- **Вложенные сущности.** «Bank of America Tower» одновременно является ORG и FACILITY. Стандартная BIO не может представить перекрывающиеся спаны. Нужен вложенный NER (многопроходные или span-based модели).
- **Длинные сущности.** «United States Federal Deposit Insurance Corporation.» Токен-уровневые модели иногда разбивают это на части. Используйте `aggregation_strategy` или постобработку.
- **Разреженные типы.** Медицинские метки NER вроде DRUG_BRAND, ADVERSE_EVENT, DOSE. Универсальные модели понятия не имеют о них. Scispacy и BioBERT — отправная точка здесь.

## Публикация

Сохраните как `outputs/skill-ner-picker.md`:

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## Упражнения

1. **Лёгкое.** Реализуйте `bio_to_spans` (обратное к `spans_to_bio`) и проверьте согласованность на циклическом преобразовании на 10 предложениях.
2. **Среднее.** Обучите приведённый выше CRF из sklearn-crfsuite на датасете CoNLL-2003 English NER. Приведите F1 по каждой сущности с помощью `seqeval`. Типичный результат: ~84 F1.
3. **Сложное.** Дообучите `distilbert-base-cased` на доменно-специфичном датасете NER (медицинском, юридическом или финансовом). Сравните с маленькой моделью spaCy. Документируйте проверки на утечку данных и опишите, что вас удивило.

## Ключевые термины

| Термин | Что говорят | Что это на самом деле значит |
|------|-----------------|-----------------------|
| NER | Извлечение имён | Разметка спанов токенов типами (PERSON, ORG, GPE, DATE, ...). |
| BIO | Схема разметки | `B-X` — начало, `I-X` — продолжение, `O` — вне сущности. |
| BILOU | Улучшенная BIO | Добавляет `L-X` (последний), `U-X` (единичный) для более чётких границ. |
| CRF | Структурированный классификатор | Моделирует переходы между метками, а не только эмиссии. Обеспечивает валидные последовательности. |
| Вложенный NER | Перекрывающиеся сущности | Один спан — это другая сущность, чем его под-спан. BIO не может это выразить. |
| F1 на уровне сущностей | Правильная метрика NER | Предсказанный спан должен точно совпадать с истинным. F1 на уровне токенов завышает точность. |

## Дополнительное чтение

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — статья про BiLSTM-CRF. Каноническая.
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — вводит паттерн токен-классификации, ставший стандартом.
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — практический справочник по каждому атрибуту `Doc.ents` и `Span`.
- [seqeval](https://github.com/chakki-works/seqeval) — корректная библиотека метрик. Используйте всегда её.