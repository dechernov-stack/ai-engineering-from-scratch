# POS-разметка и синтаксический парсинг

> Грамматика какое-то время была не в моде. Потом каждому LLM-пайплайну понадобилась валидация структурированного извлечения — и она вернулась.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 минут

## Проблема

Урок 01 обещал, что для лемматизации нужен тег части речи. Не зная, что `running` — глагол, лемматизатор не может свести его к `run`. Не зная, что `better` — прилагательное, он не может свести его к `good`.

За этим обещанием скрывалась целая подобласть. POS-разметка (part-of-speech tagging) присваивает грамматические категории. Синтаксический парсинг восстанавливает древовидную структуру предложения: какое слово что модифицирует, какой глагол какими аргументами управляет. Классическая NLP потратила двадцать лет на доработку обеих задач. Затем deep learning свёл их к задаче классификации токенов поверх предобученного трансформера, и исследовательское сообщество двинулось дальше.

Но не прикладное сообщество. Каждый пайплайн структурированного извлечения по-прежнему использует POS-теги и деревья зависимостей под капотом. JSON, сгенерированный LLM, валидируется по грамматическим ограничениям. Системы question-answering раскладывают запросы, используя деревья зависимостей. Оценщики качества машинного перевода проверяют выравнивание деревьев парсинга.

Это стоит знать. Этот урок знакомит с наборами тегов, базовыми подходами и точкой, в которой вы перестаёте реализовывать всё с нуля и обращаетесь к spaCy.

## Концепция

**POS-разметка** помечает каждый токен грамматической категорией. Набор тегов **Penn Treebank (PTB)** — стандарт по умолчанию для английского языка. 36 тегов с различиями, которые случайному читателю покажутся излишне дотошными: `NN` — существительное в единственном числе, `NNS` — существительное во множественном числе, `NNP` — собственное существительное в единственном числе, `VBD` — глагол в прошедшем времени, `VBZ` — глагол в 3-м лице единственного числа настоящего времени, и так далее. Набор тегов **Universal Dependencies (UD)** более грубый (17 тегов) и языконезависимый; он стал стандартом для кросс-языковой работы.

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**Синтаксический парсинг** строит дерево. Два основных стиля:

- **Парсинг составляющих (constituency parsing).** Именные группы, глагольные группы, предложные группы вложены друг в друга. На выходе — дерево нетерминальных категорий (NP, VP, PP) со словами в качестве листьев.
- **Парсинг зависимостей (dependency parsing).** У каждого слова есть одно главное слово, от которого оно зависит, с меткой грамматического отношения. На выходе — дерево, где каждое ребро представляет собой тройку (главное слово, зависимое слово, отношение).

В 2010-х годах парсинг зависимостей победил, потому что он чисто обобщается на разные языки, особенно на языки со свободным порядком слов.

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

```figure
pos-tagger
```

```figure
dependency-arcs
```

## Реализация

### Шаг 1: базовый метод — наиболее частый тег

Самый простой работающий POS-теггер. Для каждого слова предсказывается тег, который чаще всего встречался у него на обучающих данных.

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

На корпусе Brown этот базовый метод достигает точности ~85%. Не очень хорошо, но это порог, ниже которого не должна опускаться ни одна серьёзная модель.

### Шаг 2: биграммный HMM-теггер

Моделируем совместную вероятность последовательности:

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

Две таблицы: вероятности переходов (тег при условии предыдущего тега), вероятности эмиссии (слово при условии тега). Обе оцениваются по частотам со сглаживанием Лапласа. Декодирование — алгоритмом Витерби (динамическое программирование по решётке тегов).

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Биграммный HMM на Brown достигает точности ~93%. Скачок с 85% до 93% в основном обеспечивают вероятности переходов — модель учится, что `DET NOUN` встречается часто, а `NOUN DET` — редко.

### Шаг 3: почему современные теггеры превосходят этот подход

Вероятности переходов и эмиссии локальны. Они не могут уловить, что `saw` — существительное в "I bought a saw", но глагол в "I saw the movie". CRF с произвольными признаками (суффикс, форма слова, предыдущее и следующее слово, само слово) достигает ~97%. BiLSTM-CRF или трансформер — ~98%+.

Потолок этой задачи задаётся расхождением между аннотаторами. Люди-аннотаторы согласны друг с другом примерно в 97% случаев на Penn Treebank. Модели с точностью выше 98%, вероятно, переобучаются под тестовый набор.

### Шаг 4: набросок парсинга зависимостей

Полный парсинг зависимостей с нуля выходит за рамки этого урока; каноническое изложение в учебнике — у Джурафски и Мартина. Стоит знать два классических семейства:

- **Трансишн-парсеры (transition-based)** (arc-eager, arc-standard) работают как shift-reduce парсер: они читают токены, помещают их в стек и применяют reduce-действия, создающие дуги. Жадное декодирование быстрое. Классическая реализация — MaltParser. Современная нейросетевая версия — трансишн-парсер Чена и Мэннинга.
- **Графовые парсеры (graph-based)** (алгоритм Айснера, биаффинный парсер Дозата-Мэннинга) оценивают каждое возможное ребро "главное слово — зависимое слово" и выбирают максимальное остовное дерево. Медленнее, но точнее.

Для большинства прикладных задач достаточно обратиться к spaCy:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

Читайте столбец `dep` снизу вверх — и грамматическая структура предложения становится видна.

## Применение

Каждая промышленная NLP-библиотека включает POS-теггеры и парсеры зависимостей как часть стандартного пайплайна.

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`). Быстрый, точный, интегрирован с токенизацией + NER + лемматизацией. `token.tag_` (Penn), `token.pos_` (UD), `token.dep_` (отношение зависимости).
- **Stanford NLP (stanza)**. Преемник Stanford CoreNLP. Лучшие результаты на 60+ языках.
- **trankit**. На основе трансформеров, хорошая точность по UD.
- **NLTK**. `pos_tag`. Пригоден к использованию, медленный, устаревший. Подходит для обучения.

### Где это всё ещё важно в 2026 году

- **Лемматизация.** Уроку 01 нужны POS-теги для корректной лемматизации. Всегда.
- **Структурированное извлечение из выходов LLM.** Проверка того, что сгенерированное предложение соблюдает грамматические ограничения (например, согласование подлежащего и сказуемого, обязательные модификаторы).
- **Аспектно-ориентированный анализ тональности.** Деревья зависимостей показывают, какое прилагательное модифицирует какое существительное.
- **Понимание запросов.** "фильмы в постановке Уэса Андерсона с Биллом Мюрреем в главной роли" раскладывается на структурированные ограничения через дерево парсинга.
- **Кросс-языковой перенос.** UD-теги и отношения зависимостей языконезависимы, что позволяет проводить структурный анализ новых языков в режиме zero-shot.
- **Низкоресурсные пайплайны.** Если нельзя развернуть трансформер, POS-теги + парсинг зависимостей + газеттир позволяют продвинуться на удивление далеко.

## Готовый продукт

Сохраните как `outputs/skill-grammar-pipeline.md`:

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## Упражнения

1. **Лёгкое.** Используя базовый метод "наиболее частый тег" на небольшом размеченном корпусе (например, подвыборке Brown из NLTK), измерьте точность на отложенных предложениях. Проверьте результат ~85%.
2. **Среднее.** Обучите описанный выше биграммный HMM и получите precision/recall по каждому тегу. Какие теги HMM путает чаще всего?
3. **Сложное.** Используйте парсинг зависимостей spaCy для извлечения троек "подлежащее-глагол-дополнение" из выборки в 1000 предложений. Оцените результат на 50 вручную размеченных тройках. Задокументируйте случаи, где извлечение не срабатывает (часто это пассивные конструкции, сочинительные связи и опущенные подлежащие).

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|-----------------------|
| POS-тег | Тип слова | Грамматическая категория. В PTB — 36 тегов, в UD — 17. |
| Penn Treebank | Стандартный набор тегов | Специфичен для английского языка. Детализированные времена глаголов и число существительных. |
| Universal Dependencies | Многоязычный набор тегов | Более грубый, чем PTB; языконезависимый; стандарт для кросс-языковой работы. |
| Дерево зависимостей | Дерево предложения | У каждого слова одно главное слово, каждое ребро имеет грамматическое отношение. |
| Витерби | Динамическое программирование | Находит последовательность тегов с наибольшей вероятностью при заданных эмиссиях и переходах. |

## Дополнительное чтение

- [Джурафски и Мартин — Speech and Language Processing, главы 8 и 18](https://web.stanford.edu/~jurafsky/slp3/) — каноническое изложение POS-разметки и парсинга в учебнике.
- [Проект Universal Dependencies](https://universaldependencies.org/) — кросс-языковой набор тегов и коллекция treebank'ов, используемые в каждом многоязычном парсере.
- [Руководство spaCy по лингвистическим признакам](https://spacy.io/usage/linguistic-features) — практический справочник по каждому атрибуту, доступному в `Token`.
- [Chen и Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — статья, которая вывела нейросетевые парсеры в мейнстрим.