# Субсловная токенизация — BPE, WordPiece, Unigram, SentencePiece

> Пословные токенизаторы ломаются на незнакомых словах. Посимвольные токенизаторы раздувают длину последовательности. Субсловные токенизаторы — компромисс между ними. На них работает каждая современная LLM.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 5 · 04 (GloVe / FastText / Subword)
**Time:** ~60 минут

## Проблема

Ваш словарь содержит 50 000 слов. Пользователь вводит «untokenizable». Токенизатор возвращает `[UNK]`. У модели теперь нет никакого сигнала об этом слове. Хуже того: в 90-м перцентиле документов вашего корпуса встречается 40 редких слов — это 40 бит потерянной информации на документ.

Субсловная токенизация решает эту проблему. Частые слова остаются единым токеном. Редкие слова раскладываются на осмысленные части: `untokenizable` → `un`, `token`, `izable`. Обучающие данные покрывают всё, потому что любая строка в конечном счёте — это последовательность байтов.

Каждая передовая LLM в 2026 году работает на одном из трёх алгоритмов (BPE, Unigram, WordPiece), обёрнутом в одну из трёх библиотек (tiktoken, SentencePiece, HF Tokenizers). Выпустить языковую модель, не выбрав один из них, невозможно.

## Концепция

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding).** Начните со словаря на уровне символов. Подсчитайте все соседние пары. Объедините самую частую пару в новый токен. Повторяйте, пока не достигнете целевого размера словаря. Доминирующий алгоритм: GPT-2/3/4, Llama, Gemma, Qwen2, Mistral.

**Byte-level BPE.** Тот же алгоритм, но применяется к сырым байтам (256 базовых токенов), а не к символам Unicode. Гарантирует отсутствие токенов `[UNK]` — любая последовательность байтов кодируется. GPT-2 использует 50 257 токенов (256 байт + 50 000 слияний + 1 специальный).

**Unigram.** Начните с огромного словаря. Присвойте каждому токену вероятность униграммы. Итеративно отсекайте токены, удаление которых минимально увеличивает логарифмическое правдоподобие корпуса. Вероятностный на этапе инференса: может сэмплировать разные токенизации (полезно для аугментации данных через субсловную регуляризацию). Используется в T5, mBART, ALBERT, XLNet, Gemma.

**WordPiece.** Объединяет пары, максимизирующие правдоподобие обучающего корпуса, а не просто сырую частоту. Используется в BERT, DistilBERT, ELECTRA.

**SentencePiece vs tiktoken.** SentencePiece — это библиотека, которая *обучает* словари (BPE или Unigram) прямо на сыром Unicode-тексте, кодируя пробелы как `▁`. tiktoken — быстрый *кодировщик* OpenAI для уже построенных словарей; он не обучает.

Эмпирическое правило:

- **Обучение нового словаря:** SentencePiece (многоязычность, без предварительной токенизации) или HF Tokenizers.
- **Быстрый инференс со словарём GPT:** tiktoken (cl100k_base, o200k_base).
- **И то, и другое:** HF Tokenizers — одна библиотека для обучения и обслуживания.

```figure
bpe-merge
```

## Реализация

### Шаг 1: BPE с нуля

См. `code/main.py`. Цикл:

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

Три факта, которые кодирует алгоритм. `</w>` обозначает конец слова, чтобы «low» (суффикс) и «lower» (префикс) оставались различными. Взвешивание по частоте позволяет частым парам побеждать раньше. Список слияний упорядочен — на этапе инференса слияния применяются в порядке обучения.

### Шаг 2: кодирование с помощью изученных слияний

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

Наивная реализация — O(n·|merges|). Промышленные реализации (tiktoken, HF Tokenizers) используют поиск ранга слияния с приоритетными очередями и работают почти за линейное время.

### Шаг 3: SentencePiece на практике

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

Обратите внимание: предварительная токенизация не требуется, пробел кодируется как `▁`, `character_coverage` управляет тем, насколько агрессивно редкие символы сохраняются вместо отображения в `<unk>`.

### Шаг 4: tiktoken для словарей, совместимых с OpenAI

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

Только кодирование. Быстро (бэкенд на Rust). Точное совпадение с токенизацией GPT-4/5 для подсчёта байтов, оценки стоимости, бюджетирования контекстного окна.

## Подводные камни, актуальные и в 2026 году

- **Дрейф токенизатора.** Обучение на словаре A, развёртывание со словарём B. Идентификаторы токенов различаются; модель выдаёт бессмыслицу. Проверяйте хеш `tokenizer.json` в CI.
- **Неоднозначность пробелов.** BPE «hello» и « hello» дают разные токены. Всегда явно указывайте `add_special_tokens` и `add_prefix_space`.
- **Недообучение на многоязычных данных.** Корпуса с преобладанием английского языка порождают словари, которые разбивают нелатинские письменности на в 5-10 раз больше токенов. Один и тот же промпт стоит в 5-10 раз дороже на японском/арабском в GPT-3.5. o200k_base частично это исправил.
- **Разбиение эмодзи.** Один эмодзи может занимать 5 токенов. Учитывайте обработку эмодзи при бюджетировании контекста.

## Применение

Стек 2026 года:

| Ситуация | Выбор |
|-----------|------|
| Обучение одноязычной модели с нуля | HF Tokenizers (BPE) |
| Обучение многоязычной модели | SentencePiece (Unigram, `character_coverage=0.9995`) |
| Обслуживание API, совместимого с OpenAI | tiktoken (`o200k_base` для GPT-4+) |
| Специализированный словарь (код, математика, белки) | Обучить кастомный BPE на предметном корпусе, объединить с базовым словарём |
| Инференс на периферии, маленькая модель | Unigram (меньшие словари работают лучше) |

Размер словаря — это решение, связанное с масштабированием, а не константа. Грубая эвристика: 32k для <1B параметров, 50-100k для 1-10B, 200k+ для многоязычных/передовых моделей.

## Запуск в продакшен

Сохраните как `outputs/skill-bpe-vs-wordpiece.md`:

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## Упражнения

1. **Лёгкое.** Обучите BPE с 500 слияниями на крошечном корпусе `code/main.py`. Закодируйте три слова из отложенной выборки. Сколько из них получили ровно 1 токен, а сколько — более 1?
2. **Среднее.** Сравните количество токенов на 100 предложениях английской Википедии между `cl100k_base`, `o200k_base` и SentencePiece BPE, который вы обучите со словарём=32k. Приведите коэффициент сжатия для каждого варианта.
3. **Сложное.** Обучите один и тот же корпус с помощью BPE, Unigram и WordPiece. Измерьте точность на нижележащей задаче при использовании каждого варианта в небольшом классификаторе тональности. Меняет ли выбор результат более чем на 1 пункт F1?

## Ключевые термины

| Термин | Что говорят люди | Что это на самом деле означает |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | Жадное слияние самых частых пар символов до достижения целевого размера словаря. |
| Byte-level BPE | Никогда нет неизвестных токенов | BPE над сырыми 256 байтами; используется в GPT-2 / Llama. |
| Unigram | Вероятностный токенизатор | Отсекает кандидатов из большого набора по логарифмическому правдоподобию; используется в T5, Gemma. |
| SentencePiece | «Тот, что с пробелами» | Библиотека, обучающая BPE/Unigram на сыром тексте; пробел кодируется как `▁`. |
| tiktoken | «Быстрый вариант» | Кодировщик BPE от OpenAI на бэкенде Rust для готовых словарей. Не обучает. |
| Список слияний | «Магические числа» | Упорядоченный список слияний `(a, b) → ab`; на инференсе применяется по порядку. |
| Покрытие символов | Насколько редкий — это слишком редкий? | Доля символов обучающего корпуса, которую должен покрывать токенизатор; обычно ~0.9995. |

## Дополнительное чтение

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — статья о BPE.
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — статья об Unigram.
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — статья о библиотеке.
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — краткий справочник.
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — кукбук и список кодировок.