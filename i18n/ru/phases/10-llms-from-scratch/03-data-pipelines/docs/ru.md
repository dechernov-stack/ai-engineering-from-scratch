# Конвейеры данных для предварительного обучения

> Модель — это зеркало. Она отражает всё, что вы в неё подаёте. Подайте мусор — она с идеальной беглостью отразит мусор.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lessons 01-02 (Tokenizers, Building a Tokenizer)
**Time:** ~90 минут

## Цели обучения

- Построить потоковый конвейер данных, который токенизирует, разбивает на чанки, перемешивает и батчит терабайты текста, не загружая их целиком в память
- Реализовать фильтры качества данных (дедупликация, определение языка, фильтрация контента), используемые в реальных конвейерах предварительного обучения
- Создать обучающие последовательности фиксированной длины с корректными масками внимания и обработкой границ документов
- Профилировать пропускную способность конвейера, чтобы загрузчик данных успевал за скоростью обучения на GPU

## Проблема

У вас есть токенизатор. Теперь нужны данные.

Не датасет. Не CSV-файл. Терабайты текста — очищенного, дедуплицированного, отфильтрованного по качеству, токенизированного в последовательности фиксированной длины и подаваемого случайными батчами достаточно быстро, чтобы ваш кластер из 8 GPU никогда не простаивал в ожидании следующего батча.

Большинство людей думают, что обучение LLM — это вопрос архитектуры модели. Это не так. Llama 3 использовала 15,6 триллиона токенов. GPT-3 использовала 300 миллиардов. DeepSeek-V2 использовала 8,1 триллиона. Архитектура во всех трёх случаях примерно одинакова: стопка трансформерных блоков с attention и feedforward-слоями. Разница в качестве выходных данных определяется в подавляющем большинстве случаев именно данными.

Статья Chinchilla от DeepMind сформулировала это точно. Для заданного вычислительного бюджета существует оптимальное соотношение между количеством параметров модели и количеством обучающих токенов. Chinchilla показала, что большинство моделей 2022 года были драматически недообучены — у них было слишком много параметров относительно объёма данных, которые они видели. Модель с 70 млрд параметров, обученная на 1,4 триллиона токенов (Chinchilla-оптимально), превосходила модель с 280 млрд параметров, обученную на 300 миллиардах токенов (Gopher).

Ваш конвейер данных определяет, научится ли модель языку или научится шуму.

## Концепция

### Откуда берутся данные

Каждая большая языковая модель обучается на смеси источников. Точный состав для большинства лабораторий держится в строгом секрете, но мы знаем достаточно, чтобы понять категории.

| Источник | Размер | Качество | Используется в |
|--------|------|---------|-----|
| Common Crawl | ~250 ТБ в сыром виде | Низкое (требует тщательной фильтрации) | GPT-3, Llama, большинство открытых моделей |
| Wikipedia | ~20 ГБ | Высокое | Каждая крупная LLM |
| Код с GitHub | ~1 ТБ+ | Среднее (много дубликатов, мёртвого кода) | StarCoder, CodeLlama, DeepSeek-Coder |
| Книги (BookCorpus, Pile) | ~100 ГБ | Высокое | GPT-2, GPT-3, ранние модели |
| Научные статьи (arXiv, S2ORC) | ~100 ГБ | Высокое для STEM | Llama, Galactica |
| StackOverflow, Reddit | ~100 ГБ | Среднее | Llama, Falcon |
| Отобранный веб-контент (C4, RefinedWeb) | ~5 ТБ | Средне-высокое (предварительно отфильтровано) | T5, Falcon |

Llama 3 раскрыла свою смесь данных: примерно 50% веб-данных, 25% кода, 13% книг и научных статей, 8% математических данных и 4% многоязычных веб-данных. Общий объём составил 15,6 триллиона токенов из источников объёмом более 5 ТБ сырого текста.

Соотношение имеет значение не меньшее, чем общий размер. Слишком много веб-данных — и модель превращается в попугая Reddit. Слишком мало кода — и она не может программировать. Слишком мало математики — и она проваливается в рассуждениях. Правильный подбор этой смеси — одна из самых сложных частей обучения LLM, и формулы здесь нет — требуются эксперименты и оценка.

### Очистка данных

Сырые веб-данные — это грязь. Типичный дамп Common Crawl содержит:

- HTML-теги и JavaScript
- Шаблонные заголовки, подвалы, навигационные меню
- Дублирующиеся страницы (точные и почти идентичные)
- Машинно-сгенерированный спам
- Персонально идентифицируемую информацию (PII)
- Низкокачественный текст (списки ключевых слов, SEO-спам)
- Нетекстовый контент, закодированный как текст

Очистка этого — не опция. Это разница между моделью, генерирующей связные абзацы, и моделью, выдающей HTML-теги вперемешку со списками товаров.

```mermaid
graph TD
    A[Raw Text] --> B[HTML Strip]
    B --> C[Language Detection]
    C --> D[Quality Filter]
    D --> E[Deduplication]
    E --> F[PII Removal]
    F --> G[Clean Text]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style G fill:#1a1a2e,stroke:#e94560,color:#fff
```

Каждый шаг устраняет определённую категорию шума:

**Удаление HTML:** Удалите всю разметку. Оставьте только видимое текстовое содержимое. Библиотеки вроде `trafilatura` или `readability` извлекают содержимое статьи, отбрасывая навигацию, рекламу и шаблонные элементы.

**Определение языка:** Используйте модель определения языка fastText (lid.176.bin) для классификации каждого документа. Отфильтруйте по целевым языкам. Документ, классифицированный как английский с уверенностью менее 0,8, вероятно, не является чистым английским текстом.

**Фильтрация по качеству:** Здесь становится интереснее. RefinedWeb (датасет, лежащий в основе Falcon) использует фильтр на основе перплексии: обучите небольшую языковую модель на Wikipedia, затем оцените каждый документ. Высокая перплексия означает, что документ не похож на Wikipedia — вероятно, это спам, списки ключевых слов или машинно-сгенерированный контент. Документы с перплексией выше порога удаляются.

**Дедупликация:** Самый влиятельный этап очистки. Common Crawl содержит огромное количество дублирующихся страниц — юридические отказы от ответственности, уведомления о cookie, условия использования. Обучение на дубликатах тратит вычислительные ресурсы впустую и может привести к тому, что модель будет запоминать и дословно воспроизводить конкретные фрагменты.

**Удаление PII:** Имена, адреса электронной почты, номера телефонов, номера социального страхования. Детекция на основе регулярных выражений для структурированных PII, NER-модели для имён в контексте.

### Дедупликация с помощью MinHash

Точная дедупликация проста: захешируйте каждый документ, удалите дубликаты. Но настоящая проблема — почти идентичные документы. Две копии одной и той же новостной статьи с немного разной рекламой вокруг неё — это почти дубликаты. Содержимое совпадает на 95%, но побайтово они различаются.

MinHash + Locality-Sensitive Hashing (LSH) эффективно решает эту проблему.

```mermaid
graph LR
    A[Document] --> B[Shingling]
    B --> C[MinHash Signature]
    C --> D[LSH Buckets]
    D --> E[Candidate Pairs]
    E --> F[Jaccard Similarity]
    F --> G[Deduplicated Set]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style G fill:#1a1a2e,stroke:#e94560,color:#fff
```

Идея такова:

1. **Шинглирование:** Преобразуйте каждый документ в набор n-грамм (например, 5-грамм слов или символов). «the quick brown fox» при использовании 3-словных шинглов превращается в {"the quick brown", "quick brown fox"}.

2. **MinHash:** Для набора шинглов каждого документа вычислите k хеш-значений. Каждое хеш-значение — это минимум хеша по всем шинглам при разных хеш-функциях. Это создаёт «сигнатуру» фиксированного размера, которая аппроксимирует коэффициент сходства Жаккара между любыми двумя документами.

3. **LSH:** Группируйте документы в бакеты на основе полос (bands) их MinHash-сигнатуры. Документы в одном бакете являются кандидатами на почти-дубликаты. Это позволяет избежать сравнения каждой пары — вы сравниваете только кандидатов.

4. **Проверка:** Для каждой кандидатной пары вычислите точный коэффициент сходства Жаккара. Удалите одну копию, если сходство превышает порог (обычно 0,8).

Команда Llama сообщила об удалении примерно 38% своих веб-данных благодаря дедупликации. Это немалое число. Более трети Common Crawl — это дублирующийся или почти дублирующийся контент.

### Упаковка последовательностей (Sequence Packing)

Ваша модель ожидает входные последовательности фиксированной длины. Ваши документы имеют переменную длину. Некоторые — 50 токенов. Некоторые — 50 000 токенов.

Наивный подход: дополнить каждый документ до максимальной длины последовательности паддингом. Это тратит огромное количество вычислений на токены паддинга, которые ничего не вносят в обучение.

Лучший подход: упаковать несколько документов в одну последовательность, разделив их токенами конца последовательности. Последовательность из 2048 токенов может содержать три коротких документа, конкатенированных с токенами [EOS] между ними.

```mermaid
graph TD
    subgraph Naive Packing
        A1["Doc A (200 tokens)"] --> P1["[PAD] x 1848"]
        A2["Doc B (500 tokens)"] --> P2["[PAD] x 1548"]
        A3["Doc C (100 tokens)"] --> P3["[PAD] x 1948"]
    end

    subgraph Efficient Packing
        B1["Doc A (200) | Doc B (500) | Doc C (100) | Doc D (400) | Doc E (848)"]
    end

    style A1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style A2 fill:#1a1a2e,stroke:#e94560,color:#fff
    style A3 fill:#1a1a2e,stroke:#e94560,color:#fff
    style P1 fill:#333,stroke:#666,color:#999
    style P2 fill:#333,stroke:#666,color:#999
    style P3 fill:#333,stroke:#666,color:#999
    style B1 fill:#1a1a2e,stroke:#16c784,color:#fff
```

Маска внимания должна быть настроена правильно. Токены документа A не должны обращать внимание на токены документа B внутри одной упакованной последовательности. Это требует блочно-диагональной маски внимания.

Длинные документы обрезаются или разбиваются на чанки на границах последовательностей. Точка разбиения важна: разбиение посреди предложения заставляет модель видеть незавершённые мысли. Некоторые конвейеры по возможности выравнивают разбиения по границам абзацев или предложений.

### Закон масштабирования Chinchilla

Для фиксированного вычислительного бюджета C (измеряемого во FLOPs) оптимальный размер модели N и размер датасета D подчиняются следующему:

```
N_opt ~ C^0.5
D_opt ~ C^0.5
```

На практике это означает, что размер модели и размер датасета следует масштабировать примерно одинаково. Модели с в 10 раз большим числом параметров требуется примерно в 10 раз больше обучающих токенов, чтобы достичь того же значения функции потерь.

| Модель | Параметры | Обучающие токены | Chinchilla-оптимальна? |
|-------|-----------|----------------|-------------------|
| GPT-3 | 175 млрд | 300 млрд | Нет (недообучена в 3-4 раза) |
| Chinchilla | 70 млрд | 1,4 трлн | Да (по замыслу) |
| Llama 2 | 70 млрд | 2 трлн | Переобучена (намеренно) |
| Llama 3 | 70 млрд | 15 трлн | Сильно переобучена |

Llama 3 намеренно нарушает закон Chinchilla. Meta обнаружила, что переобучение на большем объёме данных — значительно превышающем вычислительно-оптимальное соотношение — даёт лучшие модели для инференса. Дополнительные затраты на обучение оплачиваются один раз, но меньшая модель дешевле в эксплуатации навсегда. Иногда это называют подходом «инференс-оптимального» масштабирования, и он стал отраслевым стандартом с 2024 года.

```figure
l5-data-pipeline
```

## Постройте это

### Шаг 1: Очистка текста

Удалите HTML, нормализуйте пробельные символы, удалите нетекстовый контент. Мы используем общедоступный текст (Project Gutenberg) в качестве нашего небольшого корпуса.

```python
import re

def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def quality_filter(text, min_words=50, max_ratio_caps=0.3, max_ratio_special=0.1):
    words = text.split()
    if len(words) < min_words:
        return False
    caps_ratio = sum(1 for w in words if w.isupper()) / len(words)
    if caps_ratio > max_ratio_caps:
        return False
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if special_chars / max(len(text), 1) > max_ratio_special:
        return False
    return True
```

Фильтр качества отлавливает SEO-спам (ВСЕ ЗАГЛАВНЫЕ), машинно-сгенерированный шум (высокая доля специальных символов) и заглушки страниц (слишком короткие). Эти три проверки сами по себе удаляют удивительно много мусора из веб-краулингов.

### Шаг 2: Дедупликация с помощью MinHash

Реализуйте MinHash с нуля. Внешние библиотеки не требуются — только `hashlib`.

```python
import hashlib
from collections import defaultdict

def get_shingles(text, k=5):
    words = text.lower().split()
    if len(words) < k:
        return set()
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}

def minhash_signature(shingles, num_hashes=128):
    signature = []
    for i in range(num_hashes):
        min_hash = float("inf")
        for shingle in shingles:
            h = int(hashlib.sha256(f"{i}:{shingle}".encode()).hexdigest(), 16)
            min_hash = min(min_hash, h)
        signature.append(min_hash)
    return signature

def lsh_buckets(signature, bands=16):
    rows_per_band = len(signature) // bands
    buckets = []
    for b in range(bands):
        start = b * rows_per_band
        band_data = tuple(signature[start:start + rows_per_band])
        bucket_hash = hashlib.md5(str(band_data).encode()).hexdigest()
        buckets.append((b, bucket_hash))
    return buckets

def deduplicate(documents, threshold=0.8, num_hashes=128, bands=16):
    signatures = []
    shingle_sets = []
    for doc in documents:
        shingles = get_shingles(doc)
        shingle_sets.append(shingles)
        signatures.append(minhash_signature(shingles, num_hashes))

    bucket_map = defaultdict(list)
    for doc_idx, sig in enumerate(signatures):
        for band_id, bucket_hash in lsh_buckets(sig, bands):
            bucket_map[(band_id, bucket_hash)].append(doc_idx)

    duplicate_pairs = set()
    for bucket_docs in bucket_map.values():
        if len(bucket_docs) < 2:
            continue
        for i in range(len(bucket_docs)):
            for j in range(i + 1, len(bucket_docs)):
                duplicate_pairs.add((bucket_docs[i], bucket_docs[j]))

    removed = set()
    for i, j in duplicate_pairs:
        if i in removed or j in removed:
            continue
        s1, s2 = shingle_sets[i], shingle_sets[j]
        if not s1 or not s2:
            continue
        jaccard = len(s1 & s2) / len(s1 | s2)
        if jaccard >= threshold:
            removed.add(j)

    return [doc for idx, doc in enumerate(documents) if idx not in removed], len(removed)
```

Параметры `num_hashes=128` и `bands=16` управляют компромиссом между точностью и полнотой. Больше хешей дают более точные оценки сходства. Больше полос (bands) увеличивают полноту (ловят больше дубликатов) ценой большего числа ложных срабатываний. Эти значения хорошо подходят для типичного веб-текста.

### Шаг 3: Токенизация и упаковка последовательностей

Возьмите чистый, дедуплицированный текст, токенизируйте его и упакуйте в последовательности фиксированной длины для обучения.

```python
def tokenize_corpus(documents, tokenizer):
    all_tokens = []
    for doc in documents:
        tokens = tokenizer.encode(doc)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_id)
    return all_tokens

def pack_sequences(token_ids, seq_length, pad_id=0):
    sequences = []
    attention_masks = []
    for i in range(0, len(token_ids), seq_length):
        seq = token_ids[i:i + seq_length]
        mask = [1] * len(seq)
        if len(seq) < seq_length:
            pad_count = seq_length - len(seq)
            seq = seq + [pad_id] * pad_count
            mask = mask + [0] * pad_count
        sequences.append(seq)
        attention_masks.append(mask)
    return sequences, attention_masks
```

### Шаг 4: DataLoader для обучения

Выдавайте случайные батчи упакованных последовательностей. Именно это потребляет цикл обучения.

```python
import random

class PreTrainingDataLoader:
    def __init__(self, sequences, attention_masks, batch_size, shuffle=True):
        self.sequences = sequences
        self.attention_masks = attention_masks
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.sequences) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.sequences)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            batch_seqs = [self.sequences[i] for i in batch_idx]
            batch_masks = [self.attention_masks[i] for i in batch_idx]
            yield batch_seqs, batch_masks
```

### Шаг 5: Статистика датасета

Вычислите числа, которые имеют значение: общее количество токенов, уникальные токены, коэффициент сжатия, распределение длины документов.

```python
from collections import Counter

def compute_statistics(documents, token_ids, sequences, tokenizer_vocab_size):
    total_chars = sum(len(d) for d in documents)
    total_tokens = len(token_ids)
    unique_tokens = len(set(token_ids))
    compression_ratio = total_chars / total_tokens

    doc_lengths = [len(d.split()) for d in documents]
    avg_doc_length = sum(doc_lengths) / max(len(doc_lengths), 1)
    max_doc_length = max(doc_lengths) if doc_lengths else 0
    min_doc_length = min(doc_lengths) if doc_lengths else 0

    token_counts = Counter(token_ids)
    top_tokens = token_counts.most_common(10)

    non_pad_tokens = sum(sum(1 for t in seq if t != 0) for seq in sequences)
    total_positions = sum(len(seq) for seq in sequences)
    utilization = non_pad_tokens / max(total_positions, 1)

    stats = {
        "total_documents": len(documents),
        "total_characters": total_chars,
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "vocab_utilization": unique_tokens / tokenizer_vocab_size,
        "compression_ratio": compression_ratio,
        "avg_doc_length_words": avg_doc_length,
        "max_doc_length_words": max_doc_length,
        "min_doc_length_words": min_doc_length,
        "num_sequences": len(sequences),
        "sequence_utilization": utilization,
        "top_10_tokens": top_tokens,
    }
    return stats
```

Коэффициент сжатия показывает, насколько эффективен токенизатор на данном корпусе. Английский текст обычно сжимается примерно до 3-4 символов на токен. Если вы видите 1,5 символа на токен, токенизатор режет слишком агрессивно. Если 8+ — он выучил очень специфичные для домена слияния.

Утилизация последовательностей показывает, какая доля ваших упакованных последовательностей — реальные данные, а не паддинг. Значение ниже 90% означает, что упаковка неэффективна — вы тратите вычисления на токены паддинга.

## Используйте это

### Сравнение с HuggingFace Datasets

Загрузите тот же корпус через библиотеку datasets от HuggingFace и сравните скорость конвейера.

```python
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

import time

start = time.time()
tokenized = ds.map(
    lambda x: tokenizer(x["text"], truncation=True, max_length=2048),
    batched=True,
    num_proc=4,
)
hf_time = time.time() - start
total_tokens = sum(len(t) for t in tokenized["input_ids"])
print(f"HuggingFace: {total_tokens:,} tokens in {hf_time:.2f}s ({total_tokens/hf_time:,.0f} tokens/sec)")
```

Конвейер HuggingFace использует токенизаторы на Rust «под капотом» и параллельную обработку на 4 ядрах. Ваш конвейер на чистом Python будет в 10-50 раз медленнее. Этот разрыв — причина, по которой production-команды используют компилируемые токенизаторы. Алгоритм тот же самый. Разница — в языке реализации.

## Используйте в деле

Этот урок создаёт промпт для валидации и отладки качества данных в конвейерах обучения LLM. См. `outputs/prompt-data-quality-checker.md`.

## Упражнения

1. **Лёгкое:** Добавьте определение языка в конвейер очистки с помощью простой эвристики (анализ набора символов). Отфильтруйте только английские документы и измерьте, сколько документов было удалено.
2. **Среднее:** Реализуйте точную дедупликацию с помощью хешей SHA-256 наряду с почти-дедупликацией MinHash. Сравните число дубликатов, обнаруженных каждым методом, на корпусе, собранном из веба.
3. **Сложное:** Постройте фильтр качества на основе перплексии. Обучите небольшую биграммную языковую модель на тексте Wikipedia, оцените каждый документ по перплексии и удалите нижние 20%. Сравните качество вывода модели при обучении на отфильтрованных и неотфильтрованных данных.

## Ключевые термины

| Термин | Что говорят люди | Что это на самом деле означает |
|------|----------------|----------------------|
| Common Crawl | «Интернет» | Некоммерческая организация, ежемесячно сканирующая веб — ~250 ТБ сырых данных, отправная точка для большинства обучающих данных LLM |
| MinHash | «Какой-то хеш-трюк» | Техника оценки сходства Жаккара между множествами с помощью сигнатур фиксированного размера — обеспечивает обнаружение почти-дубликатов в масштабе |
| LSH | «Locality-Sensitive Hashing» | Метод группировки похожих элементов в один бакет — сокращает число попарных сравнений с O(n^2) до почти линейного |
| Упаковка последовательностей | «Конкатенация документов» | Размещение нескольких документов в последовательностях фиксированной длины с правильными масками внимания — устраняет потери на паддинге |
| Масштабирование Chinchilla | «Обучаться на большем объёме данных» | При фиксированном вычислительном бюджете оптимальная производительность требует примерно одинакового масштабирования размера модели и количества обучающих токенов |
| Фертильность (fertility) | «Токенов на слово» | Среднее число токенов на слово — 1,3 для английского в GPT-4, выше для нелатинских письменностей |
| Смешивание данных | «Выбор обучающих данных» | Соотношение кода, текста, математики и многоязычных данных — формулы нет, требуются эксперименты |
| Фильтр по перплексии | «Оценка качества» | Использование небольшой языковой модели для оценки документов — высокая перплексия означает, что текст не похож на чистые эталонные данные |
| Дедупликация | «Удаление копий» | Устранение точных и почти-дублирующихся документов — обычно удаляет 30-40% сырых веб-данных |
| Маска внимания | «Какие токены учитывать» | Бинарная маска, предотвращающая внимание через границы документов в упакованных последовательностях |

## Дополнительное чтение

- [Hoffmann et al., 2022 -- Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556) -- статья, изменившая наш взгляд на масштаб данных
- [Penedo et al., 2023 -- The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116) -- как фильтровать Common Crawl для получения высокого качества
- [Touvron et al., 2023 -- Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) -- детали конвейера данных для Llama 2
- [Lee et al., 2022 -- Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) -- почему дедупликация важнее, чем вы думаете
- [Broder, 1997 -- On the Resemblance and Containment of Documents](https://ieeexplore.ieee.org/document/666900) -- оригинальная статья про MinHash
- [Meta, 2024 -- Llama 3 Technical Report](https://arxiv.org/abs/2407.21783) -- 15,6 трлн токенов, соотношения смешивания данных, конвейер фильтрации