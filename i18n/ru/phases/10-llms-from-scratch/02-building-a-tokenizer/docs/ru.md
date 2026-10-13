# Создание токенизатора с нуля

> Урок 01 дал вам игрушку. Этот урок даёт вам оружие.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lesson 01 (Tokenizers: BPE, WordPiece, SentencePiece)
**Time:** ~90 минут

## Цели обучения

- Построить BPE-токенизатор промышленного уровня, обрабатывающий Unicode, нормализацию пробелов и специальные токены
- Реализовать байтовый fallback, чтобы токенизатор мог закодировать любой ввод (включая эмодзи, иероглифы CJK и код) без неизвестных токенов
- Добавить регулярные выражения предтокенизации, разбивающие текст по границам слов перед применением BPE-слияний
- Обучить собственный токенизатор на корпусе и оценить коэффициент сжатия в сравнении с tiktoken на многоязычном тексте

## Проблема

Ваш BPE-токенизатор из Урока 01 работает на английском тексте. Теперь бросьте в него японский. Или эмодзи. Или код на Python со смешанными табами и пробелами.

Он ломается.

Не потому, что BPE неправильный — а потому, что реализация неполная. Промышленный токенизатор обрабатывает сырые байты в любой кодировке, нормализует Unicode перед разбиением, управляет специальными токенами, которые никогда не сливаются, выстраивает цепочку из предтокенизации и разбиения на подслова — и всё это достаточно быстро, чтобы не создавать узкое место в пайплайне обучения, обрабатывающем 15 триллионов токенов.

Токенизатор GPT-2 содержит 50 257 токенов. У Llama 3 — 128 256. У GPT-4 — около 100 000. Это не игрушечные цифры. Таблицы слияний за этими словарями обучены на сотнях гигабайт текста, а окружающая инфраструктура — нормализация, предтокенизация, вставка специальных токенов, форматирование шаблонов чата — это то, что отличает токенизатор, обрабатывающий "hello world", от токенизатора, обрабатывающего весь интернет.

Вы собираетесь построить эту инфраструктуру.

## Концепция

### Полный пайплайн

Промышленный токенизатор — это не один алгоритм. Это пайплайн из пяти стадий, каждая из которых решает свою задачу.

```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

Каждая стадия выполняет конкретную работу:

| Стадия | Что делает | Почему это важно |
|-------|-------------|----------------|
| Нормализация | NFKC Unicode, опционально приведение к нижнему регистру, опциональное удаление диакритики | Лигатура "fi" (U+FB01) превращается в "fi" (два символа). Без этого одинаковое слово получает разные токены. |
| Предтокенизация | Разбиение текста на фрагменты перед BPE | Предотвращает слияние BPE через границы слов. "the cat" никогда не должно давать токен "e c". |
| BPE-слияние | Применение выученных правил слияния к байтовым последовательностям | Ядро сжатия. Превращает сырые байты в подсловные токены. |
| Специальные токены | Вставка [BOS], [EOS], [PAD], маркеров шаблона чата | Эти токены имеют фиксированные ID. Они никогда не участвуют в BPE-слияниях. Модели они нужны для структуры. |
| Отображение в ID | Преобразование строк токенов в целочисленные ID | Модель видит целые числа, а не строки. |

### Байтовый BPE

Токенизатор из Урока 01 работал с байтами UTF-8. Это было правильное решение. Но мы пропустили кое-что важное: что происходит, когда эти байты не являются валидным UTF-8?

Байтовый BPE решает это, рассматривая каждое возможное значение байта (0–255) как валидный токен. Ваш базовый словарь — ровно 256 записей. Любой файл — текст, бинарные данные, повреждённые данные — можно токенизировать без получения неизвестного токена.

GPT-2 добавила трюк: сопоставить каждый байт с печатаемым символом Unicode, чтобы словарь оставался читаемым человеком. Байт 0x20 (пробел) превращается в символ "G" в их отображении. Это чисто косметическая мера. Алгоритму всё равно.

Настоящая сила в том, что байтовый BPE обрабатывает любой язык на земле. Китайские иероглифы — по 3 байта UTF-8 каждый. Японские символы могут занимать 3–4 байта. Арабский, деванагари, эмодзи — всё это просто байтовые последовательности. Алгоритм BPE находит паттерны в этих байтовых последовательностях точно так же, как находит паттерны в английских ASCII-байтах.

### Предтокенизация

Прежде чем BPE коснётся вашего текста, его нужно разбить на фрагменты. Это предотвращает создание алгоритмом слияния токенов, пересекающих границы слов.

GPT-2 использует шаблон регулярного выражения для разбиения текста:

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Этот шаблон разбивает текст по сокращениям ("don't" становится "don" + "'t"), словам с опциональным ведущим пробелом, числам, пунктуации и пробельным символам. Ведущий пробел остаётся прикреплённым к слову — поэтому "the cat" становится [" the", " cat"], а не ["the", " ", "cat"].

Llama использует SentencePiece, которая полностью пропускает регулярные выражения. Она рассматривает сырой поток байтов как одну длинную последовательность и позволяет алгоритму BPE самому находить границы. Это проще, но даёт BPE больше свободы для создания токенов, пересекающих слова.

Этот выбор важен. Регулярное выражение GPT-2 предотвращает обучение токенизатора тому, что "the" в конце одного слова и "the" в начале следующего должны сливаться. SentencePiece это допускает, что иногда даёт более эффективное сжатие, но менее интерпретируемые токены.

### Специальные токены

Каждый промышленный токенизатор резервирует ID токенов для структурных маркеров:

| Токен | Назначение | Используется в |
|-------|---------|---------|
| `[BOS]` / `<s>` | Начало последовательности | Llama 3, GPT |
| `[EOS]` / `</s>` | Конец последовательности | Все модели |
| `[PAD]` | Заполнитель для выравнивания батча | BERT, T5 |
| `[UNK]` | Неизвестный токен (байтовый BPE исключает это) | BERT, WordPiece |
| `<\|im_start\|>` | Начало границы сообщения чата | ChatGPT, Qwen |
| `<\|im_end\|>` | Конец границы сообщения чата | ChatGPT, Qwen |
| `<\|user\|>` | Маркер хода пользователя | Llama 3 |
| `<\|assistant\|>` | Маркер хода ассистента | Llama 3 |

Специальные токены никогда не разбиваются BPE. Они точно сопоставляются до запуска алгоритма слияния, заменяются своим фиксированным ID, а окружающий текст токенизируется обычным образом.

### Шаблоны чата

Именно здесь большинство людей путается, а большинство реализаций ломается.

Когда вы отправляете сообщения чат-модели, API принимает список сообщений:

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

Модель не видит JSON. Она видит плоскую последовательность токенов. Шаблон чата преобразует сообщения в эту плоскую последовательность с использованием специальных токенов. Каждая модель делает это по-своему:

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

Ошибётесь с шаблоном — модель выдаст бессмыслицу. Она обучена на одном точном формате. Любое отклонение — пропущенный перенос строки, переставленный токен, лишний пробел — выводит ввод за пределы распределения обучающих данных.

### Скорость

Python слишком медленный для промышленной токенизации.

tiktoken (OpenAI) написан на Rust с привязками для Python. HuggingFace tokenizers тоже на Rust. SentencePiece — на C++. Они достигают 10–100-кратного ускорения по сравнению с чистым Python.

Для масштаба: токенизация 15 триллионов токенов для предобучения Llama 3 со скоростью 1 миллион токенов в секунду (быстрый Python) заняла бы 174 дня. Со скоростью 100 миллионов токенов в секунду (Rust) это занимает 1,7 дня.

Вы строите на Python, чтобы понять алгоритм. В продакшене вы бы использовали скомпилированную реализацию и трогали бы только Python-обёртку.

```figure
weight-tying
```

## Постройте это

### Шаг 1: Байтовое кодирование

Фундамент. Преобразуйте любую строку в последовательность байтов, сопоставьте каждый байт с печатаемым символом для отображения и выполните обратное преобразование.

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

Проверьте на многоязычном тексте, чтобы увидеть количество байтов:

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello" — 5 байт. "你好" — 6 байт (по 3 на символ). Эмодзи огня — 4 байта. Байтовому токенизатору всё равно, какой это язык. Байты есть байты.

### Шаг 2: Предтокенизатор с регулярными выражениями

Разбейте текст на фрагменты, используя шаблон регулярного выражения GPT-2. Каждый фрагмент токенизируется независимо с помощью BPE.

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

Модуль `regex` поддерживает escape-последовательности свойств Unicode (`\p{L}` для букв, `\p{N}` для цифр). Стандартный модуль `re` этого не поддерживает, поэтому мы переключаемся на классы ASCII-символов. Для промышленных многоязычных токенизаторов установите `regex`.

Попробуйте:

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

Ведущий пробел остаётся прикреплённым к слову. Сокращения разбиваются по апострофу. Пунктуация становится отдельным фрагментом. BPE никогда не сольёт токены через эти границы.

### Шаг 3: BPE на байтовых последовательностях

Основной алгоритм из Урока 01, но теперь работающий на предтокенизированных фрагментах независимо.

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### Шаг 4: Обработка специальных токенов

Специальным токенам нужно точное сопоставление и фиксированные ID. Они полностью минуют BPE.

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### Шаг 5: Класс полного токенизатора

Соедините всё вместе: нормализация, разбиение по специальным токенам, предтокенизация, BPE-слияние, отображение в ID.

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### Шаг 6: Многоязычный тест

Настоящая проверка. Бросьте в него английский, китайский, эмодзи и код.

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

Китайские иероглифы дают по 3 байта каждый. Эмодзи даёт 4 байта. Ничто из этого не ломает токенизатор. Ничто не даёт неизвестных токенов. В этом сила байтового BPE.

## Используйте это

### Сравнение реальных токенизаторов

Загрузите настоящие токенизаторы Llama 3, GPT-4 и Mistral. Посмотрите, как каждый обрабатывает один и тот же многоязычный абзац.

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

Вы увидите разное количество токенов для одного и того же текста. Llama 3 со словарём 128K более агрессивно сливает частые паттерны. GPT-4 со 100K находится посередине. Mistral с 32K даёт больше токенов, но имеет меньший слой эмбеддингов.

Компромисс всегда один и тот же: больший словарь означает более короткие последовательности, но больше параметров.

## Поставка

Этот урок производит промт для построения и отладки промышленных токенизаторов. См. `outputs/prompt-tokenizer-builder.md`.

## Упражнения

1. **Лёгкое:** Добавьте метод `get_token_bytes(id)`, показывающий сырые байты для любого ID токена. Используйте его, чтобы посмотреть, что на самом деле представляют собой ваши самые частые слитые токены.
2. **Среднее:** Реализуйте предтокенизатор в стиле Llama, который разбивает текст по пробелам и цифрам, но сохраняет ведущие пробелы. Сравните его словарь с подходом на основе регулярных выражений GPT-2 на том же корпусе.
3. **Сложное:** Добавьте метод шаблона чата, принимающий список сообщений `{"role": ..., "content": ...}` и выдающий правильную последовательность токенов для формата чата Llama 3. Проверьте его на соответствие реализации HuggingFace.

## Ключевые термины

| Термин | Что говорят люди | Что это на самом деле означает |
|------|----------------|----------------------|
| Byte-level BPE | "Токенизатор, работающий с байтами" | BPE с базовым словарём из 256 значений байтов — обрабатывает любой ввод без неизвестных токенов |
| Pre-tokenization | "Разбиение перед BPE" | Разбиение на основе регулярных выражений или правил, предотвращающее слияние BPE через границы слов |
| NFKC normalization | "Очистка Unicode" | Каноническая декомпозиция с последующей композицией совместимости — лигатура "fi" превращается в "fi", полноширинная "A" становится "A" |
| Chat template | "Как сообщения превращаются в токены" | Точный формат преобразования списка сообщений роль/содержимое в плоскую последовательность токенов — специфичен для модели и должен соответствовать формату обучения |
| Special tokens | "Управляющие токены" | Зарезервированные ID токенов, минующие BPE — [BOS], [EOS], [PAD], маркеры чата — точно сопоставляются до слияния |
| Fertility | "Токенов на слово" | Отношение выходных токенов к входным словам — 1.3 для английского в GPT-4, 2–3 для корейского, чем выше, тем больше потрачено впустую контекста |
| tiktoken | "Токенизатор OpenAI" | Реализация BPE на Rust с привязками для Python — в 10–100 раз быстрее чистого Python |
| Merge table | "Словарь" | Упорядоченный список слияний байтовых пар, выученных во время обучения — это и ЕСТЬ выученные знания токенизатора |

## Дополнительное чтение

- [Исходный код OpenAI tiktoken](https://github.com/openai/tiktoken) — реализация BPE на Rust, используемая GPT-3.5/4
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) — библиотека токенизаторов на Rust, поддерживающая BPE, WordPiece, Unigram
- [Статья о Llama 3 (Meta, 2024)](https://arxiv.org/abs/2407.21783) — подробности о словаре 128K и обучении токенизатора
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) — языконезависимая токенизация
- [Исходный код токенизатора GPT-2](https://github.com/openai/gpt-2/blob/master/src/encoder.py) — исходное отображение байтов в Unicode