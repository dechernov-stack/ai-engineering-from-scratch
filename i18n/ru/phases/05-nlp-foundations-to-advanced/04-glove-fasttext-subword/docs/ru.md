# GloVe, FastText и субсловные эмбеддинги

> Word2Vec обучал по одному эмбеддингу на слово. GloVe факторизовал матрицу совместной встречаемости. FastText встраивал части слов. BPE навёл мост к трансформерам.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec from Scratch)
**Time:** ~45 минут

## Проблема

Word2Vec оставил два открытых вопроса.

Во-первых, существовало параллельное направление исследований, которое факторизовало матрицу совместной встречаемости напрямую (LSA, HAL), вместо онлайн-обновлений skip-gram. Был ли итеративный подход Word2Vec принципиально лучше, или разница была артефактом того, как эти два метода обрабатывали счётчики? **GloVe** ответил на этот вопрос: факторизация матрицы с продуманно подобранной функцией потерь не уступает Word2Vec или превосходит его, и обучается дешевле.

Во-вторых, ни один из методов не имел решения для слов, которых они никогда не видели. `Zoomer-approved`, `dogecoin`, любое имя собственное, придуманное на прошлой неделе, любая словоформа редкого корня. **FastText** решил эту проблему, встраивая символьные n-граммы: слово — это сумма его частей, включая морфемы, поэтому даже слова вне словаря получают осмысленный вектор.

В-третьих, с появлением трансформеров вопрос сместился снова. Словарные словари на уровне слов упираются в потолок около миллиона записей; реальный язык более открыт, чем это. **Кодирование пар байтов (BPE)** и родственные ему методы решили эту проблему, обучая словарь из частых субсловных единиц, покрывающий всё. Каждый современный токенизатор для каждой современной LLM — это субсловный токенизатор.

Этот урок разберёт все три метода, а затем объяснит, к какому из них обращаться в каком случае.

## Концепция

**GloVe (Global Vectors).** Постройте матрицу совместной встречаемости слово-слово `X`, где `X[i][j]` — это насколько часто слово `j` встречается в контексте слова `i`. Обучите векторы так, чтобы `v_i · v_j + b_i + b_j ≈ log(X[i][j])`. Взвесьте функцию потерь так, чтобы частые пары не доминировали. Готово.

**FastText.** Слово — это сумма его символьных n-грамм плюс само слово. `where` становится `<wh, whe, her, ere, re>, <where>`. Вектор слова — это сумма этих векторов-компонентов. Обучается как Word2Vec. Преимущество: невиданные слова (`whereupon`) составляются из известных n-грамм.

**BPE (Byte-Pair Encoding).** Начните со словаря отдельных байтов (или символов). Посчитайте каждую соседнюю пару в корпусе. Слейте наиболее частую пару в новый токен. Повторите `k` итераций. Результат: словарь из `k + 256` токенов, где частые последовательности (`ing`, `tion`, `the`) являются отдельными токенами, а редкие слова разбиваются на знакомые части. Любое предложение токенизируется во что-то.

```figure
n5-subword-merge
```

## Реализация

### GloVe: факторизация матрицы совместной встречаемости

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

Стоит отметить два ключевых компонента. Весовая функция `f(x) = (x/x_max)^alpha` понижает вес очень частых пар (таких как `(the, and)`), чтобы они не доминировали в функции потерь. Итоговый эмбеддинг — это сумма таблиц `W` (центральное слово) и `W_tilde` (контекст). Суммирование обеих таблиц — это опубликованный приём, который обычно превосходит использование только одной таблицы.

### FastText: эмбеддинги с учётом субслов

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

Каждое слово представлено набором его n-грамм (обычно от 3 до 6 символов). Эмбеддинг слова — это сумма эмбеддингов его n-грамм. При skip-gram обучении подставьте это туда, где Word2Vec использовал единственный вектор.

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

Для невиданного слова вы всё равно получаете вектор, пока хотя бы часть его n-грамм известна. `whereupon` разделяет `<wh`, `her`, `ere` и `<where` с `where`, поэтому эти два слова оказываются рядом.

### BPE: обучаемый субсловный словарь

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

Первая итерация сливает наиболее распространённую соседнюю пару. После достаточного числа итераций частые подстроки (`low`, `est`, `tion`) становятся отдельными токенами, а редкие слова разбиваются аккуратно.

Настоящие токенизаторы GPT / BERT / T5 обучают 30-100 тысяч слияний. Результат: любой текст токенизируется в последовательность известных ID ограниченной длины, никогда не возникает слов вне словаря.

## Применение

На практике вы редко обучаете что-либо из этого самостоятельно. Вы загружаете предобученные чекпоинты.

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

Для субсловной токенизации в стиле BPE в эпоху трансформеров:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

Префикс `Ġ` обозначает границы слов (соглашение GPT-2). Каждый современный токенизатор — это вариант BPE, WordPiece (BERT) или SentencePiece (T5, LLaMA).

### Что выбрать в каком случае

| Ситуация | Выбор |
|-----------|------|
| Предобученные общего назначения словесные векторы, толерантность к OOV не нужна | GloVe 300d |
| Предобученные общего назначения словесные векторы, нужно обрабатывать опечатки / неологизмы / морфологически богатые языки | FastText |
| Что угодно, что идёт в трансформер (обучение или инференс) | Тот токенизатор, с которым поставлялась модель. Никогда не меняйте. |
| Обучение собственной языковой модели с нуля | Сначала обучите BPE или SentencePiece токенизатор на своём корпусе |
| Промышленная классификация текста с линейной моделью | По-прежнему TF-IDF. Урок 02. |

## Финализация

Сохраните как `outputs/skill-embeddings-picker.md`:

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## Упражнения

1. **Лёгкое.** Запустите `char_ngrams("playing")` и `char_ngrams("played")`. Вычислите пересечение Жаккара двух наборов n-грамм. Вы увидите существенное количество общих частей (`pla`, `lay`, `play`), именно поэтому FastText хорошо переносится между морфологическими вариантами.
2. **Среднее.** Расширьте `learn_bpe` для отслеживания роста словаря. Постройте график количества токенов на символ корпуса в зависимости от числа слияний. Вы увидите быстрое сжатие вначале, выходящее на асимптоту около ~2-3 символов на токен.
3. **Сложное.** Обучите BPE с 1000 слияний на полном собрании сочинений Шекспира. Сравните токенизацию обычных слов с редкими именами собственными. Измерьте среднее число токенов на слово до и после. Опишите, что вас удивило.

## Ключевые термины

| Термин | Что говорят люди | Что это на самом деле означает |
|------|-----------------|-----------------------|
| Матрица совместной встречаемости | Таблица частот слово-слово | `X[i][j]` = насколько часто слово `j` встречается в окне вокруг слова `i`. |
| Субслово | Часть слова | Символьная n-грамма (FastText) или обученный токен (BPE/WordPiece/SentencePiece). |
| BPE | Кодирование пар байтов | Итеративное слияние наиболее частых соседних пар до достижения целевого размера словаря. |
| OOV | Вне словаря | Слово, которое модель никогда не видела. Word2Vec/GloVe с этим не справляются. FastText и BPE обрабатывают такие случаи. |
| Byte-level BPE | BPE на сырых байтах | Схема GPT-2. Словарь начинается с 256 байтов, поэтому ничто никогда не оказывается вне словаря. |

## Дополнительное чтение

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — статья о GloVe, семь страниц, всё ещё лучший вывод функции потерь.
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText.
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — статья, которая представила BPE в современной NLP.
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — как BPE, WordPiece и SentencePiece на самом деле отличаются на практике.