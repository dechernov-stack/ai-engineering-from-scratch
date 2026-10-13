# BERT  Моделирование языка в маске

> GPT предсказывает следующее слово. BERT предсказывает отсутствующее слово. Один предложение разницы  и пол десятилетия все встроенной формы.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 5 · 02 (Text Representation)
**Time:** ~45 minutes

## Проблема

В 2018 году каждая NLP-задача  sentiment, NER, QA, entailment  обучала свою собственную модель с нуля на собственных данных.

BERT (Devlin et al. 2018) спросил: что, если мы возьмем кодировщик трансформатора, обучим его каждому предложению в интернете и заставим предсказать отсутствующие слова из контекста с обеих сторон?

Результат: в течение 18 месяцев BERT и его варианты (RoBERTa, ALBERT, ELECTRA) доминировали над каждой существующей лидеровкой NLP.

В 2026 году модели, использующие только кодировщик, по-прежнему являются правильным инструментом для классификации, извлечения и структурированной извлечения. Они работают в 510x быстрее, чем декодеры, и их вкладки являются основой каждой современной пачки извлечения.

## Концепция

![Masked language modeling: pick tokens, mask them, predict originals](../assets/bert-mlm.svg)

### Сигнал обучения

Сделайте предложение:`the quick brown fox jumps over the lazy dog`- Я не знаю .

15% токенов маскируются случайным образом:

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

Учите модель предсказывать оригинальные токены на маскированных позициях.`[MASK]`на позиции 1 может использовать `brown fox jumps`Это то, что GPT не может сделать.

### Правила маски BERT

Из 15% токенов, выбранных для прогнозирования:

- 80% заменяются `[MASK]`- Я не знаю .
- 10% заменяются случайным токеном.
- 10% остаются неизменными.

Почему не всегда ?`[MASK]`Потому что ...`[MASK]`В то время как модель не может быть выведена из этого, она не может быть выведена из этого.`[MASK]`При 100% маскированных позициях произойдет смена распределения между предварительной тренировкой и настройкой. 10% случайности + 10% неизменности сохраняют модель честной.

### Следующее предсказание предложения (NSP)  и почему оно было отменено

Оригинальный BERT также обучался NSP: при двух предложениях A и B, предсказать, если B следует за A. RoBERTa (2019) отменил его и показал, что NSP вредит, а не помогает.

### Что изменилось в 2026 году: ModernBERT

В газете ModernBERT 2024 года был перестроен блок с приватизацией 2026 года:

| Component | Original BERT (2018) | ModernBERT (2024) |
|-----------|----------------------|-------------------|
| Positional | Learned absolute | RoPE |
| Activation | GELU | GeGLU |
| Normalization | LayerNorm | Pre-norm RMSNorm |
| Attention | Full dense | Alternating local (128) + global |
| Context length | 512 | 8192 |
| Tokenizer | WordPiece | BPE |

И в отличие от стэка 2018 года, он является Flash-Attention-native. Инференция 23x быстрее на длину последовательности 8K, чем DeBERTa-v3 с лучшими баллами GLUE.

### Использование случаев, которые все еще выбирают кодировщик в 2026 году

| Task | Why encoder beats decoder |
|------|---------------------------|
| Retrieval / semantic search embeddings | Bidirectional context = better embedding quality per token |
| Classification (sentiment, intent, toxicity) | One forward pass; no generation overhead |
| NER / token labeling | Per-position output, natively bidirectional |
| Zero-shot entailment (NLI) | Classifier head on top of encoder |
| Reranker for RAG | Cross-encoder scoring, 10x faster than LLM rerankers |

```figure
transformer-residual
```

## Создайте его

### Шаг 1: логика маскировки

Посмотрите .`code/main.py`. Функция `create_mlm_batch`Вносит список идентификаторов токенов, размер словаря и вероятность маски. Возвращает идентификаторы ввода (с применением масок) и этикеты (только на маскированных позициях, -100 в другом месте  Не обращайте внимание на индекс конвенции PyTorch).

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### Шаг 2: запустить прогноз MLM на крошечном корпусе

Учите 2-слойный кодировщик + MLM-главу на словарном запасе 20 слов, 200 предложений.

### Шаг 3: сравните типы масок

Покажите , как правило трехстороннего движения делает модель полезной без `[MASK]`- Предсказать о немаскированном предложении и о маскированном предложении.

### Шаг 4: тонкостроительная голова

Заменить главу MLM главой классификации на наборе данных о чувстве игрушки. Только голова загружается; кодер заморожен.

## Используйте его

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**Embedding models are fine-tuned BERT.** `sentence-transformers`модели типа `all-MiniLM-L6-v2`- У них есть контрастные потери, кодировщик тот же.

**Cross-encoder rerankers are also fine-tuned BERT.**Пара-классификация на `[CLS] query [SEP] doc [SEP]`Двустороннее внимание между запросом и документом именно то, что дает кросс-кодеры их качественный преимущество над биенкодеры.

**When not to pick BERT in 2026.**Все, что генерирует. Кодер не имеет разумного способа авторегрессивно производить токены.

## Погрузи его

Посмотрите .`outputs/skill-bert-finetuner.md`. Умение определяет тонкую настройку BERT (выбор спинной кости, спецификация головы, данные, оценка, остановка) для новой классификации или добычи задач.

## Упражнения

1. **Easy.**Беги .`code/main.py`И печатать распространение маски по 10 000 токенов.`[MASK]`- Я не знаю .
2. **Medium.**Используйте маскирование целого слова: если слово символизируется в подсловия, маскируйте все подсловы вместе или ни одного. Измерить, улучшает ли это точность MLM на корпусе 500 предложений.
3. **Hard.**Обучить крошечный (2-слойный, d=64) BERT на 10 000 предложений из общедоступного набора данных.`[CLS]`Сравните с базовой линией только для декодера в сопоставленных параметрах, что выигрывает?

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| MLM | "Masked language modeling" | Training signal: randomly replace 15% of tokens with `[MASK]`, predict the originals. |
| Bidirectional | "Looks both ways" | Encoder attention has no causal mask — every position sees every other position. |
| `[CLS]` | "The pooler token" | A special token prepended to every sequence; its final embedding is used as the sentence-level representation. |
| `[SEP]` | "Segment separator" | Separates paired sequences (e.g. query/doc, sentence A/B). |
| NSP | "Next sentence prediction" | BERT's second pretraining task; shown to be useless in RoBERTa, dropped after 2019. |
| Fine-tuning | "Adapt to a task" | Keep the encoder mostly frozen; train a small head on top for the downstream task. |
| Cross-encoder | "A reranker" | A BERT that takes both query and doc as input, outputs a relevance score. |
| ModernBERT | "2024 refresh" | Encoder rebuilt with RoPE, RMSNorm, GeGLU, alternating local/global attention, 8K context. |

## Дальнейшее чтение

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) оригинальная бумага.
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) как правильно обучить БЕРТ; убивает НСП.
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) замещение токенной обнаружения превосходит MLM при сопоставленном вычислении.
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) ModernBERT бумаги.
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) канонический кодировщик ссылка.
