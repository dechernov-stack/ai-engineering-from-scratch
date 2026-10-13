# T5, BART  Модели кодирования-декодирования

> Кодеры понимают. Декодеры генерируют. Вставьте их вместе, и вы получите модель, созданную для задач ввода → вывода: перевод, резюме, перепись, транскрипция.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT), Phase 7 · 07 (GPT)
**Time:** ~45 minutes

## Проблема

GPT только для декодера и BERT только для кодера, каждый из которых выделяет архитектуру 2017 года для другой цели.

- Перевод: английский → французский.
- Подведение итогов: 5 000 токенных статей → 200 токенных резюме.
- Распознавание речи: аудиотокены → текстовые токены.
- Структурная экстракция: проза → JSON.

Для этих, кодировщик-декодер делает самый чистый подход. Кодировщик производит плотную представление источника. Декодер генерирует выход, перекрещающийся с этим представлением на каждом шаге.

Две статьи определили современный учебник:

1. **T5**(Raffel et al. 2019). "Трансформатор передачи текста на текст". Каждая задача NLP переоформлена как текст-в, текст-из. Единая архитектура, единый словарный запас, единая потеря. Претренирован на маскированном прогнозировании времени (коррупционные сроки в вводе, декодируйте их в выводе).
2. **BART**(Lewis et al. 2019). "Двусторонний и авторегрессивный трансформатор". Отрицание автокодер: поврежденный вход в нескольких способах (перемешивание, маска, удаление, вращение), попросите декодер восстановить оригинал.

В 2026 году формат кодера-декодера будет жить там, где имеет значение структура ввода:

- Шмыкание (речь → текст).
- Переводный станок Google.
- Некоторые модели пополнения / ремонта кода, которые имеют различные структуры контекста и редактирования.
- Flan-T5 и варианты для задач структурированного рассуждения.

Только декодер выиграл свет, но декодер никогда не ушел.

## Концепция

![Encoder-decoder with cross-attention](../assets/encoder-decoder.svg)

### Передняя петля

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

Ключевое значение имеет то, что кодер работает один раз на вход. Декодер работает авторегрессивно, но на каждом шаге выполняется один и тот же кодерный выход.

### T5 Претренирование  Коррупция в течение срока

Выберите случайные интервалы ввода (средняя длина 3 токенов, 15% в общей сложности).`<extra_id_0>`- Я не знаю .`<extra_id_1>`, и т. Д. Декодер выпускает только поврежденные шины с их предварительным предварительным предварительным:

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

Это более дешевый сигнал, чем предсказание всей последовательности.

### BART предварительное обучение  многошумное отредактирование

BART использует пять звуковых функций:

1. Маскировка токенов.
2. Удаление токенов.
3. Наполнение текста (закрыть протяженность, декодер вставляет правильную длину).
4. Перемена предложения.
5. Документная ротация.

Сочетание текстового наполнения + пермутации предложений привело к наилучшим нижепоточным цифрам. Декодер всегда восстанавливает оригинальный.

### Инференция

То же самое авторегрессивное поколение, что и GPT. Применяется выборка изображений из изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображений изображения изображения изображения изображения изображения изображения изображения изображения изображения изображения изображения изображения изображения изображения изображения изображения изображения изображения изображения изображения изображения изображения изображения изображения изображения изображения изображатого.

### Когда выбрать каждый вариант в 2026 году

| Task | Encoder-decoder? | Why |
|------|------------------|-----|
| Translation | Yes, usually | Clear source sequence; fixed output distribution; beam search works |
| Speech-to-text | Yes (Whisper) | Input modality differs from output; encoder shapes audio features |
| Chat / reasoning | No, decoder-only | No persistent "input" — the conversation is the sequence |
| Code completion | Usually no | Decoder-only with long context wins; code models like Qwen 2.5 Coder are decoder-only |
| Summarization | Either works | BART, PEGASUS beat earlier decoder-only baselines; modern decoder-only LLMs match them |
| Structured extraction | Either | T5 is clean because "text → text" absorbs any output format |

Тенденция с ~2022 года: только декодер берет на себя задачи, которые раньше принадлежали кодеру-декодеру, потому что (а) LLM, настраиваемые только на инструкции, декодеру, обобщаются на что-либо через прописку, (б) одна архитектура масштабируется проще, чем две, (в) RLHF предполагает декодер.

```figure
encoder-decoder
```

## Создайте его

Посмотрите .`code/main.py`Мы реализуем коррупцию с интервалом в стиле T5 для корпуса игрушек, наиболее полезный элемент этого урока, потому что он появляется в каждом рецепте перед тренировкой кодера-декодера с тех пор.

### Шаг 1: коррупция протяженности

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

Целевой формат - конвенция T5: `<sent0> span0 <sent1> span1 ...`. Поврежденный вход перемещает неизменные токены с токены Sentinel в разрывных местах.

### Шаг 2: проверка обратного пути

Учитывая поврежденный вход и цель, восстановить оригинальное предложение. Если ваша коррупция является обратимой, пропуск вперед хорошо определен. Это проверка здравомыслия  реальная подготовка никогда не делает это, но тест дешевый и поймает по одному ошибок в вашем учете времени.

### Шаг 3: БАРТ-шум

Пять функций: `token_mask`- Я не знаю .`token_delete`- Я не знаю .`text_infill`- Я не знаю .`sentence_permute`- Я не знаю .`document_rotate`Составьте два и покажите результат.

## Используйте его

Ссылка на HuggingFace:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

Трик T5: название задачи входит в входный текст. Такой же модель обрабатывает десятки задач, потому что каждая задача - текстовое ввождение, текстовое вывождение.

## Погрузи его

Посмотрите .`outputs/skill-seq2seq-picker.md`. Умение выбирает между кодировщиком-декодировщиком и декодировщиком-только для новой задачи, учитывая структуру ввода-вывода, задержку и цели качества.

## Упражнения

1. **Easy.**Беги .`code/main.py`, применить разрыв интервала на 30-токенное предложение, проверить, что соединение несентителных источников с декодированными целевыми интервалами воспроизводит оригинал.
2. **Medium.**Внедрение программы BART `text_infill`шум: заменить случайные пробелы одним `<mask>`Токен, и декодер должен вывести правильную длину пробега плюс содержание. Покажите один пример.
3. **Hard.**Прекрасно настроенная`flan-t5-small`на крошечном английском → свино-латинском корпусе (200 пар).`Llama-3.2-1B`на одних и тех же данных с той же вычислительной системой.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Encoder-decoder | "Seq2seq transformer" | Two stacks: bidirectional encoder for input, causal decoder with cross-attention for output. |
| Cross-attention | "Where source talks to target" | Decoder's Q × encoder's K/V. The only place encoder information enters the decoder. |
| Span corruption | "T5's pretraining trick" | Replace random spans with sentinel tokens; decoder outputs the spans. |
| Denoising objective | "BART's game" | Apply a noise function to the input, train the decoder to reconstruct the clean sequence. |
| Sentinel token | "The `<extra_id_N>` placeholder" | Special tokens that tag corrupted spans in the source and re-tag them in the target. |
| Flan | "Instruction-tuned T5" | T5 fine-tuned on >1,800 tasks; made encoder-decoder competitive at instruction-following. |
| Beam search | "Decoding strategy" | Keep top-k partial sequences at each step; standard for translation/summarization. |
| Teacher forcing | "Training-time input" | During training, feed the true previous output token to the decoder, not the sampled one. |

## Дальнейшее чтение

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) Т5.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) БАРТ.
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) Флан-Т5.
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) Whisper, канонический кодировщик-декодер 2026 года.
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) Референцированная реализация.
