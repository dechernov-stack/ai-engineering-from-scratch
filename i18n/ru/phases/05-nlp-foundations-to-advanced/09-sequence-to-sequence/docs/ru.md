# Последовательность-в-последовательность (Seq2Seq)

> Две RNN, притворяющиеся переводчиком. Узкое место, в которое они упираются, — причина, по которой существует внимание (attention).

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 08 (CNNs + RNNs for Text), Phase 3 · 11 (PyTorch Intro)
**Time:** ~75 минут

## Проблема

Классификация отображает последовательность переменной длины в одну метку. Перевод отображает последовательность переменной длины в другую последовательность переменной длины. Вход и выход живут в разных словарях, возможно, на разных языках, без гарантии совпадения длин.

Архитектура seq2seq (Sutskever, Vinyals, Le, 2014) решила эту задачу намеренно простым рецептом. Две RNN. Одна читает исходное предложение и создаёт вектор контекста фиксированного размера. Другая читает этот вектор и генерирует целевое предложение токен за токеном. Тот же код, что вы писали в уроке 08, только склеенный иначе.

Это стоит изучить по двум причинам. Во-первых, узкое место вектора контекста — самый поучительный провал в NLP. Он мотивирует всё, в чём хороши attention и трансформеры. Во-вторых, рецепт обучения (teacher forcing, scheduled sampling, beam search на инференсе) всё ещё применим к любой современной системе генерации, включая LLM.

## Концепция

**Энкодер.** RNN, которая читает исходное предложение. Её финальное скрытое состояние — это **вектор контекста**: сводка всего входа фиксированного размера. Теоретически не теряется ничего, кроме исходного текста.

**Декодер.** Другая RNN, инициализированная вектором контекста. На каждом шаге она принимает ранее сгенерированный токен в качестве входа и выдаёт распределение по целевому словарю. Сэмплируем или берём argmax, чтобы выбрать следующий токен. Подаём его обратно на вход. Повторяем, пока не будет сгенерирован токен `<EOS>` или не достигнута максимальная длина.

**Обучение:** кросс-энтропийная потеря на каждом шаге декодера, суммированная по всей последовательности. Стандартный backprop through time через обе сети.

**Teacher forcing.** Во время обучения входом декодера на шаге `t` является *истинный* токен на позиции `t-1`, а не собственное предыдущее предсказание декодера. Это стабилизирует обучение; без этого ранние ошибки каскадно накапливаются, и модель никогда не обучается. На инференсе приходится использовать собственные предсказания модели, поэтому всегда есть разрыв между распределениями обучения и инференса. Этот разрыв называется **exposure bias** (смещение из-за экспозиции).

**Узкое место.** Всё, что энкодер узнал об исходном тексте, должно быть сжато в этот единственный вектор контекста. Длинные предложения теряют детали. Редкие слова размываются. Перестановку порядка слов (chat noir против black cat) приходится запоминать, а не вычислять.

Attention (урок 10) решает это, позволяя декодеру смотреть на *все* скрытые состояния энкодера, а не только на последнее. В этом вся суть.

```figure
lstm-gates
```

## Реализация

### Шаг 1: энкодер

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` имеет форму `[batch, seq_len, hidden_dim]` — одно скрытое состояние на каждую входную позицию. `hidden` имеет форму `[1, batch, hidden_dim]` — финальный шаг. В уроке 08 говорилось «пулинг по выходам для классификации». Здесь мы сохраняем последнее скрытое состояние как вектор контекста и игнорируем выходы на каждом шаге.

### Шаг 2: декодер

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

Декодер вызывается по одному шагу за раз. Вход: батч отдельных токенов и текущее скрытое состояние. Выход: логиты словаря для следующего токена и обновлённое скрытое состояние.

### Шаг 3: цикл обучения с teacher forcing

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

Стоит назвать два параметра. `ignore_index=0` пропускает потерю на токенах паддинга. `teacher_forcing_ratio` — это вероятность использования истинного токена вместо предсказания модели на каждом шаге. Начните с 1.0 (полный teacher forcing) и постепенно снижайте примерно до 0.5 в ходе обучения, чтобы сократить разрыв exposure bias.

### Шаг 4: цикл инференса (жадный)

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

Жадное декодирование выбирает токен с наибольшей вероятностью на каждом шаге. Оно может сбиться с пути: как только вы зафиксировали токен, отменить это нельзя. **Beam search** удерживает топ-`k` частичных последовательностей и в конце выбирает полную последовательность с наилучшей оценкой. Ширина луча 3-5 — стандартное значение.

### Шаг 5: демонстрация узкого места

Обучите модель на игрушечной задаче копирования: источник `[a, b, c, d, e]`, цель `[a, b, c, d, e]`. Увеличивайте длину последовательности. Наблюдайте за точностью.

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

Одно скрытое состояние GRU не может безошибочно запомнить вход из 40 токенов. Информация присутствует на каждом шаге энкодера, но декодер видит только последнее состояние. Attention решает это напрямую.

## Применение

В PyTorch есть шаблоны seq2seq на основе `nn.Transformer` и `nn.LSTM`. Библиотека Hugging Face `transformers` поставляет полноценные энкодер-декодерные модели (BART, T5, mBART, NLLB), обученные на миллиардах токенов.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

Современные энкодер-декодеры отказались от RNN в пользу трансформеров. Высокоуровневая форма (энкодер, декодер, генерация токен за токеном) идентична статье seq2seq 2014 года. Механизм внутри каждого блока — другой.

### Когда всё ещё стоит использовать seq2seq на основе RNN

Почти никогда для новых проектов. Конкретные исключения:

- Потоковый перевод, где вход потребляется по одному токену с ограниченной памятью.
- Генерация текста на устройстве, где затраты памяти трансформера неприемлемы.
- Педагогика. Понимание узкого места энкодер-декодера — самый быстрый путь к пониманию того, почему победили трансформеры.

### Exposure bias и способы его смягчения

- **Scheduled sampling.** Постепенно снижайте коэффициент teacher forcing в ходе обучения, чтобы модель училась восстанавливаться после собственных ошибок.
- **Minimum risk training.** Обучение на оценке BLEU на уровне предложения вместо кросс-энтропии на уровне токенов. Ближе к тому, чего вы на самом деле хотите.
- **Дообучение с помощью обучения с подкреплением.** Вознаграждайте генератор последовательностей по метрике. Используется в современном RLHF для LLM.

Все три подхода по-прежнему применимы к генерации на основе трансформеров.

## Сдача работы

Сохраните как `outputs/prompt-seq2seq-design.md`:

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## Упражнения

1. **Лёгкое.** Реализуйте игрушечную задачу копирования. Обучите GRU seq2seq на парах вход-выход, где цель равна источнику. Измерьте точность на длинах 5, 10, 20. Воспроизведите узкое место.
2. **Среднее.** Добавьте декодирование beam search с шириной луча 3. Измерьте BLEU на небольшом параллельном корпусе в сравнении с жадным декодированием. Задокументируйте, где beam search выигрывает (обычно на последних токенах), а где не даёт разницы.
3. **Сложное.** Дообучите `facebook/bart-base` на наборе данных из 10 тыс. пар перефразирований. Сравните вывод дообученной модели с beam-4 с выводом базовой модели на отложенных входных данных. Приведите BLEU и выберите 10 качественных примеров.

## Ключевые термины

| Термин | Что говорят люди | Что это на самом деле означает |
|------|-----------------|-----------------------|
| Энкодер | Входная RNN | Читает источник. Производит скрытые состояния на каждом шаге и финальный вектор контекста. |
| Декодер | Выходная RNN | Инициализируется вектором контекста. Генерирует целевые токены по одному. |
| Вектор контекста | Сводка | Финальное скрытое состояние энкодера. Фиксированный размер. Узкое место, которое решает attention. |
| Teacher forcing | Использование истинных токенов | Подача истинного предыдущего токена во время обучения. Стабилизирует обучение. |
| Exposure bias | Разрыв обучение/тест | Модель, обученная на истинных токенах, никогда не практиковалась в восстановлении после собственных ошибок. |
| Beam search | Лучшее декодирование | Удержание топ-k частичных последовательностей на каждом шаге вместо жадной фиксации. |

## Дополнительное чтение

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — оригинальная статья seq2seq. Четыре страницы.
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — представила GRU и концепцию энкодер-декодера.
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — статья об attention. Читайте сразу после этого урока.
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — код seq2seq + attention, который можно собрать самому.