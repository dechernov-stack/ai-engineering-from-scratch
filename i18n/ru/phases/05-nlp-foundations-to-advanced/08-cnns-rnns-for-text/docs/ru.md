# CNN и RNN для текста

> Свёртки выучивают n-граммы. Рекурренты запоминают. Оба подхода вытеснены вниманием. Оба всё ещё важны на ограниченном железе.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 11 (PyTorch Intro), Phase 5 · 03 (Word Embeddings), Phase 4 · 02 (Convolutions from Scratch)
**Time:** ~75 минут

## Проблема

TF-IDF и Word2Vec давали плоские векторы, игнорирующие порядок слов. Классификатор, построенный на них, не мог отличить `dog bites man` от `man bites dog`. Порядок слов иногда несёт сигнал.

Два семейства архитектур заполнили этот пробел до появления трансформеров.

**Свёрточные сети для текста (TextCNN).** Применяют одномерные свёртки к последовательностям эмбеддингов слов. Фильтр шириной 3 — это обучаемый детектор триграмм: он охватывает три слова и выдаёт оценку. Собирайте фильтры разной ширины (2, 3, 4, 5), чтобы обнаруживать паттерны разного масштаба. Max-pooling сжимает результат в представление фиксированного размера. Плоско, параллельно, быстро.

**Рекуррентные сети (RNN, LSTM, GRU).** Обрабатывают токены по одному, поддерживая скрытое состояние, которое переносит информацию вперёд. Последовательны, обладают памятью, гибки по длине входа. Доминировали в моделировании последовательностей с 2014 по 2017 год, затем появилось внимание.

Этот урок строит оба подхода, а затем называет сбой, который мотивировал появление внимания.

## Концепция

**TextCNN** (Kim, 2014). Токены превращаются в эмбеддинги. Одномерная свёртка шириной `k` скользит фильтром по последовательным `k`-граммам эмбеддингов, порождая карту признаков. Глобальный max-pooling по этой карте выбирает самую сильную активацию. Конкатенируйте max-pooled выходы от нескольких ширин фильтров. Передайте в классификационную голову.

Почему это работает. Фильтр — это обучаемая n-грамма. Max-pooling инвариантен к позиции, поэтому "not good" активирует один и тот же признак в начале или середине отзыва. Три ширины фильтра по 100 фильтров каждая дают 300 обученных детекторов n-грамм. Обучение параллельное; нет последовательной зависимости.

**RNN.** На каждом шаге времени `t` скрытое состояние `h_t = f(W * x_t + U * h_{t-1} + b)`. Параметры `W`, `U`, `b` разделяются во времени. Скрытое состояние в момент `T` — это сводка всего предыдущего префикса. Для классификации выполняется пулинг по `h_1 ... h_T` (max, mean или last).

Обычные RNN страдают от затухающих градиентов. **LSTM** добавляет гейты, решающие, что забыть, что сохранить и что выдать на выход, стабилизируя градиенты на длинных последовательностях. **GRU** упрощает LSTM до двух гейтов; работает похоже, но с меньшим числом параметров.

**Двунаправленные RNN** запускают одну RNN вперёд и другую назад, конкатенируя скрытые состояния. Представление каждого токена видит контекст и слева, и справа. Критично для задач разметки последовательностей.

```figure
rnn-unroll
```

## Реализация

### Шаг 1: TextCNN на PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` изменяет форму `[batch, seq_len, embed_dim]` на `[batch, embed_dim, seq_len]`, потому что `nn.Conv1d` рассматривает среднюю ось как каналы. Выход пулинга имеет фиксированный размер независимо от длины входа.

### Шаг 2: LSTM-классификатор

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

Используйте max-pooling по последовательности, а не пулинг по последнему состоянию. Для классификации max-pooling обычно превосходит взятие последнего скрытого состояния, потому что информация в конце длинной последовательности имеет тенденцию доминировать над последним состоянием.

### Шаг 3: демонстрация затухающего градиента (интуиция)

Обычная RNN без гейтинга не может выучить дальние зависимости. Рассмотрим игрушечную задачу: предсказать, встретился ли токен `A` где-либо в последовательности. Если `A` находится на позиции 1, а последовательность длиной 100 токенов, градиент от функции потерь должен пройти обратно через 99 умножений рекуррентного веса. Если вес меньше 1, градиент затухает. Если больше 1 — взрывается.

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTM решают эту проблему с помощью **состояния ячейки** (cell state), которое проходит через сеть только с аддитивными взаимодействиями (forget-гейт масштабирует его мультипликативно, но градиенты всё равно текут по этой "магистрали"). GRU делают нечто похожее с меньшим числом параметров. Оба подхода дают стабильное обучение на последовательностях из 100+ шагов.

### Шаг 4: почему этого всё равно было недостаточно

Три проблемы сохранялись даже с LSTM.

1. **Последовательное узкое место.** Обучение RNN на последовательности длиной 1000 требует 1000 последовательных шагов прямого/обратного прохода. Невозможно распараллелить по времени.
2. **Контекстный вектор фиксированного размера в схемах кодировщик-декодировщик.** Декодировщик видит только финальное скрытое состояние кодировщика, сжатое по всему входу. Длинные входы теряют детали. Урок 09 рассматривает это напрямую.
3. **Потолок точности для дальних зависимостей.** LSTM превосходят обычные RNN, но всё равно с трудом переносят конкретную информацию через 200+ шагов.

Внимание решило все три проблемы. Трансформеры полностью отказались от рекуррентности. Урок 10 — это точка перехода.

## Применение

PyTorch-реализации `nn.LSTM`, `nn.GRU` и `nn.Conv1d` готовы к продакшену. Код обучения — стандартный.

Hugging Face поставляет предобученные эмбеддинги, которые вы подключаете как входной слой:

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

Чеклист "используйте, когда подходит под ограничения".

- **Инференс на устройстве / edge.** TextCNN с эмбеддингами GloVe в 10-100 раз меньше трансформера. Если цель развёртывания — телефон, это ваш стек.
- **Потоковая / онлайн-классификация.** RNN обрабатывает по одному токену за раз; трансформерам нужна вся последовательность целиком. Для текста, поступающего в реальном времени, LSTM всё ещё выигрывают.
- **Маленькие модели для базовых решений.** Быстрые итерации на новой задаче. Обучите TextCNN за 5 минут на CPU.
- **Разметка последовательностей при ограниченных данных.** BiLSTM-CRF (урок 06) всё ещё остаётся архитектурой NER промышленного уровня для 1-10 тысяч размеченных предложений.

Всё остальное — к трансформеру.

## Отправка в продакшн

Сохраните как `outputs/prompt-text-encoder-picker.md`:

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## Упражнения

1. **Лёгкое.** Обучите TextCNN на игрушечном датасете из 3 классов (придумайте данные сами). Убедитесь, что ширины фильтров (2, 3, 4) превосходят единственную ширину (3) по среднему F1.
2. **Среднее.** Реализуйте max-pool, mean-pool и пулинг по последнему состоянию для LSTM-классификатора. Сравните на небольшом датасете; задокументируйте, какой пулинг побеждает, и выдвиньте гипотезу почему.
3. **Сложное.** Постройте BiLSTM-CRF теггер NER (объедините урок 06 и этот). Обучите на CoNLL-2003. Сравните с базовым решением на чистом CRF из урока 06 и с дообученным BERT. Отчитайтесь о времени обучения, памяти и F1.

## Ключевые термины

| Термин | Что говорят | Что это на самом деле означает |
|------|-----------------|-----------------------|
| TextCNN | CNN для текста | Стек одномерных свёрток над эмбеддингами слов с глобальным max-pooling. Kim (2014). |
| RNN | Рекуррентная сеть | Скрытое состояние, обновляемое на каждом шаге времени: `h_t = f(W x_t + U h_{t-1})`. |
| LSTM | RNN с гейтами | Добавляет входной / forget / выходной гейты + состояние ячейки. Обучается стабильно на длинных последовательностях. |
| GRU | Упрощённый LSTM | Два гейта вместо трёх. Похожая точность, меньше параметров. |
| Bidirectional | Оба направления | Прямая + обратная RNN, конкатенированные. Каждый токен видит обе стороны своего контекста. |
| Vanishing gradient | Сигнал обучения затухает | Повторное умножение на веса <1 в обычных RNN делает градиенты на ранних шагах практически нулевыми. |

## Дополнительное чтение

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — статья о TextCNN. Восемь страниц. Читается легко.
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — статья о LSTM. Неожиданно ясно изложена.
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — диаграммы, которые сделали LSTM понятными для всех.