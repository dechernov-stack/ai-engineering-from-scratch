# Почему трансформаторы  Проблемы с РНН

> RNN обрабатывает токены один за другим. Трансформаторы обрабатывают все токены сразу. Эта единая архитектурная ставка изменила каждую кривую масштабирования в глубоком обучении после 2017 года.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 3 (Deep Learning Core), Phase 5 · 09 (Sequence-to-Sequence), Phase 5 · 10 (Attention Mechanism)
**Time:** ~45 minutes

## Проблема

До 2017 года каждая современная модель последовательности на планете  язык, перевод, речь  была повторяющейся нейронной сетью. LSTM и GRU получили эквивалентные переводным стандартам ImageNet в течение полудесяти лет. Они были единственным инструментом, который у кого-либо был.

У них было три фатальных слабости.`t+1`нуждается в скрытом состоянии от символа `t`. 1,024-токенная последовательность означает 1,024 серийных шагов на GPU, которые могут выполнять 1,000,000 плавающих точек операций в цикл.

Избыточные градиенты означали, что информация 50 токенов назад уже сжималась через 50 нелинейных. Загруженные повторяющиеся единицы (LSTM, GRU) смягчили схватку, но никогда не устранили ее.

Фиксированная ширина скрытых состояний означает, что кодировщик сжимает всю исходную последовательность в один вектор, прежде чем декодер видит что-либо.

В статье 2017 года "Уверенность - это все, что вам нужно" было предложено что-то радикальное: полностью отбросить рецидив. Пусть каждая позиция будет прислушиваться к каждой другой позиции параллельно.

Результат доминирует во всех формах к 2026 году. Язык (GPT-5, Claude 4, Llama 4), видение (ViT, DINOv2, SAM 3), аудио (Whisper), биология (AlphaFold 3), робототехника (RT-2).

## Концепция

![RNN sequential compute vs Transformer parallel attention](../assets/rnn-vs-transformer.svg)

**Recurrence as a bottleneck.**RNN вычисляет `h_t = f(h_{t-1}, x_t)`Каждый шаг зависит от предыдущего.`h_5`раньше`h_4`На современных графических процессорах с более чем 10 000 параллельными ядрами, это тратит 99% кремния на длинную последовательность.

**Attention as a broadcast.**Самовнимание - это вычислитель .`output_i = sum_j(a_ij * v_j)`за каждую пару`(i, j)`И это не значит, что мы должны быть в состоянии делать что-то другое.

**The speedup is not a constant.**Это разница между `O(N)`серийная глубина и `O(1)`На практике трансформаторы тренируются 510x быстрее на эпоху на совпадающем оборудовании на N=512, и разрыв расширяется с длиной последовательности, пока вы не достигнете `O(N²)`память стенки внимания (которая Flash Attention позже исправила  см. Урок 12).

**What transformers cost.**Степень памяти внимания:`O(N²)`Для контекста 2K, хорошо. Для контекста 128K, вам нужно скользящие окна, экстраполяция RoPE, пластинка Flash Attention или линейные варианты внимания.`O(N)`как в времени, так и в памяти; трансформаторы обмениваются временем на память, а затем возвращают время через параллелизм.

**The inductive bias shift.**РНН предполагают локальность и скорость. Трансформаторы ничего не предполагают.  каждая пара является кандидатом на внимание. Вот почему трансформаторы нуждаются в большем количестве данных для хорошей подготовки, но масштабируются дальше, как только они получат его. Чинчилла (2022) формализовал это: при наличии достаточно токенов, трансформатор всегда побеждает RNN равного количества параметров.

```figure
rnn-vs-parallel
```

## Создайте его

Здесь нет нейронной сети. Мы смоделируем ключевой горшок, чтобы вы почувствовали разрыв на ноутбуке.

### Шаг 1: измерять серийную глубину

Посмотрите .`code/main.py`Мы создаем две функции. Одна кодирует последовательность как цепочка дополнений (сериальный, как RNN).

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

Мы делаем обоих последовательностей до 100 000 элементов. Версия RNN - O(N) и один процессорный трубопровод. Даже в чистом Python, уменьшение внимания стиля превзойдет его в длину ≥ 1000, потому что Python `sum()`выполняется в C и повторяется без переводчика за каждой ступени.

### Шаг 2: подсчитывать теоретические операции

В результате, в процессе сбора данных, используемых в процессе сбора данных, используется диапазон диапазона, который определяется в процессе сбора данных.

### Шаг 3: эмпирическое масштабирование на длинных последовательностях

Мы печатаем таблицу времени, которая делает разрыв O(N) видимым. На ноутбуке Mac 2026 последовательности под 1000 элементов слишком быстрые для измерения. Последовательности 100 000 показывают чистый линейный сканирование.

## Используйте его

Когда еще выбрать RNN в 2026 году:

| Situation | Pick |
|-----------|------|
| Streaming inference, one token at a time, constant memory | RNN or state-space model (Mamba, RWKV) |
| Very long sequences (>1M tokens) where attention memory explodes | Linear attention, Mamba 2, Hyena |
| Edge device with no matmul accelerator | Depthwise-separable RNN still wins on FLOPs/watt |
| Anything else (training, batched inference, context up to 128K) | Transformer |

Модели государственного пространства (ММОС), такие как Mamba, по существу являются RNNs с структурированной параметризацией , которая дает им лучшее из обоих: `O(N)`В 2026 году большинство лабораторий пограничных стран обучают гибридные модели трансформаторов SSM+ (например, Jamba, Samba)  рецидивы не мертвы, это составляет компонент.

## Погрузи его

Посмотрите .`outputs/skill-architecture-picker.md`. Умение выбирает архитектуру для новой проблемы последовательности с учетом длины, пропускной способности и ограничений на бюджет обучения.

## Упражнения

1. **Easy.**Возьми .`rnn_style`от `code/main.py`И заменять скалярное скрытое состояние на вектор длины-64 скрытых состояний.
2. **Medium.**Используйте параллельную префикс-суму (сканирование Хиллиса-Стиле) в чистом Python.
3. **Hard.**Приведите снижение внимания на PyTorch на GPU. Время, как вы протираете длину последовательности от 64 до 65 536.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Recurrence | "RNNs are sequential" | Computation where step `t` depends on step `t-1`, forcing serial execution along the time axis. |
| Serial depth | "How deep the graph is" | Longest chain of dependent ops; bounds wall-clock even on infinite hardware. |
| Attention | "Let tokens look at each other" | Weighted sum `sum_j a_ij v_j` where `a_ij` comes from a similarity score between positions i and j. |
| Context window | "How much the model sees" | Number of positions an attention layer can take as input; quadratic memory cost scales here. |
| Inductive bias | "Assumptions baked into the architecture" | Prior about what the data looks like; CNNs assume translation invariance, RNNs assume recency. |
| State-space model | "RNN with algebra behind it" | Recurrence parameterized for parallel training via structured state-space matrices. |
| Quadratic bottleneck | "Why context costs so much" | Attention memory = `O(N²)` in sequence length; Flash Attention hides the constants, not the scaling. |

## Дальнейшее чтение

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) статья, которая уничтожила рецидивы в основном НЛП.
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) где родилось внимание, завязанное на RNN.
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) оригинальная газета LSTM, для протокола.
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) современный ответ на трансформаторы.
