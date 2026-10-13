# Распознавание речи (ASR) — CTC, RNN-T, Attention

> Распознавание речи — это классификация аудио на каждом временном шаге, склеенная воедино секвенциальной моделью, которая знает английский и тишину. CTC, RNN-T и attention — три способа это сделать. Выберите один и поймите почему.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Спектрограммы и Mel), Phase 5 · 08 (CNN и RNN для текста), Phase 5 · 10 (Attention)
**Time:** ~45 минут

## Проблема

У вас есть 10-секундный клип с частотой 16 кГц. Вы хотите получить строку: «turn on the kitchen lights». Проблема структурная: аудиофреймы не соответствуют символам один к одному. Слово «okay» может занимать 200 мс или 1200 мс. Тишина пунктуирует высказывание. Одни фонемы длиннее других. Количество выходных токенов заранее неизвестно.

Эти три формулировки решают проблему:

1. **CTC (Connectionist Temporal Classification).** Выдавать вероятности токенов для каждого фрейма, включая специальный *blank* (пустой токен). При декодировании схлопывать повторы и пустые токены. Неавторегрессионный, быстрый. Используется в wav2vec 2.0, MMS.
2. **RNN-T (Recurrent Neural Network Transducer).** Совместная сеть предсказывает следующий токен на основе фрейма энкодера и предыдущих токенов. Поддерживает потоковую обработку. Используется в ASR Google на устройстве, NVIDIA Parakeet.
3. **Attention encoder-decoder.** Энкодер сжимает аудио в скрытые состояния, декодер выполняет cross-attention для авторегрессионной генерации токенов. Используется в Whisper, SeamlessM4T.

В 2026 году SOTA WER на LibriSpeech test-clean составляет 1,4% (Parakeet-TDT-1.1B, NVIDIA) и 1,58% (Whisper-Large-v3-turbo). Различия крошечные; различия в развёртывании — огромные.

## Концепция

![Three ASR formulations: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**Интуиция CTC.** Пусть энкодер выдаёт `T` распределений на уровне фреймов по `V+1` токенам (V символов + blank). Для целевой строки `y` длины `U < T` учитывается любое выравнивание по фреймам, которое схлопывается до `y`. Функция потерь CTC суммирует по всем таким выравниваниям. Инференс: argmax для каждого фрейма, схлопывание повторов, удаление пустых токенов.

Преимущества: неавторегрессионный, потоковый, нулевой lookahead. Недостаток: *предположение об условной независимости* — каждое предсказание фрейма независимо от остальных, поэтому отсутствует внутренняя языковая модель. Исправляется внешней LM через beam search или shallow fusion.

**Интуиция RNN-T.** Добавляет сеть-*предиктор*, которая встраивает историю токенов, и *joiner*, который объединяет состояние предиктора с фреймом энкодера в совместное распределение по `V+1` (`+1` — это null / отсутствие эмиссии). Явно моделирует условную зависимость, которую CTC игнорирует. Потоковый, потому что каждый шаг обусловлен только прошлыми фреймами и прошлыми токенами.

Преимущества: потоковость + внутренняя LM. Недостаток: обучение сложнее и требует больше памяти (3D-решётка потерь); ядра для вычисления RNN-T loss — это целая отдельная категория библиотек.

**Attention encoder-decoder.** Энкодер (6-32 слоя трансформера) обрабатывает log-mel фреймы. Декодер (6-32 слоя трансформера) выполняет cross-attention к выходам энкодера для авторегрессионной генерации токенов. Нет ограничения на выравнивание — attention может смотреть в любое место аудио. Непотоковый, если не ограничить attention (chunked Whisper-Streaming, 2024).

Преимущества: наивысшее качество на офлайн-ASR, легко обучать стандартными seq2seq-инструментами. Недостаток: авторегрессионная задержка пропорциональна длине выхода; без дополнительной инженерии потоковость невозможна.

### WER: главное число

**Word Error Rate** = `(S + D + I) / N`, где S=замены, D=удаления, I=вставки, N=количество слов в эталоне. Соответствует расстоянию Левенштейна на уровне слов. Чем меньше, тем лучше. WER выше 20% обычно непригоден к использованию; ниже 5% — паритет с человеком для чтения вслух. Цифры 2026 года на стандартных бенчмарках:

| Модель | LibriSpeech test-clean | LibriSpeech test-other | Размер |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B параметров |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

Все они основаны на encoder-decoder или RNN-T. Системы с чистым CTC (wav2vec 2.0) показывают около 1,8–2,1% на test-clean.

```figure
ctc-collapse
```

## Постройте это

### Шаг 1: жадное декодирование CTC

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

Два правила: схлопывать последовательные повторы, отбрасывать пустые токены. Пример: `a a _ _ a b b _ c` → `a a b c`.

### Шаг 2: beam-search для CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

В продакшене используется beam search по префиксному дереву с LM fusion; это концептуальный скелет.

### Шаг 3: WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### Шаг 4: инференс против Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

Однострочник для сильнейшего общего ASR в 2026 году. Работает на GPU с 24 ГБ со скоростью ~20× реального времени.

### Шаг 5: потоковая обработка с Parakeet или wav2vec 2.0

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

Потоковому ASR нужна внимание энкодера по блокам (chunked attention) и перенос состояния; используйте библиотеку, которая это поддерживает (NeMo для Parakeet, `transformers` pipeline с `chunk_length_s`).

## Используйте это

Стек 2026 года:

| Ситуация | Выбор |
|-----------|------|
| Английский, офлайн, максимум качества | Whisper-large-v3-turbo |
| Многоязычный, устойчивый | SeamlessM4T v2 |
| Потоковый, низкая задержка | Parakeet-TDT-1.1B или Riva |
| Edge, мобильные устройства, <500 мс задержка | Whisper-Tiny квантованный или Moonshine (2024) |
| Длинные записи | Whisper с чанкингом на основе VAD (WhisperX) |
| Специфическая область (медицина, юриспруденция) | Файнтюн wav2vec 2.0 + LM fusion для домена |

## Подводные камни, которые всё ещё встречаются в 2026 году

- **Отсутствие VAD.** Запуск Whisper на тишине приводит к галлюцинациям («Thanks for watching!»). Всегда фильтруйте с помощью VAD.
- **WER по символам vs словам vs подсловам.** Сообщайте WER на уровне слов *после* нормализации (нижний регистр, удалённая пунктуация).
- **Дрейф определения языка.** Авто-LID Whisper ошибочно маршрутизирует зашумлённые клипы в японский или валлийский; принудительно задавайте `language="en"`, если язык известен.
- **Длинные клипы без чанкинга.** У Whisper окно 30 секунд. Используйте `chunk_length_s=30, stride=5` для всего, что длиннее.

## Воплотите это

Сохраните как `outputs/skill-asr-picker.md`. Выберите модель, стратегию декодирования, чанкинг и LM fusion для конкретной цели развёртывания.

## Упражнения

1. **Лёгкое.** Запустите `code/main.py`. Это жадно декодирует вручную составленный вывод CTC и вычисляет WER относительно эталона.
2. **Среднее.** Реализуйте beam search по префиксному дереву из Шага 2 как следует (учтите правило слияния пустых токенов). Сравните с жадным декодированием на синтетическом датасете из 10 примеров.
3. **Сложное.** Используйте `whisper-large-v3-turbo` на [LibriSpeech test-clean](https://www.openslr.org/12). Вычислите WER на первых 100 высказываниях. Сравните с опубликованными цифрами.

## Ключевые термины

| Термин | Что говорят люди | Что это на самом деле означает |
|------|-----------------|-----------------------|
| CTC | Loss на основе пустого токена | Маргинализация по всем выравниваниям фреймов к токенам; неавторегрессионный. |
| RNN-T | Потоковый loss | CTC + предиктор следующего токена; учитывает порядок слов. |
| Attention enc-dec | В стиле Whisper | Энкодер + декодер с cross-attention; лучшее офлайн-качество. |
| WER | Число, которое вы приводите | `(S+D+I)/N` на уровне слов. |
| Blank | Пустота | Специальный токен в CTC, сигнализирующий «нет эмиссии на этом фрейме». |
| LM fusion | Внешняя языковая модель | Добавление взвешенных лог-вероятностей LM во время beam search. |
| VAD | Фильтр тишины | Детектор голосовой активности; обрезает не-речевые участки. |

## Дополнительное чтение

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — статья о CTC.
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — статья о RNN-T.
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — каноническая статья 2022 года; расширение v3-turbo в 2024.
- [NVIDIA NeMo — карточка Parakeet-TDT](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — лидер Open ASR Leaderboard 2026 года.
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — живой бенчмарк по более чем 25 моделям.