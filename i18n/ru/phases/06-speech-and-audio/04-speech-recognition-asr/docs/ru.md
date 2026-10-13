# Признание речи (ASR)  CTC, RNN-T, внимание

> Распознавание речи - это классификация звука на каждом этапе времени, склеенная вместе с помощью модели последовательности, которая знает английский и тишину. CTC, RNN-T и внимание - это три способа сделать это. Выберите один и поймите, почему.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 08 (CNNs & RNNs for Text), Phase 5 · 10 (Attention)
**Time:** ~45 minutes

## Проблема

У вас есть 10-секундный клип 16 кГц. Вы хотите строку: "заключите кухню". Проблема структурная: аудиокадры не выровняются с символами. Слово "okay" может занять 200 мс или 1200 мс. Тишина пунктует произношение. Некоторые фонемы длиннее других. Количество выходной токены неизвестно заранее.

Три формулировки решают это:

1. **CTC (Connectionist Temporal Classification).**Выделять вероятности токенов на кадры, включая специальный *побезопасность*. Колапс повторяется и пустые в момент декодирования. Неавторегрессивный, быстрый. Используется wav2vec 2.0, MMS.
2. **RNN-T (Recurrent Neural Network Transducer).**Совместная сеть предсказывает следующий токен, данный кодерной рамкой и предыдущими токенами.
3. **Attention encoder-decoder.**Кодёр сжимает аудио в скрытые состояния, декодер перекрестно обслуживает для генерации токенов авторегрессивно.

В 2026 году SOTA WER на книспеч-тест-чистоте составляет 1,4% (Parakeet-TDT-1.1B, NVIDIA) и 1,58% (Whisper-Large-v3-turbo).

## Концепция

![Three ASR formulations: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**CTC intuition.**Пусть кодер выйдет `T`распределение на уровне рамы по `V+1`токены (V символы + пустое). Для целевой строки `y`длины `U < T`, любой рамовой выровненности , которая рухнет на `y`Счет. СТК потеря суммы на все такие выровнения. Инференция: argmax на кадр, повторение коллапса, удаление пустоты.

Преимущества: неавторегресивное, потоковое, нулевое зрение. Недостатки: *условное предположение независимости*  каждое предсказание кадров независимо от других, поэтому нет внутренней языковой модели.

**RNN-T intuition.**Добавляет сеть * предсказатель * , которая внедряет историю токенов и * соединитель * , который сочетает состояние предсказателя с рамкой кодера в совместное распределение `V+1`(в `+1`CTC игнорирует условную зависимость. Стримеваемая, потому что каждый шаг обусловлен только прошлыми рамками и прошлыми токеном.

Преимущества: потоковое + внутреннее LM. Недостатки: обучение более сложно и голодно памятью (3D-счет потерь); ядра потерь RNN-T являются целым категориями библиотеки само по себе.

**Attention encoder-decoder.**Кодёр (6-32 слоя трансформатора) над рамками логин-меля. Декодер (6-32 слоя трансформатора) перекрестно обслуживает кодирующие выходы для генерации токенов авторегрессивно.

Преимущества: высокое качество на офлайн ASR, легко обучаться с использованием стандартных инструментов seq2seq. Недостатки: авторегрессивная задержка пропорциональна длине выхода; не может транслироваться без инженерии.

### WER: один номер

**Word Error Rate**= `(S + D + I) / N`, где S=замещения, D=удаления, I=вставки, N=счет ссылочных слов. Совпадает с расстоянием редактирования Левенштейна на уровне слов. Ниже лучше. WER выше 20% обычно не используется; ниже 5% - человеческое равновесие для чтения речи.

| Model | LibriSpeech test-clean | LibriSpeech test-other | Size |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B params |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

Все эти системы основаны на кодировании-декодере или RNN-T. Чистые системы CTC (wav2vec 2.0) находятся примерно на 1,82,1% на испытательном уровне.

```figure
ctc-collapse
```

## Создайте его

### Шаг 1: жадный декод CTC

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

Два правила: разваливание последовательных повторений, отпуск пустых.`a a _ _ a b b _ c`→ `a a b c`- Я не знаю .

### Шаг 2: СТК по поиску лучей

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

Производство использует поиск префикса деревянного луча с LM-фьюзией; это концептуальный скелет.

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

### Шаг 4: вывод против Шпылка

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

Однолинейный для сильнейшего общего ASR в 2026 году.

### Шаг 5: потоковый режим с Parakeet или wav2vec 2.0

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

Для потокового трансляции ASR требуется внимание кодера и состояние перевозки; используйте библиотеку, которая поддерживает его (NeMo для Parakeet, `transformers`трубопровода с `chunk_length_s`)).

## Используйте его

Склад 2026 года:

| Situation | Pick |
|-----------|------|
| English, offline, max quality | Whisper-large-v3-turbo |
| Multilingual, robust | SeamlessM4T v2 |
| Streaming, low latency | Parakeet-TDT-1.1B or Riva |
| Edge, mobile, <500 ms latency | Whisper-Tiny quantized or Moonshine (2024) |
| Long-form | Whisper with VAD-based chunking (WhisperX) |
| Domain-specific (medical, legal) | Fine-tune wav2vec 2.0 + domain LM fusion |

## Ловушки, которые по-прежнему будут в 2026 году

- **No VAD.**Бег Взвинитель на молчание производит галлюцинации ("Спасибо за наблюдение!").
- **Character vs word vs subword WER.**Отчет о WER на уровне слов *после* нормализации (нижепись, пунктуация снята).
- **Language ID drift.**Авто-ЛИД Whisper неправильно направляет шумные клипы на японский или валлийский язык; сила `language="en"`Когда ты знаешь.
- **Long clips without chunking.**У Whisper есть 30-секундное окно.`chunk_length_s=30, stride=5`на что угодно дольше.

## Погрузи его

Сделайте это .`outputs/skill-asr-picker.md`- Выберите модель, стратегию декодирования, разбивку и LM-фьюзию для конкретной цели развертывания.

## Упражнения

1. **Easy.**Беги .`code/main.py`- Он жадно декодирует ручной CTC выход и вычисляет WER с точки зрения ссылки.
2. **Medium.**Правильно выполните поиск предварительного ветка в шаге 2 (с учетом правила пустого слияния).
3. **Hard.**Использование`whisper-large-v3-turbo`На [LibriSpeech test-clean](https://www.openslr.org/12)- Вычислить WER на первых 100 высказываний.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| CTC | The blank-token loss | Marginal over all frame-to-token alignments; non-AR. |
| RNN-T | The streaming loss | CTC + next-token predictor; handles word-order. |
| Attention enc-dec | Whisper-style | Encoder + cross-attending decoder; best offline quality. |
| WER | The number you report | `(S+D+I)/N` at word level. |
| Blank | The emptiness | Special token in CTC signalling "no emission this frame". |
| LM fusion | External language model | Add weighted LM log-probs during beam search. |
| VAD | The silence gate | Voice activity detector; trims non-speech. |

## Дальнейшее чтение

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) бумаги CTC.
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) бумаги RNN-T.
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) каноническая газета 2022 года; v3-турбо-расширение в 2024 году.
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) 2026 Лидер открытого рейтингового рейтинга ASR.
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) Live Benchmark по более чем 25 моделям.
