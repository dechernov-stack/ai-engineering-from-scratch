# Аудиотрансформаторы  Архитектура шепот

> Аудио - это изображение частоты с течением времени.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 08 (Encoder-Decoder), Phase 7 · 09 (ViT)
**Time:** ~45 minutes

## Проблема

До Whisper (OpenAI, Radford et al. 2022), современное автоматическое распознавание речи (ASR) означало wav2vec 2.0 и HuBERT  самоконтролируемые экстракторы функций плюс тонко настроенная голова.

Уисспер сделал три ставки:

1. **Train on everything.**680.000 часов слабо обозначенного аудио снято с интернета на 97 языках.
2. **Multi-task single model.**Один декодер совместно обучался транскрипции, переводу, обнаружению голосовой активности, идентификации языка и времени через токены задач.
3. **Standard encoder-decoder transformer.**Кодировщик использует спектрограммы логин-меля. Декодер производит текстовые токены авторегрессивно.

Результат: Whisper large-v3 прочно используется в акцентах, шумах и языках с нулевой чистой маркировкой данных. Это является дефолтным голосовым фронт-эндем для каждого голосового помощника с открытым исходным кодом и большинства коммерческих в 2026 году.

## Концепция

![Whisper pipeline: audio → mel → encoder → decoder → text](../assets/whisper.svg)

### Шаг 1  повторный образец + окно

Аудио на 16 кГц. Клип/пад до 30 секунд. Вычислить спектрограмму логин-мель: 80 мель бин, 10 мс шаг → ~ 3000 кадров × 80 функций. Это "входное изображение", которое видит Whisper.

### Шаг 2  сгибальный ствол

Два слоя Conv1D с ядром 3 и шагом 2 уменьшают 3000 кадров до 1500. Уменьшает длину последовательности без добавления много параметров.

### Шаг 3  кодировщик

24-слойный (для больших) трансформаторный кодировщик более 1500 временных этапов. Синусоидальное позиционное кодирование, самоотверженность, GELU FFN. Производит 1500 × 1280 скрытых состояний.

### Шаг 4  декодер

24-слойный декодер трансформатора. Он авторегрессивно производит токены из лексики BPE, которая является суперкомплектом GPT-2 с несколькими специальными токенями для звука.

### Шаг 5  токены задач

Промокт декодера начинается с токенов управления, которые говорят модели, что делать:

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

или

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

Модель была обучена на этой конвенции. Вы управляете задачей по префиксу. эквивалент 2026 инструкции-турнировки, но применяется к речи.

### Шаг 6  выход

Поиск луча (ширина 5) с порогом лог-проб. Временные знаки предсказываются каждые 0,02 секунды звука, когда `<|notimestamps|>`Токен отсутствует.

### Размеры шепот

| Model | Params | Layers | d_model | Heads | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB (4-layer decoder) |

Большой v3-turbo (2024) сократил декодер с 32 слоев до 4,8x быстрее декодирования с регрессией <1 WER-точки.

### Что не делает Шмыкание

- Не нужно вести дневник, чтобы это было возможно.
- Не существует потокового потока в режиме реального времени на родном языке  30-секундное окно фиксировано.`faster-whisper`- Я не знаю .`WhisperX`) сжимание потока через VAD + перекрытие.
- Не требуется длинный контекст за 30 секунд без внешнего разлага.

### 2026 ландшафт

| Task | Model | Notes |
|------|-------|-------|
| English ASR | Whisper-turbo, Moonshine | Moonshine is 4× faster on edge |
| Multilingual ASR | Whisper-large-v3 | 97 languages |
| Streaming ASR | faster-whisper + VAD | 150 ms latency targets achievable |
| TTS | Piper, XTTS-v2, Kokoro | Encoder-decoder pattern, but Whisper-shaped |
| Audio + language | AudioLM, SeamlessM4T | Text tokens + audio tokens in one transformer |

```figure
n5-mel-decode
```

## Создайте его

Посмотрите .`code/main.py`Мы не тренируем Whisper, мы строим цепочку логин-мель + форматист задач-токена.

### Шаг 1: синтезировать аудио

Выделять синусовую волну на 440 Гц, пробивать 16 кгц. 16 000 образцов.

### Шаг 2: спектрограммы с логин-мелем (упрощенный)

Полный спектрограмм мела требует FFT. Мы делаем упрощенное рамочное + энергетическое варианты на рамочный, которое показывает трубопровод без необходимости `librosa`В следующем случае:

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

Рама = 25 мс, хоп = 10 мс. Совпадает с окном Уиспера.

### Шаг 3: подкладка до 30 секунд

Whisper всегда обрабатывает 30-секундные кусочки.

### Шаг 4: создать проскочные токены

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

Это вся поверхность управления задачей.

## Используйте его

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

Быстрее, совместимо с OpenAI:

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**When to pick Whisper in 2026:**

- Многоязычный ASR с одной моделью.
- Устойчивая транскрипция шумного, разнообразного звука.
- Исследование / прототип ASR  самый быстрый исходный пункт.

**When to pick something else:**

- Сверхнизкая задержка потока на краю  Moonshine превосходит Whisper при соответствующем качестве.
- ИИ-сообщения в режиме реального времени, требующий <200 ms  специального потокового ASR.
- Спикерная дневная запись  Шмыкание не делает этого; болт на пианоте.

## Погрузи его

Посмотрите .`outputs/skill-asr-configurator.md`. Умение выбирает модель ASR, параметры декодирования и предварительной обработки для нового речевого приложения.

## Упражнения

1. **Easy.**Беги .`code/main.py`. Подтвердите количество кадров для 1 секунды сигнал на 16 кГц с 10 мс скольжение ~100 кадров.
2. **Medium.**Создайте полный спектрограмм с использованием `numpy.fft`- Проверьте , что 80 мельбанов соответствуют .`librosa.feature.melspectrogram(n_mels=80)`в пределах цифровой ошибки.
3. **Hard.**Используйте потоковую вывод: загрузка звука в окна 10 секунд с переплетом 2 секунд, запускайте Whisper на каждом из них, объединяйте транскрипты. Измерите частоту ошибок слов против однопропуска на 5-минутном примере подкаста.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Mel spectrogram | "Audio image" | 2D representation: frequency bins on one axis, time frames on the other; log-scaled energy per cell. |
| Log-mel | "What Whisper sees" | Mel spectrogram passed through log; approximates human perception of loudness. |
| Frame | "One time slice" | A 25 ms window of samples; overlapping at 10 ms stride. |
| Task token | "Prompt prefix for speech" | Special tokens like `<\|transcribe\|>` / `<\|translate\|>` in the decoder prompt. |
| Voice activity detection (VAD) | "Find the speech" | Gate that removes silence before ASR; cuts cost massively. |
| CTC | "Connectionist Temporal Classification" | Classic ASR loss for alignment-free training; Whisper does NOT use it. |
| Whisper-turbo | "Small decoder, full encoder" | large-v3 encoder + 4-layer decoder; 8× faster decoding. |
| Faster-whisper | "The production wrapper" | CTranslate2 reimplementation; int8 quantization; 4× faster than OpenAI's reference. |

## Дальнейшее чтение

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) Шмыкание бумаги.
- [OpenAI Whisper repo](https://github.com/openai/whisper) Справочный код + вес модели.`whisper/model.py`чтобы увидеть ствол Conv1D + кодировщик + декодер сверху наниз в ~ 400 строках.
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) логика "поиск луча" + "токен задачи", описанная в шагах 56, представлена здесь; 500 строков, полностью читаемый.
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) прекурсор; все еще содержит функции SOTA в некоторых настройках.
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) производственная упаковка, 4x быстрее, чем эталон.
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) 2024 ASR, удобный к краям, в форме Шмыкания, но меньший.
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) канонический рецепт настройки, включающий в себя предпроцессор спектраграммы мел и обработку символов-часовых печат.
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) полная реализация (кодер, декодер, перекрестное внимание, генерация), отражающая схему архитектуры урока.
