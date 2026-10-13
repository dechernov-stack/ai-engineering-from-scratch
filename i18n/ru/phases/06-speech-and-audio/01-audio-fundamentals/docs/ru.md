# Основы звука  Волнообразования, образец образец, преобразование Фурье

> Волнообразования - сырой сигнал. Спектрограммы - это представление. Мел-функции - это форма, дружественная к ML. Каждый современный АСР и ТТС-провод проходит по этой лестнице, и первым шагом является понимание выборки образцов и Фурье.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Vectors & Matrices), Phase 1 · 14 (Probability Distributions)
**Time:** ~45 minutes

## Проблема

Микрофон создает сигнал давления против времени. Ваша нейронная сеть потребляет тенсоры. Между ними находится куча конвенций, которые, когда нарушаются, производят тихие ошибки: модель хорошо работает, но WER удвоится, или TTS отправляет сись, или голосовая система клонирования запоминает микрофон вместо динамика.

Каждый бог в речевых системах восходит к одному из трех вопросов:

1. На какой скорости были зафиксированы данные, и чего ожидает модель?
2. Сигнал под псевдонимом?
3. Вы работаете на сырых образцах или на частотном представлении?

Если вы сделаете это правильно, остальная часть фазы 6 будет легко обработать, если вы сделаете это неправильно, даже Whisper-Large-v4 будет производить мусор.

## Концепция

![Waveform, sampling, DFT, and frequency bins visualized](../assets/audio-fundamentals.svg)

**Waveform.**Одномерная массивная сеть плавучих лодок в `[-1.0, 1.0]`Для преобразования в секунды делите по скорости пробки:`t = n / sr`10-секундный клип на 16 кГц - это массив из 160 000 плавущих батарей.

**Sampling rate (sr).**Сколько образцов в секунду.

| Rate | Use |
|------|-----|
| 8 kHz | Telephony, legacy VOIP. Nyquist at 4 kHz kills consonants. Avoid for ASR. |
| 16 kHz | ASR standard. Whisper, Parakeet, SeamlessM4T v2 all consume 16 kHz. |
| 22.05 kHz | TTS vocoder training for older models. |
| 24 kHz | Modern TTS (Kokoro, F5-TTS, xTTS v2). |
| 44.1 kHz | CD audio, music. |
| 48 kHz | Film, pro audio, high-fidelity TTS (VALL-E 2, NaturalSpeech 3). |

**Nyquist-Shannon.**Уровень выборки `sr`может однозначно представлять частоты до `sr/2`- В .`sr/2`При этом, если вы не можете получить энергию, которая не является источником энергии, вы можете получить энергию, которая не является источником энергии, и которая не является источником энергии, вы можете получить энергию, которая не является источником энергии.

**Bit depth.**16-битный PCM (подписанный int16, диапазон ±32,767) - универсальный формат обмена. 24-битный для музыки, 32-битный плавающий для внутреннего DSP.`soundfile`Читайте в 16 , но выставьте массивы float32 в `[-1, 1]`- Я не знаю .

**Fourier Transform.**Любой конечный сигнал - это сумма синусоидов на разных частотах.`N`образцы, `N`сложные коэффициенты  один на частотное колодце. `bin k`карты на частоту `k · sr / N`Гц. Величина - это амплитуда при этой частоте, угол - фаза.

**FFT.**Быстрая преобразование Фурье: `O(N log N)`алгоритм для DFT , когда `N`При этом, в каждой аудиобиблиотеке используется FFT под капотом.

**Framing + window.**Мы не FFT целый клип. Мы разрезаем его на перекрывающиеся *frames* (обычно 25 мс с 10 мс скользком), умножаем каждую картинку функцией окна (Ханн, Хамминг) для устранения несконтезий края, затем FFT каждый кадр. Это Short-Time Fourier Transform (STFT).

```figure
mel-scale
```

## Создайте его

### Шаг 1: прочитайте клип и сделайте схему формы волны

`code/main.py`использует только stdlib `wave`Модуль, чтобы сохранить демо-зависимость.`soundfile`или `torchaudio.load`(оба возвращаются `(waveform, sr)`Тюплей:

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### Шаг 2: синтезировать синусовую волну из первых принципов

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

Синус 440 Гц (концерт А) на 16 кГц в течение 1 секунды составляет 16 000 плавучих.`wave.open(..., "wb")`используя 16-битный кодирование PCM.

### Шаг 3: вычислить DFT вручную

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)` штраф за `N=256`Для подтверждения точности, бесполезный для настоящего аудио.`numpy.fft.rfft`или `torch.fft.rfft`- Я не знаю .

### Шаг 4: Найти доминирующую частоту

Индекс пика величины `k_star`карты на частоту `k_star * sr / N`- Если запустить это на синусе 440 Гц , то вернём пик на бине .`440 * N / sr`- Я не знаю .

### Шаг 5: продемонстрировать алиазинг

Образец синуса 7 кГц при 10 кГц (Найквист = 5 кГц).`10 − 7 = 3 kHz`Это классическая демонстрация прозвища и причина, по которой каждый DAC/ADC отправляется с кирпичным фильтром низкого прохода.

## Используйте его

Склад, который вы отправляете в 2026 году:

| Task | Library | Why |
|------|---------|-----|
| Read/write WAV/FLAC/OGG | `soundfile` (libsndfile wrapper) | Fastest, stable, returns float32. |
| Resample | `torchaudio.transforms.Resample` or `librosa.resample` | Correct anti-aliasing built in. |
| STFT / Mel | `torchaudio` or `librosa` | GPU-friendly; PyTorch ecosystem. |
| Real-time streaming | `sounddevice` or `pyaudio` | Cross-platform PortAudio bindings. |
| Inspect a file | `ffprobe` or `soxi` | CLI, fast, reports sr/channels/codec. |

Правило принятия решений: **match sample rate before you match anything else**Взгляните на стерео с частотой 44,1 кГц, и вы получите мусор, который выглядит как модельная ошибка.

## Погрузи его

Сделайте это .`outputs/skill-audio-loader.md`. Умение помогает проверить, соответствует ли аудио вход ожиданиям модели вдоль потока и правильно повторяет образцы, когда это не так.

## Упражнения

1. **Easy.**Сынтезируйте 1-секундную смесь 220 Гц + 440 Гц + 880 Гц при 16 кГц. Запустите DFT. Подтвердите три пика в ожидаемых контейнерах.
2. **Medium.**Запишите 3-секундный WAV вашего голоса на 48 кГц.`torchaudio.transforms.Resample`(с антиалиазированием), затем до 16 кГц с использованием наивной десяточности (каждый третий образец).
3. **Hard.**Создайте STFT с нуля , используя только `math`И DFT из Шага 3. Размер кадра 400, хоп 160, окно Хан.`matplotlib.pyplot.imshow`Это спектрограмма урока 02.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Sample rate | How many samples per second | Frequency in Hz at which the ADC measures the signal. |
| Nyquist | The max frequency you can represent | `sr/2`; energy above it aliases back down. |
| Bit depth | Resolution of each sample | `int16` = 65,536 levels; `float32` = 24-bit precision in `[-1, 1]`. |
| DFT | The Fourier transform for sequences | `N` samples → `N` complex frequency coefficients. |
| FFT | The fast DFT | `O(N log N)` algorithm requiring `N` = power of 2. |
| Bin | Frequency column | `k · sr / N` Hz; resolution = `sr / N`. |
| STFT | Spectrogram under the hood | Framed + windowed FFT over time. |
| Aliasing | Weird frequency ghosts | Energy above Nyquist mirroring down to lower bins. |

## Дальнейшее чтение

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) документ, стоящий за теоремой выборки образцов.
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) бесплатный, канонический учебник DSP.
- [librosa docs — audio primer](https://librosa.org/doc/latest/auto_tutorials/index.html) практический проход с кодом.
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.taylorfrancis.com/books/mono/10.1201/9781315372150/room-acoustics-heinrich-kuttruff) ссылка на то, почему аудио в реальном мире не является чистым синусоидом.
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/)Интуиция частотного бина устранена за 10 минут.
