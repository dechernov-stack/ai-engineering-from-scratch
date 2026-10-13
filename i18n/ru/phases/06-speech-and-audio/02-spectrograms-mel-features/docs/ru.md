# Спектрограммы, мель-скаль и аудио

> Нейронные сети не потребляют сырые волны хорошо. Они потребляют спектрограммы. Они потребляют мел-спектрограммы еще лучше. Каждый ASR, TTS и аудиоклассификатор в 2026 году живет или умирает от этого единственного предварительного выбора.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 01 (Audio Fundamentals)
**Time:** ~45 minutes

## Проблема

Сделайте 10-секундный клип 16 кГц. Это 160 000 плавущих батарей, все в`[-1, 1]`В сырой форме волны есть информация, но в форме, которую модель не может легко извлечь.

Спектрограмма исправляет это. Она разрушает временную деталь, где человеческое восприятие игнорирует ее (микросекундный джиттер) и сохраняет структуру, где присутствует восприятие (частоты, которые являются энергичными, по времевым окнам ~ 1025 мс).

Мель-спектрограммы продвигаются дальше. Люди воспринимают звук логарифмически: 100 Гц против 200 Гц звучат "в том же расстоянии" как 1000 Гц против 2000 Гц. Мель-спектрограммы искажают частотную ось, чтобы соответствовать. Мель-спектрограммы являются единственной самой важной особенностью речи ML с 2010 по 2026 год.

## Концепция

![Waveform to STFT to mel spectrogram to MFCC ladder](../assets/mel-features.svg)

**STFT (Short-Time Fourier Transform).**Разрезать форму волны на перекрывающиеся кадры (типичный: 25 мс окно, 10 мс хоп = 400 образцов / 160 образцов при 16 кГц). Умножить каждый кадр функцией окна (Ханн является дефолтом; Hamming немного отличается компромиссом). FFT каждый кадр. Складывать спектры величины в матрицу формы `(n_frames, n_freq_bins)`Это твоя спектраграма.

**Log-magnitude.**Существуют величины, которые составляют 5-6 порядков величины.`log(|X| + 1e-6)`или `20 * log10(|X|)`Каждый производственный трубопровод использует величину лога, а не величину сырой.

**Mel scale.**Частота`f`в Hz карты к мелу `m``m = 2595 * log10(1 + f / 700)`. Карта приблизительно линейная ниже 1 кГц и приблизительно логарифмическая выше. 80 мельбин, охватывающих 08 кГц, является стандартным входом ASR.

**Mel filterbank.**Набор треугольных фильтров, размещенных в равной степени на шкале Мел. Каждый фильтр представляет собой взвешенную сумму соседних контейнеров FFT. Умножение величины STFT матрицей фильтровой баки дает спектрограмму Мел в одной мамуле.

**Log-mel spectrogram.** `log(mel_spec + 1e-10)`Вход Whisper, вход Parakeet, вход SeamlessM4T, всеобщий аудио фронтэнд 2026.

**MFCCs.**Возьмите спектрограмму лог-меля, примените DCT (тип II), сохраните первые 13 коэффициентов. Декорелирует функции и дополнительно сжимает. Доминирующая функция до примерно 2015 года, когда CNN / Трансформаторы на сырых лог-мелях были задержаны.

**Resolution trade.**Большая FFT = лучшая частотная разрешение, но худшая временная разрешение. 25 ms / 10 ms является default audio-ML; 50 ms / 12,5 ms для музыки; 5 ms / 2 ms для временного обнаружения (биты барабанов, плосивы).

```figure
spectrogram-window
```

## Создайте его

### Шаг 1: закладка формы волны

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

10-секундный 16 кГц клип с `frame_len=400, hop=160`Вырабатывает 998 кадров.

### Шаг 2: оконное окно

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

Умножение элементов до FFT. Устраняет спектральные утечки, вызванные сокращением на не нулевых конечных точках.

### Шаг 3: величина STFT

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

Использование в производстве `torch.stft`или `librosa.stft`(ФФТ-поддержка, векторизированная) цикл здесь педагогический; он работает на коротких клипах в `code/main.py`- Я не знаю .

### Шаг 4: мел фильтрбанк

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

80 мелс, охватывающие 08 кГц с `n_fft=400`дает `(80, 201)`Матрица.`(n_frames, 201)`Величина STFT по транспозу получается `(n_frames, 80)`Мел-спектрограмм.

### Шаг 5: логи-мель

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

Общие альтернативы: `librosa.power_to_db`(справочно-нормализованный dB), `10 * log10(power + eps)`. Whisper использует более вовлеченный клип + нормализует рутинную (см. Whisper's `log_mel_spectrogram`)).

### Шаг 6: МФК

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

Применяйте DCT на каждую рамку, сохраняйте первые 13 коэффициентов. Это ваша матрица MFCC. Первый коэффициент обычно падает (это кодирует общую энергию).

## Используйте его

Склад 2026 года:

| Task | Features |
|------|----------|
| ASR (Whisper, Parakeet, SeamlessM4T) | 80 log-mels, 10 ms hop, 25 ms window |
| TTS acoustic model (VITS, F5-TTS, Kokoro) | 80 mels, 5–12 ms hop for fine temporal control |
| Audio classification (AST, PANNs, BEATs) | 128 log-mels, 10 ms hop |
| Speaker embedding (ECAPA-TDNN, WavLM) | 80 log-mels or raw-waveform SSL |
| Music (MusicGen, Stable Audio 2) | EnCodec discrete tokens (not mels) |
| Keyword spotting | 40 MFCCs for tiny devices |

Правило: **if you are not working on music, start with 80 log-mels.**Но на всех отклонениях возлагается доказательство.

## Ловушки, которые по-прежнему будут в 2026 году

- **Mel count mismatch.**Тренировка с 80 м, вывод с 128 м. Тихий отказ.
- **Sample-rate mismatch upstream.**Mels, вычисленные на 22,05 кГц, выглядят по-другому, чем 16 кГц.
- **dB vs log.**Виспер ожидает логин-мель, а не дб-мель.
- **Normalization drift.**Нормализация на протяжении тренировок, глобальная нормализация во время выводов.
- **Leakage from padding.**Нулевая подкладка конца клипа создает плоский спектр в задней раме.

## Погрузи его

Сделайте это .`outputs/skill-feature-extractor.md`. Умение выбирает тип характеристик, количество мел, рамки/скок и нормализацию для данной модели цели.

## Упражнения

1. **Easy.**Беги .`code/main.py`. Он синтезирует крип (частотный скачок 200 → 4000 Гц) и печатает аргмакс мел бин на картинку.
2. **Medium.**Повторяйте с `n_mels`В `{40, 80, 128}`и `frame_len`В `{200, 400, 800}`Какая комбинация решает крип лучше?
3. **Hard.**Реализация`power_to_db`и сравнить точность ASR крошечного классификатора CNN на AudioMNIST с использованием (а) сырой логинной мели, (б) dB-мели с `ref=max`, (c) MFCC-13 + delta + delta-delta.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Frame | A slice | 25 ms chunk of waveform fed to one FFT. |
| Hop | Stride | Samples between consecutive frames; 10 ms is ASR default. |
| Window | Hann/Hamming thing | Point-wise multiplier that tapers the frame edges to zero. |
| STFT | Spectrogram generator | Framed + windowed FFT; yields time × frequency matrix. |
| Mel | Warped frequency | Log-perception scale; `m = 2595·log10(1 + f/700)`. |
| Filterbank | The matrix | Triangular filters that project STFT onto mel bins. |
| Log-mel | Whisper's input | `log(mel_spec + eps)`; standardized in 2026. |
| MFCC | Old-school feature | DCT of log-mel; 13 coeffs, decorrelated. |

## Дальнейшее чтение

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) документ МФКЦ.
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) оригинальная мельская шкала.
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) прочитать справочную реализацию.
- [librosa feature extraction docs](https://librosa.org/doc/latest/api/feature.html) ссылка на `mfcc`- Я не знаю .`melspectrogram`, и подскочить / окно.
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) трубопровод в производственной масштабе для моделей Parakeet + Canary.
