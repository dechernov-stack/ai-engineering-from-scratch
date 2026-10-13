# Текст-то-говой (ТТС)  От Такотрона до F5 и Кокоро

> ASR переворачивает речь в текст; TTS переворачивает текст в речь. Склад 2026 состоит из трех частей: текст → токены, токены → mel, mel → форма волны.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 09 (Seq2Seq), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## Проблема

У вас есть строка: "Пожалуйста, напомни мне, чтобы поливать растения в 6 вечера". Вам нужен 3-секундный аудиоклип, который звучит естественно, имеет правильную прозодию (паузы, стресс), произносит "растения" правильным гласным, и работает менее чем на 300 мс на процессоре для живого голосового помощника.

Современные трубопроводы ТТС выглядят вот так:

1. **Text frontend.**Нормализуйте текст (данные, номера, электронные письма), преобразуйте в фонемы или токены подсловий, предскажите функции прозодии.
2. **Acoustic model.**Текст → мель спектрограммы. Такотрон 2 (2017), Быстрый разговор 2 (2020), VITS (2021), F5-TTS (2024), Кокоро (2024).
3. **Vocoder.**Mel → форма волны. WaveNet (2016), WaveRNN, HiFi-GAN (2020), BigVGAN (2022), нейронные кодеки в 2024+

В 2026 году акустический + вокодер размывается с концовой диффузией и моделями соответствия потока.

## Концепция

![Tacotron, FastSpeech, VITS, F5/Kokoro side-by-side](../assets/tts.svg)

**Tacotron 2 (2017).**Seq2seq: char-embedding → BiLSTM кодировщик → местоположение-чувствительная внимание → авторегресивный LSTM декодер излучает мел рамки. медленный (AR), колеблющийся на длинном тексте.

**FastSpeech 2 (2020).**Неавторегресивное. Продолжительность предсказующий выпускает, сколько мел кадров получает каждый фонема. 1 проход, 10x быстрее, чем Такотрона.

**VITS (2021).**Совместное обучение кодера + длительность на основе потока + голосователь HiFi-GAN с концами с вариационным выводом. Высокое качество, единая модель. Доминирующий открытый код TTS 20222024. Варианты: YourTTS (многовуковой нулевой выстрел), XTTS v2 (2024, Coqui).

**F5-TTS (2024).**Диффузионный трансформатор с помощью сборки потоков. Природная просодия, клонирование голоса с нулевым выстрелом с 5 секунд ссылочного звука.

**Kokoro (2024).**Маленький (82 М), управляемый процессором, лучший в своем классе английский ТТС для использования в режиме реального времени.

**OpenAI TTS-1-HD, ElevenLabs v2.5, Google Chirp-3.**Коммерческое состояние искусства. Эмоционные теги ElevenLabs v2.5 ("[шептали]", "[смеялись]") и голоса персонажей доминируют в производстве аудиокниг в 2026 году.

### Эволюция Vocoder

| Era | Vocoder | Latency | Quality |
|-----|---------|---------|---------|
| 2016 | WaveNet | offline only | SOTA at release |
| 2018 | WaveRNN | ~realtime | good |
| 2020 | HiFi-GAN | 100× realtime | near-human |
| 2022 | BigVGAN | 50× realtime | generalizes across speakers/langs |
| 2024 | SNAC, DAC (neural codecs) | integrated with AR models | discrete tokens, bit-efficient |

К 2026 году большинство моделей "ТТС" будут переходить от текста к форме волны; спектрограмм мела является внутренним представлением.

### Оценка

- **MOS (Mean Opinion Score).**1-5 масштаба, массовый.
- **CMOS (Comparative MOS).**Предпочтение A против B. Уменьшение интервалов доверия на аннотацию.
- **UTMOS, DNSMOS.**Неврологические предсказатели MOS без ссылки, используются для рейтинговых таблиц.
- **CER (Character Error Rate) via ASR.**Запустите выход TTS через Whisper, вычислите CER с входящим текстом.
- **SECS (Speaker Embedding Cosine Similarity).**Качество клонирования голоса.

Номеры 2026 на испытательном очистке LibriTTS:

| Model | UTMOS | CER (via Whisper) | Size |
|-------|-------|-------------------|------|
| Ground truth | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

```figure
sp-tts-stack
```

## Создайте его

### Шаг 1: фонемизируйте вход

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

Позаботьтесь о том, чтобы не вводить сырой текст ниже уровня VITS.

### Шаг 2: запустить Kokoro (2026 CPU по умолчанию)

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

Он работает в автономном режиме, один файл, 82М параметров.

### Шаг 3: запустить F5-TTS с голосовым клонированием

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

Продавайте 5-секундный ссылочный клип + его транскрипт; F5 клонирует просодию и тембры.

### Шаг 4: Вокодер HiFi-GAN с нуля

Слишком большой, чтобы поместиться в учебный сценарий, но форма:

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

Обучение: противопоказание (дискриминатор на коротких окнах) + потеря восстановления мел-спектрограммы + потеря совпадения функций.`hifi-gan`repo или nvidia-neMo.

### Шаг 5: полный трубопровод (псевдокод)

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## Используйте его

Склад 2026 года:

| Situation | Pick |
|-----------|------|
| Real-time English voice assistant | Kokoro (CPU) or XTTS v2 (GPU) |
| Voice cloning from 5 s reference | F5-TTS |
| Commercial character voices | ElevenLabs v2.5 |
| Audiobook narration | ElevenLabs v2.5 or XTTS v2 + fine-tune |
| Low-resource language | Train VITS on 5–20 h target-lang data |
| Expressive / emotion tags | ElevenLabs v2.5 or StyleTTS 2 fine-tune |

Лидер системы открытого кода с 2026 года: **F5-TTS for quality, Kokoro for efficiency**Не дотягивайся до Такотрона, если ты не историк.

## Ловушки

- **No text normalizer.**"Доктор Смит" - "Доктор" или "Драйв"? "2026" - "двадцать двадцать шесть" или "две нуля два шесть"?
- **OOV proper nouns.**"Гумаре" → "гю-майр"? отправьте модель отпадания графема к фонему для неизвестных токенов.
- **Clipping.**Выход Vocoder редко клипы, но мел масштабирование несоответствия при выводе может перейти ±1.0.`np.clip(wav, -1, 1)`- Я не знаю .
- **Sample-rate mismatch.**Kokoro выпускает 24 кГц; ваш потоковый трубопровод ожидает 16 кГц → повторного образца или получение прозвища.

## Погрузи его

Сделайте это .`outputs/skill-tts-designer.md`. Разработка трубопровода TTS для данного голоса, задержки и языка.

## Упражнения

1. **Easy.**Беги .`code/main.py`Создает фонемовый словарь из игрушечного словаря, оценивает продолжительность на фонеме и печатает поддельный график "мель".
2. **Medium.**Установите Кокоро, синтезируйте одно и то же предложение на голосе.`af_bella`и `am_adam`- Сравните продолжительность звука и субъективное качество.
3. **Hard.**Запишите 5-секундный ссылочный клип, используйте F5-TTS для клонирования, сообщите SECS между ссылочным и клонированным выходом.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Phoneme | Sound unit | Abstract sound class; 39 in English (ARPABet). |
| Duration predictor | How long each phoneme lasts | Non-AR model output; integer frames per phoneme. |
| Vocoder | Mel → waveform | Neural net mapping mel-spec to raw samples. |
| HiFi-GAN | Standard vocoder | GAN-based; dominant 2020–2024. |
| MOS | Subjective quality | 1–5 mean opinion score from human raters. |
| SECS | Voice-clone metric | Cosine similarity between target and output speaker embedding. |
| F5-TTS | 2024 open-source SOTA | Flow-matching diffusion; zero-shot cloning. |
| Kokoro | CPU English leader | 82M-param model, Apache 2.0. |

## Дальнейшее чтение

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) базовый показатель sequ2seq.
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) на основе потока от конца к концу.
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) текущий SOTA с открытым исходным кодом.
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) Вокодер, который по-прежнему будет работать в 2026 году.
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) 2024 г. - английский TTS, поддерживающий процессоры.
