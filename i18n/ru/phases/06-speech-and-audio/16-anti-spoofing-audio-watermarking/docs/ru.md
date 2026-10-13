# Противозвуковые антисмешивания и аудиоводометки  ASVspoof 5, AudioSeal, WaveVerify

> Клонирование голоса доставляется быстрее, чем обороны. В 2026 году производственным голосовым системам нужны две вещи: детектор (AASIST, RawNet2) который классифицирует реальную и фальшивую речь, и водоотпечаток (AudioSeal), который выживает сжатия и редактирования.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 08 (Voice Cloning)
**Time:** ~75 minutes

## Проблема

Три связанные с этим обороны:

1. **Anti-spoofing / deepfake detection.**Учитывая аудиоклип, является ли он синтетическим или реальным?
2. **Audio watermarking.**Внедрить незаметный сигнал в генерируемый аудио, который детектор может вытащить позже.
3. **Authenticated provenance.**Криптовалютная подпись аудиофайлов + метаданные.

Определение обрабатывает противников, которые не сотрудничают. Водометка обрабатывает соответствие  Аудио, генерируемое ИИ, должно быть идентифицировано как таковое.

## Концепция

![Anti-spoofing vs watermarking vs provenance — three defense layers](../assets/spoofing-watermark.svg)

### ASVspoof 5  эталон 2024-2025

Самое большое изменение по сравнению с предыдущими изданиями:

- **Crowdsourced data**(не студия чистый)  реалистичные условия.
- **~2000 speakers**(против ~ 100 ранее).
- **32 attack algorithms.**TTS + конверсия голоса + противоречивое возмущение.
- **Two tracks.**Контрмеры (CM) самостоятельное обнаружение; СПУФИГ-прочный ASV (SASV) для биометрических систем.

Современные технологии на ASVspoof 5: ~ 7,23% EER. На более старых ASVspoof 2019 LA: 0,42% EER. Реальное развертывание: ожидайте 5-10% EER на диких клипах.

### Семейства моделей обнаружения AASIST и RawNet2 

**AASIST**(2021, обновленный до 2026 гг.). Графическое внимание к спектральным особенностям.

**RawNet2.**Конвульционный фронт-энд над сырой формой волны + TDNN спинной кости.

**NeXt-TDNN + SSL features.**2025 вариант: ECAPA-стиль + WavLM функции + фокусная потеря.

### AudioSeal  по умолчанию 2024 водный знак

Мета **AudioSeal**(Jan 2024, v0.2 Dec 2024).

- **Localized.**Выявляет водозначение на кадре при разрешении 16 кГц (1/16000 с) образца.
- **Generator + detector jointly trained.**Генератор учится встраивать нечуванный сигнал; детектор учится находить его посредством увеличения.
- **Robust.**Выживает сжатие MP3 / AAC, EQ, скоростная смена ±10%, смесь шума +10 dB SNR.
- **Fast.**Детektor работает в 485x в реальном времени, 1000x быстрее, чем WavMark.
- **Capacity.**16-битная полезная нагрузка (может кодировать идентификатор модели, штамп времени генерации, идентификатор пользователя) встроенная в каждое изложение.

### Уайвмарк

Предусмотренная база открытой за AudioSeal.

- Синографика грубой силы медленна.
- Может быть удалено гауссианским шумом или сжатием MP3.
- Не дружелюбный в реальном времени.

### WaveVerify (июль 2025)

Устраняет слабости AudioSeal  специально временные манипуляции (обратная смена, скорость). использует генератор на базе FiLM + детектор смеси экспертов. Конкурентный с AudioSeal на стандартных атаках; обрабатывает временные изменения.

### Противоположность , которую противники используют

От AudioMarkBench: "в случае смены звуковой мощности все водометки показывают точность восстановления bits ниже 0,6, что указывает на почти полную удаление". **Pitch-shift is the universal attack.**Водомет No 2026 полностью устойчив к агрессивному изменению звуковой точки.

### C2PA / Инициатива по подлинности контента

Не ML-техника  манифестирующий формат. Аудиофайлы содержат криптографически подписанные метаданные о создании инструмента, автора, даты. Аудобокс / Seamless используют его. Хорошее для происхождения; ничего не делает, если плохой актер перекодирует и вычеркивает метаданные.

```figure
v4-audio-watermark
```

## Создайте его

### Шаг 1: простой детектор спектральных характеристик (играчка)

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

Синтетическая речь часто имеет необычно плоскую высокочастотную энергию.

### Шаг 2: AudioSeal встроен + обнаружен

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float in [0, 1] — probability of watermark presence
# decoded_payload: 16 bits; match against embedded payload
```

### Шаг 3: оценка  ЕЭР

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### Шаг 4: интеграция производства

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

Каждое судно поколения: (1) водный знак, (2) подписанный манифест, (3) учетный журнал, соответствующий политике хранения.

## Используйте его

| Use case | Defense |
|----------|---------|
| Shipping TTS / voice cloning | AudioSeal embed on every output (non-negotiable) |
| Biometric voice unlock | AASIST + ECAPA ensemble; liveness challenge |
| Call-center fraud detection | AASIST on 20% sample of incoming calls |
| Podcast authenticity | C2PA signing on upload, AudioSeal if AI-generated |
| Research / training detectors | ASVspoof 5 train/dev/eval sets |

## Ловушки

- **Watermark without detector ever running.**Пошлите детектор в свой информационный центр.
- **Detection without calibration.**АИСИСТ тренируется на аварийных перебоях в Лос-Анджелесе, точность падает в реальном мире.
- **Pitch-shift gap.**Агрессивная смена звука удаляет большинство водных знаков.
- **Metadata strip-and-rehost.**C2PA легко обойтись путем перезашифровки.
- **Liveness as detection.**Проси пользователя произнести случайную фразу. Предотвращает повторные атаки, но не клонирование в режиме реального времени.

## Погрузи его

Сделайте это .`outputs/skill-spoof-defender.md`. Выберите модель обнаружения, водный знак, маннифест происхождения и операционный справочник для развертывания голосового поколения.

## Упражнения

1. **Easy.**Беги .`code/main.py`. Детektor игрушек + водознака игрушек встроен/открыт на синтетическом аудио.
2. **Medium.**Установка`audioseal`, встроять 16-битное полезное нагрузку в выход TTS, передекодировать.
3. **Hard.**Настроить RawNet2 или AASIST на ASVspoof 2019 LA. Измерить EER. Проверьте на выдержанном наборе клипов, генерируемых F5-TTS  посмотрите, как деградация обнаружения OOD.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| ASVspoof | The benchmark | Biennial challenge; 2024 = ASVspoof 5. |
| CM (countermeasure) | Detector | Classifier: real speech vs synthetic / converted. |
| SASV | Speaker verif + CM | Integrated biometric + spoof detection. |
| AudioSeal | Meta watermark | Localized, 16-bit payload, 485× faster than WavMark. |
| Bit Recovery Accuracy | Watermark survival | Fraction of payload bits recovered after attack. |
| C2PA | Provenance manifest | Cryptographic metadata about creation / authorship. |
| AASIST | Detector family | Graph-attention-based anti-spoofing SOTA. |

## Дальнейшее чтение

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) текущий эталон.
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) дефолтный водоназначный знак.
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150)Детektor MoE для временных атак.
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) позвоночник обнаружения SOTA.
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) оценка надежности.
- [C2PA specification](https://spec.c2pa.org/specifications/specifications/2.4/index.html) Формат манифеста происхождения.
