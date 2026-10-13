# Открытие и сбор голосовой активности  Силеро, Кобра и флеш-трик

> Каждый голосовой агент живет или умирает по двум решениям: говорит ли пользователь сейчас, и они закончены? VAD отвечает на первое. Датчик поворота (VAD + тишина-перевешивание + семантическая модель конечного пункта) отвечает на второе.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 11 (Real-Time Audio), Phase 6 · 12 (Voice Assistant)
**Time:** ~45 minutes

## Проблема

Три различных решения, которые принимает голосовой агент на каждые 20 мс:

1. **Is this frame speech?**ВАД, бинарный, на кадры.
2. **Has the user started a new utterance?** обнаружение начала.
3. **Has the user finished?** конечная направленность (вернутая точка).

Наивный ответ (энергетический порог) не удается при любых шумах  трафик, клавиатуры, болтовня толпы. Ответ 2026 года: Silero VAD (открытый, глубоко изученный) + модель обнаружения поворота (семантическое концапоказание) + калибрированное VAD молчание.

## Концепция

![VAD cascade: energy → Silero → turn-detector → flush trick](../assets/vad-turn-taking.svg)

### Три уровня каскада ВАД

**Tier 1: energy gate.**Прозрачный RMS на -40 дБФС, фильтрует очевидное молчание, но зажигает любые звуки, выходящие за пределы порога.

**Tier 2: Silero VAD**(2020-2026, MIT). 1M параметров. Обучен на 6000+ языках. Запускается в ~1 мс на 30 мс на одну нить процессора. 87,7% TPR при 5% FPR.

**Tier 3: semantic turn detector.**LiveKit - модель обнаружения поворотов (2024-2026) или собственный небольшой классификатор.

### Ключевые параметры и их дефолты

- **Threshold.**Силеро выпускает вероятность; классифицировать речь на &gt; 0,5 (по умолчанию) или &gt; 0,3 (чувствительный). Нижний порог = меньше клипов первого слова, больше ложных положительных.
- **Minimum speech duration.**Отклоняйте речь меньше 250 мс , обычно кашль или шум стула.
- **Silence hangover (end-pointing).**После того, как VAD возвращается к 0, подождите 500-800 мс, прежде чем объявить окончание поворота.
- **Pre-roll buffer.**Сохраняйте 300-500 мс звука перед VAD-сгоревшим.

### Трюк "Флаши" (Кютай 2025)

Стриминговые модели STT имеют задержку в ожидании (500 мс для Kyutai STT-1B, 2,5 с для STT-2.6B).**send a flush signal to the STT**СТТ обрабатывается в режиме ~4× в режиме реального времени, поэтому буфер 500 мс заканчивается в ~125 мс.

От конца к концу: 125 ms VAD + flush STT = задержка разговора.

### Сравнение ВАД 2026

| VAD | TPR @ 5% FPR | Latency | License |
|-----|--------------|---------|---------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 ms | MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 ms | commercial |
| pyannote segmentation | 95% | ~10 ms | MIT-ish |

Силеро - это правильный дефолт. Кобра - это обновление соответствия / точности.

```figure
sp-vad-cascade
```

## Создайте его

### Шаг 1: энергетическая шлюз

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### Шаг 2: Silero VAD в Python

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### Шаг 3: машина состояния "верн-энд"

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### Шаг 4: скелет фишки

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT (Kyutai, Deepgram, AssemblyAI) должна поддерживать flush для этого.

## Используйте его

| Situation | VAD choice |
|-----------|-----------|
| Open, fast, general | Silero VAD |
| Commercial call center | Cobra VAD |
| On-device (phone) | Silero VAD ONNX |
| Research / diarization | pyannote segmentation |
| Zero-dependency fallback | WebRTC VAD (legacy) |
| Need turn-ending quality | Silero + LiveKit turn-detector layered |

Правило: никогда не отправляйте энергосберегающий VAD, если у вас нет другого варианта.

## Ловушки

- **Fixed threshold.**Работает в тишине, не работает в шуме.
- **Too-short silence hangover.**Агент перерывает середину предложения. 500-800 мс - это сладкое место для беседы.
- **Too-long hangover.**А/Б-тест с целевыми пользователями.
- **No pre-roll buffer.**Первые 200-300 мс аудио пользователя потеряно.
- **Ignoring semantic endpointing.**"Хм, позволь мне подумать"... содержит длинные паузы.

## Погрузи его

Сделайте это .`outputs/skill-vad-tuner.md`Выберите модель VAD, порог, похмелье, предварительную свертывание и стратегию обнаружения поворота для рабочей нагрузки.

## Упражнения

1. **Easy.**Беги .`code/main.py`Он моделирует последовательность речи + молчания + речи + кашля и тестирует три уровня VAD.
2. **Medium.**Установка`silero-vad`, обработать запись в 5 минут, настроить порог, чтобы минимизировать как клипы первого слова, так и ложные триггеры.
3. **Hard.**Создайте мини-детектор поворотов: Silero VAD + 3-слойный MLP на встроении последних 10 слов (используйте преобразователи предложений).

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| VAD | Voice detector | Binary per-frame: is this speech? |
| Turn detection | End-pointing | VAD + silence-hangover + semantic endpoint. |
| Silence hangover | Wait-after-speech | Time to wait before declaring turn end; 500-800 ms. |
| Pre-roll | Pre-speech buffer | Keep 300-500 ms audio before VAD fires. |
| Flush trick | Kyutai hack | VAD → flush-STT → 125 ms instead of 500 ms delay. |
| Semantic endpoint | "Did they mean to stop?" | ML classifier that looks at words, not just silence. |
| TPR @ FPR 5% | ROC point | Standard VAD benchmark; 87.7% for Silero, 50% WebRTC. |

## Дальнейшее чтение

- [Silero VAD](https://github.com/snakers4/silero-vad) ссылка на открытый VAD.
- [Picovoice Cobra VAD](https://picovoice.ai/products/voice/voice-activity-detection/) лидер в области коммерческой точности.
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) под-200 мс инженерный трюк.
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) семантическое окончательное указание в производстве.
- [WebRTC VAD](https://webrtc.googlesource.com/src/) базовый показатель.
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) сегментация по уровню дневничества.
