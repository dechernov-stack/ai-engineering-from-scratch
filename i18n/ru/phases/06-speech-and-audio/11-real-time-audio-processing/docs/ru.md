# Обработка звука в режиме реального времени

> Каждый разговорный ИИ, радиостудия и телефонный бот живут и умирают с этим бюджетом задержки.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## Проблема

Вы хотите голосового помощника, который чувствует себя живым. человеческая разговоры задержка по смену составляет ~ 230 ms (тихость-ответы). Все, что выше 500 ms чувствует себя роботизированным; выше 1500 ms чувствует себя сломанным. Бюджет для полного **hear → understand → respond → speak**цикл в 2026 году:

| Stage | Budget |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → speaker | 20 ms |
| **Total** | **~400 ms** |

Моши (Киутай, 2024) установил полный дуплекс в 200 мс. Часы GPT-4o в реальном времени (2024) ~ 320 мс. Каскадированные трубопроводы в 2022 году были отправлены на 2500 мс. Улучшение в 10 раз было вызвано тремя методами: (1) потоковой передачей везде, (2) асинхронной трубопроводной линии с частичными результатами, (3) прерываемой генерацией.

## Концепция

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**Frame / chunk / window.**Реальный аудиопоток в режиме реального времени как блоки фиксированного размера.

**Ring buffer.**Фиксированный размер круговой буфера. Продукт-навод пишет новые рамки, потребительская навод читает. Предотвращает выделения в горячей трассе. Размер ≈ максимальная задержка × скорость пробы; 2-секундный кольцо 16 кГц = 32 000 образцов.

**VAD (Voice Activity Detection).**Gates работает вниз по течению, когда никто не говорит. Silero VAD 4.0 (2024) работает < 1 мс на 30 мс кадров на процессоре. `webrtcvad`Это старый вариант.

**Streaming ASR.**Модели, которые выпускают частичные транскрипты, когда приходит аудио. Parakeet-CTC-0.6B в режиме потокового потока (NeMo, 2024) делает 25% WER при 320 мс задержке.

**Interruption.**Когда пользователь говорит, пока помощник говорит, вы должны (а) обнаружить барж-ин, (б) остановить TTS, (в) отказаться от оставшегося выхода LLM. Все это в течение 100 мс, иначе пользователь воспринимает глухого помощника.

**WebRTC Opus transport.**20 мс кадров, 48 кГц, адаптивная битовальность 8128 кГц. Стандарт для браузера и мобильных устройств. LiveKit, Daily.co, Pion являются 2026 стаками для создания голосовых приложений.

**Jitter buffer.**Сетевые пакеты прибывают из строя / поздно. Буфер jitter перезагружается и гладко; слишком маленькие → звуковые пробелы, слишком большие → задержка. 6080 мс типично.

### Общие готы

- **Thread contention.**Гигиена и графика Python могут уничтожить аудиофайл. Используйте аудиобиблиотеку C-callback (звуковое устройство, PortAudio) и удерживайте Python от горячего пути.
- **Sample-rate conversion latency.**Внутри трубопровода повторное пробку добавляется 520 мс.`soxr_hq`)).
- **TTS priming.**Даже быстрые TTS, такие как Kokoro, могут нагреваться на 100200 мс при первом запросе.
- **Echo cancellation.**Без AEC выход TTS возвращается в микрофон и запускает ASR на собственном голосе бота.

```figure
nyquist-aliasing
```

## Создайте его

### Шаг 1: буферный кольцо

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

Пропускная способность определяет максимальную латентность буфера. 32 000 образцов при 16 кГц = 2 секунды.

### Шаг 2: ВАД-ворот

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

Замена Silero VAD в производстве:

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### Шаг 3: потоковый ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### Шаг 4: управляющий перерывом

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

Зависит от синхронного ввода/вывода и отмененного потока TTS. WebRTC peerconnection.stop() на аудиотреке является каноническим способом.

## Используйте его

Склад 2026 года:

| Layer | Pick |
|-------|------|
| Transport | LiveKit (WebRTC) or Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B or Whisper-Streaming |
| LLM first-token | Groq, Cerebras, vLLM-streaming |
| Streaming TTS | Kokoro or ElevenLabs Turbo v2.5 |
| Echo cancel | WebRTC AEC3 |
| End-to-end native | OpenAI Realtime API or Moshi |

## Ловушки

- **Buffering 500 ms to be safe.**Буфер - это ваш пол задержки.
- **Not pinning threads.**Аудио-зазвонок на нижеприоритетной ниже UI нигере = сбои под загрузкой.
- **TTS chunks too small.**Под-200 мс кусочки делают звукозаписи. 320 мс кусочки - это сладкое место.
- **No jitter buffer.**Настоящие сети нервничают, без сглаживания вы получаете поп-пап.
- **Single-shot error handling.**Аудиопроводные трубы должны быть неисправительными.

## Погрузи его

Сделайте это .`outputs/skill-realtime-designer.md`- Разработка аудиопроводного канала в режиме реального времени с конкретными бюджетами на задержку на каждом этапе.

## Упражнения

1. **Easy.**Беги .`code/main.py`. Симулирует буферный кольцо + энергетический VAD; отпечатывает задержки этапов для поддельного 10-секундного потока.
2. **Medium.**Используя `sounddevice`, создать цикл прохождения, который обрабатывает микрофон в 20 мс кадров и отпечатки VAD состояние на каждой кадре.
3. **Hard.**Создайте полный дуплексный экзотест с `aiortc`: браузер → WebRTC → Python → WebRTC → браузер. Измерить задержку стекло-стекло с 1 кГц импульсом.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Ring buffer | The circular queue | Fixed-size, lock-free (or SPSC-locked) FIFO for audio frames. |
| VAD | Silence gate | Model or heuristic marking speech vs non-speech. |
| Streaming ASR | Real-time STT | Emits partial text as audio arrives; bounded lookahead. |
| Jitter buffer | Network smoother | Queue reordering out-of-order packets; 60–80 ms typical. |
| AEC | Echo cancellation | Subtracts speaker-to-mic feedback path. |
| Barge-in | User interrupt | System detects user speech mid-TTS; must cancel playback. |
| Full duplex | Simultaneous both ways | User and bot can talk at the same time; Moshi is full duplex. |

## Дальнейшее чтение

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) Круглый шпутер.
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) полная дуплексная задержка 200 мс.
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) Оркестрация аудио-агента производства.
- [Silero VAD repo](https://github.com/snakers4/silero-vad) Sub-1 ms VAD, Apache 2.0.
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) отмена эхо в режиме открытого исходного кода.
