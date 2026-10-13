# Создать трубопровод для голосовых помощников  Капстон 6 фазы

> Все из уроков 01-11, зашитые вместе. Создайте голосовой помощник, который слушает, рассуждает и говорит. В 2026 году это решена инженерная проблема, а не исследовательская проблема  но детали интеграции решают, будет ли она работать.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 05, 06, 07, 11; Phase 11 · 09 (Function Calling); Phase 14 · 01 (Agent Loop)
**Time:** ~120 minutes

## Проблема

Создайте помощника от конца к концу:

1. Захватывает микрофонный вход (16 кГц моно).
2. Открывает начало и конец речи пользователя.
3. Переписывает потоки.
4. Проходит транскрипт на LLM, который может вызвать инструменты (таймер, погода, календарь).
5. Получает текст LLM в TTS.
6. Засматривает аудио обратно к пользователю.
7. Прекращается, если пользователь прерывает ответ в середине.

Цель задержки: первый аудиобайт TTS в течение 800 мс после того, как пользователь завершил произношение на ноутбуке. Цель качества: без пропущенных слов, без галлюцинационных субтитров на тишине, без утечки голосового клонирования, без успеха в введении вводных средств.

## Концепция

![Voice assistant pipeline: mic → VAD → STT → LLM+tools → TTS → speaker](../assets/voice-assistant.svg)

### Семь компонентов

1. **Audio capture.**Мик → 16 кГц моно → 20 мс кусочки.`sounddevice`в Python или нативный AudioUnit/ALSA/WASAPI в производстве.
2. **VAD (Lesson 11).**Силеро VAD @ порог 0,5, мн речь 250 мс, молчание завязаться 500 мс. Сигналы "начало" и "кончание".
3. **Streaming STT (Lesson 4-5).**Шмыкание-трансляция, Parakeet-TDT, или Deepgram Nova-3 (API). Частичные + окончательные транскрипты.
4. **LLM with tool calling.**GPT-4o / Claude 3.5 / Gemini 2.5 Flash. JSON схема для инструментов. потоковые токены.
5. **Streaming TTS (Lesson 7).**Kokoro-82M (самый быстрый открытый) или Cartesia Sonic (коммерческий).
6. **Playback.**Выход на высокоскоростной связи, кодирование для сетей с низкой пропускной способностью.
7. **Interruption handler.**Если VAD зажжет во время воспроизведения TTS, прекратите воспроизведение, отмените LLM, перезагрузите STT.

### Три режима неудачи вы попадете

1. **First-word clip.**ВАД начинает бит слишком поздно. "Эй" пользователя отсутствует.
2. **Mid-response interrupt confusion.**LLM продолжает генерировать после перерыва пользователя; помощник разговаривает с пользователем.
3. **Silence hallucination.**Вскрытие "Спасибо за наблюдение" на тихих рамах отопления.

### 2026 производственные ссылочные пары

| Stack | Latency | License | Notes |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | commercial API | Industry default 2026 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | mostly open | DIY-friendly |
| Moshi (full-duplex) | 200-300 ms | CC-BY 4.0 | Single-model; different architecture, lesson 15 |
| Vapi / Retell (managed) | 300-500 ms | commercial | Fastest to launch; limited customization |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | offline | open | Privacy / edge |

```figure
v4-voice-latency
```

## Создайте его

### Шаг 1: запечатление микрофона с помощью разбивки (псевдокод)

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### Шаг 2: Захватка поворота с VAD-открытием

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### Шаг 3: потоковый STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### Шаг 4: инструмент для вызова в кругу LLM

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### Шаг 5: обработка перерыва

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## Используйте его

Посмотрите .`code/main.py`для выполняемой модели, которая связывает все семь компонентов с моделями стуб, так что вы можете видеть форму трубопровода даже без оборудования. для реальной реализации, обменьте стуб с:

- `silero-vad`(`pip install silero-vad`)
- `deepgram-sdk`или `openai-whisper`
- `openai`(`gpt-4o`) или `anthropic`
- `kokoro`или `cartesia`
- `sounddevice`для В/В

## Ловушки

- **Logging PII forever.**В большинстве стран полноценный аудиоинформация - это личное личное сообщение.
- **No barge-in.**У вас есть помощник, который должен перестать говорить.
- **TTS that blocks.**Синхронный ТТС блокирует цикл событий. Используйте асинхронный или отдельный нить.
- **No tool-call error handling.**Инструменты не работают. LLM должен вернуть ошибку + попробовать снова, а затем грациозно понизить.
- **Overzealous hallucination filters.**Слишком фильтруйте, а помощник повторяет: "Я не могу помочь с этим".
- **No wake-word option.**Всегда слушать - это ответственность за конфиденциальность.

## Погрузи его

Сделайте это .`outputs/skill-voice-assistant-architect.md`. При наличии бюджетных + масштабных + языковых + соответственных ограничений, выделять полную спецификацию.

## Упражнения

1. **Easy.**Беги .`code/main.py`Он симулирует полный поворот от конца к концу с помощью модулей и отпечатков за каждым этапом.
2. **Medium.**Заменить STT-постку на настоящую модель Whisper на заранее записанной `.wav`- Измерение WER и отсрочки от конца к концу.
3. **Hard.**Добавить инструмент призыв: реализовать `get_weather`(любой API) и `set_timer`. Проверьте LLM через инструменты и убедитесь, что, когда пользователь говорит "заставить 5-минутный таймер", запускается правильная функция и разговорный ответ подтверждает это.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Turn | A user + assistant round-trip | One VAD-bounded user speech + one LLM-TTS response. |
| Barge-in | Interruption | User speaks while assistant talks; assistant stops. |
| Wake word | "Hey assistant" | Short keyword detector; Porcupine, Snowboy, openWakeWord. |
| End-pointing | Turn ending | VAD + min-silence decision that user has finished. |
| Pre-roll | Pre-speech buffer | Keep 200-400 ms of audio before VAD fires to avoid first-word clip. |
| Tool call | Function invocation | LLM emits JSON; runtime dispatches; result feeds back in-loop. |

## Дальнейшее чтение

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) Ссылка на уровень производства.
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat) Совместная с DIY структура.
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) управляемый голосовой путь.
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) полная дуплексная ссылка (урок 15).
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) "Слушайте".
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) LLM - функционное вызов.
