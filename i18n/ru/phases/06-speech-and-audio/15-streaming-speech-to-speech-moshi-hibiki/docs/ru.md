# Стриминг речь-с-речь  Моши, хибики и полный дуплексный диалог

> 2024-2026 переопределенное голосовое ИИ. Моши отправляет единую модель, которая слушает и говорит одновременно с 200 мс задержкой. Хибики делает перевод речи-на-говоры кусочек-по-кусочке. Оба отказываются от ASR → LLM → TTS трубопровода для единой полной дуплексной архитектуры над токеном Мими. Это новый дизайн ссылки.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 13 (Neural Audio Codecs), Phase 6 · 11 (Real-Time Audio), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## Проблема

Каждый голосовой агент, созданный из уроков 11 + 12, имеет фундаментальный уровень задержки около 300-500 мс: VAD-пожар, STT-процессы, LLM-причины, TTS-генерации.

Моши (Киутай, 2024-2026) задает другой вопрос: что, если нет трубопровода? что, если одна модель принимает аудио и выпускает аудио напрямую, непрерывно, с текстом как промежуточным "внутренний монолог" вместо требуемой стадии?

Ответ:**full-duplex speech-to-speech**Теоретическая латентность 160 мс (80 мс. кадра Мими + 80 мс. акустическая задержка). Практическая латентность 200 мс. на одном графическом процессоре L4. Это половина того, что достигает лучший в своем классе пипелированный голосовой агент.

## Концепция

![Moshi architecture: two parallel Mimi streams + inner-monologue text](../assets/moshi-hibiki.svg)

### Архитектура Моши

**Inputs.**Два потока кодека Mimi, оба на 12,5 Гц × 8 кодовых книг:

- Поток 1: аудио пользователя (кодированный в мими, постоянно поступающий)
- Поток 2: собственный аудио Моши (генерированный Моши)

**The transformer.**Временный трансформатор с параметром 7B обрабатывает как потоки, так и текстовый поток "внутреннего монолога".

1. Использует новейшие пользовательские токены Mimi (8 книг кодов).
2. Использует самые последние токены Moshi Mimi (8 кодовых книг, как произведено).
3. Создает следующий текстовый токен Моши (внутренний монолог).
4. Генерарует следующие токены Моши Мими (8 кодовых книг через маленький глубокий трансформатор).

Все три потока  пользовательский аудио, аудио Моши, текст Моши  работают параллельно. Моши может слышать пользователя во время речи; может прерывать себя, когда пользователь прерывает; может перевести канал ("mhm") без нарушения его основного высказывания.

**The depth transformer.**В рамках рамки 8 кодекса не предсказываются параллельно  они имеют межкодексальные зависимости. Небольшой 2-слойный "глубокий трансформатор" предсказывает их последовательно в течение 80 мс. Это стандартная факторизация для AR кодекса LMs (также используется VALL-E, VibeVoice).

### Почему полезно использовать внутренний монолог

Без ясного текста модель должна подразумевать язык в своем акустическом потоке. Моши понимает: заставить его выпускать текстовые токены вместе с аудио. Текстовый поток по существу является транскриптом того, что говорит Моши. Это улучшает семантическую согласованность, облегчает обмен головой языка и дает транскрипты бесплатно.

### Hibiki: трансляция речь-на-речь

Одно и то же архитектура, обученная на переводных парах. Источник аудио в, аудио на целевом языке, непрерывно. Hibiki-Zero (февраль 2026) устраняет необходимость в данных обучения на уровне слов, используя данные на уровне предложения + GRPO усиление обучения для оптимизации задержки.

Первоначально поддерживаются четыре языковых пары; могут быть адаптированы к новому языку с ≈1000 часов.

### Широкий станок Кютай (2026)

- **Moshi** полный дуплексный диалог (на первом французский, с хорошей поддержкой английский)
- **Hibiki / Hibiki-Zero** одновременный перевод речи
- **Kyutai STT** потоковый ASR (500 мс или 2,5 секунды вперед)
- **Kyutai Pocket TTS** 100 М-парамерный ТТС работает на ЦП (январь 2026)
- **Unmute** полный трубопровод, объединяющий их на государственных серверах

Пропускная способность на графическом процессоре L40S: 64 одновременных сеансов в 3x реального времени.

### Сезам CSM  кузен

Сезам CSM (2025) использует аналогичную идею  позвоночник Llama-3 с головой кодека Mimi. Но CSM является однонаправленным (принимает контекст + текст, производит речь) вместо полного дуплекса. Это лучший "голосовой присутствие" TTS на рынке; не совсем то же самое, что полной дуплекс возможности Моши.

### 2026 показатели

| Model | Latency | Use case | License |
|-------|---------|----------|---------|
| Moshi | 200 ms (L4) | full-duplex English / French dialogue | CC-BY 4.0 |
| Hibiki | 12.5 Hz framerate | French ↔ English streaming translation | CC-BY 4.0 |
| Hibiki-Zero | same | 5 language-pairs, no aligned data | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | context-conditioned TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | closed, OpenAI API | commercial |
| Gemini 2.5 Live | ~350 ms | closed, Google API | commercial |

```figure
sp-fullduplex
```

## Создайте его

### Шаг 1: интерфейс

Моши раскрывает сервер WebSocket, который получает 80 мс от зашифрованного звука Мими и возвращает 80 мс от зашифрованного звука Мими.

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### Шаг 2: полная дуплексная петля

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

Оба направления работают одновременно.

### Шаг 3: цель обучения (концепционная)

За каждую 80 мс картинку `t`В следующем случае:

- Ввод: `user_mimi[0..t]`- Я не знаю .`moshi_mimi[0..t-1]`- Я не знаю .`moshi_text[0..t-1]`
- Предсказание: `moshi_text[t]`, тогда`moshi_mimi[t, codebook_0..7]`

Текст предсказывается до аудио (внутренний монолог); аудио предсказывается в секунциальном коде-буке в глубине трансформатора.

### Шаг 4: где Моши побеждает и где нет

Моши выигрывает:

- Под-250 мс от конца к концу на дешевом оборудовании.
- Естественные обратные каналы и прерывания.
- Никакого кода клейка трубопровода.

Моши не выигрывает:

- Привлечение к инструментам (не подготовлено к этому; вам нужен отдельный путь LLM).
- Длинное рассуждение (Моши - это модель диалога 8B, а не Клод/ГПТ-4).
- Фактическая точность на нишевые темы.
- Большинство случаев использования производственными предприятиями (по-прежнему используют трубопроводы в 2026 году).

## Используйте его

| Situation | Pick |
|-----------|------|
| Lowest-latency voice companion | Moshi |
| Live translation call | Hibiki |
| Voice demo / research | Moshi, CSM |
| Enterprise agent with tools | Pipeline (Lesson 12), not Moshi |
| Custom-voice TTS in context | Sesame CSM |
| Speech-to-speech, any languages | GPT-4o Realtime or Gemini 2.5 Live (commercial) |

## Ловушки

- **Limited tool calling.**Моши - это модель диалога, а не агентская структура.
- **Specific-voice conditioning.**Моши использует одну тренированную персонаж; клонирование - это отдельная тренировка.
- **Language coverage.**Французский + английский отличный, другие ограничены.
- **Resource cost.**Полная сессия Моши содержит слот для GPU, а не дешевый шаблон развертывания совместных арендаторов.

## Погрузи его

Сделайте это .`outputs/skill-duplex-pipeline.md`Выберите трубопровод против полной дуплексной архитектуры для рабочей нагрузки голосового агента, с уважением.

## Упражнения

1. **Easy.**Беги .`code/main.py`Символично имитирует архитектуру двух потоков + внутреннего монолога.
2. **Medium.**Вытащи Моши из HuggingFace, запусти сервер, провери один разговор, измеряй задержку с конца речи пользователя до начала ответа Моши.
3. **Hard.**Возьмите свой агент трубопровода урока 12 и сравнивайте задержку P50 против Moshi на 20 совпадающих тестовых заявлениях.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Full-duplex | Hear-and-speak at once | Two audio streams active simultaneously on the same model. |
| Inner monologue | Model's text stream | Moshi emits text tokens alongside its audio output. |
| Depth transformer | Inter-codebook predictor | Small transformer that predicts 8 codebooks within one 80 ms frame. |
| Mimi | Kyutai's codec | 12.5 Hz × 8 codebooks; semantic+acoustic; powers Moshi. |
| Streaming S2S | Audio → audio live | Chunk-by-chunk translation/dialogue, no pipeline stages. |
| Back-channeling | "Mhm" reactions | Moshi can emit small acknowledgments without breaking its turn. |

## Дальнейшее чтение

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) газету.
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) потоковый перевод без согласованных данных.
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) Спецификация ЦСМ.
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) установка + сервер.
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) закрытый коммерческий партнёр.
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) рамки STT/TTS под капотом.
