# Модели аудиоязыка  Qwen2.5-Omni, аудио фламинго, GPT-4o аудио

> 2026 аудиоязычные модели рассуждают о речи + окружающем звуке + музыке. Qwen2.5-Omni-7B совпадает с GPT-4o Audio на MMAU-Pro. Audio Flamingo Next превосходит Gemini 2.5 Pro на LongAudioBench.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 12 · 03 (Vision-Language Models), Phase 7 · 10 (Audio Transformers)
**Time:** ~45 minutes

## Проблема

У вас есть 5 секунд звука: собака лает, кто-то кричит "стоп!", затем молчание.

- **Transcription.**"Что было сказано?"  Территория АССР.
- **Semantic reasoning.**"Человек в опасности?"  требует совместного понимания ладания + крика + молчания.
- **Music reasoning.**"Какие инструменты играют на мелодию?"
- **Long-audio retrieval.**"Где в этой 90-минутной лекции инструктор объяснил спуск по ступеням?"

Одна модель , которая отвечает на все эти вопросы одним просьбой , - это **audio-language model**Отдельно от чистой АСР: ЛАЛМ дают свободные ответы на естественном языке, а не просто транскрипты.

## Концепция

![Audio-language model: audio encoder + projector + LLM decoder](../assets/alm-architecture.svg)

### Трехкомпонентный шаблон

Каждый ЛАЛМ 2026 года имеет один и тот же скелет:

1. **Audio encoder.**Шпустящий кодировщик · BEATs · CLAP · WavLM · или пользовательский кодировщик на модель.
2. **Projector.**Линейный или MLP-мостный аудио-экодер включает в себя пространство вставки токенов LLM.
3. **LLM.**Дикодер на основе Llama / Qwen / Gemma. Занимает переплетенный текст + аудио-токены; генерирует текст.

Обучение:

- **Stage 1.**Заморозный кодировщик + LLM; проектор поезда только на данных ASR / подзаписи.
- **Stage 2.**Плохая / LoRA тонкая настройка на аудио задачи, следующие за инструкцией (QA, рассуждение, понимание музыки).
- **Stage 3 (optional).**Включение голоса / выключение голоса добавляет декодер речи. Qwen2.5-Omni и AF3-Chat делают это.

### Карта модели 2026

| Model | Backbone | Audio encoder | Output modality | Access |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Custom + Whisper | text + speech | Apache-2.0 |
| Qwen3-Omni | Qwen3 | Custom | text + speech | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | text | NVIDIA non-commercial |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | text | NVIDIA non-commercial |
| SALMONN | Vicuna | Whisper + BEATs | text | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | text | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | text | Apache-2.0 |
| Gemini 2.5 Flash/Pro (closed) | Gemini | proprietary | text + speech | API |
| GPT-4o Audio (closed) | GPT-4o | proprietary | text + speech | API |

### Сравнительная проверка реальности (2026)

**MMAU-Pro.**1800 паров QA, охватывающих речь / звук / музыка / смешанный.

| Model | Overall | Speech | Sound | Music | Multi-audio |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | SOTA on LongAudioBench | — | — | — | — |

- Да .**multi-audio column is damning for everyone.**Случайный шанс на 4- вариант многократного выбора = 25%; большинство моделей получают баллы в этом направлении.

### Где ЛАЛМ будут полезны в 2026 году

- **Compliance audit of call-center recordings.**"Агент упомянул ли о необходимости раскрытия информации?"
- **Accessibility.**Опишите звуковые события для глухих пользователей (не только транскрипцию).
- **Content moderation.**Выявление насильственного языка + угрожающий тон + контекст фона.
- **Podcast / meeting chaptering.**Семантическое резюме, а не просто спикерные повороты.
- **Music catalog analysis.**"Находи все треки с изменением клавиша в B-секции".

### Где они (еще) не полезны

- Финозерно-музыкальная теория (ниже уровня аккордов).
- Рассуждение, приписываемое докладчиком после длительных разговоров (спустя 10 минут).
- Многозвуковое сравнение (22-26% - это чуть более случайное).
- Реальное время потокового рассуждения (большинство из них являются неактивными выводами партии).

```figure
v4-alm-tokens
```

## Создайте его

### Шаг 1: запрос Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Шаг 2: схема проектора

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

Проектор обычно состоит из 1-3 линейных слоев.

### Шаг 3: сравнительное определение MMAU / LongAudioBench

```python
from datasets import load_dataset
mmau = load_dataset("gamma-lab-umd/MMAU-Pro", split="test")
mcq = mmau.filter(lambda item: len(item["choices"] or []) > 1)

correct = 0
for item in mcq:
    answer = call_model(item["audio_path"], item["question"], item["choices"])
    if answer == item["answer"]:
        correct += 1
print(f"Accuracy: {correct / len(mcq):.3f}")
```

`audio_path`указывают на repo набор данных `data.zip`(около 47 ГБ), так что скачивайте и разжигайте его перед тем, как забить. Эта схема точной совпадения является проверкой здравомыслия, а не показателем показателя, поэтому ее количество не сравнимо с опубликованными результатами MMAU-Pro. Официальный оценщик сравнивает ответы с множественным выбором, внедряя сходство (NV-Embed-v2), оценивает ответы с открытым конец с судьей LLM, и проверяет ответы, следующие за инструкцией с правилами regex: пишите предсказания для `model_output`колонка и запуска `evaluate_mmau_pro_comprehensive.py`из [MMAU-Pro repo](https://github.com/sonalkum/MMAUPro)- Отчет о каждом .`category`(говоры, звук, музыка, мульти и т.д.) отдельно.

## Используйте его

| Task | 2026 pick |
|------|-----------|
| Free-form audio QA (open) | Qwen2.5-Omni-7B |
| Best open on long audio | Audio Flamingo Next |
| Best closed | Gemini 2.5 Pro |
| Voice-in / voice-out agent | Qwen2.5-Omni or GPT-4o Audio |
| Music reasoning | Audio Flamingo 3 or 2 (music-specialized AF-CLAP) |
| Call-center audit | Gemini 2.5 Pro via API, with RAG over your policy docs |

## Ловушки

- **Over-trust on multi-audio.**Если ваша задача требует "какой клип имеет X", то производительность на случайном уровне является реальной.
- **Long-audio degradation.**После 10 минут большинство моделей перерывают свойство динамика.
- **Hallucinations on silence.**Те же проблемы, которые унаследовали ЛАЛМ, использующие код Whisper.
- **Benchmark cherry-picking.**В блогах продавцов выделяются категории лучших случаев.

## Погрузи его

Сделайте это .`outputs/skill-alm-picker.md`. Выберите LALM + подмножество ссылки + выходную модуль (текст против речи) для задачи определения аудиопонимания.

## Упражнения

1. **Easy.**Беги .`code/main.py`чтобы увидеть шаблон игрового проектора + поддельное направление LALM (аудио-встроенные, текстовые токены) → выходной токены.
2. **Medium.**Сравните Qwen2.5-Omni-7B на 100 MMAU-Pro выступлений.
3. **Hard.**Создайте минимальную базовую линию заголовков для звука: BEATs кодировщик + 2-слойный проектор + замороженный Llama-3.2-1B. Настроить только проектор на AudioCaps. Сравните с SALMONN на Clotho-AQA.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| LALM | Audio ChatGPT | Audio encoder + projector + LLM decoder. |
| Projector | Adapter | Small MLP mapping audio features into LLM embedding space. |
| MMAU | The benchmark | 10k audio-QA pairs across speech, sound, music. |
| MMAU-Pro | Harder MMAU | 1800 multi-audio / reasoning-heavy questions. |
| LongAudioBench | Long-form eval | Multi-minute clips with semantic queries. |
| Voice-in / voice-out | Speech-native | Model ingests speech and emits speech without text detour. |

## Дальнейшее чтение

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) архитектура ссылки.
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) речь в речь.
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) открытый длинный аудио лидер.
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) LongAudioBench SOTA.
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) пионером двойного кодирования.
- [MMAU-Pro leaderboard](https://sonalkum.github.io/mmau-pro/) живые рейтинги 2026 года.
