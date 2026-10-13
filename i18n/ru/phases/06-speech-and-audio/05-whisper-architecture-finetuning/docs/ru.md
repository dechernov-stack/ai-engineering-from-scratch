# Шмыкание  Архитектура и тонкая настройка

> Whisper - это 30-секундный трансформатор окна кодировщик-декодер, обученный на 680 тыс. часов многоязычных слабо контролируемых аудио-текстовых пар.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 5 · 10 (Attention), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## Проблема

Whisper, выпущенный OpenAI в сентябре 2022 года, был первой моделью ASR, которая была поставлена в качестве товара: вставьте аудио, получите текст, 99 языков, прочность к шуму, работает на ноутбуке. К 2024 году OpenAI выпустила варианты Large-v3 и Turbo; к 2026 году Whisper является дефолтным базовым для всего, начиная от транскрипции подкастов до голосовых помощников до субтитров YouTube.

Но "Шиппер" не трубопровод, который можно рассматривать как черное ящик навсегда.

1. Что там внутри.
2. Как правильно передать его в виде фрагмента, потока или длинной формы.
3. Когда и как настроить.

## Концепция

![Whisper encoder-decoder, tasks, chunked inference, fine-tune](../assets/whisper.svg)

**Architecture.**Стандартный трансформатор кодировщик-декодер.

- Ввод: 30-секундный лог-мель спектрограмм, 80 мел, 10 мс hop → 3000 кадров.
- Кодировщик: конво-надовый образец (шаг 2) + `N`Для большого V3: 32 слоя, 1280-темный, 20 голов.
- Декодер: `N`Трансформаторные блоки с причинным самоотключением + перекрестным отключением к выходу кодера.
- Выход: токены BPE над словарным знаком 51 865.

Большой v3 имеет параметры 1,55B. Turbo использует 4-слойный декодер (с 32), сокращая задержку 8x с ударом WER < 1%.

**The prompt format.**Whisper - это многозадачная модель , управляемая специальными токеном в запросе декодера:

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` языковой тег; принуждает поведение перевода против транскрипции.
- `<|transcribe|>`или `<|translate|>` переводить английский выход из ввода на любом языке или буквально.
- `<|notimestamps|>` пропустить временные знаки на уровне слова (быстрее).

Промост - это то, что позволяет одной модели выполнять много задач.`<|en|>`к `<|fr|>`и переписывает на французском.

**30-second window.**Все закреплено на 30 секунд. Длиншие клипы нуждаются в разбивке; короткие клипы наполнены. Windows не передаются на родном языке.

**Log-mel normalization.** `(log_mel - mean) / std`Вы должны использовать предварительную обработку Whisper.`whisper.audio.log_mel_spectrogram`), нет `librosa.feature.melspectrogram`- Я не знаю .

### Варианты в 2026 году

| Variant | Params | Latency (A100) | WER (LibriSpeech-clean) |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 1× realtime | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | streaming | 2.0% |

### Прекрасное настройка

Канонический рабочий процесс в 2026 году:

1. Собирать 10100 часов аудио для целевого домена с выравниваемыми транскриптами.
2. Беги .`transformers.Seq2SeqTrainer`с `generate_with_loss`- Отзывы.
3. Параметровый эффект: LoRA на `q_proj`- Я не знаю .`k_proj`- Я не знаю .`v_proj`Снижение памяти GPU 4x с < 0,3 WER стоимости.
4. Заморозите кодер, если у вас есть <10 часов.
5. Используйте собственный токенер и формат запроса Whisper; никогда не обменивайте токенеры.

Результаты сообщества: тонкая настройка Средняя на 20 часов медицинского диктанта снижает WER с 12% до 4,5% на медицинский словарь.

```figure
sp-asr-attention
```

## Создайте его

### Шаг 1: запустить Whisper из коробки

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # prevents runaway repetition
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

Ключевые дефолты , которые вы должны всегда отменить: `temperature=0.0`(уборка дефолтов до 0,0 → 0,2 → 0,4 ... цепочки обратного возврата), `condition_on_previous_text=False`(предотвращает каскадную проблему галлюцинации), и `no_speech_threshold=0.6`(отчет тишины).

### Шаг 2: длинная форма в кусочках

```python
# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX добавляет (1) Silero VAD gating, (2) выравнивание уровня слова через wav2vec 2.0, (3) дневную систему через `pyannote.audio`. Рабочий конь 2026 для переписи производства.

### Шаг 3: настройка с помощью LoRA

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total
```

Затем стандартный цикл тренера, контрольный пункт каждые 1000 шагов, оценка WER на протяжении.

### Шаг 4: проверьте, что выучивает каждый слой

```python
# Grab cross-attention weights during decode to see what the decoder attends to.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

Визуализируйте с помощью тепловой карты  вы увидите диагональное выравнивание, как декодер сканирует шаги через рамки кодера.

## Используйте его

Склад 2026 года:

| Situation | Pick |
|-----------|------|
| General English, offline | Large-v3-turbo via `whisperx` |
| Mobile / edge | Whisper-Tiny quantized (int8) or Moonshine |
| Multilingual long-form | Large-v3 via `whisperx` + diarization |
| Low-resource language | Fine-tune Medium or Turbo with LoRA |
| Streaming (2 s latency) | Whisper-Streaming or Parakeet-TDT |
| Word-level timestamps | WhisperX (forced alignment via wav2vec 2.0) |

`faster-whisper`(CTranslate2 backend) - это самый быстрый срок запуска вывода CPU+GPU в 2026 году  4x быстрее, чем ванильная с идентичным выходом.

## Ловушки, которые по-прежнему будут в 2026 году

- **Hallucinated text on silence.**В песню "Whisper" входят слова "Thank you for watching!", "Subscribe!", всегда VAD-gate перед звонком.
- **`condition_on_previous_text` cascade.**Один галлюцинация загрязняет следующие окна.`False`Если только вам не нужна свободная проницательность по кусочкам.
- **Short-clip padding.**Двухсекундный клип, заполненный до 30 секунд, может галлюцинации в последующем тишине.`pad=False`или VAD-gate.
- **Wrong mel stats.**Использование мель "Либроза" вместо "Шиппа" дает почти случайный выход.`whisper.audio.log_mel_spectrogram`- Я не знаю .

## Погрузи его

Сделайте это .`outputs/skill-whisper-tuner.md`- Разрабатывать тонкую настройку или выводную трубопроводу для данного домена.

## Упражнения

1. **Easy.**Беги .`code/main.py`Он символизирует запрос в стиле "Шипп", вычисляет расшифрованные бюджеты форм и печатает график за 10 минут клипа.
2. **Medium.**Установка`faster-whisper`, переписывать 10-минутный подкаст, сравнивать WER с человеческой транскрипцией.`language="auto"`В противном случае .`language="en"`- Я не знаю .
3. **Hard.**Использование HF `datasets`, выбирать язык, с которым Whisper борется (например, урду), настроить Medium на LoRA на 2 эпохи в 2 часа и сообщить о WER delta.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| 30-sec window | Whisper's limit | Hard input cap; chunk longer audio. |
| SOT | Start-of-transcript | `<\|startoftranscript\|>` kicks off the decoder prompt. |
| Timestamps token | Temporal alignment | Every 0.02 s offset is a special token in the 51k vocab. |
| Turbo | The fast variant | 4-decoder layers, 8× faster, <1% WER regression. |
| WhisperX | The long-form wrapper | VAD + Whisper + wav2vec alignment + diarization. |
| LoRA fine-tune | Efficient tuning | Add low-rank adapters to attention; train ~0.3% of params. |
| Hallucination | The silent failure | Whisper produces fluent English from noise/silence. |

## Дальнейшее чтение

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) оригинальная архитектура и рецепт обучения.
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363)4 слоя декодера, 8x скорость.
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) длинная форма, словно выраженная, дневная.
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) CTranslate2 поддерживается, 4x быстрее.
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) каноническая LoRA / полная FT прохождение.
