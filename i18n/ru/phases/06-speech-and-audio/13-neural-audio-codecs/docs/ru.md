# Нейронные аудиокодеки  EnCodec, SNAC, Mimi, DAC и Семантико-акустический раздел

> 2026 г. - это почти все токены. EnCodec, SNAC, Mimi и DAC превращают непрерывные формы волн в дискретные последовательности, которые трансформатор может предсказать.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 10 · 11 (Quantization), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 minutes

## Проблема

Модели языка работают на дискретных токенах. Аудио непрерывно. Если вы хотите модель LLM-стиля для речи / музыки  MusicGen, Moshi, Sesame CSM, VibeVoice, Orpheus  вам сначала нужен **neural audio codec**: обученный кодировщик, который дискретирует аудио в небольшом словарном запасе токенов, и соответствующий декодер, который восстанавливает форму волны.

Сформировалось две семьи:

1. **Reconstruction-first codecs** EnCodec, DAC. Оптимизируйте восприятие качества звука. Токены "акустичны"  они захватывают все, включая идентификацию динамика, темп, фоновый шум.
2. **Semantic-first codecs** Мими (Кютай), SpeechTokenizer. Наставьте первую кодовую книгу кодировать языковой / фонетический контент (часто путем дистилляции из WavLM).

Взгляды на 2024-2026 годы: **a pure reconstruction codec gives you blurry speech when you try to generate from text.**LLM над токеном кодека должен изучить как структуру языка, так и акустическую структуру в одной книге кодекса, которая не масштабируется.

## Концепция

![Four codec landscape: EnCodec, DAC, SNAC (multi-scale), Mimi (semantic+acoustic)](../assets/codec-comparison.svg)

### Основной трюк: остаточная векторная квантовость (RVQ)

Вместо одной большой книги кодов (которая потребует миллионов кодов для хорошего качества), все современные аудиокодеки используют **RVQ**Каскад небольших книг кода. Первый код-книга квантовывает выход кодера; второй квантовывает остаток; и т. Д. Каждый код-книга 1024 кодов. 8 код-книг = эффективный словарный запас 1024 ^ 8 = 10^24.

В момент вывода декодер суммирует все выбранные коды на кадры для реконструкции.

### Четыре кодека, которые будут иметь значение в 2026 году

**EnCodec (Meta, 2022).**Базовый уровень. Энкодер-декодер по волновой форме, узлое шее RVQ. 24 кГц, 32 кодовые книги возможны, по умолчанию 4 кодовые книги @ 1,5 кбит/с. Использует `1D conv + transformer + 1D conv`Используется компанией MusicGen.

**DAC (Descript, 2023).**RVQ с L2-нормализованными кодовыми книгами, периодическими функциями активации, улучшенными потерями. Наибольшая верность реконструкции любого открытого кодека  иногда неотличима от оригинальной речи с 12 кодовыми книгами. 44,1 кГц полная полоса.

**SNAC (Hubert Siuzdak, 2024).**Многоуровневые RVQ  грубые кодовые книги работают на более низкой частоте кадров, чем тонкие. Эффективно моделируют аудиоиерархически: грубый "sketch" при ~ 12 Гц плюс детали при 50 Гц. Используется Орфеем-3Б, потому что иерархическая структура хорошо отображает на основе LM генерации.

**Mimi (Kyutai, 2024).**Изменение игры 2026 г. 12,5 Гц частота кадров (очень низкая), 8 кодовых книг @ 4,4 kbps.**distilled from WavLM** обучен предсказывать функции речи-содержания WavLM. Код-книги 1-7 являются акустическими остатками.

### Рама-точки имеют значение для моделирования языка

Низкая частота кадров = короче последовательность = быстрее LM.

| Codec | Frame rate | 1 s = N frames | Good for |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75 Hz | 75 | music, general audio |
| DAC-44.1k | 86 Hz | 86 | high-fidelity music |
| SNAC-24k (coarse) | ~12 Hz | 12 | AR-LM efficient |
| Mimi | 12.5 Hz | 12.5 | streaming speech |

При 12,5 Гц 10-секундный выговор составляет всего 125 кадров кодека. Трансформатор может легко их предсказать.

### Семантические и акустические токены

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **Semantic token (codebook 0 in Mimi).**Кодирует то, что было сказано  фонемы, слова, содержание.
- **Acoustic tokens (codebooks 1-7).**Кодирующий темб, идентификация динамика, просодия, фоновый шум, тонкие детали.

AR LM сначала предсказывает семантический токен (усложненный текстом), затем предсказывает акустические токены (усложненные семантическим + ссылочным ссылочным).

### 2026 качество реконструкции (бит в секунду, более низкая бит-скорость лучше)

| Codec | Bitrate | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

Традиционные кодеки, такие как Opus, все еще выигрывают по уровню восприятия.**discrete tokens**(который не производится Опусом) и **generative-model quality**(что LM может сделать с этими токены).

```figure
rvq-codec-cascade
```

## Создайте его

### Шаг 1: кодировать с помощью EnCodec

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

`n_codebooks=8`Каждый код - 0-1023 (10-бит).

### Шаг 2: декодирование и измерение реконструкции

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### Шаг 3: семантико-акустическое разделение (в стиле Мими)

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

Семантический кодник 0 выполнил WavLM. Вы можете обучить текстовый-семантический трансформатор  гораздо меньший словарный запас, чем прямой-на аудио.

### Шаг 4: почему AR LM над токенами кодека работает

Для 10-секундного разговора на 12,5 Гц × 8 кодовых книгах Мими:

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000 токенов - это тривиальный контекст для трансформатора. 256М-параметровый трансформатор может генерировать 10 секунд речи в миллисекундах на современном графическом процессоре.

## Используйте его

Проблема карты → кодек:

| Task | Codec |
|------|-------|
| General music generation | EnCodec-24k |
| Highest-fidelity reconstruction | DAC-44.1k |
| AR LM over speech (TTS) | SNAC or Mimi |
| Streaming full-duplex speech | Mimi (12.5 Hz) |
| Sound-effect library with text | EnCodec + T5 condition |
| Fine-grained audio editing | DAC + inpainting |

Правило: **if you're building a generative model, start with Mimi or SNAC. If you're building a compression pipeline, use Opus.**

## Ловушки

- **Too many codebooks.**Добавление кодовых книг увеличивает верность линейно, но и длина последовательности LM линейно.
- **Frame-rate mismatch.**Тренировка LM на 12,5 Гц Мими затем настройка на 50 Гц EnCodec молча не работает.
- **Assuming all codebooks equal.**В Мими код-бук 0 несет содержание, потеря его разрушает понятность.
- **Using reconstruction quality as the only metric.**Кодек может иметь большую реконструкцию, но не может быть полезным для LM-генерации, если семантическая структура плоха.

## Погрузи его

Сделайте это .`outputs/skill-codec-picker.md`Выберите кодек для задачи генерации или сжатия.

## Упражнения

1. **Easy.**Беги .`code/main.py`Он использует скалярный и остаточный квантизатор и измеряет ошибку восстановления при добавлении книг кодов.
2. **Medium.**Установка`encodec`Сравните 1, 4, 8, 32 книги кодов на протянутом видео.
3. **Hard.**Загрузите Мими. Зашифруйте клип. Замените кодбук 0 случайными целыми числами; декодируйте. Затем замените кодбук 7 аналогичным образом. Сравните два коррупции  Кодбук 0 коррупция должна уничтожить понятность; Кодбук 7 коррупция почти ничего не должна менять.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| RVQ | Residual quantization | Cascade of small codebooks; each quantizes the previous residual. |
| Frame rate | Codec speed | How many token-frames per second. Lower = faster LM. |
| Semantic codebook | Codebook 0 (Mimi) | Codebook distilled from SSL features; encodes content. |
| Acoustic codebooks | Everything else | Timbre, prosody, noise, fine detail. |
| PESQ / ViSQOL | Perceptual quality | Objective metrics correlating with MOS. |
| EnCodec | Meta codec | The RVQ baseline; used by MusicGen. |
| Mimi | Kyutai codec | 12.5 Hz frame rate; semantic-acoustic split; powers Moshi. |

## Дальнейшее чтение

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) базовый показатель RVQ.
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) Открытое место с высочайшей верностью.
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) многоуровневый RVQ.
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) семантико-акустическое разделение, дестилляция WavLM.
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143) двухэтапная семантическая/акустическая парадигма.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) оригинальный стриймный кодек RVQ.
