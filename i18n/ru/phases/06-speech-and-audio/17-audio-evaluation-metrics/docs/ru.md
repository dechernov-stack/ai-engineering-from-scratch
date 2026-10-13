# Аудиооценка  WER, MOS, UTMOS, MMAU, FAD и открытые лидерские таблицы

> Вы не можете отправить то, что не можете измерить. В этом уроке указаны показатели 2026 для каждой аудиозадачи: ASR (WER, CER, RTFx), TTS (MOS, UTMOS, SECS, WER-on-ASR-round-trip), аудиоязык (MMAU, LongAudioBench), музыка (FAD, CLAP) и динамик (EER).

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 06, 07, 09, 10; Phase 2 · 09 (Model Evaluation)
**Time:** ~60 minutes

## Проблема

Каждое аудиозадание имеет несколько показателей, каждый из которых измеряет другую ось. Используя неправильную метрику, вы отправляете модель, которая выглядит отлично на панели управления и ужасно в производстве.

| Task | Primary | Secondary |
|------|---------|-----------|
| ASR | WER | CER · RTFx · first-token latency |
| TTS | MOS / UTMOS | SECS · WER-on-ASR-round-trip · CER · TTFA |
| Voice cloning | SECS (ECAPA cosine) | MOS · CER |
| Speaker verification | EER | minDCF · FAR / FRR at operating point |
| Diarization | DER | JER · speaker confusion |
| Audio classification | top-1 · mAP | macro F1 · per-class recall |
| Music generation | FAD | CLAP · listening panel MOS |
| Audio language model | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| Streaming S2S | latency P50/P95 | WER · MOS |

## Концепция

![Audio evaluation matrix — metrics vs tasks vs 2026 leaderboards](../assets/eval-landscape.svg)

### Счетные показатели ASR

**WER (Word Error Rate).** `(S + D + I) / N`- Малочисленные, прописки, нормализация числа перед забитом.`jiwer`или OpenAI `whisper_normalizer`. &lt;5% = человеческое равноправие читать речь.

**CER (Character Error Rate).**Одинаковая формула, уровень символов. Используется для тонных языков (мандаринский, кантонский), где сегментация слов неоднозначна.

**RTFx (inverse real-time factor).**Аудиосекунды, обработанные на секунду с помощью стенового часа.

**First-token latency.**Стены от ввода до первого транскрипта.

### Счетные показатели ТТС

**MOS (Mean Opinion Score).**1-5 человека. Золотой стандарт, но медленный. Сбор 20+ слушателей на образцы, 100+ образцов на модель.

**UTMOS (2022-2026).**Учился предсказатель MOS. Соотносится с MOS человека на стандартных эталонах. F5-TTS: UTMOS 3.95; основание правды: 4.08.

**SECS (Speaker Encoder Cosine Similarity).**Для голосового клонирования. ECAPA встроя косинус между референтной и клонированной выходом. &gt; 0,75 = узнаваемый клон.

**WER-on-ASR-round-trip.**Запустить Whisper над выходом TTS, вычислить WER против входящего текста. Захватывает регрессии понятности. 2026 SOTA: &lt; 2% CER.

**TTFA (time-to-first-audio).**Кокоро-82М: ~100 мс; F5-ТТТС: ~1 с.

### Специфическая для клонирования голоса

**SECS + MOS + CER**Клонирование с высоким SECS, но низким MOS означает "правильный, но неестественный" темб; противоположное означает естественный голос, но неправильный динамик.

### Проверка динамика

**EER (Equal Error Rate).**Показатель, при котором процент ложного принятия равен процент ложного отказа.

**minDCF (min Detection Cost).**Смотренные затраты на выбранной рабочей точке (часто FAR=0,01).

### Диаризация

**DER (Diarization Error Rate).** `(FA + Miss + Confusion) / total_speaker_time`. Невыполненная речь + ложно-алармовая речь + звукозападок, каждый как часть.

**JER (Jaccard Error Rate).**Альтернативная DER, прочная к короткосегментным предвзятости.

### Классификация звука

Многозначные: **mAP (mean Average Precision)**AudioSet: 0,548 mAP для BEATs-iter3.

Эксклюзивный для нескольких классов: **top-1, top-5 accuracy**. Речь команды v2: 99,0% верхний-1 (Авдио-MAE).

Неравновешенность: **macro F1**+ **per-class recall**. Отчет на класс  совокупная точность скрывает, какие классы неудачи.

### Музыкальное поколение

**FAD (Fréchet Audio Distance).**Расстояние между VGGish-встроенными дистрибутивами реального и генерируемого аудио.

**CLAP Score.**Скорь для текстовой-аудиовой подсознания с использованием встроенных CLAP. &gt; 0,3 = разумное подсознание.

**Listening panel MOS.**Сунно V5 ELO 1293 на TTS Arena (из паров человеческих предпочтений).

### Сравнительные показатели аудиоязыка

**MMAU (Massive Multi-Audio Understanding).**10 тысяч пар аудио-QA.

**MMAU-Pro.**1800 твердых предметов, четыре категории: речь / звук / музыка / мульти-аудио. случайный шанс 25% на 4-сторонний. Близнецы 2.5 Pro в целом ~ 60%; мульти-аудио ~ 22% по всем моделям.

**LongAudioBench.**Многоминутные клипы с семантическими запросами.

**AudioCaps / Clotho.**Сборник показателей SPICE, CIDER, FENSE.

### Поток речи-на-говоры

**Latency P50 / P95 / P99.**Стены от конца речи пользователя до первого звукового ответа.

**WER / MOS**на выходе.

**Barge-in responsiveness.**Время от перерыва пользователя до помощника безмолвного.

### Лидеры 2026

| Leaderboard | Tracks | URL |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | English + multilingual + long-form | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | English TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT, ELO from paired votes | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM reasoning | `sonalkum.github.io/mmau-pro` |
| SpeakerBench / VoxSRC | Speaker recognition | `voxsrc.github.io` |
| MMAU music subset | Music LALM | (within MMAU) |
| HEAR benchmark | Self-supervised audio | `hearbenchmark.com` |

```figure
sp-wer-align
```

## Создайте его

### Шаг 1: WER с нормализацией

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### Шаг 2: TTS обратный и обратный WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### Шаг 3: SECS для клонирования голоса

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### Шаг 4: FAD для музыкального поколения

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### Шаг 5: ЭЭР для проверки динамиков (то же код, что и урок 6)

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## Используйте его

Соедините каждое развертывание с фиксированным арнесом оценки, который работает на каждом обновлении модели.

1. **Normalize before scoring.**Малобусловная, пунктуационная полоса, число расширенное.
2. **Report distributions, not averages.**P50/P95/P99 для задержки. Засвл. за класс для классификации. За категорию для MMAU.
3. **Run one canonical public benchmark.**Даже если ваши данные о производстве отличаются, отчетность на Open ASR / TTS Arena / MMAU позволяет рецензентам сравнивать яблоки с яблоками.

## Ловушки

- **UTMOS extrapolation.**Обучается в стиле VCTK чистой речи; плохо оценивает шумный / клонированный / эмоциональный аудио.
- **MOS panel bias.**20 сотрудников Amazon Mechanical Turk ≠ 20 целевых пользователей.
- **FAD depends on reference set.**Сравните с тем же ссылочным распределением между моделями.
- **Aggregate WER.**В целом 5% WER может скрыть 30% WER при акцентированной речи.
- **Public benchmark saturation.**Большинство моделей на границе находятся вблизи потолка по стандартным показателям.

## Погрузи его

Сделайте это .`outputs/skill-audio-evaluator.md`. Выберите показатели, эталоны и формат отчетности для любой аудиомодели.

## Упражнения

1. **Easy.**Беги .`code/main.py`. Вычислить WER / CER / EER / SECS / FAD-ish / MMAU-ish на входящих в игрушки.
2. **Medium.**Создайте TTS-поездный WER-сборник. Запустите выход Kokoro или F5-TTS через Whisper.
3. **Hard.**Оценьте свой выбор ЛАЛМ для урока 10 на MMAU-Pro речи + много аудио подмножеств (каждый из них - 50 пунктов).

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| WER | ASR score | `(S+D+I)/N` at word level after normalization. |
| CER | Character WER | For tone languages or char-level systems. |
| MOS | Human opinion | 1-5 rating; 20+ listeners × 100 samples. |
| UTMOS | ML MOS predictor | Learned model; correlates ~0.9 with human MOS. |
| SECS | Voice-clone similarity | ECAPA cosine between reference and clone. |
| EER | Speaker verif score | Threshold where FAR = FRR. |
| DER | Diarization score | (FA + Miss + Confusion) / total. |
| FAD | Music-gen quality | Fréchet distance on VGGish embeddings. |
| RTFx | Throughput | Audio seconds per wall-clock second. |

## Дальнейшее чтение

- [jiwer](https://github.com/jitsi/jiwer) библиотека WER/CER с приборами нормализации.
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) научился предсказывать MOS.
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) музыкальный стандарт.
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) 2026 живой рейтинг.
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) рейтинг TTS с человеческим голосом.
- [MMAU-Pro benchmark](https://sonalkum.github.io/mmau-pro/) ЛАЛМ рейтинговый план рассуждений.
- [HEAR benchmark](https://hearbenchmark.com/) аудио-справочные показатели SSL.
