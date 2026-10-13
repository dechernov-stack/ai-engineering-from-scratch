# Распознавание и верификация диктора

> ASR отвечает на вопрос «что сказали?». Распознавание диктора отвечает на вопрос «кто это сказал?». Математика выглядит одинаково — эмбеддинги плюс косинус — но каждое производственное решение упирается в одно число EER.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Спектрограммы и Mel), Phase 5 · 22 (Embedding Models)
**Time:** ~45 минут

## Проблема

Пользователь произносит кодовую фразу. Вы хотите знать: это тот человек, за которого он себя выдаёт (*верификация*, 1:1), или это первый человек в вашей базе зарегистрированных (*идентификация*, 1:N)? Или ни то, ни другое — это неизвестный диктор (*open-set*)?

До 2018: GMM-UBM + i-vectors. Приемлемый EER, но хрупкость к смене канала (телефон vs ноутбук) и эмоциям. 2018–2022: x-vectors (TDNN-основа, обученная с angular margin). 2022+: эмбеддинги ECAPA-TDNN и WavLM-large. К 2026 году в области доминируют три модели и одна метрика.

Эта метрика — **EER** (Equal Error Rate, равная частота ошибок). Установите порог принятия решения так, чтобы частота ложных принятий (False Accept Rate) равнялась частоте ложных отклонений (False Reject Rate). Точка пересечения — это EER. Используется в каждой статье, на каждом лидерборде, в каждом тендере.

## Концепция

![Enrollment + verification pipeline with embedding + cosine + EER](../assets/speaker-verification.svg)

**Пайплайн.** Регистрация (enrollment): записываете 5–30 секунд речи целевого диктора; вычисляете эмбеддинг фиксированной размерности (192-d для ECAPA-TDNN, 256-d для WavLM-large). Верификация: получаете эмбеддинг тестового высказывания; вычисляете косинусное сходство; сравниваете с порогом.

**ECAPA-TDNN (2020, всё ещё доминирует в 2026).** Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network. Блоки 1D-свёртки со squeeze-excitation, multi-head attention pooling, затем линейный слой до 192-d. Обучена на VoxCeleb 1+2 (2700 дикторов, 1.1M высказываний) с функцией потерь Additive Angular Margin (AAM-softmax).

**WavLM-SV (2022+).** Дообучение предобученной SSL-основы WavLM-large с AAM-loss. Выше качество, но медленнее — 300+ МБ против 15 МБ.

**x-vector (базовая модель).** TDNN + statistics pooling. Классика; всё ещё полезна на CPU / edge-устройствах.

**AAM-softmax.** Стандартный softmax с добавленным margin `m` в угловом пространстве: `cos(θ + m)` для правильного класса. Заставляет увеличивать угловое разделение между классами. Типичное значение `m=0.2`, масштаб `s=30`.

### Скоринг

- **Косинус** между эмбеддингами регистрации и теста. Решение на основе порога.
- **PLDA (Probabilistic LDA).** Проекция эмбеддингов в латентное пространство, где для пары «тот же диктор / разные дикторы» существует отношение правдоподобия в замкнутой форме. Добавляется поверх косинуса для снижения EER на 10–20%. Стандарт до 2020 года; сейчас используется только в closed-set сценариях.
- **Нормализация оценок.** `S-norm` или `AS-norm`: нормализация каждой оценки относительно когорты средних и стандартных отклонений «самозванцев». Существенна при междоменной оценке.

### Цифры, которые стоит знать (2026)

| Модель | VoxCeleb1-O EER | Параметры | Пропускная способность (A100) |
|-------|-----------------|--------|-------------------|
| x-vector (классика) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 segmentation + embedding | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### Диаризация

«Кто говорил и когда» в клипе с несколькими дикторами. Пайплайн: VAD → сегментация → эмбеддинг каждого сегмента → кластеризация (агломеративная или спектральная) → сглаживание границ. Современный стек: `pyannote.audio` 3.1, который объединяет сегментацию дикторов + эмбеддинг + кластеризацию в одном вызове. SOTA DER 2026 года на AMI составляет ~15% (против 23% в 2022).

```figure
sp-eer-crossover
```

## Реализация

### Шаг 1: игрушечный эмбеддинг из статистик MFCC

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

Далеко не SOTA — только для обучения. `code/main.py` использует это как proof-of-concept на синтетических данных дикторов.

### Шаг 2: косинусное сходство + порог

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### Шаг 3: EER из пар сходства

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

Возвращает (eer, threshold_at_eer). Сообщайте оба значения.

### Шаг 4: продакшн со SpeechBrain

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### Шаг 5: диаризация с pyannote

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## Применение

Стек 2026 года:

| Ситуация | Выбор |
|-----------|------|
| Closed-set верификация 1:1, edge | ECAPA-TDNN + косинусный порог |
| Open-set верификация, облако | WavLM-SV + AS-norm |
| Диаризация (совещания, подкасты) | `pyannote/speaker-diarization-3.1` |
| Защита от спуфинга (обнаружение replay-атак / дипфейков) | AASIST или RawNet2 |
| Миниатюрное встраиваемое решение (KWS + регистрация) | Titanet-Small (NeMo) |

## Подводные камни

- **Рассогласование канала.** Модель, обученная на VoxCeleb (веб-видео) ≠ аудио телефонного звонка. Всегда оценивайте на целевом канале.
- **Короткие высказывания.** EER резко ухудшается при длине тестового аудио менее 3 секунд.
- **Регистрация с шумом.** Один зашумлённый образец при регистрации отравляет якорь. Используйте ≥3 чистых образца и усредняйте.
- **Фиксированный порог для разных условий.** Всегда настраивайте порог на отложенном dev-наборе из целевого домена.
- **Косинус на ненормализованных эмбеддингах.** Сначала выполните L2-нормализацию; иначе доминирует величина.

## Финализация

Сохраните как `outputs/skill-speaker-verifier.md`. Выберите модель, протокол регистрации, план настройки порога и меры защиты от мошенничества.

## Упражнения

1. **Лёгкое.** Запустите `code/main.py`. Строит синтетических «дикторов» (разные тональные профили), регистрирует их, вычисляет EER на списке из 100 пар испытаний.
2. **Среднее.** Используйте SpeechBrain ECAPA на 30 высказываниях VoxCeleb1 (5 дикторов × по 6 каждый). Вычислите EER с косинусом и с PLDA.
3. **Сложное.** Постройте полный пайплайн «регистрация → диаризация → верификация» с помощью `pyannote.audio`. Оцените DER на dev-наборе AMI.

## Ключевые термины

| Термин | Что говорят | Что это на самом деле означает |
|------|-----------------|-----------------------|
| EER | Основная метрика | Порог, при котором False Accept = False Reject. |
| Верификация | 1:1 | «Это Алиса?» |
| Идентификация | 1:N | «Кто говорит?» |
| Open-set | Возможен неизвестный диктор | Тестовый набор может содержать незарегистрированных дикторов. |
| Регистрация (enrollment) | Регистрация | Вычисление эталонного эмбеддинга диктора. |
| AAM-softmax | Функция потерь | Softmax с additive angular margin; заставляет разделять кластеры. |
| PLDA | Классический скоринг | Probabilistic LDA; скоринг на основе отношения правдоподобия поверх эмбеддингов. |
| DER | Метрика диаризации | Diarization Error Rate — пропуски + ложные тревоги + путаница. |

## Дополнительное чтение

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — классическая статья про глубокие эмбеддинги.
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — доминирующая архитектура 2020–2026.
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — SSL-основа для SV и диаризации.
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — производственный стек диаризации + эмбеддингов.
- [VoxCeleb leaderboard (обновлено в 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — текущие позиции по EER среди моделей.