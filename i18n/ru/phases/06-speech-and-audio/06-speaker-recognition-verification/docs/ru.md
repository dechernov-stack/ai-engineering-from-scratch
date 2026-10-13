# Признание и проверка спикеров

> ASR спрашивает: "Что они сказали?" Раздел распознавания спикеров спрашивает: "Кто сказал это?" Математика выглядит одинаково  встроенные плюс косинус , но каждое решение о производстве зависит от одного EER-номера.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 22 (Embedding Models)
**Time:** ~45 minutes

## Проблема

Пользователь говорит пароль. Вы хотите знать: это ли человек, который он утверждает, что является (* проверка*, 1:1), или это первый человек в вашем банке регистрации (* идентификация*, 1:N)?

До 2018 года: GMM-UBM + i-векторы. разумная EER, но хрупкая для сдвига канала (телефон против ноутбука) и эмоции. 20182022: x-векторы (обученная с угловым маржем позвоночник TDNN). 2022+: ECAPA-TDNN и WavLM-большие вкладки. К 2026 году в поле доминируют три модели и одна метрика.

Метрика **EER** Равномерная ставка ошибок. Установить порог принятия решения так, чтобы ложная ставка принятия = ложная ставка отклонения. Кроссовер - это EER. Используется в каждой газете, каждой таблице рейтинга, каждой встрече с закупками.

## Концепция

![Enrollment + verification pipeline with embedding + cosine + EER](../assets/speaker-verification.svg)

**The pipeline.**Запись: запись 530 секунд целевого динамика; вычисление встроения фиксированного измерения (192-d для ECAPA-TDNN, 256-d для WavLM-large).

**ECAPA-TDNN (2020, still dominant 2026).**Подчеркнутое внимание канала, распространение и агрегация - Нейронная сеть с задержкой времени. 1D-объем блоков с сжатием-возбуждением, многоглавной концентрацией внимания, затем линейный слой до 192-d. Обучен на VoxCeleb 1+2 (2 700 динамиков, 1,1M выступлений) с добавленной угловой маржинальной потерей (AAM-softmax).

**WavLM-SV (2022+).**Настройка SSL-палочки с высоким качеством, но медленнее  300+ МБ против 15 МБ.

**x-vector (baseline).**TDNN + объединение статистики. Классический; все еще полезен на CPU / Edge.

**AAM-softmax.**Стандартный softmax с добавленным маржем `m`в угловом пространстве: `cos(θ + m)`Для правильного класса.`m=0.2`, масштаб `s=30`- Я не знаю .

### Сборные

- **Cosine**Включение в учетную запись и тестирование.
- **PLDA (Probabilistic LDA).**Проектные внедрения в латентное пространство, где один и тот же динамик против другого динамика имеет соотношение вероятности закрытой формы. Добавлено на поверхность косину для сокращения +1020% EER. Стандарт до 2020 года; теперь используется только в закрытых настройках.
- **Score normalization.** `S-norm`или `AS-norm`В результате, в результате чего, по мнению экспертов, в результате исследования, в результате исследования было достигнуто более высокое значение.

### Число, которое вы должны знать (2026)

| Model | VoxCeleb1-O EER | Params | Throughput (A100) |
|-------|-----------------|--------|-------------------|
| x-vector (classic) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 segmentation + embedding | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### Диаризация

"Кто говорил, когда" в многовукольном клипе. Трубопровод: VAD → сегмент → встроенный каждый сегмент → кластер (агломеративный или спектральный) → гладкие границы.`pyannote.audio`3.1, который объединяет сегментацию динамиков + вкладку + кластерирование за одним звонком.

```figure
sp-eer-crossover
```

## Создайте его

### Шаг 1: вкладка игрушек из статистики МФКК

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

Не сотни километров, только для преподавания.`code/main.py`использует это в качестве доказательства концепции на данных синтетических динамиков.

### Шаг 2: сходство косинов + порог

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### Шаг 3: ЭЭР из паров сходства

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

Отчет о возврате (eer, threshold_at_eer).

### Шаг 4: производство с помощью SpeechBrain

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### Шаг 5: записывайте дневник с помощью пианоты

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## Используйте его

Склад 2026 года:

| Situation | Pick |
|-----------|------|
| Closed-set 1:1 verification, edge | ECAPA-TDNN + cosine threshold |
| Open-set verification, cloud | WavLM-SV + AS-norm |
| Diarization (meetings, podcasts) | `pyannote/speaker-diarization-3.1` |
| Anti-spoofing (replay / deepfake detection) | AASIST or RawNet2 |
| Tiny embedded (KWS + enrollment) | Titanet-Small (NeMo) |

## Ловушки

- **Channel mismatch.**Модель, обученная на VoxCeleb (веб-видео) ≠ аудио звонка.
- **Short utterances.**EER резко ухудшается ниже 3 секунд пробного звука.
- **Enrollment with noise.**Один шумный запись отравляет якорь.
- **Fixed threshold across conditions.**Всегда настройте порог на задержанный набор разработчиков из целевого домена.
- **Cosine on non-normalized embeddings.**L2-нормализуйте сначала; в противном случае величина доминирует.

## Погрузи его

Сделайте это .`outputs/skill-speaker-verifier.md`- Выбор модели, протокол регистрации, план регулирования порогов и гарантии от мошенничества.

## Упражнения

1. **Easy.**Беги .`code/main.py`. Создает синтетические "говорящие" (разные тоновые профили), регистрирует, вычисляет EER в 100-парном испытательном списке.
2. **Medium.**Используйте SpeechBrain ECAPA на 30 высказываниях VoxCeleb1 (каждый 5 динамиков × 6).
3. **Hard.**Создать полную запись → дневник → проверить трубопровод с `pyannote.audio`Оценить DER на наборе AMI.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| EER | The headline metric | Threshold where False Accept = False Reject. |
| Verification | 1:1 | "Is this Alice?" |
| Identification | 1:N | "Who is speaking?" |
| Open-set | Unknown possible | Test set can contain unenrolled speakers. |
| Enrollment | Registering | Computing a speaker's reference embedding. |
| AAM-softmax | The loss | Softmax with additive angular margin; forces cluster separation. |
| PLDA | Classic scoring | Probabilistic LDA; likelihood-ratio scoring on top of embeddings. |
| DER | Diarization metric | Diarization Error Rate — miss + false alarm + confusion. |

## Дальнейшее чтение

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) классическая глубокозагружающаяся бумага.
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) Доминирующая архитектура 20202026.
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) SSL-пазваночная часть для SV и дневника.
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) производственная дневная работа + вкладка.
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) текущие рейтинги ЕЭР по моделям.
