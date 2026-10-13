# Классификация аудио — от k-NN на MFCC до AST и BEATs

> Всё, начиная с «лай собаки против сирены» и заканчивая «на каком это языке», — это классификация аудио. Признаки — мел-спектры. Архитектура меняется каждое десятилетие. Оценка остаётся прежней: AUC, F1 и per-class recall.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Спектрограммы и Mel), Phase 3 · 06 (CNN), Phase 5 · 08 (CNN и RNN для текста)
**Time:** ~75 минут

## Проблема

У вас есть 10-секундный клип. Вы хотите узнать: «что это?» Городской звук (сирена, дрель, собака), голосовая команда (yes/no/stop), определение языка (en/es/ar), эмоция говорящего (angry/neutral) или звук окружения (indoor/outdoor, babble). Всё это — *классификация аудио*, и в 2026 году базовая архитектура уже зрелая: log-mel → CNN или Transformer → softmax.

Основная сложность не в сети. Она в данных. Аудиодатасеты страдают от жёсткого дисбаланса классов, сильного доменного сдвига (чистый звук против шумного) и шума в разметке (кто решил, что это «городской гул», а не «шум ресторана»?). 80% проблемы — это курирование данных, аугментация и оценка, а не замена CNN на Transformer.

## Концепция

![Audio classification ladder: k-NN on MFCCs to AST to BEATs](../assets/audio-classification.svg)

**k-NN на MFCC (бейзлайн 1990-х).** Разворачиваем MFCC каждого клипа в вектор, считаем косинусное сходство с размеченным банком, возвращаем голосование большинства по топ-K. Удивительно хорошо работает на чистых, небольших датасетах (Speech Commands, ESC-50). Работает без GPU.

**2D CNN на log-mel (2015-2019).** Рассматриваем log-mel `(T, n_mels)` как изображение. Применяем ResNet-18 или VGG-подобную архитектуру. Global mean pool по оси времени. Softmax по классам. Всё ещё базовый подход в большинстве kaggle-соревнований 2026 года.

**Audio Spectrogram Transformer, AST (2021-2024).** Разбиваем log-mel на патчи (например, 16×16), добавляем позиционные эмбеддинги, подаём в ViT. Состояние искусства на AudioSet (mAP 0.485) для обучения с учителем.

**BEATs и WavLM-base (2024-2026).** Самообучение без учителя на миллионах часов. Файнтюнинг на вашей задаче с использованием 1-10% данных с разметкой, которые понадобились бы иначе. В 2026 году это стандартная отправная точка для неречевого аудио. BEATs-iter3 обходит AST на 1-2 mAP на AudioSet, используя 1/4 вычислений.

**Whisper-encoder как замороженный бэкбон (2024).** Берём энкодер Whisper, отбрасываем декодер, прикрепляем линейный классификатор. Почти SOTA на определении языка и простой классификации событий без какой-либо аудиоаугментации. Бейзлайн «бесплатного обеда».

### Дисбаланс классов — вот реальная проблема

ESC-50: 50 классов, по 40 клипов каждый — сбалансировано, просто. UrbanSound8K: 10 классов, дисбаланс 10:1. AudioSet: 632 класса с длинным хвостом 100 000:1. Работающие техники:

- Сбалансированная выборка во время обучения (не в оценке).
- Mixup: линейная интерполяция двух клипов (и их меток) как аугментация.
- SpecAugment: маскирование случайных временных и частотных полос. Простой приём; критически важен.

### Оценка

- Мультикласс исключающий (Speech Commands): top-1 accuracy, top-5 accuracy.
- Мультикласс мульти-лейбл (AudioSet, UrbanSound-стиль): mean average precision (mAP).
- Сильный дисбаланс: per-class recall + macro F1.

Цифры 2026 года, которые стоит знать:

| Бенчмарк | Бейзлайн | SOTA 2026 | Источник |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs paper (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR leaderboard 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2 results |

```figure
mfcc-pipeline
```

## Построение

### Шаг 1: извлечение признаков

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### Шаг 2: сводка фиксированной длины

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

Просто, но эффективно: среднее + дисперсия по времени даёт 26-мерный фиксированный эмбеддинг для 13-коэффициентного MFCC. Выполняется мгновенно. Превосходил state-of-the-art нейросетевые бейзлайны на ESC-50 ещё в 2017 году.

### Шаг 3: k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### Шаг 4: переход на CNN на log-mel

На PyTorch:

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

3M параметров. Обучается ~10 минут на ESC-50 на одной RTX 4090. 80%+ accuracy.

### Шаг 5: файнтюнинг предобученного аудиотрансформера (показан AST)

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

В примере файнтюнится AST из Hub. BEATs, стандарт 2026 года, не доступен на Hugging Face Hub: скачайте чекпойнт из [релиза BEATs в microsoft/unilm](https://github.com/microsoft/unilm/tree/master/beats) и загрузите его классами `BEATs` и `BEATsConfig` из этого репозитория; цикл файнтюнинга сохраняет ту же форму.

## Применение

Стек 2026 года:

| Ситуация | Начать с |
|-----------|-----------|
| Крошечный датасет (<1000 клипов) | k-NN на средних MFCC (ваш бейзлайн) + аудиоаугментация |
| Средний датасет (1K–100K) | Файнтюнинг BEATs или AST |
| Большой датасет (>100K) | Обучение с нуля или файнтюнинг Whisper-encoder |
| Реальное время, edge | 40-MFCC CNN, квантизация в int8 (KWS-стиль) |
| Мульти-лейбл (AudioSet) | BEATs-iter3 с BCE loss + mixup + SpecAugment |
| Определение языка | MMS-LID, SpeechBrain VoxLingua107 бейзлайн |

Правило принятия решения: **начинайте с замороженного бэкбона, а не с новой модели**. Файнтюнинг головы BEATs даёт 95% от SOTA за часы, а не недели.

## Отгрузка

Сохраните как `outputs/skill-classifier-designer.md`. Выберите архитектуру, аугментации, стратегию балансировки классов и метрику оценки для заданной задачи классификации аудио.

## Упражнения

1. **Лёгкое.** Запустите `code/main.py`. Это обучает бейзлайн k-NN MFCC на синтетическом датасете из 4 классов (чистые тоны разной высоты). Сообщите матрицу ошибок.
2. **Среднее.** Замените `summarize` на [mean, var, skew, kurtosis]. Превосходит ли 4-моментное пулинг mean+var на том же синтетическом датасете?
3. **Сложное.** Используя `torchaudio`, обучите 2D CNN на ESC-50 fold 1. Сообщите точность 5-fold кросс-валидации. Добавьте SpecAugment (time mask = 20, freq mask = 10) и сообщите изменение.

## Ключевые термины

| Термин | Что говорят люди | Что это на самом деле значит |
|------|-----------------|-----------------------|
| AudioSet | ImageNet для аудио | Датасет Google из 2М клипов, 632 классов со слабой разметкой с YouTube. |
| ESC-50 | Небольшой бенчмарк классификации | 50 классов × 40 клипов звуков окружения. |
| AST | Audio Spectrogram Transformer | ViT на патчах log-mel; SOTA 2021 года. |
| BEATs | Самообучение аудио без учителя | Модель Microsoft, iter3 лидирует на AudioSet по состоянию на 2026 год. |
| Mixup | Парная аугментация | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`. |
| SpecAugment | Аугментация на основе маскирования | Обнуление случайных временных и частотных полос спектрограммы. |
| mAP | Основная метрика для мульти-лейбл | Mean average precision по классам и порогам. |

## Дополнительное чтение

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — эталонная архитектура 2021–2024 годов.
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — стандарт 2024+.
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — доминирующая аугментация аудио.
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — живучий бенчмарк из 50 классов.
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — таксономия YouTube из 632 классов; всё ещё золотой стандарт.