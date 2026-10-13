# Аудиоклассификация  От k-NN на MFCC до AST и BEAT

> Все, от "собачьи лапы против сирены" до "какой язык это" - это аудиоклассификация.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 3 · 06 (CNNs), Phase 5 · 08 (CNNs & RNNs for Text)
**Time:** ~75 minutes

## Проблема

Вы получаете 10-секундный клип. Вы хотите знать: "что это такое?" городский звук (сирень, драйл, собака), речевое командование (да/нет/стоп), языковой идентификатор (en/es/ar), эмоции говорящего (злой/нейтральный), или окружающий мир (внутренний/вне, бабл). Все это * аудио классификация*, и в 2026 году базовая архитектура зрела: log-mel → CNN или Transformer → softmax.

Основная сложность не в сети. Это данные. Аудио-данные имеют жестокий дисбаланс класса, сильное сдвиг домена (чистый против шумных), и шум на этикетках (кто решил "городской болтовня" против "шум ресторана"?). 80% проблемы - это курация, увеличение и оценка, а не обмен CNN на Трансформатор.

## Концепция

![Audio classification ladder: k-NN on MFCCs to AST to BEATs](../assets/audio-classification.svg)

**k-NN on MFCCs (the 1990s baseline).**Плоские MFCC на клип, вычислить соизновое сходство с маркированным банком, вернуть большинство голосов верхнего K. Удивительно сильный на чистых, небольших наборах данных (Speech Commands, ESC-50).

**2D CNN on log-mels (2015-2019).**Выздоравливать`(T, n_mels)`В этом случае, если вы хотите получить информацию о своем роде, используйте ResNet-18 или VGG.

**Audio Spectrogram Transformer, AST (2021-2024).**Закрепить запись (например, 16×16 пачков), добавить вкладки позиций, подать в ViT. Современное использование AudioSet (mAP 0.485) для контролируемого обучения.

**BEATs and WavLM-base (2024-2026).**Самоконтролируемое предварительное обучение на миллионы часов. Настройка на задачу с 1-10% контролируемых данных, которые вам понадобятся. В 2026 году это является дефолтным отправным пунктом для аудио, не использующего звуки речи. BEATs-iter3 уступает AST на 1-2 mAP на AudioSet при использовании 1/4 вычисления.

**Whisper-encoder as a frozen backbone (2024).**Возьмите кодировку Whisper, отбросьте декодер, прикрепите линейный классификатор.

### Реальный вызов - классовый дисбаланс

ESC-50: 50 классов, 40 клипов каждый  сбалансированный, легкий. UrbanSound8K: 10 классов, сбалансированный 10:1. AudioSet: 632 классы с длинным хвостом 100,000: 1.

- Сбалансированная выборка образцов во время обучения (не в ходе оценки).
- Смешивание: линейный интерполяция двух клипов (и их этикетки) в качестве дополнения.
- Спецификация: маскируйте случайный время и частотные полосы.

### Оценка

- Эксклюзивный в многоклассах (Speech Commands): точная точность 1 - 5, точная точность 5.
- Многоклассная многознаковая (AudioSet, UrbanSound-style): средняя точность (mAP).
- Сильное дисбаланс: отзыв на класс + макро F1.

2026 числа, которые вы должны знать:

| Benchmark | Baseline | SOTA 2026 | Source |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs paper (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR leaderboard 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2 results |

```figure
mfcc-pipeline
```

## Создайте его

### Шаг 1: фигурировать

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### Шаг 2: краткое содержание фиксированной длины

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

Простая, но сильная: средняя + варианс в течение времени дает 26-размерную фиксированную встроенность для 13-коевого MFCC.

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

### Шаг 4: обновление на CNN на "Лог-Мелс"

В PyTorch:

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

3М параметров. Поезда в ~10 минут на ESC-50 с одним RTX 4090.

### Шаг 5: настройка предварительно обученного аудиотрансформатора (наказано АСТ)

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

Пример отредактирования AST из Hub. BEATs, дефолт 2026 года, не находится в Hub Hugging Face: скачать контрольный пункт из [BEATs release in microsoft/unilm](https://github.com/microsoft/unilm/tree/master/beats)и загрузить его с этим репо `BEATs`и `BEATsConfig`классы; петля тонкого настройки сохраняет ту же форму.

## Используйте его

Склад 2026 года:

| Situation | Start with |
|-----------|-----------|
| Tiny dataset (<1000 clips) | k-NN on MFCC means (your baseline) + audio augmentation |
| Medium dataset (1K–100K) | BEATs or AST fine-tune |
| Large dataset (>100K) | Train from scratch or fine-tune Whisper-encoder |
| Real-time, edge | 40-MFCC CNN, quantized to int8 (KWS-style) |
| Multi-label (AudioSet) | BEATs-iter3 with BCE loss + mixup + SpecAugment |
| Language ID | MMS-LID, SpeechBrain VoxLingua107 baseline |

Правило принятия решений: **start with a frozen backbone, not a fresh model**Прекрасное настройка головы БЭТс дает 95% соты за несколько часов, а не недель.

## Погрузи его

Сделайте это .`outputs/skill-classifier-designer.md`. Выберите архитектуру, дополнения, стратегию баланса классов и оценку метрики для данной задачи классификации звука.

## Упражнения

1. **Easy.**Беги .`code/main.py`. Он обучает базовую линию MFCC k-NN на 4-классном синтетическом наборе данных (чистые тоны на разных тонах).
2. **Medium.**Заменить`summarize`4 м м м б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б б
3. **Hard.**Используя `torchaudio`, обучить 2D CNN на ESC-50 складывать 1. Отчет 5-кратной точности перекрестной проверки. Добавить SpecAugment (время маска = 20, частота маска = 10) и сообщить дельта.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| AudioSet | The ImageNet of audio | Google's 2M-clip, 632-class weakly-labeled YouTube dataset. |
| ESC-50 | Small classification benchmark | 50 classes × 40 clips of environmental sounds. |
| AST | Audio Spectrogram Transformer | ViT on log-mel patches; 2021 SOTA. |
| BEATs | Self-supervised audio | Microsoft model, iter3 leads AudioSet as of 2026. |
| Mixup | Pair augmentation | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`. |
| SpecAugment | Mask-based augmentation | Zero-out random time and frequency bands of the spectrogram. |
| mAP | Main multi-label metric | Mean average precision across classes and thresholds. |

## Дальнейшее чтение

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) архитектура рекордов с 20212024.
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) дефолт 2024+
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) доминирующее звуковое увеличение.
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) 50 классов, которые живут.
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/)Таксономия YouTube класса 632; все еще золотой стандарт.
