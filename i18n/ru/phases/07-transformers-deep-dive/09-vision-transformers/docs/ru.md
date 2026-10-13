# Трансформаторы зрения (ViT)

> Изображение - это сетка с пачками, предложение - это сетка с токенами, один и тот же трансформатор съедает оба.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 4 · 03 (CNNs), Phase 4 · 14 (Vision Transformers intro)
**Time:** ~45 minutes

## Проблема

До 2020 года компьютерное зрение означало свертывание. Каждый SOTA на ImageNet, COCO и детекционные стандарты использовали спинноколон CNN. Трансформаторы были для языка.

Досовичский и другие (2020)  "Образ стоит 16х16 слов"  показал, что вы можете полностью сбросить свертывания.

ViT был началом более широкой модели в 2026 году: одна архитектура, много способов. Шмышка символизирует аудио. ViT символизирует изображения. Токены действия для робототехники. Токены пикселей для видео. Трансформатор не заботится  накормить его последовательностью и он учится.

К 2026 году ViT и его потомки (DeiT, Swin, DINOv2, ViT-22B, SAM 3) владеют большинством зрения.

## Концепция

![Image → patches → tokens → transformer](../assets/vit.svg)

### Шаг 1  закрепить

Раздел `H × W × C`изображение в `N × (P·P·C)`последовательность плоских пятен.`224 × 224`изображение `16 × 16`пасты → 196 пасты по 768 значений каждый.

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

Размер пакета - это рычаг. Малые пасты = больше токенов, лучшее разрешение, стоимость квадратического внимания. Большие пасты = грубейшие, дешевле.

### Шаг 2  линейная вкладка

Одной научной матрицей проецируется каждый плоский пласт к `d_model`. эквивалент свертыванию размера ядра `P`И шаг вперед .`P`В PyTorch это буквально`nn.Conv2d(C, d_model, kernel_size=P, stride=P)` реализация двухстроек.

### Шаг 3  предварительный `[CLS]`Токен, добавление позиционных вкладки

- Подготовить обучающийся `[CLS]`Его последнее скрытое состояние - это представление изображения, используемое для классификации.
- Добавьте обучаемые позиционные встроенные (оригинальные ВиТ) или синусоидные 2D (позже варианты).
- В 2024+ RoPE расширился до 2D для позиции, иногда без явных встроек.

### Шаг 4  стандартный кодировщик трансформаторов

Складки L блоков `LayerNorm → Self-Attention → + → LayerNorm → MLP → +`- Идентичный с BERT. Нет специальных слоев зрения.

### Шаг 5  голова

Для классификации: принимать `[CLS]`скрытое состояние → линейное → мягкое максимум. для DINOv2 или SAM, выбрасывайте `[CLS]`, используйте встроенные пластыря непосредственно.

### Варианты, которые имели значение

| Model | Year | Change |
|-------|------|--------|
| ViT | 2020 | The original. Fixed patch size, full global attention. |
| DeiT | 2021 | Distillation; trainable on ImageNet-1k only. |
| Swin | 2021 | Hierarchical with shifted windows. Fixed sub-quadratic cost. |
| DINOv2 | 2023 | Self-supervised (no labels). Best general vision features. |
| ViT-22B | 2023 | 22B params; scaling laws apply. |
| SigLIP | 2023 | ViT + language pair, sigmoid contrastive loss. |
| SAM 3 | 2025 | Segment anything; ViT-Large + promptable mask decoder. |

### Почему это заняло некоторое время

В результате, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате, в результате чего, в результате чего, в результате, в результате чего, в результате чего, в результате, в результате чего, в результате, в результате чего, в результате, в результате чего, в результате, в результате чего-то, что было достигнуто, в результате чего, в результате, в результате чего, в результате, было достигнуто, в результате чего, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате, в результате

```figure
n5-patch-stream
```

## Создайте его

Посмотрите .`code/main.py`. Чистый стеллиб-пачтификация + линейная вкладка + проверка психики.

### Шаг 1: поддельный образ

Изображение в 24 × 24 RGB в качестве списка рядов `(R, G, B)`Мы используем 6×6 пачек → 16 пачек, каждый встроенный вектор 108-D.

### Шаг 2: закрепиться

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

Растерный порядок: главный ряд по сетке.

### Шаг 3: линейная вкладка

Умножьте каждый плоский пятно случайным образом `(patch_flat_size, d_model)`Матрица. проверьте, что выходной формат `(N_patches + 1, d_model)`после предварительного заказа `[CLS]`- Я не знаю .

### Шаг 4: пересчитывать параметры для реалистичного ViT

Печать параметров для ViT-Базы: 12 слоев, 12 голов, d=768, патч=16. Сравните с ResNet-50 (~25М).

## Используйте его

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**DINOv2 embeddings are the 2026 default for image features.**Заморозите позвоночник, обучите крошечную голову. Работает для классификации, извлечения, обнаружения, подзаписи. Контрольные пункты Meta DINOv2 превосходят CLIP на каждой задаче нетекстового зрения.

**Patch-size picking.**Малые модели используют 16×16 (ViT-B/16). Плотный прогноз (сегментация) использует 8×8 или 14×14 (SAM, DINOv2).

## Погрузи его

Посмотрите .`outputs/skill-vit-configurator.md`. Умение выбирает вариант ViT и размер пакета для новой задачи зрения, учитывая размер наборов данных, разрешение и бюджет вычислений.

## Упражнения

1. **Easy.**Беги .`code/main.py`. Проверьте количество пачек равно`(H/P) * (W/P)`и размер плоской плетки равен `P*P*C`- Я не знаю .
2. **Medium.**Внедрение 2D синусоидальных позиционных встроек  два независимых синусоидальных кода для `row`и `col`Включите их в крошечный PyTorch ViT и сравнивайте точность и обучаемые позиционные встроения на CIFAR-10.
3. **Hard.**Создайте 3-слойный ViT (PyTorch), тренируйте на 1000 изображениях MNIST с 4×4 пачками. Измерите точность теста. Теперь добавьте DINOv2 предварительное тренирование на тех же 1000 изображениях (упрощенно: просто тренируйте кодер, чтобы предсказать вкладки пачек из маскированных пачек). Улучшается ли точность?

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Patch | "The vision-transformer token" | Flat vector of pixel values for a `P × P × C` region of the image. |
| Patchify | "Chop + flatten" | Slice image into non-overlapping patches, flatten each to a vector. |
| `[CLS]` token | "The image summary" | Prepended learnable token; its final embedding is the image representation. |
| Inductive bias | "What the model assumes" | ViT has fewer priors than CNNs; needs more data to make up the gap. |
| DINOv2 | "Self-supervised ViT" | Trained without labels using image augmentation + momentum teacher. Best general image features in 2026. |
| SigLIP | "CLIP's successor" | ViT + text encoder trained with sigmoid contrastive loss; better than CLIP on matched compute. |
| Swin | "Windowed ViT" | Hierarchical ViT with local attention + shifted windows; sub-quadratic. |
| Register tokens | "2023 trick" | A few extra learnable tokens that soak up attention sinks; improves DINOv2 features. |

## Дальнейшее чтение

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) ВИТ-бумага.
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) ОТВЕРЖДЕНИЕ.
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) Свин.
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) ДИНОВ2.
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) регистрационный токен DINOv2.
