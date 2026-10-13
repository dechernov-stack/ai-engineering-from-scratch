# Build a Complete Vision Pipeline — Capstone

> A production vision system is a chain of models and rules stitched with data contracts. The pieces are already in this phase; the capstone wires them together end-to-end.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lessons 01-15
**Time:** ~120 минут

## Learning Objectives

- Спроектировать производственный vision-пайплайн, который детектирует объекты, классифицирует их и выдаёт структурированный JSON — с обработкой каждого пути отказа
- Встроить детектор (Mask R-CNN или YOLO), классификатор (ConvNeXt-Tiny) и контракт данных (Pydantic) в единый сервис
- Провести бенчмарк end-to-end пайплайна и выявить первое узкое место (обычно это препроцессинг, затем детектор)
- Выпустить минимальный сервис на FastAPI, который принимает загрузку изображения, прогоняет пайплайн и возвращает детекции с классификациями

## The Problem

Отдельные vision-модели полезны сами по себе; vision-продукты — это их цепочки. Аудит полок в ритейле — это детектор плюс классификатор товаров плюс пайплайн OCR для цен. Автономное вождение — это 2D-детектор плюс 3D-детектор плюс сегментатор плюс трекер плюс планировщик. Медицинский предварительный скрининг — это сегментатор плюс классификатор областей плюс интерфейс для врача.

Именно соединение этих цепочек отличает ML-прототип от продукта. Каждый интерфейс между моделями — новое место для багов. Каждое преобразование координат, каждая нормализация, каждое изменение размера маски — кандидат на тихий отказ. Пайплайн настолько надёжен, насколько надёжен его самый слабый интерфейс.

Этот capstone выстраивает минимально жизнеспособный пайплайн: детекция + классификация + структурированный вывод + слой обслуживания. Всё остальное в Phase 4 встраивается в этот каркас: замените Mask R-CNN на YOLOv8, добавьте голову OCR, добавьте ветку сегментации, добавьте трекер. Архитектура стабильна; компоненты подключаемы.

## The Concept

### The pipeline

```mermaid
flowchart LR
    REQ["HTTP request<br/>+ image bytes"] --> LOAD["Decode<br/>+ preprocess"]
    LOAD --> DET["Detector<br/>(YOLO / Mask R-CNN)"]
    DET --> CROP["Crop + resize<br/>each detection"]
    CROP --> CLS["Classifier<br/>(ConvNeXt-Tiny)"]
    CLS --> AGG["Aggregate<br/>detections + classes"]
    AGG --> SCHEMA["Pydantic<br/>validation"]
    SCHEMA --> RESP["JSON response"]

    REQ -.->|error| RESP

    style DET fill:#fef3c7,stroke:#d97706
    style CLS fill:#dbeafe,stroke:#2563eb
    style SCHEMA fill:#dcfce7,stroke:#16a34a
```

Семь стадий. Две модельные стадии — дорогие; пять остальных — там, где живут баги.

### Data contracts with Pydantic

Каждая граница между моделями становится типизированным объектом. Это превращает тихие сбои в громкие.

```
Detection(
    box: tuple[float, float, float, float],   # (x1, y1, x2, y2), absolute pixels
    score: float,                              # [0, 1]
    class_id: int,                             # from detector's label map
    mask: Optional[list[list[int]]],           # RLE-encoded if present
)

PipelineResult(
    image_id: str,
    detections: list[Detection],
    classifications: list[Classification],
    inference_ms: float,
)
```

Когда детектор возвращает боксы в формате `(cx, cy, w, h)` вместо `(x1, y1, x2, y2)`, валидация Pydantic падает прямо на границе, и вы узнаёте об этом немедленно, вместо того чтобы отлаживать кроп дальше по цепочке, который молча возвращает пустые области.

### Where latency goes

Почти в любом vision-пайплайне верны три истины:

1. **Препроцессинг часто является самым большим отдельным блоком.** Декодирование JPEG, преобразование цветовых пространств, изменение размера — всё это упирается в CPU и легко забывается.
2. **Детектор доминирует по времени GPU.** 70-90% времени GPU уходит на прямой проход детекции.
3. **Постпроцессинг (NMS, кодирование/декодирование RLE) дёшев на GPU, дорог на CPU.** Всегда профилируйте на фактической целевой платформе.

Знание распределения — это то, что превращает оптимизацию в приоритизированный список.

### Failure modes

- **Пустые детекции** — верните пустой список, не падайте. Логируйте.
- **Боксы за границами изображения** — обрезайте до размера изображения перед кропом.
- **Слишком маленькие кропы** — пропускайте классификацию для боксов меньше минимального входа классификатора.
- **Повреждённая загрузка** — ответ 400 с конкретным кодом ошибки, а не 500.
- **Сбой загрузки модели** — должен происходить при старте сервиса, а не при первом запросе.

Производственный пайплайн обрабатывает каждый из этих случаев, не прибегая к общему `try/except`, который скрывает сбой. Каждому отказу присваивается именованный код и ответ.

### Batching

Производственный сервис обслуживает множество клиентов. Батчирование детекций и классификаций между запросами многократно увеличивает пропускную способность. Компромисс: дополнительная задержка из-за ожидания заполнения батча. Типичная настройка: собирать запросы в течение до 20 мс, объединять в батч, обрабатывать, распределять ответы. `torchserve` и `triton` делают это нативно; небольшие сервисы с предсказуемой нагрузкой пишут свой собственный микро-батчер.

```figure
v4-vision-pipeline
```

## Build It

### Step 1: Data contracts

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

class Detection(BaseModel):
    box: Tuple[float, float, float, float]
    score: float = Field(ge=0, le=1)
    class_id: int = Field(ge=0)
    mask_rle: Optional[str] = None


class Classification(BaseModel):
    detection_index: int
    class_id: int
    class_name: str
    score: float = Field(ge=0, le=1)


class PipelineResult(BaseModel):
    image_id: str
    detections: List[Detection]
    classifications: List[Classification]
    inference_ms: float
```

Пять секунд кода экономят час отладки на любом серьёзном пайплайне.

### Step 2: A minimal Pipeline class

```python
import time
import numpy as np
import torch
from PIL import Image

class VisionPipeline:
    def __init__(self, detector, classifier, class_names,
                 device="cpu", min_crop=32):
        self.detector = detector.to(device).eval()
        self.classifier = classifier.to(device).eval()
        self.class_names = class_names
        self.device = device
        self.min_crop = min_crop

    def preprocess(self, image):
        """
        image: PIL.Image or np.ndarray (H, W, 3) uint8
        returns: CHW float tensor on device
        """
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"))
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return tensor.to(self.device)

    @torch.no_grad()
    def detect(self, image_tensor):
        return self.detector([image_tensor])[0]

    @torch.no_grad()
    def classify(self, crops):
        if len(crops) == 0:
            return []
        batch = torch.stack(crops).to(self.device)
        logits = self.classifier(batch)
        probs = logits.softmax(-1)
        scores, cls = probs.max(-1)
        return list(zip(cls.tolist(), scores.tolist()))

    def run(self, image, image_id="anonymous"):
        t0 = time.perf_counter()
        tensor = self.preprocess(image)
        det = self.detect(tensor)

        crops = []
        detections = []
        valid_indices = []
        for i, (box, score, cls) in enumerate(zip(det["boxes"], det["scores"], det["labels"])):
            x1, y1, x2, y2 = [max(0, int(b)) for b in box.tolist()]
            x2 = min(x2, tensor.shape[-1])
            y2 = min(y2, tensor.shape[-2])
            detections.append(Detection(
                box=(x1, y1, x2, y2),
                score=float(score),
                class_id=int(cls),
            ))
            if (x2 - x1) < self.min_crop or (y2 - y1) < self.min_crop:
                continue
            crop = tensor[:, y1:y2, x1:x2]
            crop = torch.nn.functional.interpolate(
                crop.unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )[0]
            crops.append(crop)
            valid_indices.append(i)

        class_preds = self.classify(crops)

        classifications = []
        for valid_idx, (cls_id, cls_score) in zip(valid_indices, class_preds):
            classifications.append(Classification(
                detection_index=valid_idx,
                class_id=int(cls_id),
                class_name=self.class_names[cls_id],
                score=float(cls_score),
            ))

        return PipelineResult(
            image_id=image_id,
            detections=detections,
            classifications=classifications,
            inference_ms=(time.perf_counter() - t0) * 1000,
        )
```

Каждый интерфейс типизирован. У каждого пути отказа есть конкретное решение об обработке.

### Step 3: Wire a detector and a classifier

```python
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models import convnext_tiny

# Use ImageNet-pretrained weights for a realistic pipeline without training
detector = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
classifier = convnext_tiny(weights="DEFAULT")
class_names = [f"imagenet_class_{i}" for i in range(1000)]

pipe = VisionPipeline(detector, classifier, class_names)

# Smoke test with a synthetic image
test_image = (np.random.rand(400, 600, 3) * 255).astype(np.uint8)
result = pipe.run(test_image, image_id="demo")
print(result.model_dump_json(indent=2)[:500])
```

### Step 4: FastAPI service

```python
from fastapi import FastAPI, UploadFile, HTTPException
from io import BytesIO

app = FastAPI()
pipe = None  # initialised on startup

@app.on_event("startup")
def load():
    global pipe
    detector = maskrcnn_resnet50_fpn_v2(weights="DEFAULT").eval()
    classifier = convnext_tiny(weights="DEFAULT").eval()
    pipe = VisionPipeline(detector, classifier, class_names=[f"c{i}" for i in range(1000)])

@app.post("/detect")
async def detect_endpoint(file: UploadFile):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="unsupported image type")
    data = await file.read()
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="cannot decode image")
    result = pipe.run(img, image_id=file.filename or "upload")
    return result.model_dump()
```

Запустите с помощью `uvicorn main:app --host 0.0.0.0 --port 8000`. Протестируйте с помощью `curl -F 'file=@dog.jpg' http://localhost:8000/detect`.

### Step 5: Benchmark the pipeline

```python
import time

def benchmark(pipe, num_runs=20, image_size=(400, 600)):
    img = (np.random.rand(*image_size, 3) * 255).astype(np.uint8)
    pipe.run(img)  # warm up

    stages = {"preprocess": [], "detect": [], "classify": [], "total": []}
    for _ in range(num_runs):
        t0 = time.perf_counter()
        tensor = pipe.preprocess(img)
        t1 = time.perf_counter()
        det = pipe.detect(tensor)
        t2 = time.perf_counter()
        crops = []
        for box in det["boxes"]:
            x1, y1, x2, y2 = [max(0, int(b)) for b in box.tolist()]
            x2 = min(x2, tensor.shape[-1])
            y2 = min(y2, tensor.shape[-2])
            if (x2 - x1) >= pipe.min_crop and (y2 - y1) >= pipe.min_crop:
                crop = tensor[:, y1:y2, x1:x2]
                crop = torch.nn.functional.interpolate(
                    crop.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
                )[0]
                crops.append(crop)
        pipe.classify(crops)
        t3 = time.perf_counter()
        stages["preprocess"].append((t1 - t0) * 1000)
        stages["detect"].append((t2 - t1) * 1000)
        stages["classify"].append((t3 - t2) * 1000)
        stages["total"].append((t3 - t0) * 1000)

    for stage, times in stages.items():
        times.sort()
        print(f"{stage:12s}  p50={times[len(times)//2]:7.1f} ms  p95={times[int(len(times)*0.95)]:7.1f} ms")
```

Типичный результат на CPU: препроцессинг ~3 мс, детекция 300-500 мс, классификация 20-40 мс, итого 350-550 мс. На GPU детекция занимает 20-40 мс, и препроцессинг + классификация начинают иметь большее относительное значение.

## Use It

Производственные шаблоны сходятся к одной и той же структуре, плюс:

- **Версионирование моделей** — всегда логируйте имя модели и хэш весов в ответе.
- **ID трассировки по запросам** — логируйте тайминг каждой стадии для каждого запроса, чтобы можно было сопоставлять медленные ответы со стадиями.
- **Запасной путь** — если классификатор превышает тайм-аут, возвращайте детекции без классификаций, а не проваливайте весь запрос.
- **Фильтры безопасности** — фильтры NSFW/PII выполняются после классификации, перед тем как ответ покинет сервис.
- **Батч-эндпоинт** — `/detect_batch`, принимающий список URL изображений для массовой обработки.

Для продакшен-обслуживания `torchserve`, `Triton Inference Server` и `BentoML` обрабатывают батчирование, версионирование, метрики и проверки работоспособности "из коробки". Запуск `FastAPI` напрямую подходит для прототипов и небольших продуктов.

## Ship It

Этот урок производит:

- `outputs/prompt-vision-service-shape-reviewer.md` — промпт, который проверяет код vision-сервиса на нарушения формы контракта/ответа и указывает первый критичный баг.
- `outputs/skill-pipeline-budget-planner.md` — навык, который, получив целевую задержку и пропускную способность, назначает временной бюджет каждой стадии пайплайна и указывает, какая стадия первой не уложится в бюджет.

## Exercises

1. **(Easy)** Запустите пайплайн на 10 изображениях из любого открытого датасета. Сообщите среднее время по каждой стадии и распределение числа детекций на изображение.
2. **(Medium)** Добавьте поле вывода маски в `Detection` и закодируйте его как RLE. Убедитесь, что JSON остаётся меньше 1 МБ даже для изображения с 10 объектами.
3. **(Hard)** Добавьте микро-батчер перед классификатором: собирайте кропы в течение до 10 мс, классифицируйте их все одним вызовом GPU, возвращайте результаты по каждому запросу. Измерьте прирост пропускной способности при 5 одновременных запросах в секунду и добавленную задержку.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Pipeline | «Система» | Упорядоченная цепочка этапов препроцессинга, инференса и постпроцессинга с типизированным интерфейсом между каждой парой |
| Data contract | «Схема» | Определения Pydantic / dataclass, которым соответствуют вход и выход каждой стадии; ловит баги интеграции на границе |
| Preprocessing | «Перед моделью» | Декодирование, преобразование цвета, изменение размера, нормализация; обычно главный пожиратель CPU-времени |
| Postprocessing | «После модели» | NMS, изменение размера маски, пороговая обработка, кодирование RLE; дёшево на GPU, дорого на CPU |
| Microbatcher | «Собрать, потом передать» | Агрегатор, который ждёт фиксированное окно для нескольких запросов, выполняет один батчированный прямой проход |
| Trace ID | «ID запроса» | Идентификатор для каждого запроса, логируемый на каждой стадии, чтобы медленные запросы можно было отследить от начала до конца |
| Failure code | «Именованная ошибка» | Конкретный код ошибки для каждого класса сбоев вместо общего 500; позволяет клиенту реализовать логику повторных попыток |
| Health check | «Проверка готовности» | Дешёвый эндпоинт, сообщающий, может ли сервис отвечать; на него полагаются балансировщики нагрузки |

## Further Reading

- [Full Stack Deep Learning — Deploying Models](https://fullstackdeeplearning.com/course/2022/lecture-5-deployment/) — каноничный обзор развёртывания производственного ML
- [BentoML docs](https://docs.bentoml.com) — фреймворк обслуживания с батчированием, версионированием и метриками
- [torchserve docs](https://pytorch.org/serve/) — официальная библиотека обслуживания от PyTorch
- [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server) — высокопроизводительное обслуживание с батчированием и поддержкой нескольких моделей