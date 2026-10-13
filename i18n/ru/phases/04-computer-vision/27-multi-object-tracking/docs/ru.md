# Трекинг множества объектов и видеопамять

> Трекинг — это детекция плюс сопоставление (ассоциация). Детектируй каждый кадр. Сопоставляй детекции текущего кадра с треками предыдущего кадра по ID.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 06 (YOLO Detection), Phase 4 Lesson 08 (Mask R-CNN), Phase 4 Lesson 24 (SAM 3)
**Time:** ~60 минут

## Цели обучения

- Различать tracking-by-detection и query-based трекинг, называть семейства алгоритмов (SORT, DeepSORT, ByteTrack, BoT-SORT, SAM 2 memory tracker, SAM 3.1 Object Multiplex)
- Реализовать с нуля IoU + венгерское сопоставление (Hungarian assignment) для классического tracking-by-detection
- Объяснить банк памяти (memory bank) SAM 2 и почему он лучше справляется с окклюзией, чем ассоциация на основе IoU
- Понимать три метрики трекинга (MOTA, IDF1, HOTA) и выбирать подходящую для конкретного сценария использования

## Проблема

Детектор сообщает, где находятся объекты в одном кадре. Трекер сообщает, какая детекция в кадре `t` является тем же объектом, что и детекция в кадре `t-1`. Без этого нельзя посчитать объекты, пересекающие линию, проследить за мячом через окклюзию или узнать, что «машина №4 находится в полосе уже 8 секунд».

Трекинг необходим в любом продукте, работающем с видео: спортивная аналитика, видеонаблюдение, автономное вождение, анализ медицинского видео, мониторинг дикой природы, подсчёт товарных знаков. Базовые компоненты одни и те же: подетекторный детектор, модель движения (фильтр Калмана или что-то более сложное), шаг ассоциации (венгерский алгоритм по IoU / косинусному сходству / обученным признакам) и жизненный цикл трека (появление, обновление, завершение).

2026 год принёс два новых подхода: **трекинг на основе памяти SAM 2** (память признаков вместо ассоциации по модели движения) и **SAM 3.1 Object Multiplex** (общая память для множества экземпляров одного и того же концепта). Этот урок сначала рассматривает классический стек, а затем — подход на основе памяти.

## Концепция

### Tracking-by-detection

```mermaid
flowchart LR
    F1["Frame t"] --> DET["Detector"] --> D1["Detections at t"]
    PREV["Tracks up to t-1"] --> PREDICT["Motion predict<br/>(Kalman)"]
    PREDICT --> PRED["Predicted tracks at t"]
    D1 --> ASSOC["Hungarian assignment<br/>(IoU / cosine / motion)"]
    PRED --> ASSOC
    ASSOC --> UPDATE["Update matched tracks"]
    ASSOC --> NEW["Birth new tracks"]
    ASSOC --> DEAD["Age unmatched tracks; delete after N"]
    UPDATE --> NEXT["Tracks at t"]
    NEW --> NEXT
    DEAD --> NEXT

    style DET fill:#dbeafe,stroke:#2563eb
    style ASSOC fill:#fef3c7,stroke:#d97706
    style NEXT fill:#dcfce7,stroke:#16a34a
```

Любой трекер, который вы встретите в 2026 году, — это вариация этого цикла. Различия:

- **SORT** (2016): фильтр Калмана + венгерское сопоставление по IoU. Прост, быстр, без модели внешнего вида.
- **DeepSORT** (2017): SORT + признак внешнего вида на основе CNN для каждого трека (ReID-эмбеддинг). Лучше справляется с пересечениями объектов.
- **ByteTrack** (2021): сопоставляет низкоуверенные детекции на втором этапе; не требует признаков внешнего вида, но является лучшим по результатам на MOT17.
- **BoT-SORT** (2022): Byte + компенсация движения камеры + ReID.
- **StrongSORT / OC-SORT** — потомки ByteTrack с улучшенной моделью движения и внешнего вида.

### Фильтр Калмана в одном абзаце

Фильтр Калмана поддерживает для каждого трека состояние `(x, y, w, h, dx, dy, dw, dh)` с ковариацией. На каждом кадре происходит **предсказание** (predict) состояния с помощью модели постоянной скорости, затем **обновление** (update) с помощью сопоставленной детекции. Обновление доверяет детекции сильнее, когда неопределённость предсказания высока. Это даёт плавные траектории и возможность продолжать трек через короткую окклюзию (1-5 кадров).

Каждый классический трекер использует фильтр Калмана на шаге предсказания движения.

### Венгерский алгоритм

Дана матрица стоимости `M x N` (треки x детекции), нужно найти взаимно однозначное сопоставление, минимизирующее суммарную стоимость. Стоимость обычно — это `1 - IoU(track_bbox, detection_bbox)` или отрицательное косинусное сходство признаков внешнего вида. Время выполнения — O((M+N)^3); для M, N до ~1000 этого достаточно быстро в Python через `scipy.optimize.linear_sum_assignment`.

### Ключевая идея ByteTrack

Стандартные трекеры отбрасывают низкоуверенные детекции (< 0.5). ByteTrack сохраняет их как **кандидатов второго этапа**: после сопоставления треков с высокоуверенными детекциями несопоставленные треки пытаются сопоставиться с низкоуверенными детекциями с чуть более мягким порогом IoU. Это восстанавливает короткие окклюзии и переключения ID вблизи скоплений объектов.

### Трекинг на основе памяти SAM 2

SAM 2 обрабатывает видео, поддерживая **банк памяти** (memory bank) пространственно-временных признаков для каждого экземпляра. По запросу (клик, рамка, текст) на одном кадре инстанс кодируется в память. На последующих кадрах память сопоставляется через cross-attention с признаками нового кадра, и декодер выдаёт маску того же экземпляра на новом кадре.

Здесь нет ни фильтра Калмана, ни венгерского сопоставления. Ассоциация неявно заложена в операции memory-attention.

Плюсы:
- Устойчивость к сильным окклюзиям (память сохраняет идентичность экземпляра на протяжении многих кадров).
- Открытый словарь при сочетании с текстовыми запросами SAM 3.
- Работает без отдельной модели движения.

Минусы:
- Медленнее ByteTrack при трекинге множества объектов.
- Банк памяти растёт; это ограничивает окно контекста.

### SAM 3.1 Object Multiplex

Более ранние версии трекинга SAM 2 / SAM 3 хранят отдельный банк памяти для каждого экземпляра. Для 50 объектов — 50 банков памяти. Object Multiplex (март 2026) объединяет их в одну общую память с **токенами-запросами для каждого экземпляра**. Стоимость растёт сублинейно от количества экземпляров.

Multiplex стал новым стандартом для трекинга толпы в 2026 году: концертные толпы, работники склада, перекрёстки.

### Три метрики, которые нужно знать

- **MOTA (Multi-Object Tracking Accuracy)** — 1 - (FN + FP + переключения ID) / GT. Взвешена по типу ошибки; единая метрика, смешивающая ошибки детекции и ассоциации.
- **IDF1 (ID F1)** — гармоническое среднее точности и полноты по ID. Фокусируется конкретно на том, насколько хорошо каждый истинный трек сохраняет свой ID во времени. Лучше MOTA для задач, чувствительных к переключениям ID.
- **HOTA (Higher Order Tracking Accuracy)** — раскладывается на точность детекции (DetA) и точность ассоциации (AssA). Стандарт сообщества с 2020 года; наиболее полная метрика.

Для видеонаблюдения (кто есть кто): отчитываться следует по IDF1. Для спортивной аналитики (подсчёт передач): HOTA. Для общего академического сравнения: HOTA.

```figure
cv3-track-assoc
```

## Реализация

### Шаг 1: Матрица стоимости на основе IoU

```python
import numpy as np


def bbox_iou(a, b):
    """
    a, b: (N, 4) arrays of [x1, y1, x2, y2].
    Returns (N_a, N_b) IoU matrix.
    """
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-8, None)
```

### Шаг 2: Минимальный трекер в стиле SORT

Постоянно-скоростной фильтр Калмана для краткости опущен — здесь мы используем простую ассоциацию по IoU; в продакшене шаг предсказания Калмана необходим. Полную версию предоставляет Python-пакет `sort`.

```python
from scipy.optimize import linear_sum_assignment


class Track:
    def __init__(self, tid, bbox, frame):
        self.id = tid
        self.bbox = bbox
        self.last_frame = frame
        self.hits = 1

    def update(self, bbox, frame):
        self.bbox = bbox
        self.last_frame = frame
        self.hits += 1


class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_age=5):
        self.tracks = []
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    def step(self, detections, frame):
        if not self.tracks:
            for d in detections:
                self.tracks.append(Track(self.next_id, d, frame))
                self.next_id += 1
            return [(t.id, t.bbox) for t in self.tracks]

        track_boxes = np.array([t.bbox for t in self.tracks])
        det_boxes = np.array(detections) if len(detections) else np.empty((0, 4))

        iou = bbox_iou(track_boxes, det_boxes) if len(det_boxes) else np.zeros((len(track_boxes), 0))
        cost = 1 - iou
        cost[iou < self.iou_threshold] = 1e6

        matched_track = set()
        matched_det = set()
        if cost.size > 0:
            row, col = linear_sum_assignment(cost)
            for r, c in zip(row, col):
                if cost[r, c] < 1.0:
                    self.tracks[r].update(det_boxes[c], frame)
                    matched_track.add(r); matched_det.add(c)

        for i, d in enumerate(det_boxes):
            if i not in matched_det:
                self.tracks.append(Track(self.next_id, d, frame))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if frame - t.last_frame <= self.max_age]
        return [(t.id, t.bbox) for t in self.tracks]
```

60 строк. Принимает детекции по кадрам, возвращает ID треков по кадрам. В реальных системах добавляются предсказание Калмана, повторное сопоставление второго этапа ByteTrack и признаки внешнего вида.

### Шаг 3: Тест на синтетической траектории

```python
def synthetic_frames(num_frames=20, num_objects=3, H=240, W=320, seed=0):
    rng = np.random.default_rng(seed)
    starts = rng.uniform(20, 200, size=(num_objects, 2))
    velocities = rng.uniform(-5, 5, size=(num_objects, 2))
    frames = []
    for f in range(num_frames):
        dets = []
        for i in range(num_objects):
            cx, cy = starts[i] + f * velocities[i]
            dets.append([cx - 10, cy - 10, cx + 10, cy + 10])
        frames.append(dets)
    return frames


tracker = SimpleTracker()
for f, dets in enumerate(synthetic_frames()):
    tracks = tracker.step(dets, f)
```

Три объекта, движущихся по прямым линиям, должны сохранять свои ID на протяжении всех 20 кадров.

### Шаг 4: Метрика переключения ID

```python
def count_id_switches(tracks_per_frame, gt_per_frame):
    """
    tracks_per_frame:  list of list of (track_id, bbox)
    gt_per_frame:      list of list of (gt_id, bbox)
    Returns number of ID switches.
    """
    prev_assignment = {}
    switches = 0
    for tracks, gts in zip(tracks_per_frame, gt_per_frame):
        if not tracks or not gts:
            continue
        t_boxes = np.array([b for _, b in tracks])
        g_boxes = np.array([b for _, b in gts])
        iou = bbox_iou(g_boxes, t_boxes)
        for g_idx, (gt_id, _) in enumerate(gts):
            j = iou[g_idx].argmax()
            if iou[g_idx, j] > 0.5:
                t_id = tracks[j][0]
                if gt_id in prev_assignment and prev_assignment[gt_id] != t_id:
                    switches += 1
                prev_assignment[gt_id] = t_id
    return switches
```

Это упрощённая метрика, близкая к IDF1: подсчитывает, сколько раз истинному объекту присваивается другой предсказанный ID трека. Настоящие инструменты MOTA / IDF1 / HOTA находятся в `py-motmetrics` и `TrackEval`.

## Применение

Продакшен-трекеры в 2026 году:

- `ultralytics` — встроенные YOLOv8 + ByteTrack / BoT-SORT. `results = model.track(source, tracker="bytetrack.yaml")`. Вариант по умолчанию.
- `supervision` (Roboflow) — обёртки ByteTrack плюс утилиты для аннотирования.
- SAM 2 / SAM 3.1 — трекинг на основе памяти через `processor.track()`.
- Собственный стек: детектор (YOLOv8 / RT-DETR) + `sort-tracker` / `OC-SORT` / `StrongSORT`.

Выбор:

- Пешеходы / машины / коробки при 30+ fps: **ByteTrack с ultralytics**.
- Много экземпляров одного класса в толпе: **SAM 3.1 Object Multiplex**.
- Сильные окклюзии с различимым внешним видом: **DeepSORT / StrongSORT** (признаки ReID).
- Спорт / сложные взаимодействия: **BoT-SORT** или обучаемые трекеры (MOTRv3).

## Отправка в продакшен

Этот урок производит:

- `outputs/prompt-tracker-picker.md` — выбирает SORT / ByteTrack / BoT-SORT / SAM 2 / SAM 3.1 исходя из типа сцены, паттернов окклюзии и бюджета задержки.
- `outputs/skill-mot-evaluator.md` — пишет полный тестовый стенд для оценки MOTA / IDF1 / HOTA относительно истинных треков.

## Упражнения

1. **(Лёгкое)** Запустите синтетический трекер выше с 3, 10 и 30 объектами. Сообщите количество переключений ID в каждом случае. Определите, где простая ассоциация только по IoU начинает давать сбой.
2. **(Среднее)** Добавьте шаг предсказания Калмана с постоянной скоростью перед ассоциацией. Покажите, что короткие (2-3 кадра) окклюзии больше не вызывают переключений ID.
3. **(Сложное)** Интегрируйте трекер SAM 2 на основе памяти (через `transformers`) в качестве альтернативного бэкенда трекера. Запустите SimpleTracker и SAM 2 на 30-секундном клипе с толпой и сравните количество переключений ID, вручную разметив истинные ID для 5 заметных людей.

## Ключевые термины

| Термин | Как говорят люди | Что это на самом деле означает |
|------|----------------|----------------------|
| Tracking-by-detection | «Сначала детектируй, потом сопоставляй» | Подетекторный детектор + венгерское сопоставление по IoU / внешнему виду |
| Фильтр Калмана | «Предсказание движения» | Линейная динамика + ковариация для плавных предсказаний трека и обработки окклюзий |
| Венгерский алгоритм | «Оптимальное сопоставление» | Решает задачу двудольного сопоставления минимальной стоимости; `scipy.optimize.linear_sum_assignment` |
| ByteTrack | «Второй проход низкоуверенных детекций» | Повторно сопоставляет несопоставленные треки с низкоуверенными детекциями для восстановления после коротких окклюзий |
| DeepSORT | «SORT + внешний вид» | Добавляет признак ReID для сопоставления между кадрами; лучше сохраняет ID |
| Банк памяти (memory bank) | «Фишка SAM 2» | Пространственно-временные признаки для каждого экземпляра, хранящиеся по кадрам; cross-attention заменяет явную ассоциацию |
| Object Multiplex | «Общая память SAM 3.1» | Единая общая память с запросами для каждого экземпляра для быстрого трекинга множества объектов |
| HOTA | «Современная метрика трекинга» | Раскладывается на точность детекции и ассоциации; стандарт сообщества |

## Дополнительное чтение

- [SORT (Bewley et al., 2016)](https://arxiv.org/abs/1602.00763) — минимальная статья по tracking-by-detection
- [DeepSORT (Wojke et al., 2017)](https://arxiv.org/abs/1703.07402) — добавляет признак внешнего вида
- [ByteTrack (Zhang et al., 2022)](https://arxiv.org/abs/2110.06864) — второй проход низкоуверенных детекций
- [BoT-SORT (Aharon et al., 2022)](https://arxiv.org/abs/2206.14651) — компенсация движения камеры
- [HOTA (Luiten et al., 2020)](https://arxiv.org/abs/2009.07736) — декомпозированная метрика трекинга
- [SAM 2 video segmentation (Meta, 2024)](https://ai.meta.com/sam2/) — трекер на основе памяти
- [SAM 3.1 Object Multiplex (Meta, март 2026)](https://ai.meta.com/blog/segment-anything-model-3/)