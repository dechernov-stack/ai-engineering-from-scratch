# 3D Gaussian Splatting с нуля

> Сцена — это облако из миллионов 3D-гауссиан. У каждой есть позиция, ориентация, масштаб, непрозрачность и цвет, зависящий от направления обзора. Растеризуйте их, сделайте backprop через растеризацию — готово.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 13 (3D Vision & NeRF), Phase 1 Lesson 12 (Tensor Operations), Phase 4 Lesson 10 (Diffusion basics optional)
**Time:** ~90 минут

## Learning Objectives

- Объяснить, почему 3D Gaussian Splatting вытеснил NeRF в качестве продакшен-стандарта для фотореалистичной 3D-реконструкции в 2026 году
- Перечислить шесть параметров на гауссиану (позиция, кватернион поворота, масштаб, непрозрачность, цвет в виде сферических гармоник, опциональный признак) и сколько чисел с плавающей точкой даёт каждый
- Реализовать с нуля 2D-растеризатор гауссианового сплаттинга с использованием композитинга `alpha`, а затем показать, как 3D-случай проецируется на тот же цикл
- Использовать `nerfstudio`, `gsplat` или `SuperSplat` для реконструкции сцены по 20-50 фотографиям и экспортировать её в расширение glTF `KHR_gaussian_splatting` или схему OpenUSD 26.03 `UsdVolParticleField3DGaussianSplat`

## The Problem

NeRF хранит сцену как веса MLP. Каждый отрисованный пиксель — это сотни запросов к MLP вдоль луча. Обучение занимает часы, рендеринг — секунды, а веса нельзя редактировать: если нужно подвинуть стул внутри сцены, придётся переобучать модель заново.

3D Gaussian Splatting (Kerbl, Kopanas, Leimkühler, Drettakis, SIGGRAPH 2023) заменил всё это. Сцена — это явный набор 3D-гауссиан. Рендеринг — это растеризация на GPU со скоростью 100+ fps. Обучение занимает минуты. Редактирование прямое: переместите подмножество гауссиан — и стул сдвинут. К 2026 году Khronos Group утвердила расширение glTF для гауссиановых сплатов, OpenUSD 26.03 поставляется со схемой гауссиановых сплатов, Zillow и Apartments.com рендерят недвижимость с их помощью, а большинство новых исследовательских статей по 3D-реконструкции — варианты базовой идеи 3DGS.

Ментальная модель проста, но в математике достаточно движущихся частей, из-за чего большинство введений начинают сразу с растеризации, пропуская проекции и сферические гармоники. В этом уроке мы соберём всё целиком — сначала 2D-версию, затем 3D-расширение.

## The Concept

### Что несёт в себе гауссиана

Одна 3D-гауссиана — это параметрическая капля в пространстве со следующими атрибутами:

```
position         mu         (3,)    centre in world coordinates
rotation         q          (4,)    unit quaternion encoding orientation
scale            s          (3,)    log-scales per axis (exponentiated at render time)
opacity          alpha      (1,)    post-sigmoid opacity [0, 1]
SH coefficients  c_lm       (3 * (L+1)^2,)   view-dependent colour
```

Поворот + масштаб образуют матрицу ковариации 3x3: `Sigma = R S S^T R^T`. Это форма гауссианы в 3D. Сферические гармоники позволяют цвету меняться в зависимости от направления обзора — бликов, едва заметного блеска, свечения, зависящего от угла обзора — без хранения текстур для каждого ракурса. При степени SH 3 получается 16 коэффициентов на канал цвета, то есть 48 чисел на гауссиану только для цвета.

Типичная сцена содержит 1-5 миллионов гауссиан. Каждая хранит примерно 60 чисел (3 + 4 + 3 + 1 + 48 + прочее). Это 240 МБ для сцены с пятью миллионами гауссиан — намного меньше эквивалентного облака точек с текстурой на точку, и на порядок меньше, чем веса MLP в NeRF при повторном рендеринге в высоком разрешении.

### Растеризация, а не рейкастинг

```mermaid
flowchart LR
    SCENE["Millions of 3D Gaussians<br/>(position, rotation, scale,<br/>opacity, SH colour)"] --> PROJ["Project to 2D<br/>(camera extrinsics + intrinsics)"]
    PROJ --> TILES["Assign to tiles<br/>(16x16 screen-space)"]
    TILES --> SORT["Depth-sort<br/>per tile"]
    SORT --> ALPHA["Alpha-composite<br/>front-to-back"]
    ALPHA --> PIX["Pixel colour"]

    style SCENE fill:#dbeafe,stroke:#2563eb
    style ALPHA fill:#fef3c7,stroke:#d97706
    style PIX fill:#dcfce7,stroke:#16a34a
```

Пять шагов, все дружественны к GPU. Никаких запросов к MLP на пиксель. Одна RTX 3080 Ti рендерит 6 миллионов сплатов со скоростью 147 fps.

### Шаг проекции

3D-гауссиана в мировой позиции `mu` с 3D-ковариацией `Sigma` проецируется в 2D-гауссиану в экранной позиции `mu'` с 2D-ковариацией `Sigma'`:

```
mu' = project(mu)
Sigma' = J W Sigma W^T J^T          (2 x 2)

W = viewing transform (rotation + translation of camera)
J = Jacobian of the perspective projection at mu'
```

Отпечаток 2D-гауссианы — это эллипс, чьи оси являются собственными векторами `Sigma'`. Каждый пиксель внутри этого эллипса получает вклад гауссианы, взвешенный на `exp(-0.5 * (p - mu')^T Sigma'^-1 (p - mu'))`.

### Правило альфа-композитинга

Для одного пикселя гауссианы, покрывающие его, сортируются от дальних к ближним (или, эквивалентно, от ближних к дальним с обратной формулой). Цвет композируется по тому же уравнению, что использовалось в любом полупрозрачном растеризаторе с 1980-х годов:

```
C_pixel = sum_i alpha_i * T_i * c_i

T_i = prod_{j < i} (1 - alpha_j)       transmittance up to i
alpha_i = opacity_i * exp(-0.5 * d^T Sigma'^-1 d)   local contribution
c_i = eval_SH(SH_i, view_direction)    view-dependent colour
```

Это **то же самое уравнение, что и объёмный рендеринг NeRF**, только над явным разреженным набором гауссиан вместо плотных сэмплов вдоль луча. Именно это тождество объясняет, почему качество рендеринга совпадает с NeRF — оба метода интегрируют одно и то же уравнение поля яркости.

### Почему это дифференцируемо

Каждый шаг — проекция, назначение тайлов, альфа-композитинг, вычисление SH — дифференцируем по параметрам гауссиан. Имея эталонное изображение, вычисляем потери по отрисованным пикселям, делаем backprop через растеризатор, обновляем все `(mu, q, s, alpha, c_lm)` градиентным спуском. За ~30 000 итераций гауссианы находят нужные позиции, масштабы и цвета.

### Денсификация и прунинг

Фиксированный набор гауссиан не может покрыть сложную сцену. Обучение включает два адаптивных механизма:

- **Клонирование (Clone)** гауссианы на её текущей позиции, когда величина градиента высока, а масштаб мал — реконструкции здесь не хватает детализации.
- **Разделение (Split)** крупной гауссианы на две меньшие, когда её градиент высок — одна крупная гауссиана слишком гладкая, чтобы вписаться в область.
- **Прунинг (Prune)** гауссиан, чья непрозрачность падает ниже порога — они не вносят вклад.

Денсификация запускается каждые N итераций. Сцена обычно растёт от ~100 тыс. исходных гауссиан (засеянных из точек SfM) до 1-5 млн к концу обучения.

### Сферические гармоники в одном абзаце

Цвет, зависящий от направления обзора, — это функция `c(direction)` на единичной сфере. Сферические гармоники — это базис Фурье для сферы. Обрезав на степени `L`, получаем `(L+1)^2` базисных функций на канал. Вычисление цвета для нового ракурса — это скалярное произведение между обученными коэффициентами SH и базисом, вычисленным в направлении обзора. Степень 0 = один коэффициент = постоянный цвет. Степень 3 = 16 коэффициентов = достаточно для захвата ламбертовского затенения, бликов и слабого отражения. Статьи по 3D Gaussian Splatting по умолчанию используют степень 3.

### Продакшен-стек 2026 года

```
1. Capture         smartphone / DJI drone / handheld scanner
2. SfM / MVS       COLMAP or GLOMAP derives camera poses + sparse points
3. Train 3DGS      nerfstudio / gsplat / inria official / PostShot (~10-30 min on RTX 4090)
4. Edit            SuperSplat / SplatForge (clean floaters, segment)
5. Export          .ply -> glTF KHR_gaussian_splatting or .usd (OpenUSD 26.03)
6. View            Cesium / Unreal / Babylon.js / Three.js / Vision Pro
```

### 4D и генеративные варианты

- **4D Gaussian Splatting** — гауссианы как функции времени; используется для объёмного видео (Superman 2026, "Helicopter" A$AP Rocky).
- **Генеративные сплаты** — модели «текст-в-сплат» (Marble от World Labs), которые галлюцинируют целые сцены.
- **3D Gaussian Unscented Transform** — вариант NVIDIA NuRec для симуляции автономного вождения.

```figure
cv3-gaussian-splat
```

## Build It

### Шаг 1: 2D-гауссиана

Сначала построим 2D-растеризатор. 3D-случай сводится к нему после проекции.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_2d_gaussian(means, covs, points):
    """
    means:  (G, 2)      centres
    covs:   (G, 2, 2)   covariance matrices
    points: (H, W, 2)   pixel coordinates
    returns: (G, H, W)  density at every pixel for every Gaussian
    """
    G = means.size(0)
    H, W, _ = points.shape
    flat = points.view(-1, 2)
    inv = torch.linalg.inv(covs)
    diff = flat[None, :, :] - means[:, None, :]
    d = torch.einsum("gpi,gij,gpj->gp", diff, inv, diff)
    density = torch.exp(-0.5 * d)
    return density.view(G, H, W)
```

`einsum` вычисляет квадратичную форму `diff^T Sigma^-1 diff` для каждой пары (гауссиана, пиксель).

### Шаг 2: 2D-растеризатор сплаттинга

Альфа-композитинг от ближних к дальним. Глубина в 2D не имеет смысла, поэтому мы используем обучаемый скаляр на гауссиану для задания порядка.

```python
def rasterise_2d(means, covs, colours, opacities, depths, image_size):
    """
    means:     (G, 2)
    covs:      (G, 2, 2)
    colours:   (G, 3)
    opacities: (G,)     in [0, 1]
    depths:    (G,)     per-Gaussian scalar used for ordering
    image_size: (H, W)
    returns:   (H, W, 3) rendered image
    """
    H, W = image_size
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=means.device),
        torch.arange(W, dtype=torch.float32, device=means.device),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1)

    densities = eval_2d_gaussian(means, covs, points)
    alphas = opacities[:, None, None] * densities
    alphas = alphas.clamp(0.0, 0.99)

    order = torch.argsort(depths)
    alphas = alphas[order]
    colours_sorted = colours[order]

    T = torch.ones(H, W, device=means.device)
    out = torch.zeros(H, W, 3, device=means.device)
    for i in range(means.size(0)):
        a = alphas[i]
        out += (T * a)[..., None] * colours_sorted[i][None, None, :]
        T = T * (1.0 - a)
    return out
```

Не быстро — реальная реализация использует тайловые CUDA-ядра, — но математика точная и полностью дифференцируемая.

### Шаг 3: Обучаемая 2D-сцена со сплатами

```python
class Splats2D(nn.Module):
    def __init__(self, num_splats=128, image_size=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        H, W = image_size, image_size
        self.means = nn.Parameter(torch.rand(num_splats, 2, generator=g) * torch.tensor([W, H]))
        self.log_scale = nn.Parameter(torch.ones(num_splats, 2) * math.log(2.0))
        self.rot = nn.Parameter(torch.zeros(num_splats))  # single angle in 2D
        self.colour_logits = nn.Parameter(torch.randn(num_splats, 3, generator=g) * 0.5)
        self.opacity_logit = nn.Parameter(torch.zeros(num_splats))
        self.depth = nn.Parameter(torch.rand(num_splats, generator=g))

    def covs(self):
        s = torch.exp(self.log_scale)
        c, si = torch.cos(self.rot), torch.sin(self.rot)
        R = torch.stack([
            torch.stack([c, -si], dim=-1),
            torch.stack([si, c], dim=-1),
        ], dim=-2)
        S = torch.diag_embed(s ** 2)
        return R @ S @ R.transpose(-1, -2)

    def forward(self, image_size):
        covs = self.covs()
        colours = torch.sigmoid(self.colour_logits)
        opacities = torch.sigmoid(self.opacity_logit)
        return rasterise_2d(self.means, covs, colours, opacities, self.depth, image_size)
```

`log_scale`, `opacity_logit` и `colour_logits` — все это неограниченные параметры, преобразуемые через нужную функцию активации во время рендеринга. Это стандартный паттерн для каждой реализации 3DGS.

### Шаг 4: Подгонка 2D-гауссиан под целевое изображение

```python
import math
import numpy as np

def make_target(size=64):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.zeros((size, size, 3), dtype=np.float32)
    # Red circle
    mask = (xx - 20) ** 2 + (yy - 20) ** 2 < 10 ** 2
    img[mask] = [1.0, 0.2, 0.2]
    # Blue square
    mask = (np.abs(xx - 45) < 8) & (np.abs(yy - 40) < 8)
    img[mask] = [0.2, 0.3, 1.0]
    return torch.from_numpy(img)


target = make_target(64)
model = Splats2D(num_splats=64, image_size=64)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(200):
    pred = model((64, 64))
    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 40 == 0:
        print(f"step {step:3d}  mse {loss.item():.4f}")
```

За 200 шагов 64 гауссианы выстраиваются в две фигуры. Это и есть вся идея целиком — градиентный спуск по явным геометрическим примитивам.

### Шаг 5: От 2D к 3D

3D-расширение сохраняет тот же цикл. Добавления:

1. Поворот на гауссиану теперь задаётся кватернионом вместо одного угла.
2. Ковариация — это `R S S^T R^T`, где `R` построена из кватерниона и `S = diag(exp(log_scale))`.
3. Проекция `(mu, Sigma) -> (mu', Sigma')` использует внешние параметры камеры и якобиан перспективной проекции в `mu`.
4. Цвет становится разложением по сферическим гармоникам; вычисляется для направления обзора.
5. Сортировка по глубине ведётся по реальному z в системе координат камеры вместо обучаемого скаляра.

Каждая продакшен-реализация (`gsplat`, `inria/gaussian-splatting`, `nerfstudio`) делает именно это на GPU с помощью тайловых CUDA-ядер.

### Шаг 6: Вычисление сферических гармоник

Базис SH до степени 3 содержит 16 слагаемых на канал. Вычисление:

```python
def eval_sh_degree_3(sh_coeffs, dirs):
    """
    sh_coeffs: (..., 16, 3)   last dim is RGB channels
    dirs:      (..., 3)       unit vectors
    returns:   (..., 3)
    """
    C0 = 0.282094791773878
    C1 = 0.488602511902920
    C2 = [1.092548430592079, 1.092548430592079,
          0.315391565252520, 1.092548430592079,
          0.546274215296039]
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result = C0 * sh_coeffs[..., 0, :]
    result = result - C1 * y[..., None] * sh_coeffs[..., 1, :]
    result = result + C1 * z[..., None] * sh_coeffs[..., 2, :]
    result = result - C1 * x[..., None] * sh_coeffs[..., 3, :]

    result = result + C2[0] * xy[..., None] * sh_coeffs[..., 4, :]
    result = result + C2[1] * yz[..., None] * sh_coeffs[..., 5, :]
    result = result + C2[2] * (2.0 * z2 - x2 - y2)[..., None] * sh_coeffs[..., 6, :]
    result = result + C2[3] * xz[..., None] * sh_coeffs[..., 7, :]
    result = result + C2[4] * (x2 - y2)[..., None] * sh_coeffs[..., 8, :]

    # degree 3 terms omitted here for brevity; full 16-coefficient version in the code file
    return result
```

Обученные `sh_coeffs` хранят «цвет в каждом направлении» для данной гауссианы. Во время рендеринга вычисляется относительно текущего направления обзора и получается 3-вектор RGB.

## Use It

Для реальной работы с 3DGS используйте `gsplat` (Meta) или `nerfstudio`:

```bash
pip install nerfstudio gsplat
ns-download-data example
ns-train splatfacto --data path/to/data
```

`splatfacto` — это тренер 3DGS от nerfstudio. Прогон занимает 10-30 минут на RTX 4090 для типичной сцены.

Варианты экспорта, важные в 2026 году:

- `.ply` — необработанное облако гауссиан (переносимый формат, самый большой файл).
- `.splat` — квантованный формат PlayCanvas / SuperSplat.
- glTF `KHR_gaussian_splatting` — стандарт Khronos, переносимый между вьюверами (Feb 2026 RC).
- OpenUSD `UsdVolParticleField3DGaussianSplat` — нативный для USD, для пайплайнов NVIDIA Omniverse и Vision Pro.

Для 4D/динамических сцен `4DGS` и `Deformable-3DGS` расширяют тот же механизм с переменными во времени средними и непрозрачностями.

## Ship It

Этот урок производит:

- `outputs/prompt-3dgs-capture-planner.md` — промпт, который планирует сессию съёмки (количество фото, траекторию камеры, освещение) для заданного типа сцены.
- `outputs/skill-3dgs-export-router.md` — навык, который выбирает правильный формат экспорта (`.ply` / `.splat` / glTF / USD) в зависимости от целевого вьювера или движка.

## Exercises

1. **(Easy)** Запустите приведённый выше 2D-тренер сплатов на другом синтетическом изображении. Варьируйте `num_splats` в `[16, 64, 256]` и постройте график MSE от шага для каждого значения. Определите точку убывающей отдачи.
2. **(Medium)** Расширьте 2D-растеризатор для поддержки RGB-цветов на гауссиану, зависящих от скалярного «угла обзора» через гармонику степени 2. Обучите на паре целевых изображений и проверьте, что модель реконструирует оба.
3. **(Hard)** Склонируйте `nerfstudio` и обучите `splatfacto` на съёмке из 20 фотографий любой сцены, которая у вас есть (стол, растение, лицо, комната). Экспортируйте в glTF `KHR_gaussian_splatting` и откройте во вьювере (Three.js `GaussianSplats3D`, SuperSplat, Babylon.js V9). Сообщите время обучения, количество гауссиан и fps рендеринга.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| 3DGS | «гауссиановые сплаты» | Явное представление сцены в виде миллионов 3D-гауссиан с позицией, поворотом, масштабом, непрозрачностью и цветом в виде SH на гауссиану |
| Covariance | «Форма гауссианы» | `Sigma = R S S^T R^T`; ориентация и анизотропный масштаб одной гауссианы |
| Alpha compositing | «Смешивание от дальних к ближним» | То же уравнение, что и объёмный рендеринг NeRF, теперь над явным разреженным набором |
| Densification | «Клонирование и разделение» | Адаптивное добавление новых гауссиан там, где реконструкция недостаточно детализирована |
| Pruning | «Удаление низкой непрозрачности» | Удаление гауссиан, чья непрозрачность упала почти до нуля во время обучения |
| Spherical harmonics | «Цвет, зависящий от ракурса» | Базис Фурье на сфере; хранит цвет как функцию направления обзора |
| Splatfacto | «3DGS от nerfstudio» | Самый простой путь к обучению 3DGS в 2026 году |
| `KHR_gaussian_splatting` | «стандарт glTF» | Расширение Khronos 2026 года, делающее 3DGS переносимым между вьюверами и движками |

## Further Reading

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — оригинальная статья
- [gsplat (Meta/nerfstudio)](https://github.com/nerfstudio-project/gsplat) — продакшен-растеризатор на CUDA
- [nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) — эталонный рецепт обучения
- [Khronos KHR_gaussian_splatting extension](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md) — переносимый формат 2026 года
- [OpenUSD 26.03 release notes](https://openusd.org/release/) — схема `UsdVolParticleField3DGaussianSplat`
- [THE FUTURE 3D State of Gaussian Splatting 2026](https://www.thefuture3d.com/blog-0/2026/4/4/state-of-gaussian-splatting-2026) — обзор отрасли