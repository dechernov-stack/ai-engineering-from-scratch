# Настройка GPU и облако

> Для обучения ради обучения CPU вполне подходит. Для настоящего обучения моделей нужен GPU.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 минут

## Цели обучения

- Проверить доступность локального GPU с помощью `nvidia-smi` и CUDA API PyTorch
- Настроить Google Colab с GPU T4 для бесплатных экспериментов в облаке
- Сравнить производительность умножения матриц на CPU и GPU и измерить ускорение
- Оценить размер самой большой модели, помещающейся в вашу VRAM, используя эмпирическое правило для fp16

## Проблема

Большинство уроков фаз 1-3 нормально работают на CPU. Но как только вы начинаете обучать CNN, трансформеры или LLM (фазы 4+), вам нужно ускорение на GPU. Обучение, занимающее 8 часов на CPU, на GPU занимает 10 минут.

У вас есть три варианта: локальный GPU, облачный GPU или Google Colab (бесплатно).

## Концепция

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

```figure
s0-gpu-dispatch
```

## Построение

### Вариант 1: Локальный GPU NVIDIA

Проверьте, есть ли он у вас:

```bash
nvidia-smi
```

Установите PyTorch с CUDA:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Вариант 2: Google Colab

1. Перейдите на [colab.research.google.com](https://colab.research.google.com)
2. Runtime > Change runtime type > T4 GPU
3. Запустите `!nvidia-smi` для проверки

Загружайте ноутбуки из этого курса прямо в Colab.

### Вариант 3: Облачный GPU

Для Lambda Labs, RunPod или Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Нет GPU? Не проблема.

Большинство уроков работают на CPU. Те, для которых нужен GPU, будут отмечены отдельно и содержать ссылки на Colab.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## Построение: сравнение GPU и CPU

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## Упражнения

1. Запустите приведённый выше бенчмарк и сравните время на CPU и GPU
2. Если у вас нет GPU, запустите его в Google Colab и сравните результаты
3. Проверьте, сколько у вас памяти GPU, и оцените размер самой большой модели, которая туда поместится (эмпирическое правило: 2 байта на параметр для fp16)

## Ключевые термины

| Термин | Как говорят | Что это значит на самом деле |
|------|----------------|----------------------|
| CUDA | «программирование GPU» | платформа параллельных вычислений NVIDIA, позволяющая запускать код на GPU |
| VRAM | «память GPU» | видеопамять на GPU, отдельная от системной RAM. Ограничивает размер модели. |
| fp16 | «половинная точность» | 16-битное число с плавающей запятой, использует вдвое меньше памяти, чем fp32, с минимальной потерей точности |
| Tensor Core | «быстрое матричное железо» | специализированные ядра GPU для умножения матриц, в 4-8 раз быстрее обычных ядер |