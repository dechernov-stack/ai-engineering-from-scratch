# Настройка GPU и облака

> Тренировка на процессоре - это хорошо для обучения.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 minutes

## Цели обучения

- Проверьте наличие локальной графической системы с помощью `nvidia-smi`и API Пайторча
- Настройка Google Colab с GPU T4 для бесплатных облачных экспериментов
- Сравните умножение матрицы на процессоре против процессоре и измерьте скорость
- Оценить самую большую модель, которая вписывается в ваш VRAM с помощью правила fp16

## Проблема

Большинство уроков 1-3 фаз проходят хорошо на процессоре. Но как только вы начинаете тренировать CNN, трансформаторы или LLM (фазы 4+), вам нужно ускорение GPU.

У вас есть три варианта: локальный GPU, облачный GPU или Google Colab (бесплатный).

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

## Создайте его

### Выбор 1: локальный NVIDIA GPU

Проверь , есть ли у тебя:

```bash
nvidia-smi
```

Установка PyTorch с CUDA:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Выбор 2: Google Colab

1. Иди к [colab.research.google.com](https://colab.research.google.com)
2. Время запуска > Изменение типа времени запуска > T4 GPU
3. Беги .`!nvidia-smi`для проверки

Загрузите ноутбуки с этого курса прямо в Колаб.

### Выбор 3: облачный графический процессор

Для Lambda Labs, RunPod или Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Нет никаких проблем.

Большинство уроков работают на процессоре. Те, кто нуждается в GPU, скажут так и включат ссылки Colab.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## Создайте его: GPU против CPU

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

1. Запустите эталон выше и сравнивайте время процессора против процессора графики
2. Если у вас нет GPU, запустите его в Google Colab и сравнивайте
3. Проверьте, сколько у вас есть памяти GPU и оценить самую большую модель, которую вы можете поместить (половое правило: 2 байта на параметр для fp16)

## Ключевые условия

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| CUDA | "GPU programming" | NVIDIA's parallel computing platform that lets you run code on the GPU |
| VRAM | "GPU memory" | Video RAM on the GPU, separate from system RAM. Limits model size. |
| fp16 | "Half precision" | 16-bit floating point, uses half the memory of fp32 with minimal accuracy loss |
| Tensor Core | "Fast matrix hardware" | Specialized GPU cores for matrix multiplication, 4-8x faster than regular cores |
