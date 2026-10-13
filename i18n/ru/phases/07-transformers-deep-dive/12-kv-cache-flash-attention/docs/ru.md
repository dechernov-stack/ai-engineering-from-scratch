# КВ-каше, Flash Attention и Инференция оптимизация

> Тренинг параллельный и связан с FLOP. Инференция серийная и связанная с памятью.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~75 minutes

## Проблема

Наивный авторегрессивный декодер .`O(N²)`Работа для создания `N`Токены: на каждом шаге он пересчитывает внимание над полным префиксом. Для 4K-токенного ответа, который является 16M внимания операции, большинство из них избыточные. Каждое скрытое состояние префикса токены является детерминистичным после вычисления  вам нужно только запустить запрос нового токены против кешированных ключей и значения всех предыдущих.

Кроме того, внимание само по себе перемещает много данных. Стандартное внимание материализует матрицу N×N, N×d softmax выпуск, N×d окончательный выпуск  слишком много читает и пишет в HBM. Для N≥2K внимание становится связано с памятью, прежде чем оно становится FLOP-обусловленным. Классические ядра внимания недостаточно используют современные графические процессоры 410×.

Две оптимизации, оба из Дао и др., подтолкнули граничное выводение от "медленного" к "быстрому":

1. **KV cache.**Сохранить векторы K и V каждого предварительного токен.`O(N²)`к `O(N)`на каждое поколение.
2. **Flash Attention.**Наталь вычисления внимания, чтобы полная матрица N×N никогда не попала в HBM. Все мягкое максимум + мамуль происходит в SRAM. 24× ускорение стенового часа на A100; 510× на H100 с FP8.

К 2026 году оба являются универсальными. Каждый производственный сбор выводов (vLLM, TensorRT-LLM, SGLang, llama.cpp) предполагает их.

## Концепция

![KV cache growth and Flash Attention tiling](../assets/kv-cache-flash-attn.svg)

### КВ-кашевая математика

За каждый слой декодера, за каждый токен, за каждый голова:

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V
```

Для модели 7B с 32 слоями, 32 головами, d_head=128, fp16:

```
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

Для Llama 3 70B (80 слоев, d_head=128, GQA с 8 KV-головками):

```
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

Вот почему Llama 3 70B в контексте 128K требует большей части A100 40 GB только для кеша KV в размере партии 1.

**GQA is the KV-cache win.**MHA с 64 головками будет 32 ГБ. MLA сжимает еще дальше.

Сдвините размер шкалы и посмотрите, как движется размер шкалы.

```figure
kv-cache-sizer
```

### Внимание  трюк по тильцам

Стандартное внимание:

```
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

Три поездки HBM в обратную сторону. На H100, полоса пропускания HBM составляет 3 ТБ/с; SRAM составляет 30 ТБ/с. Каждое поездка HBM является фактором 10-го замедления против держать все на чипе.

Внимание:

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

Один поход по HBM на плитку.`O(N²)`к `O(N)`. Задним проходом пересчитываются некоторые значения из перехода вперед вместо того, чтобы хранить их  еще одна память выигрыш.

**Numerical trick.**Работает softmax поддерживает `(max, sum)`Не приближение  Flash Attention вычисляет битовой идентичный выход на стандартную внимание (модуль fp16 неассоциативность).

**Version evolution:**

| Version | Year | Key change | Speedup on reference hardware |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | Tiled SRAM kernel | 2× on A100 |
| Flash 2 | 2023 | Better parallelism, causal-first ordering | 3× on A100 |
| Flash 3 | 2024 | Hopper asynchrony, FP8 | 1.5–2× on H100 (~740 TFLOPs FP16) |
| Flash 4 | 2026 | Blackwell 5-stage pipeline, software exp2 | Inference-first (forward only initially) |

Flash 4 будет проходить только при запуске.

### Спекулятивная декодировка  другой латентность победить

Дешевая модель предлагает N токенов. Большая модель проверяет все N параллельно. Если проверка принимает k токены, вы заплатили 1 большой модель проход на k поколений. Типичный k=35 на код и прозу.

2026 неисправности:
- **EAGLE 2 / Medusa.**Интегрированные заголовки, которые разделяют скрытые состояния проверщика. 23x ускорение без потери качества.
- **Speculative decoding with draft model.**2×4x ускорение на потребительском оборудовании.
- **Lookahead decoding.**Якоби итерация, не требуется ниши, но бесплатная.

### Непрекращающееся выбросывание

Классическое выводное вывод: подождите, пока самая медленная последовательность закончится, а затем начните новую партию.

Продолжительная партийность (первое отправлялось в Orca, теперь в vLLM, TensorRT-LLM, SGLang): обменять новые запросы в партию, как только закончатся старые.

### PagedAttention  КВ-каш в виде виртуальной памяти

Главная функция vLLM. КВ-каш распределяется в 16-токенных блоках; таблица страниц отображает логические положения физических блоков. позволяет делиться КВ на параллельных образцах (поиск лучей, параллельное выборка образцов), префиксы горячего смена для оперативного кэширования и дефрагментативной памяти. 4x улучшение пропускной способности по сравнению с наивным сопредельным распределением.

```figure
flash-attention-memory
```

## Создайте его

Посмотрите .`code/main.py`Мы реализуем:

1. Я наивный .`O(N²)`инкрементальный декодер.
2. А .`O(N)`КВ-каш-декодер.
3. Softmax, который имитирует алгоритм Flash Attention.

### Шаг 1: КВ-каш

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

Просто: продолжайте расти векторы на токен K, V в списках на слой, на голову.

### Шаг 2: плитка мягкая

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

Выход с битом идентичный к `softmax(qK) V`в один кадр, но в любой момент рабочий набор является `tile × d_head`блок, а не полный `N × d_head`- Я не знаю .

### Шаг 3: сравните наивное и кешированное декодирование на генерации 100 токенов

Считайте операции внимания.`O(N²)`= 5050. Запчасти: `O(N)`Код отпечатывает оба.

## Используйте его

```python
# HuggingFace transformers auto-enables KV cache on decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # use FA3 if Hopper
    torch_dtype="bfloat16",
)
# generate() uses KV cache automatically
```

Производство VLLM:

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

Префиксный кешинг по запросам - это большая победа 2026 года  один и тот же системный промплет, несколько примеров или длинный контекстный документ повторно использует KV по звонкам.

## Погрузи его

Посмотрите .`outputs/skill-inference-optimizer.md`. Умение выбирает внимание внедрения, КВ кеширование стратегии, квантования и спекулятивный декодирование для новой выводы развертывания.

## Упражнения

1. **Easy.**Беги .`code/main.py`. Убедитесь, что наивные и кешированные декодеры дают одинаковый выход; обратите внимание на разницу в оптосчете.
2. **Medium.**Внедрить кеширование префикса: при задании P и нескольких завершениях, запустите один проход вперед над P, чтобы заполнить кеширование KV, а затем разделить за каждый завершение.
3. **Hard.**Внедрение игрушки PagedAttention: КВ-каш в фиксированных 16-токенных блоках с свободным списком. Когда последовательность закончится, верните ее блоки в бассейн. Симулируйте 1000 завершений чата с различной длиной. Сравните фрагментацию памяти с сопутствующим распределением.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| KV cache | "The trick that makes decoding fast" | Stored K and V from every prefix token; new queries attend to them instead of recomputing. |
| HBM | "GPU main memory" | High Bandwidth Memory; 80 GB on H100, 192 GB on B200. ~3 TB/s bandwidth. |
| SRAM | "On-chip memory" | Per-SM fast memory, ~256 KB per SM on H100. ~30 TB/s bandwidth. |
| Flash Attention | "Tiled attention kernel" | Computes attention without materializing N×N in HBM. |
| Continuous batching | "No-wait batching" | Swap finished sequences out, new ones in, without draining the batch. |
| PagedAttention | "vLLM's headline" | KV cache allocated in fixed blocks with a page table; eliminates fragmentation. |
| Prefix caching | "Reuse long prompts" | Cache KV for a shared prefix across requests; major cost cut for agents. |
| Speculative decoding | "Draft + verify" | Cheap draft model proposes tokens; big model verifies k in one pass. |

## Дальнейшее чтение

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) Флэш 1.
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) Флэш 2.
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) Флэш 3.
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) 5 этапов трубопровода Blackwell и трюк программного обеспечения; прочитайте repo README для предупреждений о запуске только вперед, упомянутых в этом уроке.
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) ВВЛМ-папировый.
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) декодирование спецификаций.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) Документ EAGLE-1/2 для комплексного подхода к проектам, который цитируется в уроке.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) подход Медузы, указанный рядом с ЭГЛЕ.
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) каноническая глубокая погружение на 16-токенный блок и дизайн таблицы страниц.
