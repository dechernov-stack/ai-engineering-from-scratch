# Смесь экспертов (МЭ)

> Плотный трансформатор 70B активирует каждый параметр для каждого токену. 671B MoE активирует только 37B на токен и побеждает его на каждом этапе.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~45 minutes

## Проблема

FLOPs плотного трансформатора при выводе равняются его количеству параметров (2 раза для пропуска вперед). Расширяйте плотную модель и каждый токен оплачивает полный счет. К 2024 году граница ударила по вычислительной стене: чтобы быть значительно умнее, вам нужно было экспоненциально больше FLOPs на токен.

Смесь экспертов разрушает эту связь.`E`независимые эксперты + маршрутизатор , который выбирает `k`эксперты на токен.`E × FFN_size`. Активные параметры на токен = `k × FFN_size`. Типичная конфигурация 2026 года: `E=256`- Я не знаю .`k=8`. Степень хранения с `E`, вычислить масштабы с `k`- Я не знаю .

Граница 2026 почти полностью MoE: DeepSeek-V3 (671B в общей сложности / 37B активный), Mixtral 8×22B, Qwen2.5-MoE, Llama 4, Kimi K2, gpt-oss.

## Концепция

![MoE layer: router selects k of E experts per token](../assets/moe.svg)

### Смена FFN

Плотный блок трансформатора:

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

Блок МОЭ:

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

Каждый эксперт является независимым FFN (обычно SwiGLU).`k`экспертов и получает закрытую смесь их выходов.

### Проблема с балансом нагрузки

Если маршрутизатор пропускает 90% токенов через эксперта 3, остальные эксперты голодают.

1. **Auxiliary load-balancing loss**Добавьте штраф пропорционально варианту в экспертном использовании.
2. **Expert capacity + token dropping**(ранний переключатель).`C × N/E`Токены, токены переполнения, не входящие в слой.
3. **Auxiliary-loss-free balancing**(DeepSeek-V3). Добавьте изученный налог на эксперта, который смещает выбор верхнего класса маршрутизатора.

Подход DeepSeek-V3: после каждого этапа обучения для каждого эксперта проверьте, превышает ли его использование цель или ниже.`±γ`. Изборные приложения `scores + bias`. Экспертные вероятности , используемые для заграждения , являются сырыми `scores`Не изменяется. Отключает маршрутизацию от выражения.

### Общие эксперты

DeepSeek-V2/V3 также делит экспертов на *shared* и *routed*. Каждый токен проходит через всех общих экспертов.

### Специалисты по мелкозрелинам

Классический MoE (GShard, Switch): каждый эксперт имеет ширину, равною полной FFN. `E`является небольшим (864), `k`является небольшим (12).

Современная мелкозерновая МОЕ (DeepSeek-V3, Qwen-MoE): каждый эксперт уменьшается (размер 1/8 FFN). `E`является большим (256+), `k`Это больше (8+).`C(256, 8) = 400 trillion`Качество повышается, задержка остается равной.

### Профиль затрат

На токен, на слой:

| Config | Active params / token | Total params |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (dense) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3 уступает Llama 3 70B (плотный) почти на всех эталонах при выполнении **fewer active FLOPs per token**Больше параметров = больше знаний. Более активные FLOPs = больше вычислений на токен.

### Улов: память

Все эксперты живут на GPU независимо от того, какой из них запускается. Модель 671B требует ~1,3 ТБ VRAM для веса fp16.

```figure
expert-routing
```

## Создайте его

Посмотрите .`code/main.py`. компактный слой MoE в чистом виде с:

- `n_experts=8`Специалисты по SWIGU (по одному линейному, для иллюстрации)
- вершина k=2 маршрутизация
- нормализованные весы замок для мягкого максимума
- балансирование без вспомогательных потерь через предвзятость эксперта

### Шаг 1: маршрутизатор

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

Это трюк DeepSeek-V3  bias корректирует дисбаланс нагрузки без управления прогнозами модели.

### Шаг 2: запустить 100 токенов через маршрутизатор

Следить за тем, какие эксперты стреляют, как часто.`-γ`для переиспользованных экспертов, `+γ`для недостаточно используемого), использование сближается с однородной распределением в течение нескольких итераций.

### Шаг 3: сравнение параметров

Печать "плотный эквивалент" конфигурации MoE. DeepSeek-V3-образный: 256 маршрутизированный + 1 разделенный, 8 активный, d_model=7168. Общий показатель параметров - это очаровательно.

## Используйте его

Погрузка лица:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026 вывод производства: vLLM поддерживает маршрутизацию MoE нативным способом. SGLang имеет самый быстрый путь эксперт-параллель.

**When to pick MoE:**
- Вы хотите высокое качество по меньшей стоимости за токен.
- У вас есть инфраструктура VRAM / эксперт-параллель.
- Ваша рабочая нагрузка - это токен-тяжелая (чаты, код) не контекст-тяжелая (длинные документы).

**When NOT to pick MoE:**
- Развертывание границы  Вы платите полный залог за любой активный FLOP.
- Критическая для латенции однопользовательская служба  экспертный маршрутизатор добавляет общие расходы.
- Малые модели (<7B)  Преимущество качества MoE представляется только выше порога вычисления (~6B активных параметров).

## Погрузи его

Посмотрите .`outputs/skill-moe-configurator.md`. Умение выбирает E, k и общий экспертный план для нового МЭ с данным параметром бюджета, обучения токенов и цели развертывания.

## Упражнения

1. **Easy.**Беги .`code/main.py`Посмотрите, как обновление без потерь и дополнительных предвзятости уравняет использование экспертов более чем в 50 итерациях.
2. **Medium.**Замените наученный маршрутизатор на хеш-на основе маршрутизатора (детерминистический, без обучения). Сравните качество и баланс. Почему наученный маршрутизатор лучше?
3. **Hard.**Реализация "рассмотренного маршрутизации" в стиле GRPO (трек DeepSeek-V3.2): запись, которую специалисты выпускают во время вывода, принуждение к тому же маршрутизации во время вычисления градиентов. Измерение влияния на настройку политики игры-градиента.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Expert | "One FFN among many" | An independent feed-forward network; parameters dedicated to a sparse slice of the FFN computation. |
| Router | "The gate" | A tiny linear layer that scores each token against each expert; top-k selection. |
| Top-k routing | "k active experts per token" | Each token's FFN computation goes through exactly k experts, weighted by gate. |
| Auxiliary loss | "Load-balance penalty" | Extra loss term that penalizes skewed expert usage. |
| Auxiliary-loss-free | "DeepSeek-V3's trick" | Balance via per-expert bias on the router's selection only; no extra gradient. |
| Shared expert | "Always on" | Extra expert through which every token passes; captures common knowledge. |
| Expert parallelism | "Shard by expert" | Distribute different experts to different GPUs; route tokens across the network. |
| Sparsity | "Active params < total params" | The ratio `k × expert_size / (E × expert_size)`; 37/671 ≈ 5.5% for DeepSeek-V3. |

## Дальнейшее чтение

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) идея.
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) Свич, классический MoE.
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) Микстральный 8×7Б.
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) MLA + вспомогательный безотказный МОЕ + МТП.
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) балансовая бумага на основе предвзятости.
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) тонкозерно-поделившийся + общий-эксперт разделил использование маршрутизатора этого урока.
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) оригинальная совместная экспертная работа.
