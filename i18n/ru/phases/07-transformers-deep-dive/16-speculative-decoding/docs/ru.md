# Спекулятивный декодирование  Проект, проверка, повторение

> Авторегрессивное декодирование - это серийное. Каждый токен ждет предыдущего. Спекулятивный декодирование ломает цепочку: дешевая модель чертает N токенов, дорогая модель проверяет все N в одном проходе вперед.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 07 (GPT Causal LM), Phase 7 · 12 (KV Cache & Flash Attention)
**Time:** ~60 minutes

## Проблема

При выборе 70B LLM один токен занимает ~30 мс на H100. При модели 3B проект занимает ~3 мс. Если мы позволим 3B проект 5 токенов вперед, то запустим 70B *однажды* для проверки всех 5, общий объем `5×3 + 30 = 45 ms`для до 5 принятых токенов  против `5×30 = 150 ms`Это полный спекулятивный декодирующий пич: обменят небольшое количество дополнительной памяти GPU (модель проекта) на 24x меньшую латентность декода.

Спекулятивная выборка, введенная Leviathan и соавторами (2023) и Chen и соавторами одновременно, гарантирует, что последовательность выхода **identically distributed**- Нет компромисса в качестве, просто быстрее.

Четыре семьи пар проверщиков проекта доминируют в выводе 2026 года:

1. **Vanilla speculative (Leviathan 2023).**Отдельная модель проекта (например, Llama 3 1B) + проверчик (например, Llama 3 70B).
2. **Medusa (Cai 2024).**Многочисленные головы декодирования на проверяющем прогнозируют позиции `t+1..t+k`Параллельно, нет отдельной модели.
3. **EAGLE family (Li 2024, 2025).**Легкий черновик, который повторно использует скрытые состояния проверщика; более близкая доля принятия, чем ваниль; 34× типичный.
4. **Lookahead decoding (Fu 2024).**Итерация Якоби, не требуется никакой модели, самооценка, ниша, но без зависимости.

Каждый производственный выводный станок в 2026 году отправляет спекулятивный декодирование по умолчанию. vLLM, TensorRT-LLM, SGLang и llama.cpp все поддерживают по крайней мере ваниль + EAGLE-2.

## Концепция

### Основной алгоритм

Учитывая проверку`M_q`и более дешевый проект `M_p`В следующем случае:

1. Пусть .`x_1..x_k`быть предварительным уже декодированным.
2. **Draft**: использование `M_p`- авторегрессивно предложить`d_{k+1}, d_{k+2}, ..., d_{k+N}`с вероятностями проекта `p_1..p_N`- Я не знаю .
3. **Verify in parallel**: бежать `M_q`- Как раз .`x_1..x_k, d_{k+1}, ..., d_{k+N}`, получая вероятности проверки `q_1..q_{N+1}`для позиций `k+1..k+N+1`- Я не знаю .
4. **Accept/reject each draft token left to right**: для каждого `i`, принимать с вероятностью `min(1, q_i(d_i) / p_i(d_i))`- Я не знаю .
5. При первом отказе на позиции `j`: образец `t_j`от "рестабельного" распределения `(q_j - p_j)_+`Все проекты после`j`выброшены.
6. Принимая все .`N`: образец один дополнительный токен `t_{N+1}`от `q_{N+1}`(бесплатный бонусный токен).

Остаточный трюк распределения - это математическая информация , которая поддерживает распределение выхода точно так же , как если бы `M_q`Я взял образцы с нуля.

### Что определяет скорость

Пусть .`α`= ожидаемый процент принятия на проектный токен.`c`= соотношение затрат проекта к проверяющему.

- Наивное поколение делает один большой модели вызов на токен.
- Спекулятивный делает 1 большой модели звонок на `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)`токены когда `α`- Высоко.

Типичное правило .`α = 0.75`и `N = 5`В целом, это будет меньше, чем в случае с большими моделями.

**α depends on:**

- Как хорошо проект приближается к проверяющему.
- Стратегия декодирования. жадный проект против жадный проверяющий: высокий α. Образец температуры: сложнее совпадать; приемность падает.
- Тип задач: код и структурированный выход принимают больше (предсказуемый); свободное творческое письмо принимает меньше.

### Медуза  проекты без проекта модели

Medusa заменяет проектную модель дополнительными выходной головками на проверяющем устройстве.`t`В следующем случае:

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

Каждая голова выпускает свои собственные логиты. При выводе вы пробираете из каждой головы, чтобы получить последовательность кандидатов, затем проверяете одним проходом вперед с помощью схемы внимания дерева, которая учитывает все продолжения кандидатов сразу.

Просы: нет второй модели. Недостатки: добавляет обучаемые параметры; требует контролируемой стадии тонкого настройки (~ 1B токенов); процент принятия немного ниже, чем ванильный спекулятив с хорошим проектом.

### ЭГИЛ  лучшее чертеж путем повторного использования скрытых состояний

EAGLE-1/2/3 (Li et al., 20242025) делает проектную модель крошечным трансформатором (обычно 1 слой), который поглощает скрытые состояния последнего слоя проверяющего. Поскольку проект видит представление характеристик проверяющего, его прогнозы сильно коррелируют с распределением выхода проверяющего.

ВЛЛМ и СГЛАНГ отправляют EAGLE-2/3 как стандартный маршрут спецификации для Llama 3/4 и Qwen 3.

### Танцы КВ

Проверка данных `N`Продолжение действия вверенного вверенного в одно первое пропуска.`N`Если некоторые проекты отклонены, вы должны перевернуть кеши обратно на принятую длину предварительного приложения.

Реализация продукции (вЛЛМ) `--speculative-config`Это не сложно, но это нелегко.

```figure
draft-verify-tokens
```

## Создайте его

Посмотрите .`code/main.py`Мы реализуем основной алгоритм спекулятивной обработки образцов (этап отторжения + остаточное распределение) с:

- "Большая модель", которая является детерминистическим-мягким максимумом по сравнению с ручным распределением (так что мы можем проверить математическую приемлемость аналитически).
- "Дизайн-модель", которая является нарушением большой модели.
- Приемная/отрицательная петля, которая производит такое же маргинальное распределение, как прямая выборка.

### Шаг 1: шаг отклонения

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u`- это однородный случайный номер.`q_prob`является вероятность проверяющего для составленного токена. `p_prob`Теорема Левиатана заключается в том, что решение Бернулли, после чего происходит отбор образцов из остатка при отклонении, точно сохраняет распределение проверщика.

### Шаг 2: остаточный распределение

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

Отчитать`p`от `q`С точки зрения элементов, сжать отрицательные значения до нуля, ренормализировать.

### Шаг 3: один спекулятивный шаг

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

Пять принятых → один бонус → шесть токенов, произведенных в одном проходе проверки.

### Шаг 4: измерять процент принятия

Выполните 10 000 спекулятивных шагов на различных уровнях качества проекта.

### Шаг 5: проверка эквивалентности распределения

Эпирически: гистограмма токенов, произведенная спекулятивной петлей, должна соответствовать гистограммам, произведенному путем отбора образцов непосредственно от проверщика.

## Используйте его

Производство:

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-config '{"method": "eagle", "model": "/models/llama-3.1-eagle-70b", "draft_tensor_parallel_size": 1, "num_speculative_tokens": 5}'

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-config '{"method": "draft_model", "model": "meta-llama/Llama-3.2-1B-Instruct", "num_speculative_tokens": 5}'
```

TensorRT-LLM имеет самый быстрый путь по Медузе к середине 2026 года. `faster-whisper`Завершает декодирование для Whisper-large с небольшим чертежом.

**Picking a draft:**

| Strategy | When to pick | Speedup |
|----------|--------------|---------|
| Vanilla draft (1B/3B Llama family) | Fast prototype, no training | 1.8–2.3× |
| Medusa heads | You can fine-tune the verifier | 2–3× |
| EAGLE-2 / 3 | Production, max speed | 3–4× |
| Lookahead | No draft, no training, no extra params | 1.3–1.6× |

**When NOT to spec-decode:**

- Однополые генерации 15 токенов.
- Дико креативное / высокотемпературное пробовое изготовление (α-падаль).
- Развертывание с ограниченными возможностями памяти (модель проекта добавляет VRAM).

## Погрузи его

Посмотрите .`outputs/skill-spec-decode-picker.md`. Умение выбирает спекулятивную стратегию декодирования (ваниль / Медуза / EAGLE / lookahead) и параметры настройки (N, температура проекта) для новой рабочей нагрузки на выводы.

## Упражнения

1. **Easy.**Беги .`code/main.py`. Подтвердить, что распределение спекулятивных токенов соответствует распределению непосредственного образца проверяющего на 50 000 токенов в пределах чи-квадрат p > 0,05.
2. **Medium.**Ускорение заряда (токены на крупную модель вперед) в функции `N`для `α = 0.5, 0.7, 0.85`- Определите оптимальный вариант .`N`для каждой α. (Спосок: ожидаемые токены на проверку = `(1 - α^{N+1}) / (1 - α)`.)
3. **Hard.**Используйте крошечную Медузу: возьмите колонную GPT из урока 14, добавьте 3 дополнительных голов LM, которые предсказывают позиции t+2, t+3, t+4. Обучайтесь на крошечном шейке с совместной потерей многоглавы. Сравните показатели принятия против ванилового чертежа, сделанного путем сокращения той же модели.
4. **Hard.**Реализация реверсирования: начинайте с 10-токенного предварительного KV-каше, подавайте 5 проектных токенов, имитируйте отказ на позиции 3.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Draft model | "The cheap one" | A smaller model that proposes candidate tokens; usually 10–50× cheaper than the verifier. |
| Verifier | "The big one" | The target model whose distribution we preserve; runs once per speculative step. |
| Acceptance rate (α) | "How often the draft is right" | Per-token probability that the verifier accepts the draft. 0.7–0.9 typical. |
| Residual distribution | "The rejection fallback" | `(q - p)_+` normalized; sampling from this on rejection preserves the verifier's distribution. |
| Bonus token | "The free one" | When all N drafts accepted, sample one more from the verifier's next-step distribution. |
| Medusa | "Draft-less speculative" | Multiple LM heads on the verifier predict positions t+1..t+k in parallel. |
| EAGLE | "Hidden-state draft" | Tiny transformer draft conditioned on the verifier's last-layer hidden states. |
| Lookahead decoding | "Jacobi iteration" | Self-speculation using a fixed-point iteration; no draft model. |
| Tree attention | "Verify many candidates at once" | Branching verification that considers several draft continuations simultaneously. |
| KV rollback | "Undo rejected drafts" | Scratch KV buffer; commit on acceptance, discard on reject. |

## Дальнейшее чтение

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) основной алгоритм и теорема эквивалентности.
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) одновременное введение; чистое доказательство отклонения Бернулли.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) Медуза-паперы; проверка внимания к деревьям.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) ЭГЛЕ-1; скрытый проект с условиями государства.
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) ОРГЛ-2; динамическая глубина дерева.
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) Орел-3.
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) Взгляните вперед, без проекта подхода.
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) каноническая производственная ссылка с всеми четырьмя стратегиями.
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) ссылный код для EAGLE-1/2/3.
