# Актер-критик  А2С и А3С

> Включите критику, которая учится.`V̂(s)`А2C выполняет его синхронно, А3C выполняет его по нитам. Оба являются ментальной моделью для каждого современного метода глубокого рельса.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (TD Learning), Phase 9 · 06 (REINFORCE)
**Time:** ~75 minutes

## Проблема

Ванильная сила работает, но её варианты ужасны.`G_t`может пересекать по фактору 10 между эпизодами.`∇ log π`и средняя производит оценку градиента, которая занимает тысячи эпизодов, чтобы переместить политику на тот же расстояние, что и с гораздо меньшим количеством обновлений DQN.

Варианс происходит при использовании сырых доходов.`b(s_t)` любая функция состояния, включая полученное значение  ожидание неизменно и варианс падает.`V̂(s_t)`Теперь количество умножается`∇ log π`является *преимущество*:

`A(s, a) = G - V̂(s)`

Действие хорошее, если оно приносит выше среднего дохода; плохое, если ниже. REINFORCE с наученным критиком является *актор-критик*. Критик дает актеру учителя низкой вариантности. Это каждый метод глубокой политики после 2015 года (A2C, A3C, PPO, SAC, IMPALA).

## Концепция

![Actor-critic: policy net plus value net, TD residual as advantage](../assets/actor-critic.svg)

**Two networks, one shared loss:**

- **Actor** `π_θ(a | s)`Ученые, которые работают в области политики, пробовали, чтобы действовать, обучались политической гравиации.
- **Critic** `V_φ(s)`Ожидаемый доход от государства.`(V_φ(s) - target)²`- Я не знаю .

**The advantage.**Два стандартных формы:

- *Преимущество MC:* `A_t = G_t - V_φ(s_t)`Непредвзятость, более высокая вариантность.
- *TD преимущество:* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`. Пристрастные (использования `V_φ`), гораздо более низкая вариантность.`δ_t`- Я не знаю .

**n-step advantage.**Интерполировать между ними:

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1`- это чистый ТД.`n = ∞`В большинстве реализаций используется `n = 5`для Atari, `n = 2048`для ППО на MuJoCo.

**Generalized Advantage Estimation (GAE).**Schulman et al. (2016) предложили экспоненциально взвешенный средний показатель для всех преимуществ n-шаг:

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

с `λ ∈ [0, 1]`- Я не знаю .`λ = 0`является TD (низкая вариантность, высокое предвзятость). `λ = 1`является MC (высокая вариантность, беспристрастная). `λ = 0.95`- это 2026 дефолтный  тон, пока диаметр предвзятости/варианса не будет там, где вы хотите его.

**A2C: synchronous advantage actor-critic.**Собирать`T`шаги через `N`Параллельные среды. Вычислить преимущества для каждого шага. Обновлять актера и критика на комбинированной партии. Повторяю. Проще, более масштабируемый брат А3C.

**A3C: asynchronous advantage actor-critic.**Mnih et al. (2016). Spawn `N`Рабочие нитки, каждый из которых выполняет окружающую среду. Каждый работник вычисляет градиенты локально на своем собственном развертывании, а затем асинхронно применяет их к совместному параметровому серверу. Нет необходимости в буфере повторного воспроизведения  рабочие декоррелируют, выполняя разные траектории. A3C доказал, что можно тренироваться на процессорах в масштабе. В 2026 году на основе GPU A2C (параллельные окружающие среды) доминирует, потому что GPU хотят больших партий.

**The combined loss.**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

Три условия: потери по политическим показателям, регрессия стоимости, энтропия бонуса. `c_v ~ 0.5`- Я не знаю .`c_e ~ 0.01`являются каноническими отправными точками.

```figure
actor-critic
```

## Создайте его

### Шаг 1: критик

Линейный критик `V_φ(s) = w · features(s)`обновленные с MSE:

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

На табличной версии критик соединяется в несколько сотен эпизодов. на Atari, заменить линейного критика с общим CNN багажник + ценность головы.

### Шаг 2: преимущество n-шага

Учитывая длину развертывания `T`и финал с запуском `V(s_T)`В следующем случае:

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns`- это критическая цель.`advantages`Это то , что умножается .`∇ log π`- Я не знаю .

### Шаг 3: совокупная обновление

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

Политическая работа, разработка на обновление, отдельные уровни обучения для актера и критика.

### Шаг 4: параллелизация (A3C против A2C)

- **A3C:**Поворачивайся`N`Каждый из них работает по своему окружающему и проходящему. периодически нажимают обновления градиента на общий мастер.
- **A2C:**бежать`N`В процессе, в котором происходит однообразная работа, наблюдатели могут составить`[N, obs_dim]`Высокое использование GPU, детерминированность, легче рассуждать.

Наш код для игрушек - однопроводный для прозрачности; перепись на A2C в партии - это три строки.

## Ловушки

- **Critic bias before actor gradient.**Если критик случайный, то его базовый показатель не является информационным, и вы тренируетесь на чистом шуме.
- **Advantage normalization.**Нормализация преимуществ до нулевой средней / единицы-стд на партию. Стабилизирует тренировку массово при почти нулевой стоимости.
- **Shared trunk.**Используйте общий экстрактор функций для актера и критика на вводах изображений. Отдельные головы.
- **On-policy contract.**A2C повторно использует данные для одного обновления. Больше и ваш градиент является предвзятым (коррекция сборки важности - это то, что добавляет PPO).
- **Entropy collapse.**Без`c_e > 0`В течение нескольких сотен обновлений политика становится почти детерминистической и перестает исследовать.
- **Reward scale.**Величины преимуществ зависят от масштаба вознаграждения. Нормизируйте вознаграждения (например, разделение run-std) для последовательных величин градиента по задачам.

## Используйте его

А2С/А3С редко являются последним выбором в 2026 году, но это архитектура, которую все позже уточняет:

| Method | Relation to A2C |
|--------|----------------|
| PPO | A2C + clipped importance ratio for multi-epoch updates |
| IMPALA | A3C + V-trace off-policy correction |
| SAC (Phase 9 · 07) | Off-policy A2C with a soft-value critic (next lesson) |
| GRPO (Phase 9 · 12) | A2C without the critic — group-relative advantage |
| DPO | A2C collapsed into a preference-ranking loss, no sampling |
| AlphaStar / OpenAI Five | A2C with league training + imitation pre-training |

Если вы видите "выгоду" в статье 2026 года, подумайте о критике актеров.

## Погрузи его

Сделайте это .`outputs/skill-actor-critic-trainer.md`В следующем случае:

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## Упражнения

1. **Easy.**Обучение актера-критика с MC преимущества (`G_t - V(s_t)`Сравните эффективность образца с базовой базовой базой REINFORCE-with-running-median из урока 06.
2. **Medium.**Переход на преимущество TD-реста (`r + γ V(s') - V(s)`Умеряйте варианты партий преимуществ.
3. **Hard.**Внедрение ГАЭ (GAE)`λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`Где же самый лучший момент для этой задачи?

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Actor | "The policy net" | `π_θ(a\|s)`, updated by policy gradient. |
| Critic | "The value net" | `V_φ(s)`, updated by MSE regression to returns / TD targets. |
| Advantage | "How much better than average" | `A(s, a) = Q(s, a) - V(s)` or its estimators. Multiplier for `∇ log π`. |
| TD residual | "δ" | `δ_t = r + γ V(s') - V(s)`; one-step advantage estimate. |
| GAE | "The interpolation knob" | Exponentially weighted sum of n-step advantages, parameterized by `λ`. |
| A2C | "Synchronous actor-critic" | Batched across envs; one gradient step per rollout. |
| A3C | "Async actor-critic" | Worker threads push gradients to a shared param server. Original paper; less common in 2026. |
| Bootstrap | "Use V at the horizon" | Truncate the rollout, add `γ^n V(s_{t+n})` to close the sum. |

## Дальнейшее чтение

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) A3C, оригинальная асинхронная актёрско-критическая статья.
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) ГАЭ.
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) основы; сочетать это с главой 9 по приближению функций, когда критик является нейронной сетью.
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) масштабируемый распределенный актор-критик с V-отслеживание вне политики коррекции.
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) реализация продукции A2C/PPO, которую стоит прочитать.
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) фундаментальный результат конвергенции для двухкратного разложения актера-критика.
