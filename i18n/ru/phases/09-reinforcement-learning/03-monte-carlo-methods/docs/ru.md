# Методы Монте-Карло  Учение из полных эпизодов

> Динамическое программирование нуждается в модели. Монте-Карло нуждается только в эпизодах. Запустить политику, посмотреть на доходы, средний их.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming)
**Time:** ~75 minutes

## Проблема

Динамическое программирование элегантно, но предполагает, что вы можете задавать вопросы.`P(s' | s, a)`В реальном мире практически ничего не работает так. Робот не может анализировать распределение по пикселям камеры после общих крутящих токов. Алгоритм ценообразования не может интегрироваться с каждой возможной реакцией клиента. LLM не может перечислить все возможные продолжения после токен.

Вам нужен метод, который требует только способности *образки* из окружающей среды.`s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`Используйте его для оценки значений.

Переход от DP к MC имеет философское значение: мы переходим от *известной модели + точного резервного копирования* до *образных развертываний + среднего дохода*.

## Концепция

![Monte Carlo: rollout, compute returns, average; first-visit vs every-visit](../assets/monte-carlo.svg)

**The core idea, in one line:** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`где`G^{(i)}(s)`наблюдаются результаты после посещения `s`в рамках политики `π`- Я не знаю .

**First-visit vs every-visit MC.**Учитывая эпизод , который посещает штат `s`В первом посещении MC рассчитывается только возвращение от первого посещения; в каждом посещении MC рассчитывается все посещения. Оба являются беспристрастными в пределах.

**Incremental mean.**Вместо того, чтобы хранить все данные, обновьте средний показатель:

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

Реорганизовать: `V_new = V_old + α · (target - V_old)`с `α = 1/n`- Сменить .`1/n`для постоянного шага `α ∈ (0, 1)`и получаешь нестационарный MC-оценщик , который отслеживает изменения в `π`Этот шаг - это полный переход от MC к TD к любому современному алгоритму RL.

**Exploration is now a problem.**DP затронула все штаты перечислением.`π`В целом, целые области пространства никогда не получают образцы, а их оценки остаются на нуле навсегда.

1. **Exploring starts.**Начинать каждый эпизод с случайной пары (s, a). Гарантирует охват; нереалистичный на практике (нельзя "переставить" робота в произвольное состояние).
2. **ε-greedy.**Действуй жадно, не отвлекаясь от текущего Q, но с вероятностью.`ε`Все пары действий состояния пробиваются асимптотически.
3. **Off-policy MC.**Собирать данные в рамках политики поведения `μ`, узнайте о целевой политике `π`Высокая вариантность, но это мост к методам повторного буфера, как DQN.

**Monte Carlo Control.**Оценить → улучшить → оценить, как и итерация политики, но оценка основана на выборке:

1. Беги .`π`- Посмотрите, что там происходит.
2. Обновление`Q(s, a)`от наблюдаемых результатов.
3. Сделайте`π`Е-самолюдный W.R.T. `Q`- Я не знаю .
4. Повторяю.

Сходство с `Q*`и `π*`с вероятностью 1 при умеренных условиях (каждая пара посещалась бесконечно часто, `α`Удовлетворяет Робинс-Монро).

```figure
epsilon-greedy
```

## Создайте его

### Шаг 1: развертывание → перечень (s, a, r)

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

Нет модели, только `env.reset()`и `env.step(s, a)`- Тот же интерфейс, что и в спортзале, но снят.

### Шаг 2: вычисления (обратный скачивание)

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

Один пропуск.`O(T)`- Возврат назад .`G_t = r_{t+1} + γ G_{t+1}`избегает повторного суммирования.

### Шаг 3: оценка MC на первом посещении

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

Работу выполняют три строки: отметка состояния, увиденная при первом посещении, количество инкрементов, среднее значение обновления.

### Шаг 4: ε-пристрастный контроль MC (политическая)

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### Шаг 5: сравните с золотым стандартом DP

Ваша оценка MC `V^π`На практике: 50 000 эпизодов на 4×4 GridWorld дает вам возможность`~0.1`Ответа на вопрос DP.

## Ловушки

- **Infinite episodes.**Если ваша политика может быть циркуляцией навсегда, то вы можете ее прекратить.`max_steps`GridWorld с случайной политикой регулярно выходит , что нормально, просто убедитесь, что вы считаете его правильно.
- **Variance.**В длинных эпизодах варианты огромны.`V(s_0)`Урок 04) уменьшает это путем загрузки.
- **State coverage.**Жадный MC на свежем Q с галстуками попробует только одно действие.
- **Non-stationary policies.**Если`π`В случае изменения (как в контроле MC), старые возвраты являются из другой политики.
- **Off-policy importance sampling.**Весы .`π(a|s)/μ(a|s)`В результате, в результате чего, по сравнению с тем, что происходит с каждым решением, можно увеличить количество размножений по траектории.

## Используйте его

Роль методов Монте-Карло в 2026 году:

| Use case | Why MC |
|----------|--------|
| Short-horizon games (blackjack, poker) | Episodes terminate naturally; returns are clean. |
| Offline evaluation of a logged policy | Average discounted returns over stored trajectories. |
| Monte Carlo Tree Search (AlphaZero) | MC rollouts from tree leaves guide selection. |
| LLM RL evaluation | Compute average reward over sampled completions for a given policy. |
| Baseline estimation in PPO | The advantage target `A_t = G_t - V(s_t)` uses an MC `G_t`. |
| Teaching RL | Simplest algorithm that actually works — strip bootstrapping to see the core. |

Современные алгоритмы глубокого р-ра (PPO, SAC) интерполируют между чистым MC (полное возвращение) и чистым TD (одноступенчатый пусковый путь) через `n`-этапные возвраты или GAE.

## Погрузи его

Сделайте это .`outputs/skill-mc-evaluator.md`В следующем случае:

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## Упражнения

1. **Easy.**Внедрить оценку MC на первом посещении политики единообразного случайного использования на 4×4 GridWorld.`V(0,0)`как функция количества эпизодов против ответа на DP.
2. **Medium.**Внедрить ε-пристрастный контроль с `ε ∈ {0.01, 0.1, 0.3}`Сравните средний доход после 20 000 эпизодов.
3. **Hard.**Внедрение *неполитических* MC с важностью выборки: сбор данных в рамках единой случайной политики `μ`, оценка `V^π`для детерминистической оптимальной политики `π`Сравните простое ИС против по-прежнему принятого решения ИС против взвешенного ИС.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Monte Carlo | "Random sampling" | Estimate expectations by averaging over iid samples from the distribution. |
| Return `G_t` | "Future reward" | Sum of discounted rewards from step `t` to episode end: `Σ_{k≥0} γ^k r_{t+k+1}`. |
| First-visit MC | "Count each state once" | Only the first visit in an episode contributes to the value estimate. |
| Every-visit MC | "Use all visits" | Every visit contributes; slightly biased but more sample-efficient. |
| ε-greedy | "Exploration noise" | Pick greedy action with prob `1-ε`; random action with prob `ε`. |
| Importance sampling | "Correcting for sampling from the wrong distribution" | Reweight returns by `π(a\|s)/μ(a\|s)` products to estimate `V^π` from `μ` data. |
| On-policy | "Learn from my own data" | Target policy = behavior policy. Vanilla MC, PPO, SARSA. |
| Off-policy | "Learn from someone else's data" | Target policy ≠ behavior policy. Importance-sampled MC, Q-learning, DQN. |

## Дальнейшее чтение

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) каноническое лечение.
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) анализ первого визита против каждого визита.
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) внеполитический MC и контроль вариантов.
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) современные оценщики ИС с низким вариантом.
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) первая масштабная эмпирическая демонстрация самоигры MC/TD, сближающаяся с сверхчеловеческой игрой; концептуальный предшественник каждого урока во второй половине этой фазы.
