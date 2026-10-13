# Методы Монте-Карло — обучение по полным эпизодам

> Динамическое программирование требует модели. Монте-Карло не требует ничего, кроме эпизодов. Запусти политику, посмотри на отдачи, усредни их. Простейшая идея в RL — и та, что открывает путь ко всему остальному.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming)
**Time:** ~75 minutes

## Проблема

Динамическое программирование элегантно, но предполагает, что вы можете запросить `P(s' | s, a)` для любого состояния и действия. Почти ничто в реальном мире так не устроено. Робот не может аналитически вычислить распределение значений пикселей камеры после приложения крутящего момента к шарниру. Алгоритм ценообразования не может проинтегрировать по всем возможным реакциям клиента. LLM не может перечислить все возможные продолжения после токена.

Нужен метод, которому требуется лишь умение *сэмплировать* из среды. Запустить политику. Получить траекторию `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`. Использовать её для оценки значений. Это и есть Монте-Карло.

Переход от DP к MC философски важен: мы переходим от *известной модели + точного backup* к *сэмплированным проходам + усреднённой отдаче*. Дисперсия резко растёт, но применимость взрывается. Каждый алгоритм RL после этого урока — TD, Q-learning, REINFORCE, PPO, GRPO — в глубине души является оценщиком Монте-Карло, иногда с добавленным поверх bootstrapping.

## Концепция

![Monte Carlo: rollout, compute returns, average; first-visit vs every-visit](../assets/monte-carlo.svg)

**Главная идея в одной строке:** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`, где `G^{(i)}(s)` — это наблюдаемые отдачи после посещений `s` при политике `π`.

**First-visit vs every-visit MC.** Дан эпизод, посещающий состояние `s` несколько раз; first-visit MC учитывает отдачу только с первого посещения, every-visit MC учитывает все посещения. Оба несмещены в пределе. First-visit проще анализировать (iid-выборки). Every-visit использует больше данных на эпизод и на практике обычно сходится быстрее.

**Инкрементальное среднее.** Вместо хранения всех отдач обновляйте бегущее среднее:

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

Перегруппируем: `V_new = V_old + α · (target - V_old)` с `α = 1/n`. Замените `1/n` на постоянный шаг `α ∈ (0, 1)` — и вы получите нестационарный оценщик MC, отслеживающий изменения в `π`. Этот переход — целиком весь скачок от MC к TD и далее к любому современному алгоритму RL.

**Исследование теперь становится проблемой.** DP охватывало каждое состояние перечислением. MC видит только состояния, которые посещает политика. Если `π` детерминирована, целые области пространства состояний никогда не сэмплируются, и их оценки значения навсегда остаются нулевыми. Три решения, в историческом порядке:

1. **Exploring starts.** Начинать каждый эпизод со случайной пары (s, a). Гарантирует покрытие; нереалистично на практике (нельзя «сбросить» робота в произвольное состояние).
2. **ε-greedy.** Действовать жадно относительно текущего Q, но с вероятностью `ε` выбирать случайное действие. Все пары состояние-действие сэмплируются асимптотически.
3. **Off-policy MC.** Собирать данные под поведенческой политикой `μ`, обучаться целевой политике `π` через importance sampling. Высокая дисперсия, но это мост к методам с replay-буфером, таким как DQN.

**Монте-Карло контроль.** Оценка → улучшение → оценка, как в policy iteration, но оценка основана на сэмплировании:

1. Запустить `π`, получить эпизод.
2. Обновить `Q(s, a)` по наблюдаемым отдачам.
3. Сделать `π` ε-greedy относительно `Q`.
4. Повторить.

Сходится к `Q*` и `π*` с вероятностью 1 при мягких условиях (каждая пара посещается бесконечно часто, `α` удовлетворяет условиям Роббинса-Монро).

```figure
epsilon-greedy
```

## Постройте это

### Шаг 1: прогон → список (s, a, r)

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

Никакой модели, только `env.reset()` и `env.step(s, a)`. Тот же интерфейс, что у среды gym, но урезанный.

### Шаг 2: вычисление отдач (обратный проход)

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

Один проход, `O(T)`. Обратная рекуррентность `G_t = r_{t+1} + γ G_{t+1}` избегает повторного суммирования.

### Шаг 3: оценка first-visit MC

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

Три строки делают всю работу: отметить состояние как увиденное при первом посещении, увеличить счётчик, обновить бегущее среднее.

### Шаг 4: ε-greedy MC control (on-policy)

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

### Шаг 5: сравнение с эталоном DP

Ваша оценка MC для `V^π` должна совпасть с результатом DP из урока 02 при эпизоды → ∞. На практике: 50 000 эпизодов на 4×4 GridWorld приводят вас в пределы `~0.1` от ответа DP.

## Подводные камни

- **Бесконечные эпизоды.** MC требует, чтобы эпизоды *завершались*. Если ваша политика может зациклиться навечно, ограничьте `max_steps` и рассматривайте ограничение как неявную неудачу. GridWorld со случайной политикой регулярно упирается в таймаут — это нормально, просто убедитесь, что вы это правильно учитываете.
- **Дисперсия.** MC использует полные отдачи. На длинных эпизодах дисперсия огромна — одна неудачная награда в конце сдвигает `V(s_0)` на ту же величину. Методы TD (урок 04) снижают это за счёт bootstrapping.
- **Покрытие состояний.** Жадный MC на свежем Q с ничьими будет всегда пробовать только одно действие. Вы *обязаны* исследовать (ε-greedy, exploring starts, UCB).
- **Нестационарные политики.** Если `π` изменяется (как в MC control), старые отдачи получены от другой политики. Constant-α MC справляется с этим; sample-average MC — нет.
- **Off-policy importance sampling.** Веса `π(a|s)/μ(a|s)` перемножаются по траектории. Дисперсия взрывается с ростом горизонта. Ограничьте per-decision weighted IS или переходите на TD.

## Применение

Роль методов Монте-Карло в 2026 году:

| Сценарий использования | Почему MC |
|----------|--------|
| Игры с коротким горизонтом (блэкджек, покер) | Эпизоды естественным образом завершаются; отдачи чистые. |
| Офлайн-оценка зарегистрированной политики | Усреднение дисконтированных отдач по сохранённым траекториям. |
| Monte Carlo Tree Search (AlphaZero) | Прогоны MC из листьев дерева направляют выбор. |
| Оценка RL для LLM | Вычисление средней награды по сэмплированным завершениям для данной политики. |
| Оценка базовой линии в PPO | Целевое значение преимущества `A_t = G_t - V(s_t)` использует MC `G_t`. |
| Обучение RL | Простейший алгоритм, который реально работает — уберите bootstrapping, чтобы увидеть ядро. |

Современные алгоритмы глубокого RL (PPO, SAC) интерполируют между чистым MC (полные отдачи) и чистым TD (одношаговый bootstrap) через `n`-шаговые отдачи или GAE. Обе крайние точки — это экземпляры одного и того же оценщика.

## Отгрузка

Сохраните как `outputs/skill-mc-evaluator.md`:

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

1. **Лёгкое.** Реализуйте оценку first-visit MC равномерно-случайной политики на 4×4 GridWorld. Запустите 10 000 эпизодов. Постройте график `V(0,0)` как функции количества эпизодов относительно ответа DP.
2. **Среднее.** Реализуйте ε-greedy MC control с `ε ∈ {0.01, 0.1, 0.3}`. Сравните среднюю отдачу после 20 000 эпизодов. Как выглядит кривая? Где здесь проявляется компромисс смещение-дисперсия?
3. **Сложное.** Реализуйте *off-policy* MC с importance sampling: соберите данные под равномерно-случайной политикой `μ`, оцените `V^π` для детерминированной оптимальной политики `π`. Сравните обычный IS, per-decision IS и weighted IS. У какого наименьшая дисперсия?

## Ключевые термины

| Термин | Что говорят люди | Что это на самом деле означает |
|------|-----------------|-----------------------|
| Monte Carlo | «Случайное сэмплирование» | Оценка математических ожиданий усреднением по iid-выборкам из распределения. |
| Отдача `G_t` | «Будущая награда» | Сумма дисконтированных наград от шага `t` до конца эпизода: `Σ_{k≥0} γ^k r_{t+k+1}`. |
| First-visit MC | «Считать каждое состояние один раз» | Только первое посещение в эпизоде вносит вклад в оценку значения. |
| Every-visit MC | «Использовать все посещения» | Каждое посещение вносит вклад; слегка смещено, но более эффективно по выборке. |
| ε-greedy | «Шум исследования» | Выбрать жадное действие с вероятностью `1-ε`; случайное действие с вероятностью `ε`. |
| Importance sampling | «Коррекция сэмплирования из неверного распределения» | Перевзвешивание отдач произведениями `π(a\|s)/μ(a\|s)` для оценки `V^π` по данным `μ`. |
| On-policy | «Обучение на своих собственных данных» | Целевая политика = поведенческая политика. Vanilla MC, PPO, SARSA. |
| Off-policy | «Обучение на чужих данных» | Целевая политика ≠ поведенческая политика. MC с importance sampling, Q-learning, DQN. |

## Дополнительное чтение

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — канонический разбор.
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — анализ first-visit vs every-visit.
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — off-policy MC и контроль дисперсии.
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — современные оценщики IS с низкой дисперсией.
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — первая масштабная эмпирическая демонстрация того, как MC/TD self-play сходится к сверхчеловеческой игре; концептуальный предшественник каждого урока во второй половине этой фазы.