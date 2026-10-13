# Временное различие  Q-Learning и SARSA

> Монте-Карло ждет, пока эпизод не закончится. TD обновляет после каждого шага, загружая следующую оценку стоимости. Q-learning не является политикой и оптимистично; SARSA является политикой и осторожным. Оба являются одной строкой кода.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming), Phase 9 · 03 (Monte Carlo)
**Time:** ~75 minutes

## Проблема

Монте-Карло работает, но у него есть два дорогих требования. Ему нужны эпизоды, которые заканчиваются, и он обновляется только после окончательного возвращения. Если ваш эпизод составляет 1000 шагов, MC ждет 1000 шагов, чтобы обновлять что-либо.

Динамическое программирование имеет противоположный профиль  нулевые варианты загруженных резервных копий , но требует известной модели.

Временная разница (TD) обучения разделяет разницу.`(s, a, r, s')`, сформировать одну-шагную цель `r + γ V(s')`и толкать`V(s)`Нет модели, нет полных эпизодов, нет предвзятости при использовании приблизительного`V`на RHS, но значительно меньше вариантов, чем MC и онлайн-обновления с первого шага.

Это центр, на котором вращается все современные RL  DQN, A2C, PPO, SAC . Остальная часть 9 фазы - это слои приближения функций и трюки, построенные на основе одноэтапного обновления TD, которое вы напишете в этом уроке.

## Концепция

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')](../assets/td.svg)

**The TD(0) update for V:**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

В скоплении количество - ошибка TD `δ = r + γ V(s') - V(s)`Онлайн-аналог `G_t - V(s_t)`в MC. Конвергенция требует`α`Удовлетворяющий Robbins-Monro (`Σ α = ∞`- Я не знаю .`Σ α² < ∞`) и все государства посещались бесконечно часто.

**Q-learning.**Метод контроля вне политики ТД:

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

- Да .`max`предполагает, что будет соблюдаться * жадная* политика с `s'`Это развязывание делает Q-учение учиться.`Q*`Mnih et al. (2015) превратили это в глубокое Q-learning на Atari (урок 05).

**SARSA.**Метод ТД по политике:

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

Название - Тупль .`(s, a, r, s', a')`SARSA использует действие .`a'`Агент *на самом деле* берет следующее, а не жадный `argmax`- Сближается с `Q^π`Для чего бы не было жадно .`π`- В пределах.`ε → 0`становится`Q*`- Я не знаю .

**The cliff-walking difference.**В классической задаче скалоходства (падение с скалы = вознаграждение -100), Q-learning учится оптимальному пути вдоль края скалы, но иногда принимает штраф во время исследования. SARSA учится более безопасному пути на шаг от скалы, потому что он учитывает шум исследования в его Q-ценности.`ε → 0`На практике это имеет значение: когда разведка фактически происходит на развертывании, поведение SARSA более консервативно.

**Expected SARSA.**Заменить`Q(s', a')`с ожидаемой стоимостью менее `π`В следующем случае:

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

Низже вариантов, чем SARSA (не было проб `a'`), одна и та же цель по политике.

**n-step TD and TD(λ).**Интерполировать между TD(0) и MC, ожидая `n`шаги перед загрузкой. `n=1`- TD,`n=∞`- MC. TD(λ) средний по всем `n`с геометрическими весями `(1-λ)λ^{n-1}`Большинство глубоких РЛ используют`n`от 3 до 20.

```figure
qlearning-gridworld
```

## Создайте его

### Шаг 1: SARSA по политике нежелания

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

Единственное отличие от Q-учения - это целевая линия.

### Шаг 2: Q-учение

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

- Да .`max`Этот символ - разница между политикой и политикой.

### Шаг 3: кривые обучения

Срок среднего возврата на 100 эпизодов. Q-учение сближается быстрее на простой детерминистической GridWorld; SARSA более консервативно на скаловой ходьбе.`code/main.py`, оба почти оптимальны после ~ 2000 эпизодов с `α=0.1, ε=0.1`- Я не знаю .

### Шаг 4: сравнить с DP truth

Итерация значения запуска (урок 02) для получения `Q*`- Проверьте .`max_{s,a} |Q_learned(s,a) - Q*(s,a)|`. Здоровый ТБ-агент приземляется в пределах `~0.5`на 4×4 GridWorld после 10 000 эпизодов.

## Ловушки

- **Initial Q values matter.**Оптимистический старт (`Q = 0`Песимистический настрой может навсегда заманить жадную политику.
- **α schedule.**Постоянно`α`- Это нормально для нестационарных проблем.`α_n = 1/n`дает сходство в теории, но слишком медленно в практике  пин `α`В `[0.05, 0.3]`и следить за кривой обучения.
- **ε schedule.**Начало высоко (`ε=1.0`), распадаются до `ε=0.05`"GLIE" (желание в пределах с бесконечной исследованием) - это условие конвергенции.
- **Max bias in Q-learning.**- Да .`max`оператор имеет предвзятость вверх , когда `Q`Это приводит к переоценке  Двойное Q-учение Хассельта (используемое DDQN в уроке 05) исправляет это двумя таблицами Q.
- **Non-terminating episodes.**TD может учиться без терминалов, но вам нужно либо перекрыть шаги, либо правильно управлять загрузкой на перекрытии.
- **State hashing.**Если состояния являются туплями/тенсорами, используйте ключевой ключ (туплей, не список; туплей плавучих кружек, не сырой).

## Используйте его

Ландшафт ТД 2026 года:

| Task | Method | Reason |
|------|--------|--------|
| Small tabular environments | Q-learning | Learns optimal policy directly. |
| On-policy safety-critical | SARSA / Expected SARSA | Conservative during exploration. |
| High-dimensional state | DQN (Phase 9 · 05) | Neural-net Q-function with replay and target net. |
| Continuous actions | SAC / TD3 (Phase 9 · 07) | TD update on a Q-network; policy net emits actions. |
| LLM RL (reward-model-based) | PPO / GRPO (Phase 9 · 08, 12) | Actor-critic with TD-style advantage via GAE. |
| Offline RL | CQL / IQL (Phase 9 · 08) | Q-learning with conservative regularization. |

90% "RL", о которых вы читаете в статьях 2026 года, - это некоторая разработка Q-learning или SARSA.

## Погрузи его

Сделайте это .`outputs/skill-td-agent.md`В следующем случае:

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## Упражнения

1. **Easy.**Внедрить Q-learning и SARSA на 4×4 GridWorld. Записать кривые обучения (средняя отдача на 100 эпизодов) на 2000 эпизодов. Кто сближается быстрее?
2. **Medium.**Составьте среду для прогулки по скале (4×12, последний ряд - скал с наградой -100 и перезагрузите для начала). Сравните конечные политики Q-learning и SARSA. Скриншот пути каждого из них.
3. **Hard.**Внедрение двойного Q-учения. На GridWorld с шумистым вознаграждением (Гауссианский шум σ=5 добавлен к каждой награде на шаг), покажите переоценки Q-учения `V*(0,0)`Учиться в двойном Q-учебе не стоит.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| TD error | "The update signal" | `δ = r + γ V(s') - V(s)`, the bootstrapped residual. |
| TD(0) | "One-step TD" | Update after every transition using only the next state's estimate. |
| Q-learning | "Off-policy RL 101" | TD update with `max` over next-state actions; learns `Q*` regardless of behavior policy. |
| SARSA | "On-policy Q-learning" | TD update using the actual next action; learns `Q^π` for current ε-greedy π. |
| Expected SARSA | "The low-variance SARSA" | Replace sampled `a'` with its expectation under π. |
| GLIE | "Correct exploration schedule" | Greedy in the Limit with Infinite Exploration; needed for Q-learning convergence. |
| Bootstrapping | "Using current estimate in the target" | What distinguishes TD from MC. Source of bias but massive variance reduction. |
| Maximization bias | "Q-learning overestimates" | `max` over noisy estimates is upward-biased; fixed by Double Q-learning. |

## Дальнейшее чтение

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) оригинальная бумага и доказательство конвергенции.
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) TD(0), SARSA, Q-learning, ожидаемый SARSA.
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) исправление предвзятости максимизации.
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) ожидаемые мотивации SARSA.
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) статья, которая создала SARSA (в то время называемую "модифицированным связующим Q-learning").
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) обобщает TD(0) в TD(n), путь от Q-учения до следов допустимости и, позже, GAE в PPO.
