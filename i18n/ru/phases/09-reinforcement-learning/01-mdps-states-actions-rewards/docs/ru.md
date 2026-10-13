# МДП, государства, действия и вознаграждения

> Марковский процесс принятия решений состоит из пяти вещей: состояния, действия, переходы, вознаграждения, скидка. Все в RL  Q-learning, PPO, DPO, GRPO  оптимизирует эту форму.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Probability & Distributions), Phase 2 · 01 (ML Taxonomy)
**Time:** ~45 minutes

## Проблема

Вы пишете шахматный бот, или планировщик запасов, или торговый агент, или цикл PPO, который обучает модели рассуждения, четыре различных области, один удивительный факт: все четыре разрушаются к одному и тому же математическому объекту.

Подконтрольное обучение дает вам`(x, y)`Учёный не дает никаких ярлыков, а только поток состояний, действия, которые вы предприняли, и скалярная награда.

Вы не можете научиться из этого потока, пока не формализуете его. "То, что я видел", "то, что я сделал", "то, что произошло дальше", "какой это был хороший"  каждый должен стать объектом, о котором вы можете рассуждать. Эта формализация является Марковским процессом принятия решений.

## Концепция

![Markov decision process: states, actions, transitions, rewards, discount](../assets/mdp.svg)

**The five objects.**

- **States** `S`В "Связной системе", в ячейке, в шахматах, в доске, в LLM, в контексте, плюс в памяти.
- **Actions** `A`Выбор, подъем, слева, справа, игра, выпускай токен.
- **Transitions** `P(s' | s, a)`- В данном случае .`s`и действия `a`Детерминистический в шахматах, стохастический в инвентаре, почти детерминистичный в декодировании LLM.
- **Rewards** `R(s, a, s')`Сквалирующий сигнал. Уигрыш = +1, убыток = -1. Доход минус стоимость. Термин соотношения вероятности в ГРПО.
- **Discount** `γ ∈ [0, 1)`Сколько будущего вознаграждения счета в сравнении с настоящим.`γ = 0.99`приобретает горизонт ~ 100 шагов; `γ = 0.9`Купишь 10 долларов.

**The Markov property** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`Будущее зависит только от нынешнего состояния. Если нет, то представление государства является неполным.

**Policies and returns.**Политика`π(a | s)`Карты состояния на распределение действия.`G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …`- сумма будущих вознаграждений, которая будет скидкой.`V^π(s) = E[G_t | s_t = s]`является ожидаемой доходностью начиная с `s`в рамках политики `π`. Q-ценность `Q^π(s, a) = E[G_t | s_t = s, a_t = a]`Каждый алгоритм RL оценивает один из этих двух, а затем улучшает `π`Соответственно.

**The Bellman equations.**Уравнения фиксированных точек, которые используют все в этой фазе:

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

Эти ожидаемые разделения возвращаются в "значение этого шага" плюс "редуктивная стоимость того, где вы приземляетесь". Рекурсивное. Каждый алгоритм в 9 фазе либо повторяет это уравнение на конвергенцию (динамическое программирование), образцы из него (Монте-Карло), либо запускает его на один шаг (временное различие).

```figure
discount-horizon
```

## Создайте его

### Шаг 1: крошечный детерминистический MDP

Агент начинается сверху слева, терминал внизу справа, вознаграждение - 1 за шаг, действия.`{up, down, left, right}`- Посмотри .`code/main.py`- Я не знаю .

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

Это вся среда, детерминистические переходы, постоянное наказание, поглощение терминального состояния.

### Шаг 2: разработать политику

Политика - это функция от распределения состояния к действию.

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

Выполните случайную политику 1000 раз. Средний доход составляет около -60 до -80 для этой доски 4×4. Оптимальный доход составляет -6 (простоличный путь вниз вправо).

### Шаг 3: вычислить `V^π`Точно по уравнению Беллмана

Для небольших MDP уравнение Беллмана является линейной системой.

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

Это итеративная оценка политики. Это первый алгоритм в Sutton & Barto и теоретическая основа каждого метода RL, который следует за ним.

### Шаг 4: `γ`является гиперпараметром с физическим значением

Эффективный горизонт примерно `1 / (1 - γ)`- Я не знаю .`γ = 0.9`→ 10 шагов. `γ = 0.99`→ 100 шагов. `γ = 0.999`→ 1000 шагов.

Слишком низкий и агент действует близоруко. Слишком высокий и кредитная наделка становится шумной, потому что многие ранние шаги разделяют ответственность за отдаленное будущее вознаграждение.`γ = 1`В процессе управления задачи используются`0.95–0.99`. Игры по стратегии с длинным горизонтом используют `0.999`- Я не знаю .

## Ловушки

- **Non-Markovian state.**Если вам нужно решить последние три наблюдения, "статус" - это не только текущее наблюдение.
- **Sparse rewards.**В результате полученных наград обучение практически невозможно в больших пространствах.
- **Reward hacking.**Оптимизация награды прокси часто приводит к патологическому поведению. Агент гонки лодок OpenAI вращается в круги, собирая мощности навсегда, вместо того, чтобы закончить гонку.
- **Discount mis-spec.** `γ = 1`При задаче с бесконечным горизонтом каждый значение делает бесконечным.`γ < 1`- Я не знаю .
- **Reward scale.**В результате возмещения {+100, -100} против {+1, -1} получается одинаковая оптимальная политика, но существенно разные величины градиента.`[-1, 1]`- перед подключением к PPO/DQN.

## Используйте его

Склад 2026 уменьшает каждый трубопровод RL до MDP, прежде чем коснуться кода:

| Situation | State | Action | Reward | γ |
|-----------|-------|--------|--------|---|
| Control (locomotion, manipulation) | Joint angles + velocities | Continuous torques | Task-specific shaped | 0.99 |
| Games (chess, Go, poker) | Board + history | Legal move | Win=+1 / loss=-1 | 1.0 (finite) |
| Inventory / pricing | Stock + demand | Order qty | Revenue - cost | 0.95 |
| RLHF for LLMs | Context tokens | Next token | Reward-model score at end | 1.0 (episode ~200 tokens) |
| GRPO for reasoning | Prompt + partial response | Next token | Verifier 0/1 at end | 1.0 |

Напишите пять туплей перед написанием любой тренировочной петли. Большинство сообщений об ошибках "RL не работает" прослеживаются до формулировки MDP, которая была сломана на бумаге.

## Погрузи его

Сделайте это .`outputs/skill-mdp-modeler.md`В следующем случае:

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## Упражнения

1. **Easy.**Реализуйте 4×4 GridWorld и расшифровка случайной политики в `code/main.py`- Запустите 10 000 эпизодов. Отчет среднего и столь высокого показателя. Сравните с оптимальным показателем (-6).
2. **Medium.**Беги .`policy_evaluation`с `γ ∈ {0.5, 0.9, 0.99}`для единообразной случайной политики.`V`Объясните, почему значения состояния вблизи терминала растут быстрее с увеличением `γ`- Я не знаю .
3. **Hard.**Поверните GridWorld стохастически: каждое действие спускается в соседнем направлении с вероятностью `p = 0.1`- Переоценить политику униформы.`V[start]`- Почему?

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| MDP | "Reinforcement learning setup" | Tuple `(S, A, P, R, γ)` satisfying the Markov property. |
| State | "What the agent sees" | Sufficient statistic for future dynamics under the chosen policy class. |
| Policy | "Agent's behavior" | Conditional distribution `π(a \| s)` or deterministic map `s → a`. |
| Return | "Total reward" | Discounted sum `Σ γ^t r_t` from the current step. |
| Value | "How good a state is" | Expected return under `π` starting from `s`. |
| Q-value | "How good an action is" | Expected return under `π` starting from `s` with first action `a`. |
| Bellman equation | "Dynamic programming recursion" | Fixed-point decomposition of value / Q into one-step reward plus discounted successor value. |
| Discount `γ` | "Future vs present" | Geometric weight on far-future reward; effective horizon `~1/(1-γ)`. |

## Дальнейшее чтение

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf)Глава 3 охватывает MDP и уравнения Беллмана; Глава 1 мотивирует гипотезу о вознаграждении, которая лежит в основе каждого последующего урока.
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) происхождение уравнения Беллмана.
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) краткий призер MDP с глубокого угла РЛ.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) ссылка на операционные исследования по MDP и методам точного решения.
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://cs.brown.edu/media/filer_public/d1/a6/d1a6f66a-289a-4b81-9596-417114843489/littman.pdf) наиболее чистое выводы MDP как специализации динамического программирования.
