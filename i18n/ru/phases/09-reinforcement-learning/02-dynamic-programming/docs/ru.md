# Динамическое программирование  Итерация политики и итерация значения

> Динамическое программирование - это RL с обманом. Вы уже знаете функции перехода и вознаграждения; вы просто повторяете уравнение Беллмана, пока `V`или `π`Это эталон, к которому пытается подходить каждый метод, основанный на выборах.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs)
**Time:** ~75 minutes

## Проблема

У вас есть MDP с известной моделью: вы можете запросить `P(s' | s, a)`и `R(s, a, s')`Игровые таблицы имеют детерминистические переходы.

Бесмодельное RL (Q-learning, PPO, REINFORCE) было изобретено для того случая, когда у вас нет модели  вы можете только взять образцы из окружающей среды. Но когда у вас есть один, есть быстрые, лучшие методы: динамическое программирование.

Вы нуждаетесь в них в 2026 году по трем причинам. Во-первых, каждая табличная среда в исследованиях RL (GridWorld, FrozenLake, CliffWalking) решается с помощью DP для получения золотого стандарта политики. Во-вторых, точные значения позволяют вам * дебгу* методы выборки образцов: если Q-learning оценка для `V*(s_0)`В-третьих, современные методы RL и планирования вне интернета (MCTS, поиск AlphaZero, RL на основе модели в фазе 9 · 10) все повторяют резервную копию Bellman над изученной или данной моделью.

## Концепция

![Policy iteration and value iteration, side by side](../assets/dp.svg)

**Two algorithms, both fixed-point iteration on Bellman.**

**Policy iteration.**Заменяется два шага, пока политика не перестанет меняться.

1. *Оценка:* данная политика `π`, вычислить`V^π`неоднократно применяя`V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`пока она не соединяется.
2. *Улучшение:* дано `V^π`, сделать`π`жадный W.R.T.`V^π`:`π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`- Я не знаю .

Конвергенция гарантируется , потому что (а) каждый шаг улучшения либо сохраняет `π`то же самое или строго увеличивается `V^π`для некоторых государств (b) пространство детерминистических политик является конечным. Обычно сходятся в ~ 520 внешних итерациях даже для больших пространств государств.

**Value iteration.**Применяйте уравнение Беллмана *оптимальности*:

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

Повторяйте до `max_s |V_{new}(s) - V(s)| < ε`. Вытащить политику в конце, предприняв жадные действия. строго быстрее за итерацию  нет внутренней цикл оценки  но обычно требуется больше итераций, чтобы сблизиться.

**Generalized policy iteration (GPI).**Функция ценности и политика заперты в двухсторонней цикле улучшения; любой метод, который приводит оба к взаимной последовательности (итерация асинхронной стоимости, модифицированная итерация политики, Q-learning, актор-критик, PPO) является примером GPI.

**Why `γ < 1` matters.**Оператор Беллмана - это `γ`- сокращение в суб-норме: `||T V - T V'||_∞ ≤ γ ||V - V'||_∞`. Снижение означает уникальную фиксированную точку и геометрическую конвергенцию.`γ < 1`и вы потеряете гарантию, вам нужен конечный горизонт или состояние поглощающего терминала.

```figure
value-iteration-gamma
```

## Создайте его

### Шаг 1: создать модель MDP GridWorld

Используйте тот же 4×4 GridWorld из урока 01. Мы добавляем стохастический вариант: с вероятностью `0.1`Агент проскальзывает в случайном перпендикулярном направлении.

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)`возвращает список `(s', r, p)`Это вся модель.

### Шаг 2: оценка политики

С учетом политики `π(s) = {action: prob}`, повторять уравнение Беллмана , пока `V`Прекращает движение:

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### Шаг 3: улучшение политики

Заменить`π`с жадной политикой W.R.T.`V`Если ...`π`Не изменилось, возвращение  мы в оптимальном.

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### Шаг 4: зашивайте их вместе

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

Типичная конвергенция на 4×4: 46 внешних итераций.`V*(0,0) ≈ -6`и политику, которая строго уменьшает количество шагов.

### Шаг 5: итерация значения (версия в одной петле)

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

То же самое фиксированное место, меньше строк кода.

## Ловушки

- **Forgetting to handle terminals.**Если привести Беллмана в состояние поглощения, он все равно будет принимать "лучшее действие", которое ничего не изменит.`if s == terminal: V[s] = 0`- Я не знаю .
- **Sup-norm vs L2 convergence.**Использование`max |V_new - V|`Теоретическая гарантия - это норма.
- **In-place vs synchronous updates.**Обновление`V[s]`В помещении (Гаус-Сейдель) сближается быстрее, чем отдельная `V_new`Продукционный код используется на месте.
- **Policy ties.**Если две действия имеют равное значение Q, `argmax`При этом, если вы не используете "политический стабильный" тест, то вы можете разорвать связи по-разному в каждом повторении, что приведет к колебанию.
- **State-space explosion.**DP - это `O(|S| · |A|)`В результате, в результате чего, по сравнению с тем, что происходит с помощью "светки" и "светки" (с 9 по 05), вы можете получить дополнительные функции.

## Используйте его

В 2026 году DP является базовой линией корректности и внутренней петлей планировщиков:

| Use case | Method |
|----------|--------|
| Solve a small tabular MDP exactly | Value iteration (simpler) or policy iteration (fewer outer steps) |
| Verify a Q-learning / PPO implementation | Compare to DP-optimal V* on a toy environment |
| Model-based RL (Phase 9 · 10) | Bellman backup on a learned transition model |
| Planning in AlphaZero / MuZero | Monte Carlo Tree Search = async Bellman backup |
| Offline RL (CQL, IQL) | Conservative Q-iteration — DP with a penalty on OOD actions |

Каждый раз, когда кто-то говорит "оптимальная функция значения", они имеют в виду "фиксированная точка DP".`V*`или `Q*`В газете представьте себе эту петлю.

## Погрузи его

Сделайте это .`outputs/skill-dp-solver.md`В следующем случае:

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## Упражнения

1. **Easy.**Запустить итерацию значения на 4×4 GridWorld с `γ ∈ {0.9, 0.99}`Сколько пробежало до `max |ΔV| < 1e-6`?`V*`как сетка 4x4.
2. **Medium.**Сравните итерацию политики с итерацией значения на *стохастическом* GridWorld (вероятность спуска `0.1`Счет: просматривания, время с стенной частью, финал `V*(0,0)`Что сближается быстрее в итерациях?
3. **Hard.**Создать измененную итерацию политики: в этапе оценки, выполнять только `k`Стереть вместо сближения.`V*(0,0)`ошибка vs `k`для `k ∈ {1, 2, 5, 10, 50}`Что кривая говорит вам о компромиссе оценки/улучшения?

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Policy iteration | "DP algorithm" | Alternating evaluation (`V^π`) and improvement (greedy `π` w.r.t. `V^π`) until the policy stops changing. |
| Value iteration | "Faster DP" | Bellman optimality backup applied in one sweep; converges to `V*` geometrically. |
| Bellman operator | "The recursion" | `(T V)(s) = max_a Σ P (r + γ V(s'))`; a `γ`-contraction in sup-norm. |
| Contraction | "Why DP converges" | Any operator `T` with `\|\|T x - T y\|\| ≤ γ \|\|x - y\|\|` has a unique fixed point. |
| GPI | "Everything is DP" | Generalized Policy Iteration: any method driving `V` and `π` to mutual consistency. |
| Synchronous update | "Jacobi-style" | Use old `V` throughout a sweep; cleanly analyzable but slower. |
| In-place update | "Gauss-Seidel-style" | Use `V` as it's being updated; converges faster in practice. |

## Дальнейшее чтение

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) каноническое представление итерации политики и итерации ценностей.
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook_athena.html) строгое рассмотрение аргументов по моделированию сокращения.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) изменение итерации политики и ее анализ конвергенции.
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) оригинальная документация по итерации политики.
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) мост от DP до приблизительно-DP / глубокий RL, используемый в каждом последующем уроке.
