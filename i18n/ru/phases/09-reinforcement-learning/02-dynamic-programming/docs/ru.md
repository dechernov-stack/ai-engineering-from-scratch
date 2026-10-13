# Динамическое программирование — Policy Iteration и Value Iteration

> Динамическое программирование — это RL с подглядыванием в ответ. Вы уже знаете функции переходов и наград; вы просто итерируете уравнение Беллмана, пока `V` или `π` не перестанут меняться. Это эталон, к которому стремится приблизиться каждый метод, основанный на сэмплировании.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs)
**Time:** ~75 минут

## Проблема

У вас есть MDP с известной моделью: вы можете запросить `P(s' | s, a)` и `R(s, a, s')` для любой пары состояние-действие. Менеджер по складским запасам знает распределение спроса. В настольной игре переходы детерминированы. Gridworld — это четыре строки на Python. У вас есть *модель*.

Model-free RL (Q-learning, PPO, REINFORCE) придумали для случая, когда модели нет — можно только сэмплировать из среды. Но когда модель есть, существуют более быстрые и качественные методы: динамическое программирование. Беллман разработал их в 1957 году. Они до сих пор задают критерий корректности: когда говорят «оптимальная политика для этого MDP», имеют в виду политику, которую вернуло бы DP.

Эти методы нужны вам в 2026 году по трём причинам. Во-первых, каждая табличная среда в исследованиях RL (GridWorld, FrozenLake, CliffWalking) решается с помощью DP, чтобы получить эталонную политику. Во-вторых, точные значения позволяют *отлаживать* методы сэмплирования: если оценка Q-learning для `V*(s_0)` расходится с ответом DP на 30%, значит в вашем Q-learning есть баг. В-третьих, современные методы offline RL и планирования (MCTS, поиск AlphaZero, model-based RL в Phase 9 · 10) — все они итерируют бэкап Беллмана по выученной или заданной модели.

## Концепция

![Policy iteration and value iteration, side by side](../assets/dp.svg)

**Два алгоритма, оба — итерация неподвижной точки на уравнении Беллмана.**

**Policy iteration.** Чередует два шага, пока политика не перестанет меняться.

1. *Оценка (Evaluation):* для данной политики `π` вычислить `V^π`, многократно применяя `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` до сходимости.
2. *Улучшение (Improvement):* для данного `V^π` сделать `π` жадной относительно `V^π`: `π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`.

Сходимость гарантирована, потому что (а) каждый шаг улучшения либо оставляет `π` неизменным, либо строго увеличивает `V^π` для какого-то состояния, (б) пространство детерминированных политик конечно. Обычно сходится за ~5–20 внешних итераций даже для больших пространств состояний.

**Value iteration.** Сворачивает оценку и улучшение в один проход. Применяет уравнение Беллмана *оптимальности*:

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

Повторять, пока `max_s |V_{new}(s) - V(s)| < ε`. В конце извлечь политику, взяв жадное действие. Строго быстрее за итерацию — нет внутреннего цикла оценки, — но обычно требует больше итераций для сходимости.

**Generalized policy iteration (GPI).** Объединяющая формулировка. Функция ценности и политика заперты в двустороннем цикле улучшения; любой метод, который приводит их к взаимной согласованности (асинхронная value iteration, modified policy iteration, Q-learning, actor-critic, PPO), является частным случаем GPI.

**Почему `γ < 1` важна.** Оператор Беллмана является `γ`-сжатием в sup-норме: `||T V - T V'||_∞ ≤ γ ||V - V'||_∞`. Сжатие подразумевает единственную неподвижную точку и геометрическую сходимость. Уберите `γ < 1` — и гарантия пропадёт: вам понадобится конечный горизонт или поглощающее терминальное состояние.

```figure
value-iteration-gamma
```

## Реализация

### Шаг 1: построение модели MDP для GridWorld

Используйте тот же GridWorld 4×4 из Урока 01. Добавим стохастический вариант: с вероятностью `0.1` агент соскальзывает в случайное перпендикулярное направление.

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

`transitions(s, a)` возвращает список `(s', r, p)`. Это и есть вся модель.

### Шаг 2: оценка политики

Для данной политики `π(s) = {action: prob}` итерируем уравнение Беллмана, пока `V` не перестанет меняться:

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

Заменяем `π` на жадную политику относительно `V`. Если `π` не изменилась, возвращаем результат — мы в оптимуме.

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

### Шаг 4: собираем вместе

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

Типичная сходимость на 4×4: 4–6 внешних итераций. На выходе `V*(0,0) ≈ -6` и политика, строго уменьшающая число шагов.

### Шаг 5: value iteration (версия с одним циклом)

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

Та же неподвижная точка, но меньше кода.

## Подводные камни

- **Забыли обработать терминальные состояния.** Если применить Беллмана к поглощающему состоянию, оно всё равно подберёт «лучшее действие», которое ничего не меняет. Защититесь с помощью `if s == terminal: V[s] = 0`.
- **Sup-норма против L2-сходимости.** Используйте `max |V_new - V|`, а не среднее. Теоретическая гарантия даётся именно для sup-нормы.
- **Обновление на месте против синхронного.** Обновление `V[s]` на месте (Гаусс-Зейдель) сходится быстрее, чем отдельный словарь `V_new` (Якоби). В продакшн-коде используют обновление на месте.
- **Ничьи в политике.** Если у двух действий равное Q-значение, `argmax` может разрешать ничью по-разному на каждой итерации, из-за чего проверка «политика стабильна» будет осциллировать. Используйте устойчивое разрешение ничьи (первое действие в фиксированном порядке).
- **Взрыв пространства состояний.** DP требует `O(|S| · |A|)` за проход. Работает до ~10⁷ состояний. Дальше нужна аппроксимация функций (Phase 9 · 05 и далее).

## Применение

В 2026 году DP — это эталон корректности и внутренний цикл планировщиков:

| Сценарий использования | Метод |
|----------|--------|
| Точно решить небольшой табличный MDP | Value iteration (проще) или policy iteration (меньше внешних шагов) |
| Проверить реализацию Q-learning / PPO | Сравнить с DP-оптимальной V* на игрушечной среде |
| Model-based RL (Phase 9 · 10) | Бэкап Беллмана на выученной модели переходов |
| Планирование в AlphaZero / MuZero | Monte Carlo Tree Search = асинхронный бэкап Беллмана |
| Offline RL (CQL, IQL) | Consistent Q-iteration — DP со штрафом за действия вне распределения (OOD) |

Каждый раз, когда кто-то говорит «оптимальная функция ценности», имеется в виду «неподвижная точка DP». Когда вы видите `V*` или `Q*` в статье — представляйте этот цикл.

## Готовый код

Сохраните как `outputs/skill-dp-solver.md`:

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

1. **Лёгкое.** Запустите value iteration на GridWorld 4×4 с `γ ∈ {0.9, 0.99}`. Сколько проходов нужно, чтобы `max |ΔV| < 1e-6`? Выведите `V*` в виде сетки 4×4.
2. **Среднее.** Сравните policy iteration и value iteration на *стохастическом* GridWorld (вероятность соскальзывания `0.1`). Посчитайте: количество проходов, время выполнения, итоговое `V*(0,0)`. Что сходится быстрее по итерациям? А по времени выполнения?
3. **Сложное.** Постройте modified policy iteration: на шаге оценки выполняйте только `k` проходов вместо схождения до конца. Постройте график ошибки `V*(0,0)` в зависимости от `k` для `k ∈ {1, 2, 5, 10, 50}`. О чём говорит эта кривая с точки зрения компромисса между оценкой и улучшением?

## Ключевые термины

| Термин | Что обычно говорят | Что это означает на самом деле |
|------|-----------------|-----------------------|
| Policy iteration | «Алгоритм DP» | Чередование оценки (`V^π`) и улучшения (жадное `π` относительно `V^π`), пока политика не перестанет меняться. |
| Value iteration | «Более быстрый DP» | Бэкап оптимальности Беллмана, применяемый за один проход; сходится к `V*` геометрически. |
| Оператор Беллмана | «Рекурсия» | `(T V)(s) = max_a Σ P (r + γ V(s'))`; `γ`-сжатие в sup-норме. |
| Сжатие (Contraction) | «Почему DP сходится» | Любой оператор `T` с `\|\|T x - T y\|\| ≤ γ \|\|x - y\|\|` имеет единственную неподвижную точку. |
| GPI | «Всё есть DP» | Generalized Policy Iteration: любой метод, приводящий `V` и `π` к взаимной согласованности. |
| Синхронное обновление | «В стиле Якоби» | Использование старых значений `V` на протяжении всего прохода; легко анализируется, но медленнее. |
| Обновление на месте | «В стиле Гаусса-Зейделя» | Использование `V` по мере его обновления; на практике сходится быстрее. |

## Дополнительное чтение

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — канонический разбор policy iteration и value iteration.
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook_athena.html) — строгое изложение аргументов о сжимающих отображениях.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — modified policy iteration и анализ её сходимости.
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — оригинальная статья о policy iteration.
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — мост от DP к приближённому DP / глубокому RL, используемый во всех последующих уроках.