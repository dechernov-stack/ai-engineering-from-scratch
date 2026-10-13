# Градиент политики — REINFORCE с нуля

> Хватит оценивать ценность. Параметризуйте политику напрямую, вычислите градиент ожидаемой отдачи, шагните в сторону его роста. Уильямс (1992) записал это в одной теореме. Именно поэтому существуют PPO, GRPO и все RL-циклы LLM.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 03 (Monte Carlo), Phase 9 · 04 (TD Learning)
**Time:** ~75 минут

## Проблема

Q-learning и DQN параметризуют функцию *ценности*. Вы выбираете действия через `argmax Q`. Это нормально для дискретных действий и дискретных состояний. Но ломается, когда действия непрерывны (как взять `argmax` по 10-мерному крутящему моменту?) или когда нужна стохастическая политика (`argmax` по построению детерминирована).

Градиенты политики параметризуют вместо этого саму *политику*. `π_θ(a | s)` — это нейросеть, выдающая распределение над действиями. Чтобы действовать, выборка из него. Вычислите градиент ожидаемой отдачи по `θ`. Шагните в сторону роста. Никакого `argmax`. Никакой рекурсии Беллмана. Просто градиентный подъём по `J(θ) = E_{π_θ}[G]`.

Теорема REINFORCE (Williams 1992) утверждает, что этот градиент вычислим: `∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`. Прогоните эпизод. Вычислите отдачу. Умножьте на `∇ log π_θ(a | s)` на каждом шаге. Усредните. Сделайте шаг градиентного подъёма. Готово.

Каждый LLM-RL-алгоритм в 2026 году — PPO, DPO, GRPO — является уточнением REINFORCE. Понимание этого «в пальцах» — необходимое условие для остальной части этой фазы, а также для Phase 10 · 07 (реализация RLHF) и Phase 10 · 08 (DPO).

## Концепция

![Policy gradient: softmax policy, log-π gradient, return-weighted update](../assets/policy-gradient.svg)

**Теорема о градиенте политики.** Для любой политики `π_θ`, параметризованной `θ`:

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

где `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` — дисконтированная отдача, начиная с шага `t`. Математическое ожидание берётся по полным траекториям `τ`, сэмплированным из `π_θ`.

**Доказательство короткое.** Продифференцируйте `J(θ) = Σ_τ P(τ; θ) G(τ)` под знаком математического ожидания. Используйте `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)` (трюк с логарифмической производной). Факторизуйте `log P(τ; θ) = Σ log π_θ(a_t | s_t) + environment terms that do not depend on θ`. Члены, связанные со средой, исчезают. Две строчки алгебры дают вам теорему.

**Приёмы снижения дисперсии.** У ванильного REINFORCE убийственная дисперсия — отдачи шумные, `∇ log π` шумный, их произведение очень шумное. Два стандартных решения:

1. **Вычитание базовой линии (baseline).** Замените `G_t` на `G_t - b(s_t)` для любой базовой линии `b(s_t)`, не зависящей от `a_t`. Несмещённость следует из того, что `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`. Типичный выбор: `b(s_t) = V̂(s_t)`, обучаемая критиком → actor-critic (Урок 07).
2. **Reward-to-go.** Замените `Σ_t G_t · ∇ log π_θ(a_t | s_t)` на `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`. Для данного действия имеют значение только будущие отдачи — прошлые награды вносят шум с нулевым средним.

В совокупности получаем:

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

это REINFORCE с базовой линией — прямой предок A2C (Урок 07) и PPO (Урок 08).

**Softmax-параметризация политики.** Для дискретных действий стандартный выбор:

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

где `f_θ` — произвольная нейросеть, выдающая оценку для каждого действия. Градиент имеет простой вид:

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

то есть оценка выбранного действия минус её ожидаемое значение по политике.

**Гауссова политика для непрерывных действий.** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`. `∇ log N(a; μ, σ)` имеет замкнутую форму. Это всё, что нужно SAC из Phase 9 · 07.

```figure
policy-gradient-landscape
```

## Реализация

### Шаг 1: сеть softmax-политики

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

Используйте линейную политику (один вектор весов на действие) для табличной среды. Для Atari замените её на CNN, сохранив softmax-голову.

### Шаг 2: сэмплирование и лог-вероятность

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### Шаг 3: rollout с сохранением лог-вероятностей

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### Шаг 4: обновление REINFORCE

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

Градиент `∇ log π(a|s) = e_a - π(·|s)` (one-hot от `a` минус вероятности) — это сердце softmax-градиентов политики. Доведите его до автоматизма.

### Шаг 5: базовые линии

Скользящего среднего `G` за последние эпизоды достаточно для снижения дисперсии, чтобы запустить обучение на 4×4 GridWorld; сходимость занимает ~500 эпизодов. Замените базовую линию на обучаемую `V̂(s)` — и получите actor-critic.

## Подводные камни

- **Взрывающиеся градиенты.** Отдачи могут быть огромными. Всегда нормализуйте `G` до `~N(0, 1)` по батчу перед умножением на `∇ log π`.
- **Коллапс энтропии.** Политика слишком рано сходится к почти детерминированному действию, перестаёт исследовать, застревает. Решение: добавьте к цели бонус энтропии `β · H(π(·|s))`.
- **Высокая дисперсия.** Ванильный REINFORCE требует тысяч эпизодов. Стандартное решение — базовая линия от критика (Урок 07) или доверительная область TRPO/PPO (Урок 08).
- **Неэффективное использование выборки.** On-policy означает, что после одного обновления вы выбрасываете каждый переход. Off-policy-коррекции через importance sampling возвращают данные ценой дисперсии (отношение в PPO — это клиппированный вес importance sampling).
- **Нестационарные градиенты.** Тот же градиент со 100 эпизодов назад использует устаревший `π`. Именно поэтому on-policy-методы обновляются каждые несколько rollout-ов.
- **Назначение заслуг (credit assignment).** Без reward-to-go прошлые награды вносят шум. Всегда используйте reward-to-go.

## Применение

В 2026 году REINFORCE редко запускают напрямую, но его формула градиента встречается повсюду:

| Сценарий использования | Производный метод |
|----------|---------------|
| Непрерывное управление | PPO / SAC с гауссовой политикой |
| LLM RLHF | PPO с KL-штрафом, работающий на уровне политики токенов |
| Рассуждения LLM (DeepSeek) | GRPO — REINFORCE с групповой относительной базовой линией, без критика |
| Мультиагентные системы | REINFORCE с централизованным критиком (MADDPG, COMA) |
| Робототехника с дискретными действиями | A2C, A3C, PPO |
| Сценарии только с предпочтениями | DPO — REINFORCE, переписанный как функция потерь правдоподобия предпочтений, без сэмплирования |

Когда вы встречаете `loss = -advantage * log_prob` в обучающем скрипте 2026 года — это REINFORCE с базовой линией. Целые статьи (DPO, GRPO, RLOO) представляют собой приёмы снижения дисперсии поверх этой одной строчки.

## Готовый результат

Сохраните как `outputs/skill-policy-gradient-trainer.md`:

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## Упражнения

1. **Лёгкое.** Реализуйте REINFORCE на 4×4 GridWorld с линейной softmax-политикой. Обучайте 1000 эпизодов без базовой линии. Постройте кривую обучения; измерьте дисперсию (ст. отклонение отдач).
2. **Среднее.** Добавьте базовую линию в виде скользящего среднего. Обучите снова. Сравните эффективность по выборке и дисперсию с ванильным запуском. Насколько базовая линия сокращает число шагов до сходимости?
3. **Сложное.** Добавьте бонус энтропии `β · H(π)`. Переберите значения `β ∈ {0, 0.01, 0.1, 1.0}`. Постройте графики итоговой отдачи и энтропии политики. Где на этой задаче находится оптимальная точка?

## Ключевые термины

| Термин | Что говорят люди | Что это на самом деле означает |
|------|-----------------|-----------------------|
| Градиент политики (Policy gradient) | «Обучаем политику напрямую» | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`; выводится из трюка с логарифмической производной. |
| REINFORCE | «Исходный алгоритм PG» | Williams (1992); Монте-Карло отдачи, умноженные на градиент лог-политики. |
| Трюк с логарифмической производной | «Оценщик функции-очков (score function estimator)» | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`; делает градиенты математических ожиданий вычислимыми. |
| Базовая линия (Baseline) | «Снижение дисперсии» | Любое `b(s)`, вычитаемое из `G`; несмещённость следует из того, что `E[b · ∇ log π] = 0`. |
| Reward-to-go | «Важны только будущие отдачи» | `G_t^{from t}` вместо полного `G_0`; корректнее и с меньшей дисперсией. |
| Бонус энтропии | «Поощряем исследование» | Член `+β · H(π(·\|s))` не даёт политике коллапсировать. |
| On-policy | «Обучаемся на том, что только что увидели» | Математическое ожидание градиента берётся относительно текущей политики — нельзя напрямую переиспользовать старые данные. |
| Преимущество (Advantage) | «Насколько лучше среднего» | `A(s, a) = G(s, a) - V(s)`; знаковая величина, на которую умножается REINFORCE с базовой линией. |

## Дополнительные материалы

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — оригинальная статья по REINFORCE.
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — современная теорема о градиенте политики с аппроксимацией функций.
- [Sutton & Barto (2018). Гл. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) — изложение в учебнике.
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — понятное педагогическое изложение с кодом на PyTorch.
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — снижение дисперсии и взгляд с точки зрения натурального градиента, связывающий REINFORCE с семейством методов доверительной области (TRPO, PPO).