# Политический градиент  ВНАБОЛЕНИЕ С НЕДОГО

> Стой от оценки стоимости. Параметризируйте политику напрямую, вычислите градиент ожидаемой доходности, шаг вверх. Уильямс (1992) написал это в одной теореме.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 03 (Monte Carlo), Phase 9 · 04 (TD Learning)
**Time:** ~75 minutes

## Проблема

Q-learning и DQN параметризируют функцию *value* . Вы выбираете действия по `argmax Q`Это хорошо для дискретных действий и дискретных состояний.`argmax`более 10-мерного крутящего момента?) или когда вы хотите стохастическую политику (`argmax`является детерминистическим по конструкции).

Вместо этого политические градиенты параметризируют *политику* .`π_θ(a | s)`Нейронная сеть, которая выпускает распределение по действиям.`θ`- Сделай шаг вверх.`argmax`Нет рекурсии Беллмана, просто подъем на градиенте.`J(θ) = E_{π_θ}[G]`- Я не знаю .

Теорема REINFORCE (Williams 1992) говорит вам, что этот градиент вычислим: `∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`- Запустите эпизод, вычислите возвращение, умножьте на`∇ log π_θ(a | s)`В среднем, по ступеням, всё готово.

Каждый алгоритм LLM-RL в 2026 году  PPO, DPO, GRPO  является совершенствованием REINFORCE.

## Концепция

![Policy gradient: softmax policy, log-π gradient, return-weighted update](../assets/policy-gradient.svg)

**The policy gradient theorem.**Для любой политики`π_θ`параметрированные `θ`В следующем случае:

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

где`G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}`является скидкой дохода от шага `t`Ожидания переходят полную траекторию .`τ`выборка из `π_θ`- Я не знаю .

**The proof is short.**Разница .`J(θ) = Σ_τ P(τ; θ) G(τ)`В соответствии с ожиданиями.`∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`(трик с производными логинами).`log P(τ; θ) = Σ log π_θ(a_t | s_t) + environment terms that do not depend on θ`Два строка алгебры дают теорему.

**Variance reduction tricks.**В ВНВ "Ванилла" есть убийственная варианция.`∇ log π`- Это очень шумный продукт.

1. **Baseline subtraction.**Заменить`G_t`с `G_t - b(s_t)`для любого базового показателя `b(s_t)`что не зависит от `a_t`Непредвзятость , потому что`E[b(s_t) · ∇ log π(a_t | s_t)] = 0`. Типичный выбор: `b(s_t) = V̂(s_t)`Учился критик → актер-критик (урок 07).
2. **Reward-to-go.**Заменить`Σ_t G_t · ∇ log π_θ(a_t | s_t)`с `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`. Только будущие доходы имеют значение для данного действия  прошлые вознаграждения вносят нулевую среднюю шуму.

В совокупности получается:

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

который является REINFORCE с базовым показателем  прямой предком A2C (урок 07) и PPO (урок 08).

**Softmax policy parameterization.**Для дискретных действий стандартный выбор:

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

где`f_θ`- это любая нейронная сеть, которая выпускает балл на действие.

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

т.е. показатель принятых мер минус его ожидаемой стоимости в рамках полиса.

**Gaussian policy for continuous actions.** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`- Я не знаю .`∇ log N(a; μ, σ)`Это все потребности SAC в фазе 9 · 07

```figure
policy-gradient-landscape
```

## Создайте его

### Шаг 1: сеть политики softmax

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

Используйте линейную политику (один вектор веса за действие) для табличного обозрения. Для Atari, обменьте в CNN и держите голову softmax.

### Шаг 2: выбор образцов и вероятность регистрации

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

### Шаг 3: развертывание с запечатленными лог-пробовыми

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

Грейдент`∇ log π(a|s) = e_a - π(·|s)`(почти `a`- вероятности) является основой политических градиентов softmax.

### Шаг 5: базовые линии

Средний средний`G`За последние эпизоды достаточно уменьшить варианты, чтобы запустить GridWorld 4×4; для сближения требуется ~ 500 эпизодов.`V̂(s)`и получаешь актеров-критиков.

## Ловушки

- **Exploding gradients.**Возврат может быть огромным.`G`к `~N(0, 1)`через партию , прежде чем умножить на `∇ log π`- Я не знаю .
- **Entropy collapse.**Политика сближается с почти детерминистическим действием слишком рано, перестает исследовать, застрял.`β · H(π(·|s))`к цели.
- **High variance.**Для этого требуется тысячи эпизодов.
- **Sample inefficiency.**Политические коррекции при помощи выборки значений возвращают данные за счет вариантности (соотношение ППО - это сокращение веса IS).
- **Non-stationary gradients.**Тот же градиент , что и 100 эпизодов назад , использует старые .`π`По этой причине методы политики обновляются каждые несколько раз.
- **Credit assignment.**Без премии, предыдущие награды вносят шум.

## Используйте его

В 2026 году REINFORCE редко работает напрямую, но его градиентная формула везде:

| Use case | Derived method |
|----------|---------------|
| Continuous control | PPO / SAC with Gaussian policy |
| LLM RLHF | PPO with KL penalty, running on token-level policy |
| LLM reasoning (DeepSeek) | GRPO — REINFORCE with group-relative baseline, no critic |
| Multi-agent | Centralized-critic REINFORCE (MADDPG, COMA) |
| Discrete action robotics | A2C, A3C, PPO |
| Preference-only settings | DPO — REINFORCE rewritten as a preference-likelihood loss, no sampling |

Когда ты читаешь`loss = -advantage * log_prob`В 2026 учебном сценарии, то есть REINFORCE с базовой линией.

## Погрузи его

Сделайте это .`outputs/skill-policy-gradient-trainer.md`В следующем случае:

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

1. **Easy.**Реализуйте REINFORCE на 4×4 GridWorld с линейной политикой softmax. Обучайтесь на 1000 эпизодов без базовой линии.
2. **Medium.**Добавьте базовую линию среднего показателя. Опять тренируйтесь. Сравните эффективность образца и варианты с ванильной. Насколько базовая линия уменьшает шаги к конвергенции?
3. **Hard.**Добавьте бонус энтропии `β · H(π)`- Стереть .`β ∈ {0, 0.01, 0.1, 1.0}`- Окончательный план и политическая энтропия.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Policy gradient | "Train the policy directly" | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`; derived from the log-derivative trick. |
| REINFORCE | "The original PG algorithm" | Williams (1992); Monte Carlo returns multiplied by log-policy gradient. |
| Log-derivative trick | "Score function estimator" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`; makes gradients of expectations tractable. |
| Baseline | "Variance reduction" | Any `b(s)` subtracted from `G`; unbiased because `E[b · ∇ log π] = 0`. |
| Reward-to-go | "Only future returns count" | `G_t^{from t}` instead of the full `G_0`; correct and lower-variance. |
| Entropy bonus | "Encourage exploration" | `+β · H(π(·\|s))` term keeps the policy from collapsing. |
| On-policy | "Train on what you just saw" | Gradient expectation is w.r.t. the current policy — cannot reuse old data directly. |
| Advantage | "How much better than average" | `A(s, a) = G(s, a) - V(s)`; the signed quantity REINFORCE-with-baseline multiplies. |

## Дальнейшее чтение

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) оригинальная статья REINFORCE.
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) современная теорема политики-градиента с приближением функций.
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) презентация учебников.
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) четкое педагогическое описание с кодом PyTorch.
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) снижение вариантности и представление о естественном градиенте, которое связывает REINFORCE с семейством региона доверия (TRPO, PPO).
