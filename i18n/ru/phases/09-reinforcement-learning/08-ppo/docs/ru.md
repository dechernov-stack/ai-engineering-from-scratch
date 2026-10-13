# Оптимизация политики приближения (PPO)

> A2C отбросает каждый развертывание после одного обновления. PPO заворачивает градиент политики в сокращенном соотношении важности, чтобы вы могли делать более 10 эпох на одних и тех же данных без взрыва политики. Schulman et al. (2017).

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~75 minutes

## Проблема

A2C (урок 07) - это политика: градиент `E_{π_θ}[A · ∇ log π_θ]`требует обработки данных из *точного* `π_θ`- Сделайте один обзор, и`π_θ`Вы не используете данные, которые вы использовали, но их используете снова, и ваш градиент является предвзятым.

На Atari один развертывание через 8 envs × 128 шагов = 1024 переходов и дюжину секунд окружающего времени.

Оптимизация политики в области доверия (TRPO, Schulman 2015) была первым решением: ограничить каждое обновление, чтобы расхождение между старой и новой политикой KL оставалось ниже `δ`Теоретически чисто, но требует конъюгатной градиентной решений на обновление.

PPO (Schulman et al. 2017) заменяет ограничение "трудное доверие" с помощью простой вырезанной цели. Одна дополнительная линия кода. Десять эпох за развертывание. Нет конъюгационных градиентов. Достаточно хороших теоретических гарантий.

## Концепция

![PPO clipped surrogate objective: ratio clipping at 1 ± ε](../assets/ppo.svg)

**The importance ratio.**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

Это соотношение вероятности новой политики против политики, которая собирает данные. `r_t = 1`Это значит, что ничего не изменится.`r_t = 2`означает, что новая политика вдвое больше вероятности`a_t`Как и старые.

**The clipped surrogate.**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

Два срока:

- Если преимущество`A_t > 0`и соотношение пытается вырасти .`1 + ε`, клип уравняет градиент  не толкать хорошее действие дальше, чем `+ε`Выше старой вероятности.
- Если преимущество`A_t < 0`и соотношение пытается вырасти .`1 - ε`(что означает, что мы сделаем плохой акт более вероятным по сравнению с его резким сокращением), клип закрывает градиент  не толкать плохой акт ниже `-ε`- Я не знаю .

- Да .`min`обрабатывает другой путь: если соотношение переместилось в *выгодное* направление, вы все равно получаете градиент (без сжатия на стороне, что повредит вам).

Типичный `ε = 0.2`. Определите цель как функцию `r_t`: фигура с плоской крышей на "хорошей стороне" и плоской поверхностью на "плохой стороне".

**The full PPO loss.**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

То же самое, что и A2C. Три коэффициента, обычно `c_v = 0.5`- Я не знаю .`c_e = 0.01`- Я не знаю .`ε = 0.2`- Я не знаю .

**The training loop.**

1. Собирать`N × T`переходы по `N`параллельные окружающие среды для `T`каждое шаг.
2. Вычислить преимущества (GAE), заморозить их как константы.
3. Заморозить`π_{θ_old}`как кратковременный снимок текущего `π_θ`- Я не знаю .
4. Для `K`эпохи, для каждой мини-пакеты `(s, a, A, V_target, log π_old(a|s))`В следующем случае:
   - Расчет`r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`- Я не знаю .
   - Применить`L^{CLIP}`+ потеря стоимости + энтропия.
   - Постепенный шаг.
5. Отбросьте развертывание, вернитесь к шагу 1.

`K = 10`ППО является надежным: точные числа редко имеют значение в пределах ±50%.

**KL-penalty variant.**В первоначальной статье предлагалась альтернатива с использованием адаптивного штрафа KL: `L = L^{PG} - β · KL(π_θ || π_old)`с `β`Клип-версия стала доминирующей; вариант KL сохранился в RLHF (где KL к референтной политике является отдельным ограничением, которое вы всегда хотите).

```figure
ppo-clip
```

## Создайте его

### Шаг 1: захватить `log π_old(a | s)`в момент запуска

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

Снимка сделанная один раз, в момент запуска.

### Шаг 2: вычислить преимущества GAE (урок 07)

То же самое, что и A2C. Нормализуйтесь по всей партии.

### Шаг 3: отключенная замена обновления

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

В основе ППО лежит "вырезанный → нулевой градиент".

### Шаг 4: значение и энтропия

Добавьте стандартный MSE к критической цели и бонус энтропии на актера, то же самое, что и A2C.

### Шаг 5: диагностика

Три вещи, которые нужно смотреть на каждое обновление:

- **Mean KL** `E[log π_old - log π_θ]`- Я должен остаться .`[0, 0.02]`Если он пройдет мимо`0.1`, сократить `K_EPOCHS`или `LR`- Я не знаю .
- **Clip fraction** доля образцов, соотношение которых находится за пределами `[1-ε, 1+ε]`- Должно быть .`~0.1-0.3`Если ...`~0`, клип никогда не запускает → подъем `LR`или `K_EPOCHS`Если ...`~0.5+`, вы переоборудовываете развертывание → снижаете их.
- **Explained variance** `1 - Var(V_target - V_pred) / Var(V_target)`Критика качества должна подняться к 1 по мере того, как критик учится.

## Ловушки

- **Clip coefficient mistuned.** `ε = 0.2`- Это фактический стандарт.`0.1`делает обновления слишком пугающими; `0.3+`Приглашает к нестабильности.
- **Too many epochs.** `K > 20`регулярно дестабилизируется , потому что политика отклоняется от `π_old`- эпохи ограничения, особенно для крупных сетей.
- **No reward normalization.**Большие размеры наград входят в диапазон клипов.
- **Forgetting advantage normalization.**Нормализация нулевой средней / единицы STD на партию является стандартной.
- **Learning rate not decayed.**Показатель PPO - это линейный LR, который снижается до нуля.
- **Importance ratio math errors.**Всегда .`exp(log_new - log_old)`для численной стабильности, не `new / old`- Я не знаю .
- **Wrong gradient sign.**Максимально увеличить суррогатную материнскую полость = *минимизировать* `-L^{CLIP}`Обернутый знак - наиболее распространенная ошибка PPO.

## Используйте его

PPO - это алгоритм RL по умолчанию 2026 года в удивительном числе доменов:

| Use case | PPO variant |
|----------|-------------|
| MuJoCo / robotics control | PPO with Gaussian policy, GAE(0.95) |
| Atari / discrete games | PPO with categorical policy, rolling 128-step rollouts |
| RLHF for LLMs | PPO with KL penalty to reference model, reward from RM at end of response |
| Large-scale game agents | IMPALA + PPO (AlphaStar, OpenAI Five) |
| Reasoning LLMs | GRPO (Lesson 12) — PPO variant without critic |
| Preference-only data | DPO — closed-form collapsing of PPO+KL, no online sampling |

PPO * потеря форма *  вырезанный суррогат + значение + энтропия  является платформой для DPO, GRPO, и почти каждый RLHF трубопровод.

## Погрузи его

Сделайте это .`outputs/skill-ppo-trainer.md`В следующем случае:

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## Упражнения

1. **Easy.**Запустить PPO на 4×4 GridWorld с `ε=0.2, K=4`. Сравните эффективность образца с A2C (одна эпоха за развертывание) на соответствующих этапах окружающей среды.
2. **Medium.**Стереть `K ∈ {1, 4, 10, 30}`- Задача вернуть сюжет против экрана и отследить средний KL на обновление.`K`КЛ взорвется на этой задаче?
3. **Hard.**Заменить вырезанную суррогатную матку адаптивным штрафом KL (`β`удвоится, если `KL > 2·target`, вдвое меньше , если `KL < target/2`Сравните окончательный доход, стабильность и безжалостность.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Importance ratio | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`; deviation from the policy that collected the data. |
| Clipped surrogate | "PPO's main trick" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`; flat gradient past the clip on beneficial side. |
| Trust region | "TRPO / PPO intent" | Limit each update's KL to guarantee monotone improvement. |
| KL penalty | "Soft trust region" | Alternative PPO: `L - β · KL(π_θ \|\| π_old)`. Adaptive `β`. |
| Clip fraction | "How often clipping triggers" | Diagnostic — should be 0.1-0.3; outside means mistuned. |
| Multi-epoch training | "Data reuse" | K epochs on each rollout; variance cost traded for sample efficiency. |
| On-policy-ish | "Mostly on-policy" | PPO is nominally on-policy but K>1 epochs uses slightly-off-policy data safely. |
| PPO-KL | "The other PPO" | KL-penalty variant; used in RLHF where KL-to-reference is already a constraint. |

## Дальнейшее чтение

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) газету.
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) TRPO, предшественник PPO.
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) отключение каждого гиперпараметра ППО.
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) InstructGPT; рецепт PPO-in-RLHF.
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) чистые современные экспозиции с PyTorch.
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) Справочная однофайльная PPO, используемая во многих документах.
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) рецепт производства ППО на языковых моделях; читайте вместе с уроком 09 (RLHF).
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) "37 оптимизаций уровня кода"; какие трюки PPO несут нагрузку и какие являются фольклором.
