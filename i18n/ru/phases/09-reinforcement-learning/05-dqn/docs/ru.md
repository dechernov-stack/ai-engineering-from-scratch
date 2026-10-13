# Deep Q-Networks (DQN)

> 2013: Mnih обучил одну сеть Q-learning на сырых пикселях, победил каждого классического агента RL на семи играх Atari. 2015: расширен до 49 игр, опубликованных в Nature, ввел в эпоху глубокого RL. DQN - Q-learning плюс три трюка, которые делают приближение функции стабильным.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 04 (Q-learning, SARSA)
**Time:** ~75 minutes

## Проблема

Таблюлярная Q-учение требует отдельного Q-ценности для каждой пары (статуса, действия). Шахматная доска имеет ~1043 состояния. Рама Atari составляет 210×160×3 = 100,800 функций. Таблюлярная RL умирает в тысячах состояний, не говоря уже о миллиардах.

В прошлом случае решение очевидно: заменить таблицу Q нейронной сетью.`Q(s, a; θ)`Но очевидное в обратном свете заняло десятилетия. Приближение наивных функций с Q-учебой расхождается под "мертвой триадой"  приближение функций + бутоукатывание + изучение вне политики. Mnih и др. (2013, 2015) определили три инженерных трюка, которые стабилизируют обучение:

1. **Experience replay**декоррелирует переходы.
2. **Target network**замораживает цель загрузки.
3. **Reward clipping**нормализует величины градиента.

DQN на Atari был первым, когда одна архитектура с одним набором гиперпараметров решила десятки проблем управления из сырых пикселей.

## Концепция

![DQN training loop: env, replay buffer, online net, target net, Bellman TD loss](../assets/dqn.svg)

**The objective.**DQN минимизирует одноэтапную TD потерю на нейронной Q-функции:

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ`= онлайн-сеть, обновляемая на каждом шагу по спускам по градиентам. `θ^-`= целевая сеть, периодически копируемая из `θ`(каждый ~ 10 000 шагов). `D`= буфер повторного воспроизведения прошлых переходов.

**The three tricks, in order of importance:**

**Experience replay.**Буферный кольцо `~10⁶`Каждый тренировочный этап пробовывает мини-пач однообразно случайным образом. Это нарушает временную корреляцию (последовательные кадры почти идентичны), позволяет сети учиться из редких полезных переходов много раз, и декоррелирует последовательные обновления градиента.

**Target network.**Используя ту же сеть `Q(·; θ)`С обеих сторон уравнения Белмана цель передвигается каждый обновление  "гоняясь за своим хвостом".`Q(·; θ^-)`с замороженными весами.`C`Шаги, копию `θ → θ^-`Это стабилизирует цель регрессии на тысячи ступеньков в один раз.`θ^- ← τ θ + (1-τ) θ^-`(используются в DDPG, SAC) являются более плавным вариантом.

**Reward clipping.**Величина премий Atari варьируется от 1 до 1000+.`{-1, 0, +1}`Неправильно, когда величина награды имеет значение, а Атари - только подпись.

**Double DQN.**Hasselt (2016) исправляет предвзятость максимизации: используйте онлайн-сеть для *выбора* действия, целевую сеть для *оценки* ее.

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

Замена в подачу, всегда лучше.

**Other improvements (Rainbow, 2017):**приоритетное воспроизведение (больше переходов с высоким уровнем TD-ошибки), архитектура дуэлирования (отдельная `V(s)`В результате, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате чего, в результате, было достигнуто более чем одно, вы получите более, чем одно процента от общего числа, вы получите более, чем одно процента от общего числа, вы получите более, чем более, чем, чем в результате чего вы получите выгоды, вы получите более, чем вы получите более, чем вы получите, чем вы получите.

```figure
f3-dqn-stability
```

## Создайте его

Код здесь - только stdlib-free numpy  мы используем ручной валк-один-скрытый слой MLP на крошечном непрерывном GridWorld, так что каждый шаг тренировки выполняется в микросекундах. Алгоритм идентичен Atari DQN в масштабе.

### Шаг 1: буфер повторного воспроизведения

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

~ 50 000 мощностей для Atari; 5000 достаточно для нашей игрушечной среды.

### Шаг 2: крошечная Q-сеть (учебная MLP)

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

Проход вперед: линейный → ReLU → линейный. Это вся сеть.

### Шаг 3: обновление DQN

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

Форма Q-учения из урока 04 с двумя различиями: а) мы подготавливаем через дифференциальную `Q(·; θ)`Вместо индексации таблицы, б) целевые использования `Q(·; θ^-)`- Я не знаю .

### Шаг 4: внешняя петля

Для каждого эпизода, действуй на `Q(·; θ)`, толкать переходы в буфер, пробовать мини-пач, сделать шаг на градиенте, периодически синхронизировать `θ^- ← θ`- Узоры:

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

На нашем крошечном GridWorld с 16-размерным одногородном состоянием агент узнает почти оптимальную политику в 500 эпизодах.

## Ловушки

- **Deadly triad.**Приближение функции + неполитическое + бутовое загрузка может расходоваться. DQN смягчает с помощью целевой сети + повторного воспроизведения; не удаляйте ни одного из них.
- **Exploration.**В течение первых ~ 10% тренировок Q-сеть должна распадаться, обычно с 1,0 до 0,01%.
- **Overestimation.** `max`В производстве всегда используйте двойный DQN.
- **Reward scale.**Снижение или нормализация вознаграждений; величина градиента пропорциональна величине вознаграждения.
- **Replay buffer coldstart.**Не тренируйтесь, пока буфер не будет несколько тысяч переходов.
- **Target sync frequency.**Слишком часто ≈ нет целевой сети; слишком редко ≈ устаревшие цели. Atari DQN использует 10 000 шагов env.
- **Observation preprocessing.**Atari DQN собирает 4 кадров, чтобы сделать состояние Markov. Любая среда с информацией о скорости нуждается в системе сбора кадров или повторяющемуся состоянии.

## Используйте его

В 2026 году DQN редко является современным, но остается ориентировым внеполитическим алгоритмом:

| Task | Method of choice | Why not DQN? |
|------|------------------|--------------|
| Discrete-action Atari-like | Rainbow DQN or Muesli | Same framework, more tricks. |
| Continuous control | SAC / TD3 (Phase 9 · 07) | DQN has no policy network. |
| On-policy / high-throughput | PPO (Phase 9 · 08) | No replay buffer; easier to scale. |
| Offline RL | CQL / IQL / Decision Transformer | Conservative Q targets, no bootstrapping blowups. |
| Large discrete action spaces (recommender) | DQN with action embedding, or IMPALA | Fine; decoration matters. |
| LLM RL | PPO / GRPO | Sequence-level, not step-level; different loss. |

Уроки все еще путешествуют. Повторное воспроизведение и целевые сети появляются в SAC, TD3, DDPG, SAC-X, буфере самопроизведения AlphaZero и каждом методе RL в офлайн-режиме.

## Погрузи его

Сделайте это .`outputs/skill-dqn-trainer.md`В следующем случае:

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## Упражнения

1. **Easy.**Беги .`code/main.py`Сколько эпизодов до тех пор, пока средняя продолжительность не превысит -10?
2. **Medium.**Отключить целевую сеть (используйте онлайн-сеть для обеих сторон цели Беллмана).
3. **Hard.**Добавьте двойный DQN: используйте онлайн-сетку для отбора `argmax a'`Сравните предвзятость `Q(s_0, best_a)`Вс истинный`V*(s_0)`После 1000 эпизодов с "Вс без Двойного ДКН" на шумной наградной сети GridWorld.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| DQN | "Deep Q-learning" | Q-learning with a neural Q-function, replay buffer, and target network. |
| Experience replay | "Shuffled transitions" | Ring buffer sampled uniformly each gradient step; decorrelates data. |
| Target network | "Frozen bootstrap" | Periodic copy of Q used in the Bellman target; stabilizes training. |
| Deadly triad | "Why RL diverges" | Function approximation + bootstrapping + off-policy = no convergence guarantee. |
| Double DQN | "Fix for maximization bias" | Online net selects action, target net evaluates it. |
| Dueling DQN | "V and A heads" | Decompose Q = V + A - mean(A); same output, better gradient flow. |
| Rainbow | "All the tricks" | DDQN + PER + dueling + n-step + noisy + distributional in one. |
| PER | "Prioritized Replay" | Sample transitions proportional to TD-error magnitude. |

## Дальнейшее чтение

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) Рабочая статья 2013 года NeurIPS, которая запустила глубокий RL.
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) "Найтер" - 49-игровый DQN.
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) DDQN.
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) Дуэль DQN.
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) бумаги с накладкой трюков.
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) учебное пособие "мертвой триады" (приближение функций + загрузка + невыполнение политики), которую цельная сеть DQN и буфер воспроизведения предназначены для обучения.
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) Справочная однофайльная DQN, используемая в исследованиях абляции; хорошо прочитать вместе с версией этого урока с нуля.
