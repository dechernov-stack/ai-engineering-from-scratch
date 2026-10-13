# Многоагентная РЛ

> Одноагентный RL предполагает, что окружающая среда стационарна. Поставьте двух агентов обучения в один мир, и это предположение ломается: каждый агент является частью окружающей среды другого, и оба меняются.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (Q-learning), Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~45 minutes

## Проблема

Робот, который учится навигировать по комнате, является проблемой RL одного агента. Футбольная команда не является проблемой. Соперники AlphaStar vs. StarCraft не являются проблемой. Рынок агентских торгов не является проблемой. Две машины, которые ведут переговоры о четырехсторонней остановке, не являются проблемой.

В каждой многоагентной среде, с точки зрения любого агента, другие агенты являются частью окружающей среды. По мере того как они учатся и меняют свое поведение, окружающая среда становится нестационарной. Состояние Маркова "следующее состояние зависит только от текущего состояния и моего действия" нарушается, потому что следующее состояние также зависит от того, что выбрали другие агенты, и их политика движется цели.

Это нарушает табличные доказательства конвергенции (гарантия Q-learning предполагает стационарную среду). Это также нарушает наивную глубокую RL: агенты преследуют друг друга в петлях, никогда не сходятся в стабильной политике.

Приложения 2026 г.: роботные ровны, маршрутизация трафика, автономные автономные автопарки, рыночные симуляторы, системы LLM с многоагентами (Фаза 16), и любая игра с более чем одним интеллектуальным игроком.

## Концепция

![Four MARL regimes: indep, centralized critic, self-play, league](../assets/marl.svg)

**Formalism: Markov Game.**Общая обобщение MDP: государства `S`, совместной акции `a = (a_1, …, a_n)`, переход `P(s' | s, a)`, и награды за агента `R_i(s, a, s')`- Каждый агент .`i`Максимально увеличивает свою доходность в рамках собственной политики `π_i`Если награды идентичны, то это так.**fully cooperative**Если сумма нуля, то это **adversarial**Если смешанная, то это так.**general-sum**- Я не знаю .

**Core challenges:**

- **Non-stationarity.** `P(s' | s, a_i)`от агента `i`Взгляд зависит от`π_{-i}`, что меняется.
- **Credit assignment.**С общим вознаграждением, кто вызвал это?
- **Exploration coordination.**Агенты должны исследовать взаимодополняющие стратегии, а не эксплуатировать одно и то же состояние.
- **Scalability.**Объединенное пространство действий растет экспоненциально в `n`- Я не знаю .
- **Partial observability.**Каждый агент видит только собственное наблюдение; глобальное состояние скрыто.

**Four dominant regimes:**

**1. Independent Q-learning / independent PPO (IQL, IPPO).**Каждый агент учится своей собственной "К" или политике, обращаясь с другими как с окружающей средой. Простая, иногда она работает (особенно с повторным опытом, действующим как трюк моделирования агента). Теоретическая конвергенция: никакой.

**2. Centralized training, decentralized execution (CTDE).**Наиболее распространенная современная парадигма.`π_i`что условия на местное наблюдение `o_i` стандартное децентрализованное выполнение при развертывании.`Q(s, a_1, …, a_n)`условия по полному глобальному состоянию и совместным действиям.
- **MADDPG**(Lowe et al. 2017): DDPG с централизованным критиком на агента.
- **COMA**(Foerster et al. 2017): контрафактная базовая линия  спросите: "Каким был бы мой вознаграждение, если бы я принял меры `a'`вместо этого?"  изолирует мой вклад.
- **MAPPO**- Я не знаю .**IPPO**с общим критиком (Yu et al. 2022): PPO с централизованной функцией ценности. Доминирующий в 2026 году для кооперативного MARL.
- **QMIX**(Rashid et al. 2018): распад стоимости  `Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))`с монотонной смесью.

**3. Self-play.**Два экземпляра одного и того же агента играют друг с другом. Политика противника *is* моя политика из прошлого момента. АльфаГо / АльфаЗеро / МуЗеро. OpenAI Five.

**4. League play.**Расширение самоигры на общую сумму / противоборство среды: сохранить популяцию прошлой и текущей политики, пробовать противника из лиги, тренироваться против них. Добавляет эксплуатантов (специализируются в победе на текущий лучший) и основных эксплуатантов (специализируются в победе на эксплуатантов). Альфа Стар (StarCraft II). Необходимо, когда игра допускает "рок-папер-ножчик" стратегические циклы.

**Communication.**Позволь агентам отправлять знакомые сообщения .`m_i`Foerster et al. (2016) показали, что дифференцируемое межагентское общение может быть обучено от конца к концу.

```figure
f3-marl-orbit
```

## Создайте его

В этом уроке используется 6×6 GridWorld с двумя кооперативными агентами. Они начинаются в противоположных углах и должны достичь общей цели.`-1`на каждом шагу, пока один из агентов все еще движется, `+10`Когда оба придут.`code/main.py`- Я не знаю .

### Шаг 1: многоагентная среда

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

* совместное * действие пространство`|A|² = 16`Глобальное состояние - это две позиции.

### Шаг 2: самостоятельное Q-учение

Каждый агент работает с собственной таблицей Q, находящейся на клавиатуре совместного состояния.

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

Работает над этой задачей, потому что награды плотно и совпадают.

### Шаг 3: централизованный Q с обновлением расщепленного значения

Используйте один Q вместо совместных действий `Q(s, a_1, a_2)`. Обновление от общих наград. Децентрализовать при исполнении путем маргинализации: `π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`. Обмен экспоненциальным пространством совместного действия для *правильного* глобального представления.

### Шаг 4: простой самоигра (двух-агентный соперник)

Один и тот же агент, два роля.`K`В эпизодах, копируйте вес А в В. Симетрическая тренировка, последовательный прогресс.

## Ловушки

- **Non-stationary replay.**Опыт повторения с независимыми агентами хуже, чем с одним агентом, потому что старые переходы были созданы устаревшими противниками.
- **Credit assignment ambiguity.**Общие награды после длительного эпизода; нет четкого способа сказать, кто-то из агентов внес свой вклад.
- **Policy drift / chasing.**Каждый агент лучше всего реагирует на обновления друг друга.
- **Reward hacking via coordination.**Агентство обнаруживает скоординированные эксплойт, не ожидаемые дизайнером. Аукционные агенты сходятся на нулевую ставку.
- **Exploration redundancy.**Оба агента изучают одинаковые пары действий.
- **League cycles.**Чистая игра может застрять в цикле доминирования.
- **Sample explosion.** `n`агенты × пространство состояния × совместные действия. Приблизительность с приближением функций; факторовочные пространства действия (один глава выхода политики на агента).

## Используйте его

Карта применения MARL 2026:

| Domain | Method | Notes |
|--------|--------|-------|
| Cooperative navigation / manipulation | MAPPO / QMIX | CTDE; shared critic + decentralized actors. |
| Two-player games (chess, Go, poker) | Self-play with MCTS (AlphaZero) | Zero-sum; symmetric training. |
| Complex multiplayer (Dota, StarCraft) | League play + imitation pretraining | OpenAI Five, AlphaStar. |
| Autonomous-vehicle fleets | CTDE MAPPO / PPO with attention | Partial obs; variable team sizes. |
| Auction markets | Game-theoretic equilibrium + RL | Mean-field RL when `n` → ∞. |
| LLM multi-agent systems (Phase 16) | Natural-language comm + role conditioning | RL loop at the agent-planning layer. |

В 2026 году наибольшим направлением роста MARL является LLM: наводнение языковых моделей, ведущих переговоры, обсуждающих, создающих программное обеспечение.

## Погрузи его

Сделайте это .`outputs/skill-marl-architect.md`В следующем случае:

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## Упражнения

1. **Easy.**Обучение независимого Q-learning на 2-агентском кооперативе GridWorld. Сколько эпизодов до среднего возвращения > 0?
2. **Medium.**Добавьте задачу "координации": цель достигается только тогда, когда оба агента находятся на ней в одном и том же направлении.
3. **Hard.**Внедрить централизованный критик для обучения в стиле MAPPO и сравнить скорость конвергенции с независимой PPO по задаче координации.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Markov game | "Multi-agent MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`; each agent has its own reward. |
| CTDE | "Centralized training, decentralized execution" | Joint critic at training time; each agent's policy uses only local obs. |
| IPPO | "Independent PPO" | Each agent runs PPO separately. Simple baseline; often underrated. |
| MAPPO | "Multi-agent PPO" | PPO with a centralized value function conditioned on global state. |
| QMIX | "Monotonic value decomposition" | `Q_tot = f_monotone(Q_1, …, Q_n)` allows decentralized argmax. |
| COMA | "Counterfactual multi-agent" | Advantage = my Q minus expected Q marginalizing over my action. |
| Self-play | "Agent vs past self" | Single agent, two roles; standard for zero-sum games. |
| League play | "Population training" | Cache past policies, sample opponents from the pool; handles strategy cycles. |

## Дальнейшее чтение

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) CTDE с централизованным критиком.
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) контрафактные базовые линии кредитования.
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) разложение значения с монотонностью.
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) ППО удивительно сильна для Марла.
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) Лига в масштабах.
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) Чистая самоигра в играх с нулевой суммой.
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) включает в себя краткое рассмотрение учебника настройки многоагентов и проблему нестационарности, которую CTDE предназначена для решения.
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) исследование, охватывающее кооперативные, конкурентные и смешанные MARL с результатами конвергенции.
