# RL для игр  AlphaZero, MuZero и эра LLM-Разум

> 1992: TD-Gammon победил чемпионов человека в бэкгамон с чистой TD. 2016: AlphaGo победил Ли Седола. 2017: AlphaZero доминировал в шахматах, шоги и Го с нуля. 2024: DeepSeek-R1 доказал тот же рецепт, с GRPO, заменив PPO, работает на рассуждениях.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 05 (DQN), Phase 9 · 08 (PPO), Phase 9 · 09 (RLHF), Phase 9 · 10 (MARL)
**Time:** ~120 minutes

## Проблема

Игры имеют все, что хочет RL. Чистая награда (победа / потеря). Бесконечные эпизоды (самоигры ресеты).

Игры - это то, как каждый крупный прорыв в RL был проверен. TD-Gammon (backgammon, 1992). Atari-DQN (2013). АльфаГо (2016) АльфаЗеро (2017). OpenAI Five (дота 2, 2019). Альфа-Стар (StarCraft II, 2019). MuZero (ученная модель, 2019). Альфатензор (множение матрицы, 2022). AlphaDev (сортные алгоритмы, 2023). DeepSeek-R1 (математическое рассуждение, 2025)  последнее доказательство того, что игровые методы RL работают на тексте.

Этот конец исследовает три знаковых архитектуры AlphaZero, MuZero и GRPO через единый объединяющий объектив:**self-play + search + policy improvement**. Каждый обобщает предыдущее; GRPO в частности, рецепт AlphaZero, применяемый к LLM-рассуждению, с токенами в качестве действий и математической проверкой в качестве сигнала выигрыша.

## Концепция

![AlphaZero ↔ MuZero ↔ GRPO: same loop, different environments](../assets/rl-games.svg)

**The unifying loop.**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero (2017).**Сильвер и др. При игре (шах, шоги, го) с известными правилами:

- Сеть политической ценности: одна башня `f_θ(s) → (p, v)`- Я не знаю .`p`- Это предшественник по юридическим действиям.`v`- ожидаемый результат игры.
- Monte Carlo Tree Search (MCTS): при каждом шаге расширяйте дерево возможных продолжений.`(p, v)`выберите узлы по UCB (PUCT): `a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`- Я не знаю .
- Играть в игры агент против агента.`t`, распределение посещений MCTS `π_t`становится целью обучения политической деятельности.
- Потеря: `L = (v - z)² - π · log p + c · ||θ||²`- Я не знаю .`z`результат игры (+1 / 0 / -1).

Никаких человеческих знаний, никаких искусственных эвристик, единственного рецепта, который осваивал шахматы, шоги и го после нескольких десятков миллионов самоигр.

**MuZero (2019).**Schrittwieser и др. Устраняет требование о знании правил.

- Вместо фиксированной среды, учитесь *латентной динамической модели* `(h, g, f)`В следующем случае:
  - `h(s)`: кодировать наблюдение в латентное состояние.
  - `g(s_latent, a)`: предсказать следующее скрытое состояние + вознаграждение.
  - `f(s_latent)`: прогнозировать политику преимущества + стоимость.
- MCTS работает в *ученном латентном пространстве*.
- Работает на Го, шахматах, шоги и Атари, один алгоритм, нет знаний правил.

**Stochastic MuZero (2022).**Добавляет стохастическую динамику и случайные узлы; распространяется на игры класса "backgammon".

**Muesli, Gumbel MuZero (2022-2024).**Улучшение эффективности образцов и детерминистического поиска.

**GRPO (2024-2025).**Рецепт DeepSeek-R1. Тот же цикл в форме АльфаЗеро, применяемый к языковой модели рассуждения:

- "Игра": ответить на математическую / кодирующую / рассудительную проблему. "Win" = проверчик (пропускные результаты теста, числовые ответы) возвращает 1.
- Политика: LLM. Действия: токены. Государство: оперативное + ответ-до сих пор.
- Нет критиков (в стиле ПОП V_φ).`G`Вычислить награду за каждого.**group-relative advantage** `A_i = (r_i - mean_r) / std_r`как сигнал для обновления в стиле REINFORCE.
- КЛ-карта по ссылке на политику по предотвращению дрейфа (например, RLHF).
- Полная потеря:

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

Никакая модель вознаграждения, ни критик, ни MCTS. Основная линия по группе заменяет все три. Совпадает или превышает качество PPO-RLHF на критериях рассуждения на части вычислений.

**The R1 recipe in full.**DeepSeek-R1 (DeepSeek 2025) представляет собой две модели в одной статье:

- **R1-Zero.**Начните с базовой модели DeepSeek-V3. Нет SFT. Применяйте GRPO напрямую с двумя компонентами вознаграждения: *предупреждение точности* (на основе правил  сделал ли окончательный ответ анализом до правильного числа / прошел ли код испытания единицы) и *форматное вознаграждение* (завершил ли цепочку мыслей в `<think>…</think>`За тысячи шагов средняя длина ответа увеличивается с ~100 до ~10,000 токенов, а результаты математических показателей поднимаются до уровня почти o1 предварительного просмотра.
- **R1.**Устранение проблем с читаемостью R1-Zero с помощью четырехступенчатого трубопровода:
  1. **Cold-start SFT.**Собирайте несколько тысяч длинных демонстраций CoT с чистым форматированием.
  2. **Reasoning-oriented GRPO.**Применять GRPO с наградами точность + формат плюс * языковой последовательности * награды, чтобы предотвратить смену кода.
  3. **Rejection sampling + SFT round 2.**Обратите пример ~ 600K траекторий рассуждения с контрольного пункта RL, сохраните только те, которые имеют правильные окончательные ответы и читаемый CoT, и соедините с ~ 200K примеров SFT без рассуждения (письменность, QA, самопознание).
  4. **Full-spectrum GRPO.**Еще один раунд RL, охватывающий как обоснование (на основе правил вознаграждения), так и общее согласование (на основе предпочтений полезности/безвредности вознаграждения).

Результат соответствует O1 на AIME и MATH-500 при открытых весах, и достаточно мал, чтобы дистиллировать. В той же статье также выпускаются шесть дистиллированных плотных моделей (Qwen-1.5B через Llama-70B) путем SFT'ing на след рассуждения R1  нет RL у студента. Дистилляция сильного учителя RL последовательно бьет RL с нуля в шкале студента.

**Why GRPO instead of PPO for reasoning.**Три причины в статье DeepSeekMath (февраль 2024): (1) отсутствие сети ценности для обучения, уменьшение памяти вдвое; (2) базовая линия группы естественно обрабатывает редкую награду в конце траектории, которую производят задачи рассуждения; (3) нормализация на промах делает преимущества сопоставимыми по проблемам с совершенно разной сложностью, что не может сделать один критик PPO.

**Search-free vs search-based.**Игры разветвлены:

- *Игры с идеальной информацией с длинными горизонтами* (Go, шахматы): все еще на основе поиска.
- *LLM-размышление*: пока нет MCTS в производстве; GRPO на полных развертываниях, лучший из N для вычисления выводов.

```figure
f3-selfplay-ladder
```

## Создайте его

Код в `code/main.py`реализует **GRPO in miniature** бандит с несколькими группами образцов. алгоритм такой же, как и на LLM; только политика и окружающая среда проще. Он учит *убытка* и *имущество по группе*, что является инновацией 2025 года.

### Шаг 1: небольшая среда проверки

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

В реальном GRPO проверяющий проводит единичные тесты или проверяет математическое равенство.

### Шаг 2: политика: softmax над K ответы токены на запрос

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

эквивалент окончательному слою LLM, обусловленному пропортом.

### Шаг 3: групповое выборка и групповое преимущество

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

Групповое преимущество - это трюк DeepSeek 2024. Нет необходимости в критике. "Основная линия" - это средняя группа, а нормализация использует групповую сд.

### Шаг 4: сравнение с базовым показателем REINFORCE (без стоимости)

Одинаковая настройка, одинаковый вычисление, простое подкрепление.

### Шаг 5: наблюдать энтропию и КЛ

Та же диагностика, что и в RLHF: средний КЛ для ссылки, энтропия политики, вознаграждение за время.

## Ловушки

- **Reward hacking via verifier gaming.**GRPO наследует риск RLHF: если проверяющий ошибается или может быть использован, LLM найдет эксплуатируемый.
- **Group size too small.**Варианция базовой линии группы выглядит следующим образом:`1/√G`- Ниже .`G = 4`, сигнал преимущества шумно; стандартный выбор `G = 8`к `64`- Я не знаю .
- **Length bias.**Завершения LLM разных длин имеют разные вероятности. Нормализуйте по числу токенов, или используйте логин-проб на уровне последовательности, или сократите до максимальной длины.
- **Pure self-play cycles.**Тренировки в стиле Альфа-Зеро могут застрять в циркуляциях доминирования в общей сумме.
- **Search-policy mismatch.**AlphaZero обучает политику имитировать результаты поиска. Если сеть политики слишком мала, чтобы представлять распределение поиска, обучение останавливается.
- **Compute floor.**МуЗеро / АльфаЗеро требует массового вычисления. Одно изображение часто составляет сотни часов GPU.
- **Verifier coverage.**Единичные тесты, которые проходят на решение ошибок, укрепляют ошибку.

## Используйте его

Ландшафт игр-RL 2026 года, по домену:

| Domain | Dominant method |
|--------|-----------------|
| Two-player zero-sum board games (Go, chess, shogi) | AlphaZero / MuZero / KataGo |
| Imperfect info card games (poker) | CFR + deep learning (DeepStack, Libratus, Pluribus) |
| Atari / pixel games | Muesli / MuZero / IMPALA-PPO |
| Large multiplayer strategy (Dota, StarCraft) | PPO + self-play + league (OpenAI Five, AlphaStar) |
| LLM math/code reasoning | GRPO (DeepSeek-R1, Qwen-RL, open replications) |
| LLM alignment | DPO / RLHF-PPO (not GRPO; verifier is preference not verifiable) |
| Robotics | PPO + DR (not game-RL, but uses same policy-gradient tools) |
| Combinatorial problems | AlphaZero variants (AlphaTensor, AlphaDev) |

* рецепт *  самоигра, поисково-увеличенное улучшение, политическая дистилляция  охватывает текст, пиксели и физический контроль. GRPO - самый молодой пример; еще многое наступает.

## Погрузи его

Сделайте это .`outputs/skill-game-rl-designer.md`В следующем случае:

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## Упражнения

1. **Easy.**Внедрить бандита ГРПО в `code/main.py`. Обучение по 2 запросам × 4 ответы токены каждый.`G=8`- Я не знаю .
2. **Medium.**Сравните эффективность образца и варианты вознаграждения с ГРПО на одном бандите.
3. **Hard.**Расширить до длины-2 "цепочка рассуждений": агент выпускает два токены, а проверяющий награждает пару. Измерить, как GRPO обрабатывает кредитную передачу в двухступенчатой последовательности. (Представление: вычислить преимущество группы на *полную последовательность*, распространить на обе позиции токенов.)

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| MCTS | "Tree search with learned net" | Monte Carlo Tree Search; UCB1/PUCT selection with learned `(p, v)` priors. |
| AlphaZero | "Self-play + MCTS" | Policy-value net trained to match MCTS visits and game outcome. |
| MuZero | "Learned-model AlphaZero" | Same loop but in latent space via learned dynamics. |
| GRPO | "Critic-free PPO" | Group Relative Policy Optimization; REINFORCE with group-mean baseline + KL. |
| PUCT | "AlphaZero's UCB" | `Q + c · p · √N / (1 + N_a)` — balances value estimate with prior. |
| Self-play | "Agent vs past self" | Standard for zero-sum; symmetric training signal. |
| League play | "Population-based self-play" | Past + current + exploiters sampled as opponents. |
| Verifier reward | "Verifiable RL" | Reward comes from a deterministic checker (tests pass, answer matches). |
| Process reward | "PRM" | Scores each reasoning step, not just the final answer. |

## Дальнейшее чтение

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270)- Я не знаю .
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404)- Я не знаю .
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4)- Я не знаю .
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z)- Я не знаю .
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) в статье, в которой была представлена ГРПО и базовая оценка по группам.
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) полный рецепт R1 в четыре этапах плюс абляция R1-Зело.
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) CFR + глубокое обучение в масштабе.
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) газета, которая начала все.
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) производственная справка для применения GRPO с функциями на заказ.
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) открытая репликация рецепта R1 в нескольких масштабах.
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) учебник, в котором содержатся рамки для самоигры, поиска и "заработанной награды", которые R1 представляет на уровне LLM.
