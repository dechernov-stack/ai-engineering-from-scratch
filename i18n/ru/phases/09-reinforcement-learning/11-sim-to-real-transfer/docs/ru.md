# Передача SIM-на реальный

> Политика, обученная в симуляторе, которая не работает на аппаратном обеспечении, является политикой, которая запоминает симулятор.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 9 · 08 (PPO), Phase 2 · 10 (Bias/Variance)
**Time:** ~45 minutes

## Проблема

Обучение настоящего робота медленно, опасно и дорого. Двухбогушкам требуется миллионы тренировочных эпизодов, чтобы научиться ходить; настоящий двухбогушка, который падает даже когда разрывает аппаратное обеспечение.

Но симуляторы ошибаются. Подшипники имеют больше трения, чем модели MuJoCo. Камеры имеют искажение объектива, который симулятор не включает. Моторы имеют задержки, реакцию и насыщение, которые 99% моделей симулятора пропустили. Ветер, пыль и изменчивое освещение саботажают политику, обученную стерильному рендерингу.**reality gap** Систематическая разница между распространением сим и реальном распространением  является центральной проблемой развернутого RL для робототехники.

Вам нужна политика, которая *прочно подходит для синхронного сдвига распределения на реальный* Три исторических подхода: рандомизировать симулятор (рандомизация домена), адаптировать политику с помощью небольших реальных данных (адаптация домена / мелкое настройка) или идентифицировать параметры реальной системы и сопоставить их (идентификация системы). В 2026 году доминирующий рецепт сочетает все три с массивной параллельной симуляцией (Исаак Сим, Исаак Лаб, Mujoco MJX на GPU).

## Концепция

![Three sim-to-real regimes: domain randomization, adaptation, system identification](../assets/sim-to-real.svg)

**Domain Randomization (DR).**Тобин и др. 2017, Пэнг и др. 2018 год. Во время тренировки, рандомизируйте каждый сим-параметр, который может отличаться от настоящего робота: массы, коэффициенты трения, двигатель PD, шум датчика, положение камеры, освещение, текстуры, контактные модели. Политика учится условно распределять "в каком состоянии она сегодня" и обобщает в полном объеме. Если настоящий робот входит в рамки обучения, политика работает.

- **Upside:**Не нужно иметь реальных данных.
- **Downside:**Слишком много шума - слишком много регуляризации.

**System Identification (SI).**Если вы можете измерить трение на локте руки на настоящем роботе, подключите его к модели. Затем тренируйте политику, которая ожидает этих значений.

- **Upside:**точная, малошумная цель обучения.
- **Downside:**остаточная ошибка модели невидима для политики; небольшие неидентифицированные эффекты (например, моторная неразвитая полоса) все еще нарушают развертывание.

**Domain Adaptation.**Тренировка в симе, тонкая настройка с небольшим количеством реальных данных.

- **Real2Sim2Real:**Научиться использовать симулятор остатков`f(s, a, z) - f_sim(s, a)`используя реальные развертывания, тренируясь в исправленном симуляторе.
- **Observation adaptation:**тренировать политику, которая отображает реальные обс → сим-подобные обс через изученный экстрактор функций (например, GAN пиксель-по-пикселю). Контроллер остается в симе.

**Privileged learning / teacher-student.**Miki et al. 2022 (ANYmal quadruped). Обучить *учителя* в моделировании, которое имеет доступ к привилегированной информации (прикосновение земной истины, высота местности, дрейф IMU). Разделить *ученика* который видит только наблюдения с реальными датчиками.

**Massively parallel simulation.**20242026. Isaac Lab, Mujoco MJX, Brax все работают тысячи параллельных роботов на одном GPU. PPO с 4,096 параллельными гуманоидами собирает годы опыта в часах. "Разрыв в реальности" сокращается по мере расширения распределения тренировок; DR становится почти свободным, когда у каждого из этих 4,096 envs есть разные рандомизированные параметры.

**The real-world 2026 recipe (quadruped walking example):**

1. Массивно параллельное моделирование с доменно-распространенной гравитацией, трением, двигательными прибылями, полезной нагрузкой.
2. Политика учителя обучается с привилегированной информацией (карта земли, скорость тела на земле).
3. Студенческая политика, выделенная из учителя с использованием только проприоцепции (кодеры в конечном суставе).
4. Факультативная адаптация наблюдений с помощью автокодера на реальной ИМУ.
5. Развертывать, снимать на 10+ средах, а если не получится, то настроить на минуты с помощью ППО.

```figure
f3-reality-gap
```

## Создайте его

Код этого урока представляет собой крошечную демонстрацию рандомизации доменов на GridWorld с * шумными * переходами. Мы тренируем политику, которая испытывает рандомизированные вероятности скольжения в "сим" и оценивает на "реальном" с уровнем скольжения, которое никогда не было замечено во время тренировки.

### Шаг 1: параметризированная сим

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip`В реальной робототехнике это может быть трение, масса, двигательная прибыль - все, что перемещается между сим и реальным.

### Шаг 2: тренировка с DR

В начале каждого эпизода, образец `slip ~ Uniform[0.0, 0.4]`- Обучайте PPO / Q-учение / что угодно.

### Шаг 3: оценить нулевые выстрелы на "настоящих" листовках

Оценить`slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}`. Первые четыре являются частью программы поддержки обучения; `0.5`и `0.7`Политики, обучающиеся на DR, должны оставаться почти оптимальными внутри поддержки и грациозно снижаться снаружи.

### Шаг 4: сравните с узким обучением

Обучить вторую политику с `slip = 0.0`Оценить на том же.`slip`Вы должны увидеть катастрофический спад, как только реальный спуск > 0.

## Ловушки

- **Too much randomization.**Поезд в движении`slip ∈ [0, 0.9]`И ваша политика настолько рискованна, что она никогда не пытается найти оптимальный путь.
- **Too little randomization.**Используйте адаптивную учебную программу (Автоматическая рандомизация доменов), которая расширяет распределение по мере улучшения политики.
- **Misidentified parameter space.**Если вы выберете неправильное (тюнь камеры, когда реальный разрыв - это двигательная задержка), то DR не поможет.
- **Privileged info leakage.**Учитель, использующий глобальное состояние для действий, а не просто наблюдений, может создать ученика, который не может догнать.
- **Sim-to-sim transfer failure.**Если ваша политика не соответствует более сложным вариантам симы, она также не соответствует реальной жизни.
- **No real-world safety envelope.**Политика, которая работает в симе и "работает в реальности" без защитного щита низкого уровня, все равно может разрушить аппаратное обеспечение.

## Используйте его

Сборник 2026 года:

| Domain | Stack |
|--------|-------|
| Legged locomotion (ANYmal, Spot, humanoid) | Isaac Lab + DR + privileged teacher / student |
| Manipulation (dexterous hands, pick-and-place) | Isaac Lab + DR + DR-GAN for vision |
| Autonomous driving | CARLA / NVIDIA DRIVE Sim + DR + real fine-tune |
| Drone racing | RotorS / Flightmare + DR + online adaptation |
| Finger/in-hand manipulation | OpenAI Dactyl (DR at unprecedented scale) |
| Industrial arms | MuJoCo-Warp + SI + small real fine-tune |

Для контроля на всех масштабах рабочий процесс последователен: приспосабливайте симулятор как можно лучше, рандомизируйте то, что не подходит, тренируйте огромные политики, дистиллируйте, развертывайте с помощью защитного щита.

## Погрузи его

Сделайте это .`outputs/skill-sim2real-planner.md`В следующем случае:

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## Упражнения

1. **Easy.**Обучить Q-учитель на фиксированном сдвигу GridWorld (сдвиг=0,0). Оценить на сдвиге ∈ {0,0,1,0,3,0,5}.
2. **Medium.**Обучить выборку агентов Q-учения DR `slip ~ Uniform[0, 0.3]`Сколько DR покупает по цене слипа=0,5 (из-за распределения)?
3. **Hard.**Внедрить учебную программу: начиная с слипа = 0,0, расширяйте диапазон DR каждый раз, когда политика достигает 90% оптимального. Измерить общие шаги в окружающей среде, чтобы достичь слипа = 0,3 нулевого ответа против фиксированного базового уровня DR.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Reality gap | "Sim-to-real difference" | Distribution shift between training and deployment physics/sensing. |
| Domain randomization (DR) | "Train across random sims" | Randomize sim parameters during training so policy generalizes. |
| System identification (SI) | "Measure real and fit sim" | Estimate real physical parameters; set sim to match. |
| Domain adaptation | "Fine-tune on real data" | Small real-world fine-tune after sim training; may adapt obs or dynamics. |
| Privileged info | "Ground truth for teacher" | Information only the sim has; student must infer it from obs history. |
| Teacher/student | "Distill privileged -> observable" | Teacher trained with shortcuts; student learns to mimic without them. |
| ADR | "Automatic Domain Randomization" | Curriculum that widens DR ranges as the policy improves. |
| Real2Sim | "Close the gap with real data" | Learn a residual to make the sim mimic real rollouts. |

## Дальнейшее чтение

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) оригинальная документация DR (визия робототехники).
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) DR для динамики, четырехкратного движения.
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) Дактил, АДР в масштабе.
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) учитель-ученик для ANYmal.
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) массивно параллельное моделирование, которое будет управлять развертыванием 2025-2026 гг.
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) Метод учебной программы ADR.
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) Оформление Dyna (используйте модель для планирования + развертывания), которое лежит в основе современных сим-реальных трубопроводов.
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) таксономия методов сим-реализации с результатами сделанных сделок.
