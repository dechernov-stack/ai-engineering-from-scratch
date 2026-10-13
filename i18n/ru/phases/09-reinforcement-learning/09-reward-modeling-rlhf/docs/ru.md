# Reward Modeling & RLHF

> Люди не могут написать функцию вознаграждения для «хорошего ответа ассистента», но они могут сравнить два ответа и выбрать лучший. Обучите модель вознаграждения на этих сравнениях, затем обучите языковую модель с помощью RL против неё. Christiano 2017. InstructGPT 2022. Рецепт, превративший GPT-3 в ChatGPT. В 2026 году его в основном вытеснил DPO — но ментальная модель остаётся актуальной.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment), Phase 9 · 08 (PPO)
**Time:** ~45 минут

## Проблема

Вы обучили языковую модель на задаче предсказания следующего токена. Она пишет грамматически правильный английский текст. Она также лжёт, уходит от темы и отказывается отказывать. Исправить это дополнительным предобучением нельзя — веб-текст и есть проблема, а не лекарство от неё.

Вам нужна *скалярная награда*, которая говорит: «ответ A лучше ответа B для инструкции X». Написать такую функцию награды вручную невозможно. «Полезность» — это не выражение в замкнутой форме над токенами. Но люди способны сравнить два вывода и отметить предпочтение. Это дёшево собирать в большом масштабе.

RLHF (Christiano et al. 2017; Ouyang et al. 2022) преобразует предпочтения в модель вознаграждения, а затем оптимизирует LM через PPO против этой награды. В три шага: SFT → RM → PPO. Это рецепт, по которому были выпущены ChatGPT, Claude, Gemini и все остальные выровненные LLM в 2023–2025 годах.

В 2026 году шаг PPO в основном заменён DPO (Phase 10 · 08), поскольку он дешевле и почти так же хорош для настройки выравнивания. Но компонент *модели вознаграждения* по-прежнему лежит в основе каждого сэмплера Best-of-N, каждого пайплайна RL из проверяемых наград и каждой reasoning-модели, использующей process reward model. Поймите RLHF — и вы поймёте весь стек выравнивания.

## Концепция

![Three-stage RLHF: SFT, RM training on pairwise prefs, PPO with KL penalty](../assets/rlhf.svg)

**Этап 1: Supervised Fine-Tuning (SFT).** Начните с предобученной базовой модели. Дообучите её на написанных человеком демонстрациях целевого поведения (ответы с соблюдением инструкций, полезные реплики и т.д.). Результат: модель `π_SFT`, которая *смещена в сторону хорошего поведения*, но всё ещё имеет неограниченное пространство действий.

**Этап 2: Обучение модели вознаграждения (Reward Model).**

- Соберите пары ответов `(y_+, y_-)` на подсказки `x`, размеченные людьми как «y_+ предпочтительнее y_-».
- Обучите модель вознаграждения `R_φ(x, y)` присваивать более высокие оценки `y_+`.
- Функция потерь: **парная логистическая функция Bradley-Terry**:

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ — сигмоида. Разница в вознаграждении соответствует логарифму отношения шансов предпочтения. BT является стандартом с 1952 года (Bradley-Terry) и остаётся доминирующим выбором в современном RLHF.

- `R_φ` обычно инициализируется из SFT-модели со скалярной «головой» поверх неё. Тот же backbone-трансформер; один линейный слой выдаёт награду.

**Этап 3: PPO против RM с KL-штрафом.**

- Инициализируйте обучаемую политику `π_θ` из `π_SFT`. Сохраните замороженную *референсную* модель `π_ref = π_SFT`.
- Награда в конце ответа `y`:

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL-штраф предотвращает произвольное отклонение `π_θ` от `π_SFT` — это *регуляризатор*, а не жёсткая доверительная область. `β` обычно `0.01`-`0.05`.
- Запустите PPO (Lesson 08) с этой наградой. Преимущества вычисляются на уровне траектории токенов, но RM оценивает только полный ответ.

**Зачем нужен KL?** Без него PPO с радостью найдёт стратегии reward hacking — RM была обучена только на завершениях, соответствующих распределению обучающих данных. Ответ вне этого распределения может получить более высокую оценку, чем любой написанный человеком. KL удерживает `π_θ` рядом с многообразием, на котором обучалась RM. Это единственный самый важный параметр в RLHF.

**Статус на 2026 год:**

- **DPO** (Rafailov 2023): алгебраическое решение в замкнутой форме сворачивает этапы 2+3 в единую supervised-функцию потерь над данными предпочтений. Без RM, без PPO. Такое же качество на бенчмарках выравнивания при малой доле вычислительных затрат. Рассмотрено в Phase 10 · 08.
- **GRPO** (DeepSeek 2024–2025): PPO с групповым относительным базовым уровнем вместо критика, награда от *верификатора* (выполнение кода / совпадение ответа в математике) вместо обученной человеком RM. Доминирует для reasoning-моделей. Рассмотрено в Phase 9 · 12.
- **Process reward models (PRM):** оценивают частичные решения (каждый шаг рассуждения), используются как в RLHF, так и в вариантах GRPO для reasoning.
- **Constitutional AI / RLAIF:** используют выровненную LLM для генерации предпочтений вместо людей. Масштабирует бюджет предпочтений.

```figure
reward-model
```

## Реализация

В этом уроке используются крошечные синтетические «подсказки» и «ответы», представленные в виде строк. RM — это линейный скорер над представлением bag-of-tokens. Никакой настоящей LLM — важна *форма* пайплайна, а не масштаб. См. `code/main.py`.

### Шаг 1: синтетические данные о предпочтениях

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

В реальном RLHF это заменяется разметчиками-людьми. Форма — `(prompt, preferred_response, rejected_response)` — идентична.

### Шаг 2: модель вознаграждения Bradley-Terry

Линейная оценка: `R(x, y) = w · bag(y)`. Обучение с минимизацией парной log-loss BT:

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

После нескольких сотен обновлений `w` присваивает положительные веса токенам хороших слов и отрицательные — плохим.

### Шаг 3: политика в стиле PPO поверх RM

Наша игрушечная политика выдаёт один токен из словаря. Мы оцениваем токен с помощью RM, вычисляем `log π_θ(token | prompt)`, добавляем штраф KL-к-референсу и применяем клиппированный суррогат PPO.

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### Шаг 4: мониторинг KL

Отслеживайте среднее значение `KL(π_θ || π_ref)` при каждом обновлении. Если оно превысит `~5-10`, политика сильно отклонилась от `π_SFT` — либо растёт `β`, либо начинается reward hacking. Это главная диагностика в реальном RLHF.

### Шаг 5: продакшен-рецепт с TRL

Разобравшись с игрушечным пайплайном, посмотрите, как выглядит тот же цикл в написании реальным пользователем библиотеки. [TRL](https://huggingface.co/docs/trl) от Hugging Face — эталонная реализация: `RewardTrainer` для этапа 2 и `PPOTrainer` (со встроенным KL-к-референсу) для этапа 3.

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

Три вещи, которые библиотека делает за вас. `adap_kl_ctrl=True` реализует адаптивное расписание β: если наблюдаемый KL превышает `target_kl`, β удваивается; если ниже половины — β уменьшается вдвое. Референсная модель заморожена по соглашению — вы не должны случайно разделить параметры с `policy`. А головка значения находится на том же backbone, что и политика (`AutoModelForCausalLMWithValueHead` прикрепляет скалярную MLP-голову), поэтому TRL сообщает `policy/kl` и `value/loss` отдельно.

## Подводные камни

- **Переоптимизация / reward hacking.** RM несовершенна; `π_θ` находит состязательные завершения, которые получают высокую оценку, но являются плохими. Симптомы: награда растёт бесконечно, в то время как оценка людьми выходит на плато или падает. Решение: остановиться раньше, повысить `β`, расширить обучающие данные RM.
- **Эксплуатация длины.** RM, обученные на полезных ответах, часто неявно вознаграждают длину. Политика учится дополнять ответы. Средство: нормализованная по длине награда или RLAIF с RM, учитывающей длину.
- **Слишком маленькая RM.** RM должна быть как минимум такого же размера, как политика. Крошечная RM не может достоверно оценивать выходы политики.
- **Настройка KL.** Слишком низкое β → дрейф и reward hacking. Слишком высокое β → политика почти не меняется. Стандартный приём — *адаптивное* β, нацеленное на фиксированный KL на шаг.
- **Шум в данных предпочтений.** Около 30% меток людей зашумлены или неоднозначны. Калибруйте, обучая RM на данных, отфильтрованных по согласию, либо используйте температуру для BT.
- **Проблемы off-policy.** Данные PPO становятся немного off-policy после первой эпохи. Отслеживайте долю клиппирования, как в Lesson 08.

## Применение

RLHF в 2026 году слоист:

| Слой | Цель | Метод |
|-------|--------|--------|
| Следование инструкциям, полезность, безвредность | Выравнивание | DPO (Phase 10 · 08) предпочтительнее RLHF-PPO. |
| Корректность рассуждений (математика, код) | Возможности | GRPO с наградой от верификатора (Phase 9 · 12). |
| Многошаговые задачи с долгим горизонтом | Агентность | PPO / GRPO с process reward models по шагам. |
| Безопасность / поведение отказа | Безопасность | RLHF-PPO с отдельной RM безопасности, или Constitutional AI. |
| Best-of-N при инференсе | Быстрое выравнивание | Использование RM на этапе декодирования; обучение политики не требуется. |
| Дистилляция награды | Вычисления при инференсе | Обучение небольшой «головы вознаграждения» поверх замороженной LM. |

RLHF был *главным* методом в 2022–2024 годах. В 2026 году продакшен-пайплайны выравнивания используют DPO по умолчанию, а PPO — только для RM-интенсивных или критичных для безопасности этапов.

## Финализация

Сохраните как `outputs/skill-rlhf-architect.md`:

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## Упражнения

1. **Лёгкое.** Обучите модель вознаграждения Bradley-Terry в `code/main.py` на 500 синтетических парах предпочтений. Измерьте парную точность на отложенных 100 парах. Должна превышать 90%.
2. **Среднее.** Запустите игрушечный цикл PPO-RLHF с `β ∈ {0.0, 0.1, 1.0}`. Для каждого случая постройте график оценки RM в зависимости от KL-к-референсу по обновлениям. Какие запуски допускают reward hacking?
3. **Сложное.** Реализуйте DPO (функция потерь правдоподобия предпочтений в замкнутой форме) на тех же данных о предпочтениях и сравните с пайплайном RLHF-PPO по затраченным вычислениям и достигнутой итоговой оценке RM.

## Ключевые термины

| Термин | Что говорят люди | Что это на самом деле означает |
|------|-----------------|-----------------------|
| RLHF | «RL для выравнивания» | Трёхэтапный пайплайн SFT + RM + PPO (Christiano 2017, Ouyang 2022). |
| Reward Model (RM) | «Оценивающая сеть» | Обученная скалярная функция, подогнанная под парные предпочтения через Bradley-Terry. |
| Bradley-Terry | «Парная логистическая функция потерь» | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`; стандартная целевая функция для RM. |
| KL-штраф | «Оставаться рядом с референсом» | `β · KL(π_θ \|\| π_ref)` в награде; регуляризатор против reward hacking. |
| Reward hacking | «Закон Гудхарта» | Политика эксплуатирует изъяны RM; симптомы: награда растёт, оценка людьми не меняется. |
| RLAIF | «Предпочтения, размеченные ИИ» | RLHF, где метки получены от другой LM вместо людей. |
| PRM | «Process Reward Model» | Оценивает частичные шаги рассуждения; используется в reasoning-пайплайнах. |
| Constitutional AI | «Метод Anthropic» | Предпочтения, сгенерированные ИИ под руководством явных правил. |

## Дополнительное чтение

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — статья, с которой начался RLHF.
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — рецепт, лежащий в основе ChatGPT.
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — более ранний RLHF для суммаризации.
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO; дефолтный метод после RLHF в 2026 году.
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF и цикл самокритики.
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — статья HH.
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — продакшен `RewardTrainer` и `PPOTrainer`. Изучите исходный код тренера для деталей адаптивного KL и value-head.
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) от Lambert, Castricato, von Werra, Havrilla — канонический разбор трёхэтапного пайплайна с диаграммами.
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — сама библиотека; `examples/` содержит сквозные скрипты RLHF для Llama, Mistral и Qwen.
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — взгляд через reward-hypothesis; необходимый предварительный материал для размышлений о reward hacking.