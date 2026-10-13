# Моделирование наград и RLHF

> Человек не может написать функцию вознаграждения для "хорошего помощника ответа", но он может сравнить два ответа и выбрать лучший. Приспособить модель вознаграждения к этим сравнениям, затем RL языковой модель против него. Кристиано 2017. Инструкция GPT 2022.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment), Phase 9 · 08 (PPO)
**Time:** ~45 minutes

## Проблема

Вы обучили языковую модель на следующей токен-предсказание цели. Она пишет грамматический английский. Она также лжет, бродит, и отказывается отказаться. Вы не можете решить это с более предварительной подготовки  веб-текст является проблемой, а не лекарством.

Вы хотите *скалярную награду*, которая гласит: "отказ А лучше, чем ответ B для инструкции X". Написать эту функцию награды вручную невозможно. "Полезность" - это не закрытое выражение по токену.

RLHF (Christiano et al. 2017; Ouyang et al. 2022) преобразует предпочтения в модель вознаграждения, затем оптимизирует LM через PPO против этой награды.

В 2026 году этап PPO в основном заменяется DPO (Фаза 10 · 08) потому что он дешевле и почти столь же полезен для настройки настраивания. Но часть * модели вознаграждения* все еще лежит в основе каждого образца Best-of-N, каждого RL-от-проверяемых-вознаграждений-проводка, и каждого модели рассуждения, использующего модель процесса вознаграждения.

## Концепция

![Three-stage RLHF: SFT, RM training on pairwise prefs, PPO with KL penalty](../assets/rlhf.svg)

**Stage 1: Supervised Fine-Tuning (SFT).**Начните с предварительно подготовленной базовой модели. Напишите на человекопись демонстрации поведения (ответы, следующие за инструкцией, полезные ответы и т.д.). Результат: модель `π_SFT`что *пристрастно к хорошему поведению* но все же имеет неограниченное пространство для действий.

**Stage 2: Reward Model training.**

- Собирать пары ответов `(y_+, y_-)`к просьбе `x`, маркированный людьми как "y_+ предпочтительнее y_-."
- Обучить модель награды`R_φ(x, y)`присвоение более высоких баллов `y_+`- Я не знаю .
- Потеря:**Bradley-Terry pairwise logistic**В следующем случае:

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  В то же время, в некоторых странах, как и в других странах, в некоторых странах, в некоторых странах, в некоторых странах, в некоторых странах, в некоторых странах, в некоторых странах, в некоторых странах, в некоторых странах, в некоторых странах, в некоторых странах, в некоторых странах, в некоторых странах, в некоторых странах, в некоторых странах, в некоторых странах, в некоторых странах, в некоторых странах, в некоторых странах, в некоторых странах, в некоторых странах, где это происходит.

- `R_φ`Обычно эталон начинается с модели SFT с скалярной головой на вершине.

**Stage 3: PPO against the RM with KL penalty.**

- Инициализировать обучаемую политику `π_θ`от `π_SFT`Сохраняйте замороженную *справку*`π_ref = π_SFT`- Я не знаю .
- Вознаграждение в конце ответа `y`В следующем случае:

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  Наказание КЛ не позволяет .`π_θ`от произвольного отвлечения от `π_SFT` это *регулятор*, а не регион с твердым доверием. `β`Обычно`0.01`- Я не знаю.`0.05`- Я не знаю .
- Используйте PPO (урок 08) с этой наградой.

**Why the KL?**Без него PPO будет с радостью найти стратегии взлома наград  RM был обучен только в процессе выполнения распределения.`π_θ`Это один из самых важных кнопок в RLHF.

**2026 status:**

- **DPO**(Рафаилов 2023): closed-form алгебра срывается этапом 2+3 в единую контролируемую потерю по данным предпочтений. Нет RM, нет PPO. То же качество на ориентировочных показателях для части вычислений.
- **GRPO**(DeepSeek 20242025): ППО с базовой базой относительно группы вместо критика, вознаграждение от *verifier* (кодные действия / математические ответы соответствовали) вместо обученного человеком RM. Доминирующий для моделей рассуждения.
- **Process reward models (PRMs):**отчастие решения (каждый шаг рассуждения), используемые как в вариантах RLHF, так и GRPO для рассуждения.
- **Constitutional AI / RLAIF:**используйте выровненный LLM для получения предпочтений вместо людей.

```figure
reward-model
```

## Создайте его

В этом уроке используются крошечные синтетические "промпты" и "ответы", представленные в виде строк. RM является линейным балловым заметником по сравнению с представлением торговой пакеты.`code/main.py`- Я не знаю .

### Шаг 1: данные о синтетических предпочтениях

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

В реальном RLHF это заменяется человеческими маркировками.`(prompt, preferred_response, rejected_response)` идентично.

### Шаг 2: модель награды Брэдли-Терри

Линейный балл: `R(x, y) = w · bag(y)`. Поезд для минимизации потерь в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в виде потери в потери в потери в потери в потери в потери в потери в потери в потери в потери в потери в потери в потери в потери в потери в потери в потери в потери в потери в потери в потери в потери в потери в потери в потери в потери.

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

После нескольких сотен обновлений,`w`Придает положительные значения хорошим знакам и отрицательные - плохим.

### Шаг 3: Политика, подобная ППО, наряду с RM

Наша политика в области игрушек производит один токен из словарного запаса.`log π_θ(token | prompt)`, добавьте KL-относительное наказание, и примените вырезанный заместитель PPO.

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

### Шаг 4: следите за КЛ

Средняя трасса `KL(π_θ || π_ref)`Каждое обновление.`~5-10`политика отклонилась от `π_SFT` ниже `β`Это лучший диагноз в реальной RLHF.

### Шаг 5: рецепт производства с ТРЛ

Как только вы поймете, что такое "провод игрушек", здесь будет та же цикл, что и настоящий пользователь библиотеки.[TRL](https://huggingface.co/docs/trl)является эталонной реализацией  `RewardTrainer`для этапа 2 и `PPOTrainer`(с встроенным KL-to-reference) для третьего этапа.

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

Три вещи, которые библиотека делает для вас.`adap_kl_ctrl=True`выполняет график адаптивных β: если наблюдаемая КЛ превышает `target_kl`Референтная модель заморожена по конвенции  вы не должны случайно делить параметры с `policy`И ценность входит в число тех же, что и в полис (`AutoModelForCausalLMWithValueHead`прикрепляет скалярную голову MLP), поэтому TRL сообщает `policy/kl`и `value/loss`отдельно.

## Ловушки

- **Over-optimization / reward hacking.**РМ не совершенна .`π_θ`обнаруживает противоречивые результаты, которые получают высокие но плохие результаты. Симптомы: вознаграждение поднимается неопределенно, а оценка человека - плато или падает.`β`, расширить данные о подготовке к РМ.
- **Length hacking.**РМ, обученные на полезных ответах, часто косвенно вознаграждают длину. Политика учится заполнять ответы.
- **Too-small RM.**Маленький РМ не может достойно оценить выходы полиса.
- **KL tuning.**Слишком низкий β → дрейф и взлом награды. Слишком высокий β → политика едва меняется. Стандартный трюк - * адаптивный* β, который нацелен на фиксированный KL на шаг.
- **Preference-data noise.**~ 30% человеческих ярлыков шумно или неоднозначно.
- **Off-policy problems.**Данные о ППО слегка не соответствуют политике после первой эпохи.

## Используйте его

RLHF в 2026 году сложен:

| Layer | Target | Method |
|-------|--------|--------|
| Instruction following, helpfulness, harmlessness | Alignment | DPO (Phase 10 · 08) preferred over RLHF-PPO. |
| Reasoning correctness (math, code) | Capability | GRPO with verifier reward (Phase 9 · 12). |
| Long-horizon multi-step tasks | Agentic | PPO / GRPO with process reward models over steps. |
| Safety / refusal behavior | Safety | RLHF-PPO with separate safety RM, or Constitutional AI. |
| Best-of-N at inference | Fast alignment | Use RM at decode time; no policy training needed. |
| Reward distillation | Inference compute | Train a small "reward head" on top of a frozen LM. |

В 2026 году трубопроводы по производственному выравниванию будут DPO-first, PPO-only для интенсивных RM или критически важных для безопасности шагов.

## Погрузи его

Сделайте это .`outputs/skill-rlhf-architect.md`В следующем случае:

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

1. **Easy.**Обучайте модель награды Брэдли-Терри в `code/main.py`на 500 синтетических парах предпочтений. измерять точность паров на 100 парах, которые не проходят.
2. **Medium.**Запустить игрушку PPO-RLHF с помощью `β ∈ {0.0, 0.1, 1.0}`За каждый, зафиксируйте результат RM против KL-ссылки по обновлениям.
3. **Hard.**Внедрить DPO (утрата вероятности предпочтения закрытой формы) на одних и тех же данных о предпочтениях и сравнить с трубопроводом RLHF-PPO в используемом вычислении и достигнутом окончательном RM-почте.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| RLHF | "Alignment RL" | Three-stage SFT + RM + PPO pipeline (Christiano 2017, Ouyang 2022). |
| Reward Model (RM) | "The scoring net" | Learned scalar function fit to pairwise preferences via Bradley-Terry. |
| Bradley-Terry | "Pairwise logistic loss" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`; the standard RM objective. |
| KL penalty | "Stay near the reference" | `β · KL(π_θ \|\| π_ref)` in the reward; the anti-reward-hacking regularizer. |
| Reward hacking | "Goodhart's law" | Policy exploits RM flaws; symptoms: reward up, human eval flat. |
| RLAIF | "AI-labeled preferences" | RLHF where labels come from another LM instead of humans. |
| PRM | "Process Reward Model" | Scores partial reasoning steps; used in reasoning pipelines. |
| Constitutional AI | "Anthropic's method" | AI-generated preferences guided by explicit rules. |

## Дальнейшее чтение

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) газета, которая начала RLHF.
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) рецепт, стоящий за ChatGPT.
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) ранее РЛХФ для обобщения.
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) DPO; дефолт после РЛХФ в 2026 году.
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) RLAIF и цикл самокритики.
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) бумаги HH.
- [Hugging Face TRL library](https://huggingface.co/docs/trl) производство `RewardTrainer`и `PPOTrainer`. Прочитайте источник тренера для подробностей о адаптивном КЛ и ценностном уровне.
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf)Ламберта, Кастрикато, фон Вера, Хаврилла  канонический проход трубопровода в трех этапах с диаграммами.
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) библиотека; `examples/`имеет полное RLHF сценарий для Llama, Mistral и Qwen.
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) взгляд на гипотезу награды; существенная предпосылка для размышления о взломе награды.
