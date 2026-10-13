# Отслеживание состояния диалога (Dialogue State Tracking)

> «Хочу недорогой ресторан на севере... хотя нет, пусть будет средний по цене... и добавьте итальянскую кухню». Три реплики, три обновления состояния. DST поддерживает словарь слот-значение в синхронном виде, чтобы бронирование сработало.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 20 (Structured Outputs)
**Time:** ~75 минут

## Проблема

В диалоговой системе, ориентированной на выполнение задач, цель пользователя кодируется как набор пар слот-значение: `{cuisine: italian, area: north, price: moderate}`. Каждая реплика пользователя может добавить, изменить или удалить слот. Система должна прочитать весь диалог и корректно вывести текущее состояние.

Ошибись в одном слоте — и система забронирует не тот ресторан, назначит не тот рейс или спишет деньги не с той карты. DST — это шарнир между тем, что сказал пользователь, и тем, что выполняет бэкенд.

Почему это всё ещё важно в 2026 году, несмотря на LLM:

- В чувствительных к комплаенсу областях (банкинг, здравоохранение, бронирование авиабилетов) требуются детерминированные значения слотов, а не свободная генерация.
- Агентам, использующим инструменты, всё ещё нужно разрешение слотов перед вызовом API.
- Многоходовая коррекция сложнее, чем кажется: «нет, на самом деле пусть будет четверг».

Современный конвейер: классические концепции DST + LLM-экстракторы + ограничения структурированного вывода.

## Концепция

![DST: dialog history → slot-value state](../assets/dst.svg)

**Структура задачи.** Схема определяет домены (ресторан, отель, такси) и их слоты (кухня, район, цена, число людей). Каждый слот может быть пустым, заполненным значением из замкнутого множества (цена: {cheap, moderate, expensive}) или свободным значением (имя: «The Copper Kettle»).

**Две формулировки DST.**

- **Классификация.** Для каждой пары (слот, значение-кандидат) предсказывается да/нет. Работает для слотов с закрытым словарём. Стандарт до 2020 года.
- **Генерация.** По диалогу генерируются значения слотов в виде свободного текста. Работает для слотов с открытым словарём. Современный вариант по умолчанию.

**Метрика.** Joint Goal Accuracy (JGA) — доля реплик, где *все* слоты верны. Всё или ничего. Лидерборд MultiWOZ 2.4 в 2026 году достигает максимум около 83%.

**Архитектуры.**

1. **На основе правил (регулярные выражения + ключевые слова).** Сильная базовая линия для узких доменов. Легко отлаживается.
2. **TripPy / BERT-DST.** Генерация на основе копирования с BERT-кодированием. Стандарт до эпохи LLM.
3. **LDST (LLaMA + LoRA).** Инструктированная LLM с промптингом по доменам и слотам. Достигает качества уровня ChatGPT на MultiWOZ 2.4.
4. **Без онтологии (2024–26).** Схема не используется; имена и значения слотов генерируются напрямую. Подходит для открытых доменов.
5. **Промпт + структурированный вывод (2024–26).** LLM со схемой Pydantic + ограниченное декодирование. 5 строк кода, готово к продакшену.

### Классические сбои

- **Кореференция между репликами.** «Давайте остановимся на первом варианте». Нужно разрешить, какой именно вариант.
- **Перезапись vs добавление.** Пользователь говорит «добавьте итальянскую». Заменить кухню или дополнить?
- **Неявные подтверждения.** «Ок, хорошо» — означает ли это согласие на предложенное бронирование?
- **Коррекция.** «На самом деле пусть будет 7 вечера». Нужно обновить время, не сбрасывая остальные слоты.
- **Кореференция к предыдущей реплике системы.** «Да, тот самый». Какой «тот самый»?

```figure
n5-slot-tracker
```

## Реализация

### Шаг 1: экстрактор слотов на основе правил

См. `code/main.py`. Регулярные выражения + словари синонимов покрывают 70% канонических реплик в узких доменах:

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

Хрупко за пределами канонического словаря. Работает для детерминированных подтверждений слотов.

### Шаг 2: цикл обновления состояния

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

Три инварианта:

- Никогда не сбрасывать слот, которого пользователь не касался.
- Явное отрицание («забудьте про кухню») должно очищать слот.
- Коррекция пользователя («на самом деле...») должна перезаписывать, а не дополнять.

### Шаг 3: DST на основе LLM со структурированным выводом

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic гарантируют корректный объект состояния. Никаких регулярных выражений, несоответствий схеме или галлюцинированных слотов.

### Шаг 4: оценка JGA

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

Откалибруйте: в какой доле реплик система правильно определяет ВСЕ слоты? Для MultiWOZ 2.4 лучшие системы 2026 года показывают 80–83%. Ваша система для конкретного домена должна превосходить этот показатель на узком словаре, иначе базовая LLM обойдёт вас.

### Шаг 5: обработка коррекции

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

При обнаружении коррекции перезаписывайте последний обновлённый слот, а не дополняйте его. Без помощи LLM сделать это правильно сложно. Современный паттерн: всегда давать LLM регенерировать всё состояние целиком из истории, а не обновлять его инкрементально — это естественным образом обрабатывает коррекции.

## Подводные камни

- **Стоимость полной регенерации истории.** Если LLM регенерирует состояние на каждой реплике, суммарные затраты токенов составляют O(n²). Ограничивайте историю или суммаризируйте старые реплики.
- **Дрейф схемы.** Добавление новых слотов постфактум ломает старые обучающие данные. Версионируйте схему.
- **Чувствительность к регистру.** «Italian» vs «italian» vs «ITALIAN» — нормализуйте везде.
- **Неявное наследование.** Если пользователь ранее указал «на 4 человек», новый запрос на другое время не должен очищать слот «люди». Всегда передавайте полную историю.
- **Свободная форма vs закрытое множество.** Имена, время и адреса требуют свободных слотов; кухни и районы — закрытые. Сочетайте оба типа в схеме.

## Применение

Стек 2026 года:

| Ситуация | Подход |
|-----------|----------|
| Узкий домен (одно-два намерения) | На основе правил + регулярные выражения |
| Широкий домен, есть размеченные данные | LDST (LLaMA + LoRA на данных в стиле MultiWOZ) |
| Широкий домен, нет меток, готово к продакшену | LLM + Instructor + схема Pydantic |
| Устная речь / голос | ASR + нормализатор + LLM-DST |
| Многодоменный процесс бронирования | LLM с управлением по схеме + Pydantic-модели для каждого домена |
| Чувствительность к комплаенсу | Основной подход на правилах, LLM как резерв с подтверждающим потоком |

## Публикация

Сохраните как `outputs/skill-dst-designer.md`:

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## Упражнения

1. **Лёгкое.** Постройте трекер состояния на основе правил в `code/main.py` для 3 слотов (кухня, район, цена). Протестируйте на 10 вручную составленных диалогах. Измерьте JGA.
2. **Среднее.** Тот же набор данных с Instructor + Pydantic + небольшой LLM. Сравните JGA. Изучите самые сложные реплики.
3. **Сложное.** Реализуйте оба подхода и организуйте маршрутизацию: основной — на правилах, резервный — LLM, когда система на правилах выдаёт <2 слотов с уверенностью. Измерьте комбинированный JGA и стоимость инференса на реплику.

## Ключевые термины

| Термин | Как говорят люди | Что это на самом деле означает |
|------|-----------------|-----------------------|
| DST | Отслеживание состояния диалога | Поддержание словаря слот-значение на протяжении реплик диалога. |
| Slot (слот) | Единица намерения пользователя | Именованный параметр, нужный бэкенду (кухня, дата). |
| Domain (домен) | Область задачи | Ресторан, отель, такси — наборы слотов. |
| JGA | Joint Goal Accuracy | Доля реплик, где все слоты верны. Всё или ничего. |
| MultiWOZ | Бенчмарк | Многодоменный WOZ-датасет; стандартная оценка DST. |
| Ontology-free DST (DST без онтологии) | Без схемы | Прямая генерация имён и значений слотов без фиксированного списка. |
| Correction (коррекция) | «На самом деле...» | Реплика, перезаписывающая ранее заполненный слот. |

## Дополнительное чтение

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — канонический бенчмарк.
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — инструктивная настройка LLaMA + LoRA для DST.
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — рабочая лошадка DST на основе копирования.
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — неконтролируемый TOD на основе EM.
- [Лидерборд MultiWOZ](https://github.com/budzianowski/multiwoz) — канонические результаты DST.