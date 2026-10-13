# Создайте трансформатор с нуля  Капстон

> Тринадцать уроков, одна модель, никаких скортинок.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 01 through 13. Don't skip.
**Time:** ~120 minutes

## Проблема

Вы прочли все статьи, вы реализовали внимание, многоголовые разделения, позиционные кодирования, блоки кодера и декодера, потери BERT и GPT, MoE, KV cache. Теперь сделайте их вместе для реальной задачи.

Ключевой камень: обучить маленького трансформатора, использующего только декодер, от конца к концу на задачу моделирования языка на уровне персонажей. Он читает Шекспира. Он генерирует нового Шекспира. Он достаточно маленький, чтобы тренироваться на ноутбуке менее чем за 10 минут.

Это "наноГПТ" курса. Это не оригинальный  Карпатия 2023 наноГПТ учебник является справочной реализации каждый студент пишет по крайней мере раз. Мы поднимаем форму и переделать ее вокруг того, что мы охватывали.

## Концепция

![Transformer-from-scratch block diagram](../assets/capstone.svg)

Архитектура, отмечается:

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### Что мы отправляем

- `GPTConfig` одно место для конфигурации всех гиперпараметров.
- `MultiHeadAttention` причинно-следственный, партийно, с опциональным Flash-стиль пути (PyTorch `scaled_dot_product_attention`)).
- `SwiGLUFFN` современная FFN.
- `Block` Пренормальное, остаточное впакованное внимание + FFN.
- `GPT` вкладки, блоки, головка LM, генерация().
- Тренировочный цикл с AdamW, косиновой LR, срезка градиента.
- Токенер уровня "Чар" на тексте Шекспира.

### Что мы не отправляем

- RoPE  реализовано концептуально в уроке 04. Здесь мы используем обученные позиционные внедрения для простоты.
- Кашевые упражнения требуют добавления кашевых упражнений.
- Внимание в блистательном виде  PyTorch 2.0+ автоматические диспетчеры, если вводы совпадают; мы используем `F.scaled_dot_product_attention`- Я не знаю .
- Вы видели МОЕ в уроке 11.

### Целевые показатели

На ноутбуке Mac M2, 4-слойный, 4-главный, д_модель=128 GPT обучен на 2000 шагов на `tinyshakespeare.txt`В следующем случае:

- Убытки тренировок совпадают с ~4,2 (случайный) до ~1,5 за около 6 минут.
- Выборка выглядит в форме Шекспира: появляются архаические слова, перерывы в строках, собственные имена, такие как "ROMEO:"
- Утрата валя (окончательный 10% текста) внимательно отслеживает потерю обучения; не переоценивается в этом размере/бюджете.

```figure
n5-block-stack
```

## Создайте его

В этом уроке используется PyTorch.`torch`(Стоварная система в порядке).`code/main.py`Сценарий:

- Скачать `tinyshakespeare.txt`если отсутствует (или читает местный экземпляр).
- Токенер на уровне байтов.
- Раздел поезда/валя на 90/10.
- Тренировочный цикл с автоматическим загрузкой bf16 на поддерживаемом оборудовании.
- Пробовая работа завершена после тренировки.

### Шаг 1: данные

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 уникальных символов, небольшой словарный запас, удобный для 4-байтовского словарного размера, без BPE, без токенызатора.

### Шаг 2: модель

Посмотрите .`code/main.py`. Блок - учебник из урока 05  преднормальный, RMSNorm, SwiGLU, причинно-следственный MHA.

### Шаг 3: тренировочный цикл

Получить случайную партию длины 256 символов окон вперед, сдвиг по кросс-энтропии, назад, шаг Адам В, запись, повторяю.

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### Шаг 4: образец

При задании просьбы, неоднократно переводим вперед, пробуем из верхних логитов, добавляем и продолжаем.

### Шаг 5: прочитайте выход

После 2000 шагов:

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

Не Шекспир, но в форме Шекспира, ясная победа за 800 тысяч параметров и 6 минут на ноутбуке.

## Используйте его

Этот камень - это архитектура ссылки.

1. **Swap the tokenizer.**Используйте BPE (например `tiktoken.get_encoding("cl100k_base")`Размер словарного класса увеличивается с 65 до 50 000 человек.
2. **Train on a bigger corpus.**Использование`OpenWebText`или `fineweb-edu`Токены 10B на одной A100 требуют около 24 часов для GPT 125M-параметров.
3. **Add RoPE + KV cache + Flash Attention.**Упражнения ниже приведут вас к каждому из них.

Это в конечном итоге является 125M-параметром GPT, который генерирует свободный английский язык. Не является приграничной моделью. Но тот же код путь  просто больше  это то, что Карпатия, EleutherAI и Институт Аллен используют для обучения исследовательских пунктов пропуска в 2026 году.

## Погрузи его

Посмотрите .`outputs/skill-transformer-review.md`. Навыки проверяют реализацию трансформатора с нуля на предмет правильности всех 13 предыдущих уроков.

## Упражнения

1. **Easy.**Беги .`code/main.py`. Убедитесь, что потеря в подтверждении в последнем этапе вашей модели ниже 2.0. Изменение `max_steps`от 2000 до 5000 , потеря вали продолжает улучшаться?
2. **Medium.**Замените выученные позиционные вкладки RoPE.`MultiHeadAttention`Поезд и проверка потерь валя не менее низкая.
3. **Medium.**Внедрить кеширование KV в петлю выборки образцов. генерировать 500 токенов с и без кеширования.
4. **Hard.**Добавьте вторую главу к модели, которая предсказывает следующий плюс-один токен (MTP  Multi-Token Prediction from DeepSeek-V3).
5. **Hard.**Заменить FFN на блок с 4-экспертной MoE. Рутер + верхний-2 маршрутизация.

## Ключевые условия

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| nanoGPT | "Karpathy's tutorial repo" | Minimal decoder-only transformer training code, ~300 LOC; the canonical reference. |
| tinyshakespeare | "The standard toy corpus" | ~1.1 MB of text; every character-LM tutorial since 2015 uses it. |
| Tied embeddings | "Share input/output matrix" | LM head weight = transpose of token embedding matrix; saves parameters, improves quality. |
| bf16 autocast | "Training precision trick" | Run forward/back in bf16, keep optimizer state in fp32; standard since 2021. |
| Gradient clipping | "Stops spikes" | Cap global grad norm at 1.0; prevents training blowups. |
| Cosine LR schedule | "The 2020+ default" | LR ramps up linearly (warmup) then decays cosine-shaped to 10% of peak. |
| MFU | "Model FLOP Utilization" | Achieved FLOPs / theoretical peak; 40% dense, 30% MoE is strong in 2026. |
| Val loss | "Held-out loss" | Cross-entropy on data the model never saw; overfit detector. |

## Дальнейшее чтение

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) классическая аннотационная реализация.
