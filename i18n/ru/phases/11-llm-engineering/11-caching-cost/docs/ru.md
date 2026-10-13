# Кэширование, ограничение скорости запросов и оптимизация затрат

> Большинство ИИ-стартапов гибнут не из-за плохих моделей. Они гибнут из-за плохой юнит-экономики. Один вызов GPT-4o стоит доли цента. Десять тысяч пользователей, делающих по десять запросов в день, обходятся в $250 только на входных токенах — еще до того, как вы заработаете хоть доллар. Выживают те компании, которые относятся к каждому API-вызову как к финансовой транзакции, а не как к вызову функции.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 Lesson 09 (Function Calling)
**Time:** ~45 минут
**Related:** Phase 11 · 15 (Prompt Caching) — этот урок посвящен кэшированию на уровне приложения (семантический кэш, кэш точных хэшей, маршрутизация моделей). Урок 15 посвящен кэшированию промптов на уровне провайдера (Anthropic cache_control, автоматическое кэширование OpenAI, Gemini CachedContent). Сочетайте оба подхода для снижения затрат на 50-95%.

## Цели обучения

- Реализовать семантическое кэширование, которое обслуживает повторяющиеся или похожие запросы из кэша вместо нового API-вызова
- Рассчитать стоимость каждого запроса у разных провайдеров и реализовать ограничение скорости запросов с учетом токенов и оповещения о бюджете
- Построить слой оптимизации затрат со сжатием промптов, маршрутизацией моделей (дорогие vs дешевые) и кэшированием ответов
- Спроектировать многоуровневую стратегию кэширования, используя точное совпадение, семантическое сходство и кэширование префиксов для разных типов запросов

## Проблема

Вы создаете RAG-чат-бота. Он прекрасно работает. Пользователям он нравится.

Затем приходит счет.

GPT-5 стоит $5 per million input tokens and $15 за миллион выходных токенов. Claude Opus 4.7 стоит $15 input / $75 за выходные токены. Gemini 3 Pro стоит $1.25 input / $5 за выходные токены. GPT-5-mini стоит $0.25/$2. Цены ниже приведены для иллюстрации; всегда проверяйте актуальную страницу цен провайдера.

Вот математика, которая убивает стартапы:

- 10 000 ежедневных активных пользователей
- 10 запросов на пользователя в день
- 1 000 входных токенов на запрос (системный промпт + контекст + сообщение пользователя)
- 500 выходных токенов на ответ

**Дневная стоимость входных токенов:** 10 000 x 10 x 1 000 / 1 000 000 x $2.50 = **$250/день**
**Дневная стоимость выходных токенов:** 10 000 x 10 x 500 / 1 000 000 x $10.00 = **$500/день**
**Итого в месяц:** **$22,500/месяц**

И это только LLM. Добавьте эмбеддинги, хостинг векторной базы данных, инфраструктуру. Получается $30 000/месяц за чат-бота.

Самое жестокое: 40-60% этих запросов — почти дубликаты. Пользователи задают одни и те же вопросы немного разными словами. Ваш системный промпт — идентичный в каждом запросе — оплачивается каждый раз заново. Контекстные документы, полученные через RAG, повторяются у пользователей, спрашивающих об одной и той же теме.

Вы платите полную цену за избыточные вычисления.

## Концепция

### Анатомия стоимости вызова LLM

У каждого API-вызова есть пять компонентов стоимости.

```mermaid
graph LR
    A[User Query] --> B[System Prompt<br/>500-2000 tokens]
    A --> C[Retrieved Context<br/>500-4000 tokens]
    A --> D[User Message<br/>50-500 tokens]
    B --> E[Input Cost<br/>$2.50/1M tokens]
    C --> E
    D --> E
    E --> F[Model Processing]
    F --> G[Output Cost<br/>$10.00/1M tokens]
```

Системные промпты — тихий убийца. Системный промпт в 1 500 токенов, отправляемый с каждым запросом, стоит $3.75 per million requests just for that prefix. At 100K requests per day, that is $375/день — $11 250/месяц — за текст, который никогда не меняется.

### Кэширование у провайдера: встроенные скидки

Все три крупнейших провайдера предлагают кэширование промптов на своей стороне в 2026 году, но механика отличается. Подробный разбор см. в Phase 11 · 15.

| Провайдер | Механизм | Скидка | Минимум | Длительность кэша |
|----------|-----------|----------|---------|----------------|
| Anthropic | Явные маркеры cache_control | 90% на попаданиях в кэш (доплата 25% при записи) | 1 024 токена (Sonnet/Opus), 2 048 (Haiku) | 5 мин по умолчанию; расширенный 1 час (двойная премия за запись) |
| OpenAI | Автоматическое сопоставление префиксов | 50% на попаданиях в кэш | 1 024 токена | До 1 часа, по принципу "лучшее усилие" |
| Google Gemini | Явный API CachedContent | снижение ~75% (плюс хранение) | 4 096 (Flash) / 32 768 (Pro) | TTL настраивается пользователем |

**Подход Anthropic** явный. Вы отмечаете разделы промпта с помощью `cache_control: {"type": "ephemeral"}`. Первый запрос оплачивается с 25%-й премией за запись. Последующие запросы с тем же префиксом получают скидку 90%. Системный промпт в 2 000 токенов, который стоит $0.005 normally costs $0.000625 при попадании в кэш. При 100 тыс. запросов это экономит $437.50/день.

**Подход OpenAI** автоматический. Любой префикс промпта, совпадающий с предыдущим запросом, получает скидку 50%. Маркеры не нужны. Компромисс: меньше скидка, меньше контроля, но нулевые усилия по внедрению.

### Семантическое кэширование: ваш собственный слой

Кэширование провайдера работает только для идентичных префиксов. Семантическое кэширование решает более сложную задачу: разные запросы с одинаковым смыслом.

«Какая у вас политика возврата?» и «Как мне вернуть товар?» — это разные строки, но идентичное намерение. Семантический кэш встраивает оба запроса, вычисляет косинусное сходство и возвращает закэшированный ответ, если сходство превышает порог (обычно 0.92-0.95).

```mermaid
flowchart TD
    A[User Query] --> B[Embed Query]
    B --> C{Similar query<br/>in cache?}
    C -->|sim > 0.95| D[Return Cached Response]
    C -->|sim < 0.95| E[Call LLM API]
    E --> F[Cache Response<br/>with Embedding]
    F --> G[Return Response]
    D --> G
```

Затраты на эмбеддинги незначительны. text-embedding-3-small от OpenAI стоит $0.02 за миллион токенов. Проверка кэша стоит почти ничего по сравнению с полным вызовом LLM.

### Точное кэширование: хэш и сопоставление

Для детерминированных вызовов (temperature=0, та же модель, тот же промпт) точное кэширование проще и быстрее. Хэшируем полный промпт, проверяем кэш, возвращаем результат, если найден.

Это отлично работает для:
- Системного промпта + фиксированного контекста + идентичных пользовательских запросов
- Вызова функций с идентичными определениями инструментов
- Пакетной обработки, когда один и тот же документ обрабатывается несколько раз

### Ограничение скорости запросов: защита вашего бюджета

Ограничение скорости — это не только про справедливость. Это про выживание.

**Алгоритм token bucket (корзины токенов):** у каждого пользователя есть корзина из N токенов, которая пополняется со скоростью R токенов в секунду. Запрос расходует токены из корзины. Если корзина пуста, запрос отклоняется. Это позволяет всплески (использовать всю корзину сразу), одновременно обеспечивая соблюдение средней скорости.

**Квоты на пользователя:** устанавливайте дневные/месячные лимиты токенов по уровням пользователей.

| Уровень | Дневной лимит токенов | Макс. запросов/мин | Доступ к моделям |
|------|------------------|------------------|-------------|
| Free | 50 000 | 10 | Только GPT-4o-mini |
| Pro | 500 000 | 60 | GPT-4o, Claude Sonnet |
| Enterprise | 5 000 000 | 300 | Все модели |

### Маршрутизация моделей: нужная модель для нужной задачи

Не каждый запрос требует GPT-4o.

«Во сколько закрывается магазин?» не требует $10/M-output model. GPT-4o-mini at $0.60/млн выходных токенов отлично справляется. Claude Haiku с ценой $1.25/млн выходных токенов тоже справляется. Простой классификатор направляет дешевые запросы к дешевым моделям, а сложные — к дорогим.

```mermaid
flowchart TD
    A[User Query] --> B[Complexity Classifier]
    B -->|Simple: lookup, FAQ| C[GPT-4o-mini<br/>$0.15/$0.60 per 1M]
    B -->|Medium: analysis, summary| D[Claude Sonnet<br/>$3.00/$15.00 per 1M]
    B -->|Complex: reasoning, code| E[GPT-4o / Claude Opus<br/>$2.50/$10.00+]
```

Хорошо настроенный маршрутизатор экономит 40-70% только на стоимости моделей.

### Отслеживание затрат: знайте, куда уходят деньги

Невозможно оптимизировать то, что не измеряется. Логируйте каждый API-вызов с:

- Временной меткой
- Названием модели
- Входными токенами
- Выходными токенами
- Задержкой (мс)
- Вычисленной стоимостью ($)
- ID пользователя
- Попадание/промах кэша
- Категорией запроса

Эти данные показывают, какие функции дорогие, какие пользователи наиболее активно потребляют ресурсы, и где кэширование дает наибольший эффект.

### Пакетная обработка: оптовые скидки

Batch API от OpenAI обрабатывает запросы асинхронно со скидкой 50%. Вы отправляете пакет до 50 000 запросов, и результаты возвращаются в течение 24 часов.

Используйте пакетную обработку для:
- Ночной обработки документов
- Массовой классификации
- Запусков оценки
- Пайплайнов обогащения данных

Не используйте для: запросов пользователей в реальном времени (важна задержка).

### Бюджетные оповещения и автоматические выключатели

Автоматический выключатель (circuit breaker) останавливает расходы при достижении лимита. Без него баг или злоупотребление могут сжечь весь месячный бюджет за часы.

Установите три порога:
1. **Предупреждение** (70% бюджета): отправить оповещение
2. **Торможение** (85% бюджета): переключиться только на более дешевые модели
3. **Остановка** (95% бюджета): отклонять новые запросы, возвращать только закэшированные ответы

### Стек оптимизации

Применяйте эти техники по порядку. Каждый слой усиливает предыдущие.

| Слой | Техника | Типичная экономия | Усилия по внедрению |
|-------|-----------|----------------|----------------------|
| 1 | Кэширование промптов провайдера | 30-50% | Низкие (добавить маркеры кэша) |
| 2 | Точное кэширование | 10-20% | Низкие (хэш + словарь) |
| 3 | Семантическое кэширование | 15-30% | Средние (эмбеддинги + сходство) |
| 4 | Маршрутизация моделей | 40-70% | Средние (классификатор) |
| 5 | Ограничение скорости запросов | Защита бюджета | Низкие (token bucket) |
| 6 | Сжатие промптов | 10-30% | Средние (переписывание промптов) |
| 7 | Пакетная обработка | 50% для подходящих случаев | Низкие (batch API) |

RAG-приложение, применяющее слои 1-5, обычно снижает затраты с $22,500/month to $4 000-6 000/месяц. Это разница между сжиганием ресурсов и построением бизнеса.

### Реальная экономия: до и после

Вот реальная разбивка для RAG-чат-бота, обслуживающего 10 000 DAU.

| Метрика | До оптимизации | После оптимизации | Экономия |
|--------|--------------------|--------------------|---------|
| Месячная стоимость LLM | $22,500 | $5 200 | 77% |
| Средняя стоимость запроса | $0.0075 | $0.0017 | 77% |
| Доля попаданий в кэш | 0% | 52% | -- |
| Запросов, направленных на mini | 0% | 65% | -- |
| P95 задержка | 2 800 мс | 900 мс (попадания в кэш: 50 мс) | 68% |
| Месячная стоимость эмбеддингов | $0 | $180 | (новая статья затрат) |
| Общая месячная стоимость | $22,500 | $5 380 | 76% |

Стоимость эмбеддингов для семантического кэширования ($180/месяц) окупается в течение первого часа попаданий в кэш.

```figure
semantic-cache
```

## Создайте это

### Шаг 1: Калькулятор стоимости

Постройте калькулятор стоимости токенов, который знает текущие цены основных моделей.

```python
import hashlib
import time
import json
import math
from dataclasses import dataclass, field


MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00, "cached_input": 1.25},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.075},
    "gpt-4.1": {"input": 2.00, "output": 8.00, "cached_input": 0.50},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cached_input": 0.10},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "cached_input": 0.025},
    "o3": {"input": 2.00, "output": 8.00, "cached_input": 0.50},
    "o3-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.55},
    "o4-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.275},
    "claude-opus-4": {"input": 15.00, "output": 75.00, "cached_input": 1.50},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-haiku-3.5": {"input": 0.80, "output": 4.00, "cached_input": 0.08},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00, "cached_input": 0.3125},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60, "cached_input": 0.0375},
}


def calculate_cost(model, input_tokens, output_tokens, cached_input_tokens=0):
    if model not in MODEL_PRICING:
        return {"error": f"Unknown model: {model}"}
    pricing = MODEL_PRICING[model]
    non_cached = input_tokens - cached_input_tokens
    input_cost = (non_cached / 1_000_000) * pricing["input"]
    cached_cost = (cached_input_tokens / 1_000_000) * pricing["cached_input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total = input_cost + cached_cost + output_cost
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "input_cost": round(input_cost, 6),
        "cached_input_cost": round(cached_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total, 6),
    }
```

### Шаг 2: Точный кэш

Хэшируйте полный промпт и возвращайте закэшированные ответы для идентичных запросов.

```python
class ExactCache:
    def __init__(self, max_size=1000, ttl_seconds=3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _hash(self, model, messages, temperature):
        key_data = json.dumps({"model": model, "messages": messages, "temperature": temperature}, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, model, messages, temperature=0.0):
        if temperature > 0:
            self.misses += 1
            return None
        key = self._hash(model, messages, temperature)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                self.hits += 1
                entry["access_count"] += 1
                return entry["response"]
            del self.cache[key]
        self.misses += 1
        return None

    def put(self, model, messages, temperature, response):
        if temperature > 0:
            return
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache, key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        key = self._hash(model, messages, temperature)
        self.cache[key] = {
            "response": response,
            "timestamp": time.time(),
            "access_count": 1,
        }

    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0,
            "cache_size": len(self.cache),
        }
```

### Шаг 3: Семантический кэш

Встраивайте запросы и возвращайте закэшированные ответы, когда сходство превышает порог.

```python
def simple_embed(text):
    words = text.lower().split()
    vocab = {}
    for w in words:
        vocab[w] = vocab.get(w, 0) + 1
    norm = math.sqrt(sum(v * v for v in vocab.values()))
    if norm == 0:
        return {}
    return {k: v / norm for k, v in vocab.items()}


def cosine_similarity(a, b):
    if not a or not b:
        return 0.0
    all_keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in all_keys)
    return dot


class SemanticCache:
    def __init__(self, similarity_threshold=0.85, max_size=500, ttl_seconds=3600):
        self.entries = []
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def get(self, query):
        query_embedding = simple_embed(query)
        now = time.time()
        best_match = None
        best_sim = 0.0
        for entry in self.entries:
            if now - entry["timestamp"] > self.ttl:
                continue
            sim = cosine_similarity(query_embedding, entry["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_match = entry
        if best_match and best_sim >= self.threshold:
            self.hits += 1
            best_match["access_count"] += 1
            return {"response": best_match["response"], "similarity": round(best_sim, 4), "original_query": best_match["query"]}
        self.misses += 1
        return None

    def put(self, query, response):
        if len(self.entries) >= self.max_size:
            self.entries.sort(key=lambda e: e["timestamp"])
            self.entries.pop(0)
        self.entries.append({
            "query": query,
            "embedding": simple_embed(query),
            "response": response,
            "timestamp": time.time(),
            "access_count": 1,
        })

    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0,
            "cache_size": len(self.entries),
        }
```

### Шаг 4: Ограничитель скорости запросов

Ограничитель скорости запросов по методу token bucket с квотами на пользователя.

```python
class TokenBucketRateLimiter:
    def __init__(self):
        self.buckets = {}
        self.tiers = {
            "free": {"capacity": 50_000, "refill_rate": 500, "max_requests_per_min": 10},
            "pro": {"capacity": 500_000, "refill_rate": 5_000, "max_requests_per_min": 60},
            "enterprise": {"capacity": 5_000_000, "refill_rate": 50_000, "max_requests_per_min": 300},
        }

    def _get_bucket(self, user_id, tier="free"):
        if user_id not in self.buckets:
            tier_config = self.tiers.get(tier, self.tiers["free"])
            self.buckets[user_id] = {
                "tokens": tier_config["capacity"],
                "capacity": tier_config["capacity"],
                "refill_rate": tier_config["refill_rate"],
                "last_refill": time.time(),
                "request_timestamps": [],
                "max_rpm": tier_config["max_requests_per_min"],
                "tier": tier,
                "total_tokens_used": 0,
            }
        return self.buckets[user_id]

    def _refill(self, bucket):
        now = time.time()
        elapsed = now - bucket["last_refill"]
        refill = int(elapsed * bucket["refill_rate"])
        if refill > 0:
            bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + refill)
            bucket["last_refill"] = now

    def check(self, user_id, tokens_needed, tier="free"):
        bucket = self._get_bucket(user_id, tier)
        self._refill(bucket)
        now = time.time()
        bucket["request_timestamps"] = [t for t in bucket["request_timestamps"] if now - t < 60]
        if len(bucket["request_timestamps"]) >= bucket["max_rpm"]:
            return {"allowed": False, "reason": "rate_limit", "retry_after_seconds": 60 - (now - bucket["request_timestamps"][0])}
        if bucket["tokens"] < tokens_needed:
            deficit = tokens_needed - bucket["tokens"]
            wait = deficit / bucket["refill_rate"]
            return {"allowed": False, "reason": "token_limit", "tokens_available": bucket["tokens"], "retry_after_seconds": round(wait, 1)}
        return {"allowed": True, "tokens_available": bucket["tokens"]}

    def consume(self, user_id, tokens_used, tier="free"):
        bucket = self._get_bucket(user_id, tier)
        bucket["tokens"] -= tokens_used
        bucket["request_timestamps"].append(time.time())
        bucket["total_tokens_used"] += tokens_used

    def get_usage(self, user_id):
        if user_id not in self.buckets:
            return {"error": "User not found"}
        b = self.buckets[user_id]
        return {
            "user_id": user_id,
            "tier": b["tier"],
            "tokens_remaining": b["tokens"],
            "capacity": b["capacity"],
            "total_tokens_used": b["total_tokens_used"],
            "utilization": round(b["total_tokens_used"] / b["capacity"], 4) if b["capacity"] else 0,
        }
```

### Шаг 5: Трекер затрат

Логируйте каждый вызов и вычисляйте текущие итоги.

```python
class CostTracker:
    def __init__(self, monthly_budget=1000.0):
        self.logs = []
        self.monthly_budget = monthly_budget
        self.alerts = []

    def log_call(self, model, input_tokens, output_tokens, cached_input_tokens=0, latency_ms=0, user_id="anonymous", cache_status="miss"):
        cost = calculate_cost(model, input_tokens, output_tokens, cached_input_tokens)
        entry = {
            "timestamp": time.time(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "latency_ms": latency_ms,
            "cost": cost["total_cost"],
            "user_id": user_id,
            "cache_status": cache_status,
        }
        self.logs.append(entry)
        self._check_budget()
        return entry

    def _check_budget(self):
        total = self.total_cost()
        pct = total / self.monthly_budget if self.monthly_budget > 0 else 0
        if pct >= 0.95 and not any(a["level"] == "stop" for a in self.alerts):
            self.alerts.append({"level": "stop", "message": f"Budget 95% consumed: ${total:.2f}/${self.monthly_budget:.2f}", "timestamp": time.time()})
        elif pct >= 0.85 and not any(a["level"] == "throttle" for a in self.alerts):
            self.alerts.append({"level": "throttle", "message": f"Budget 85% consumed: ${total:.2f}/${self.monthly_budget:.2f}", "timestamp": time.time()})
        elif pct >= 0.70 and not any(a["level"] == "warning" for a in self.alerts):
            self.alerts.append({"level": "warning", "message": f"Budget 70% consumed: ${total:.2f}/${self.monthly_budget:.2f}", "timestamp": time.time()})

    def total_cost(self):
        return round(sum(e["cost"] for e in self.logs), 6)

    def cost_by_model(self):
        by_model = {}
        for e in self.logs:
            m = e["model"]
            if m not in by_model:
                by_model[m] = {"calls": 0, "cost": 0, "input_tokens": 0, "output_tokens": 0}
            by_model[m]["calls"] += 1
            by_model[m]["cost"] = round(by_model[m]["cost"] + e["cost"], 6)
            by_model[m]["input_tokens"] += e["input_tokens"]
            by_model[m]["output_tokens"] += e["output_tokens"]
        return by_model

    def cache_savings(self):
        cache_hits = [e for e in self.logs if e["cache_status"] == "hit"]
        if not cache_hits:
            return {"saved": 0, "cache_hits": 0}
        saved = 0
        for e in cache_hits:
            full_cost = calculate_cost(e["model"], e["input_tokens"], e["output_tokens"])
            saved += full_cost["total_cost"]
        return {"saved": round(saved, 4), "cache_hits": len(cache_hits)}

    def summary(self):
        if not self.logs:
            return {"total_calls": 0, "total_cost": 0}
        total_latency = sum(e["latency_ms"] for e in self.logs)
        cache_hits = sum(1 for e in self.logs if e["cache_status"] == "hit")
        return {
            "total_calls": len(self.logs),
            "total_cost": self.total_cost(),
            "avg_cost_per_call": round(self.total_cost() / len(self.logs), 6),
            "avg_latency_ms": round(total_latency / len(self.logs), 1),
            "cache_hit_rate": round(cache_hits / len(self.logs), 4),
            "cost_by_model": self.cost_by_model(),
            "cache_savings": self.cache_savings(),
            "budget_remaining": round(self.monthly_budget - self.total_cost(), 2),
            "budget_utilization": round(self.total_cost() / self.monthly_budget, 4) if self.monthly_budget > 0 else 0,
            "alerts": self.alerts,
        }
```

### Шаг 6: Маршрутизатор моделей

Направляйте запросы к самой дешевой модели, которая может их обработать.

```python
SIMPLE_KEYWORDS = ["what time", "hours", "address", "phone", "price", "return policy", "hello", "hi", "thanks", "yes", "no"]
COMPLEX_KEYWORDS = ["analyze", "compare", "explain why", "write code", "debug", "architect", "design", "trade-off", "evaluate"]


def classify_complexity(query):
    q = query.lower()
    if len(q.split()) <= 5 or any(kw in q for kw in SIMPLE_KEYWORDS):
        return "simple"
    if any(kw in q for kw in COMPLEX_KEYWORDS):
        return "complex"
    return "medium"


def route_model(query, tier="pro"):
    complexity = classify_complexity(query)
    routing_table = {
        "simple": {"free": "gpt-4.1-nano", "pro": "gpt-4o-mini", "enterprise": "gpt-4o-mini"},
        "medium": {"free": "gpt-4o-mini", "pro": "claude-sonnet-4", "enterprise": "claude-sonnet-4"},
        "complex": {"free": "gpt-4o-mini", "pro": "gpt-4o", "enterprise": "claude-opus-4"},
    }
    model = routing_table[complexity].get(tier, "gpt-4o-mini")
    return {"query": query, "complexity": complexity, "model": model, "tier": tier}
```

### Шаг 7: Запустите демо

```python
def simulate_llm_call(model, query):
    input_tokens = len(query.split()) * 4 + 500
    output_tokens = 150 + (len(query.split()) * 2)
    latency = 200 + (output_tokens * 2)
    return {
        "model": model,
        "response": f"[Simulated {model} response to: {query[:50]}...]",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency,
    }


def run_demo():
    print("=" * 60)
    print("  Caching, Rate Limiting & Cost Optimization Demo")
    print("=" * 60)

    print("\n--- Model Pricing ---")
    for model, pricing in list(MODEL_PRICING.items())[:6]:
        cost_1k = calculate_cost(model, 1000, 500)
        print(f"  {model}: ${cost_1k['total_cost']:.6f} per 1K in + 500 out")

    print("\n--- Cost Comparison: 100K Requests ---")
    for model in ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4", "claude-haiku-3.5"]:
        cost = calculate_cost(model, 1000 * 100_000, 500 * 100_000)
        print(f"  {model}: ${cost['total_cost']:.2f}")

    print("\n--- Anthropic Cache Savings ---")
    no_cache = calculate_cost("claude-sonnet-4", 2000, 500, 0)
    with_cache = calculate_cost("claude-sonnet-4", 2000, 500, 1500)
    saving = no_cache["total_cost"] - with_cache["total_cost"]
    print(f"  Without cache: ${no_cache['total_cost']:.6f}")
    print(f"  With 1500 cached tokens: ${with_cache['total_cost']:.6f}")
    print(f"  Savings per call: ${saving:.6f} ({saving/no_cache['total_cost']*100:.1f}%)")

    exact_cache = ExactCache(max_size=100, ttl_seconds=300)
    semantic_cache = SemanticCache(similarity_threshold=0.75, max_size=100)
    rate_limiter = TokenBucketRateLimiter()
    tracker = CostTracker(monthly_budget=100.0)

    print("\n--- Exact Cache ---")
    messages_1 = [{"role": "user", "content": "What is the return policy?"}]
    result = exact_cache.get("gpt-4o-mini", messages_1, 0.0)
    print(f"  First lookup: {'HIT' if result else 'MISS'}")
    exact_cache.put("gpt-4o-mini", messages_1, 0.0, "You can return items within 30 days.")
    result = exact_cache.get("gpt-4o-mini", messages_1, 0.0)
    print(f"  Second lookup: {'HIT' if result else 'MISS'} -> {result}")
    result = exact_cache.get("gpt-4o-mini", messages_1, 0.7)
    print(f"  With temp=0.7: {'HIT' if result else 'MISS (non-deterministic, skip cache)'}")
    print(f"  Stats: {exact_cache.stats()}")

    print("\n--- Semantic Cache ---")
    test_queries = [
        ("What is the return policy?", "Items can be returned within 30 days with receipt."),
        ("How do I return an item?", None),
        ("What are your store hours?", "We are open 9am-9pm Monday through Saturday."),
        ("When does the store open?", None),
        ("Tell me about quantum computing", "Quantum computers use qubits..."),
        ("Explain quantum mechanics", None),
    ]
    for query, response in test_queries:
        cached = semantic_cache.get(query)
        if cached:
            print(f"  '{query[:40]}' -> CACHE HIT (sim={cached['similarity']}, original='{cached['original_query'][:40]}')")
        elif response:
            semantic_cache.put(query, response)
            print(f"  '{query[:40]}' -> MISS (stored)")
        else:
            print(f"  '{query[:40]}' -> MISS (no match)")
    print(f"  Stats: {semantic_cache.stats()}")

    print("\n--- Rate Limiting ---")
    for i in range(12):
        check = rate_limiter.check("user_1", 1000, "free")
        if check["allowed"]:
            rate_limiter.consume("user_1", 1000, "free")
        status = "OK" if check["allowed"] else f"BLOCKED ({check['reason']})"
        if i < 5 or not check["allowed"]:
            print(f"  Request {i+1}: {status}")
    print(f"  Usage: {rate_limiter.get_usage('user_1')}")

    print("\n--- Model Routing ---")
    routing_queries = [
        "What time do you close?",
        "Summarize this quarterly earnings report",
        "Analyze the trade-offs between microservices and monoliths",
        "Hello",
        "Write code for a binary search tree with deletion",
    ]
    for q in routing_queries:
        route = route_model(q, "pro")
        print(f"  '{q[:50]}' -> {route['model']} ({route['complexity']})")

    print("\n--- Full Pipeline: Before vs After Optimization ---")
    queries = [
        "What is the return policy?",
        "How do I return something?",
        "What are your hours?",
        "When do you open?",
        "Explain the difference between TCP and UDP",
        "Compare TCP vs UDP protocols",
        "Hello",
        "What is your phone number?",
        "Write a Python function to sort a list",
        "Analyze the pros and cons of serverless architecture",
    ]

    print("\n  [Before: no caching, single model (gpt-4o)]")
    tracker_before = CostTracker(monthly_budget=1000.0)
    for q in queries:
        result = simulate_llm_call("gpt-4o", q)
        tracker_before.log_call("gpt-4o", result["input_tokens"], result["output_tokens"], latency_ms=result["latency_ms"], cache_status="miss")
    before = tracker_before.summary()
    print(f"  Total cost: ${before['total_cost']:.6f}")
    print(f"  Avg cost/call: ${before['avg_cost_per_call']:.6f}")
    print(f"  Avg latency: {before['avg_latency_ms']}ms")

    print("\n  [After: caching + routing + rate limiting]")
    exact_c = ExactCache()
    semantic_c = SemanticCache(similarity_threshold=0.75)
    tracker_after = CostTracker(monthly_budget=1000.0)

    for q in queries:
        messages = [{"role": "user", "content": q}]
        cached = exact_c.get("gpt-4o", messages, 0.0)
        if cached:
            tracker_after.log_call("gpt-4o-mini", 0, 0, latency_ms=5, cache_status="hit")
            continue
        sem_cached = semantic_c.get(q)
        if sem_cached:
            tracker_after.log_call("gpt-4o-mini", 0, 0, latency_ms=15, cache_status="hit")
            continue
        route = route_model(q)
        result = simulate_llm_call(route["model"], q)
        tracker_after.log_call(route["model"], result["input_tokens"], result["output_tokens"], latency_ms=result["latency_ms"], cache_status="miss")
        exact_c.put(route["model"], messages, 0.0, result["response"])
        semantic_c.put(q, result["response"])

    after = tracker_after.summary()
    print(f"  Total cost: ${after['total_cost']:.6f}")
    print(f"  Avg cost/call: ${after['avg_cost_per_call']:.6f}")
    print(f"  Avg latency: {after['avg_latency_ms']}ms")
    print(f"  Cache hit rate: {after['cache_hit_rate']:.0%}")

    if before["total_cost"] > 0:
        savings_pct = (1 - after["total_cost"] / before["total_cost"]) * 100
        print(f"\n  SAVINGS: {savings_pct:.1f}% cost reduction")
        print(f"  Latency improvement: {(1 - after['avg_latency_ms'] / before['avg_latency_ms']) * 100:.1f}% faster")

    print("\n--- Budget Alerts Demo ---")
    alert_tracker = CostTracker(monthly_budget=0.01)
    for i in range(5):
        alert_tracker.log_call("gpt-4o", 5000, 2000, latency_ms=500)
    print(f"  Total spent: ${alert_tracker.total_cost():.6f} / ${alert_tracker.monthly_budget}")
    for alert in alert_tracker.alerts:
        print(f"  ALERT [{alert['level'].upper()}]: {alert['message']}")

    print("\n--- Cost Breakdown by Model ---")
    multi_tracker = CostTracker(monthly_budget=500.0)
    for _ in range(50):
        multi_tracker.log_call("gpt-4o-mini", 800, 200, latency_ms=150)
    for _ in range(30):
        multi_tracker.log_call("claude-sonnet-4", 1500, 500, latency_ms=400)
    for _ in range(10):
        multi_tracker.log_call("gpt-4o", 2000, 800, latency_ms=600)
    for _ in range(10):
        multi_tracker.log_call("claude-opus-4", 3000, 1000, latency_ms=1200)
    breakdown = multi_tracker.cost_by_model()
    for model, data in sorted(breakdown.items(), key=lambda x: x[1]["cost"], reverse=True):
        print(f"  {model}: {data['calls']} calls, ${data['cost']:.6f}, {data['input_tokens']:,} in / {data['output_tokens']:,} out")
    print(f"  Total: ${multi_tracker.total_cost():.6f}")

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
```

## Используйте это

### Кэширование промптов Anthropic

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-sonnet-5",
#     max_tokens=1024,
#     system=[
#         {
#             "type": "text",
#             "text": "You are a helpful customer support agent for Acme Corp...",
#             "cache_control": {"type": "ephemeral"},
#         }
#     ],
#     messages=[{"role": "user", "content": "What is the return policy?"}],
# )
#
# print(f"Input tokens: {response.usage.input_tokens}")
# print(f"Cache creation tokens: {response.usage.cache_creation_input_tokens}")
# print(f"Cache read tokens: {response.usage.cache_read_input_tokens}")
```

Первый вызов записывает в кэш (премия 25%). Каждый последующий вызов с тем же префиксом системного промпта читает из кэша (скидка 90%). Кэш живет 5 минут и сбрасывает таймер при каждом попадании.

### Автоматическое кэширование OpenAI

```python
# from openai import OpenAI
#
# client = OpenAI()
#
# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "system", "content": "You are a helpful customer support agent..."},
#         {"role": "user", "content": "What is the return policy?"},
#     ],
# )
#
# print(f"Prompt tokens: {response.usage.prompt_tokens}")
# print(f"Cached tokens: {response.usage.prompt_tokens_details.cached_tokens}")
# print(f"Completion tokens: {response.usage.completion_tokens}")
```

OpenAI кэширует автоматически. Любой префикс промпта длиной от 1 024 токенов, совпадающий с недавним запросом, получает скидку 50%. Изменения в коде не нужны — просто проверьте `prompt_tokens_details.cached_tokens` в ответе, чтобы убедиться, что все работает.

### OpenAI Batch API

```python
# import json
# from openai import OpenAI
#
# client = OpenAI()
#
# requests = []
# for i, query in enumerate(queries):
#     requests.append({
#         "custom_id": f"request-{i}",
#         "method": "POST",
#         "url": "/v1/chat/completions",
#         "body": {
#             "model": "gpt-4o-mini",
#             "messages": [{"role": "user", "content": query}],
#         },
#     })
#
# with open("batch_input.jsonl", "w") as f:
#     for r in requests:
#         f.write(json.dumps(r) + "\n")
#
# batch_file = client.files.create(file=open("batch_input.jsonl", "rb"), purpose="batch")
# batch = client.batches.create(input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h")
# print(f"Batch ID: {batch.id}, Status: {batch.status}")
```

Batch API дает фиксированную скидку 50% на все токены. Результаты поступают в течение 24 часов. Идеально для задач, не требующих реального времени: оценки, разметка данных, массовое суммирование.

### Продуктивный семантический кэш с Redis

```python
# import redis
# import numpy as np
# from openai import OpenAI
#
# r = redis.Redis()
# client = OpenAI()
#
# def get_embedding(text):
#     response = client.embeddings.create(model="text-embedding-3-small", input=text)
#     return response.data[0].embedding
#
# def semantic_cache_lookup(query, threshold=0.95):
#     query_emb = np.array(get_embedding(query))
#     keys = r.keys("cache:emb:*")
#     best_sim, best_key = 0, None
#     for key in keys:
#         stored_emb = np.frombuffer(r.get(key), dtype=np.float32)
#         sim = np.dot(query_emb, stored_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(stored_emb))
#         if sim > best_sim:
#             best_sim, best_key = sim, key
#     if best_sim >= threshold and best_key:
#         response_key = best_key.decode().replace("cache:emb:", "cache:resp:")
#         return r.get(response_key).decode()
#     return None
```

В продакшене замените линейное сканирование векторным индексом (Redis Vector Search, Pinecone или pgvector). Линейное сканирование работает для <1 000 записей. Сверх этого используйте ANN (approximate nearest neighbor) для поиска за O(log n).

## Публикация

Этот урок создает `outputs/prompt-cost-optimizer.md` — переиспользуемый промпт, который анализирует ваше LLM-приложение и рекомендует конкретные оптимизации затрат с прогнозируемой экономией.

Он также создает `outputs/skill-cost-patterns.md` — фреймворк принятия решений для выбора правильной стратегии кэширования, конфигурации ограничения скорости запросов и правил маршрутизации моделей для вашего случая использования.

## Упражнения

1. **Реализуйте вытеснение LRU для семантического кэша.** Замените вытеснение «самый старый первый» на вытеснение «наименее недавно использованный» (least-recently-used). Отслеживайте время последнего доступа для каждой записи и вытесняйте запись с самым старым временем доступа, когда кэш заполнен. Сравните долю попаданий между двумя стратегиями на 100 запросах.

2. **Постройте инструмент прогнозирования затрат.** На основе лога API-вызовов (логи CostTracker) спрогнозируйте месячные затраты на основе скользящего среднего за 7 дней. Учтите закономерности будни/выходные. Запускайте оповещение, если прогнозируемая месячная стоимость превышает бюджет более чем на 20%.

3. **Реализуйте многоуровневое семантическое кэширование.** Используйте два порога сходства: 0.98 для попаданий с высокой уверенностью (возврат немедленно) и 0.90 для попаданий со средней уверенностью (возврат с пометкой: «На основе похожего предыдущего вопроса...»). Отслеживайте, из какого уровня пришло каждое попадание, и измерьте разницу в удовлетворенности пользователей.

4. **Постройте классификатор маршрутизации моделей.** Замените классификатор на основе ключевых слов классификатором на основе эмбеддингов. Встройте 50 размеченных запросов (простой/средний/сложный), затем классифицируйте новые запросы, находя ближайший размеченный пример. Измерьте точность классификации на тестовом наборе из 20 запросов.

5. **Реализуйте автоматический выключатель с уровнями деградации.** При 70% бюджета логируйте предупреждение. При 85% автоматически переключайте всю маршрутизацию на самую дешевую модель (gpt-4o-mini). При 95% обслуживайте только закэшированные ответы и отклоняйте новые запросы. Протестируйте, симулируя 1 000 запросов против бюджета в $1.00, и убедитесь, что каждый порог срабатывает корректно.

## Ключевые термины

| Термин | Что говорят люди | Что это на самом деле означает |
|------|----------------|----------------------|
| Кэширование промптов | «Кэшировать системный промпт» | Кэширование на уровне провайдера, когда повторяющиеся префиксы промпта получают скидку (90% у Anthropic, 50% у OpenAI) — без изменений кода у OpenAI, с явными маркерами у Anthropic |
| Семантическое кэширование | «Умное кэширование» | Встраивание запроса, вычисление сходства с прошлыми запросами и возврат закэшированного ответа, если сходство превышает порог — ловит перефразировки, которые пропускает точное совпадение |
| Точное кэширование | «Хэш-кэширование» | Хэширование полного промпта (модель + сообщения + температура) и возврат закэшированного ответа для идентичных входов — работает только для детерминированных вызовов с temperature=0 |
| Token bucket | «Ограничитель скорости» | Алгоритм, в котором у каждого пользователя есть корзина из N токенов, пополняемая со скоростью R в секунду — позволяет всплески до N, при этом обеспечивая среднюю скорость R |
| Маршрутизация моделей | «Скупая маршрутизация» | Использование классификатора для отправки простых запросов к дешевым моделям (GPT-4o-mini, Haiku) и сложных запросов к дорогим моделям (GPT-4o, Opus) — экономит 40-70% на стоимости моделей |
| Отслеживание затрат | «Метеринг» | Логирование каждого API-вызова с указанием модели, токенов, задержки, стоимости и ID пользователя, чтобы точно знать, куда уходят деньги и какие функции дороги |
| Автоматический выключатель | «Аварийный выключатель» | Автоматическая деградация сервиса (более дешевые модели, только кэш) или полная остановка запросов при приближении расходов к лимиту бюджета |
| Batch API | «Оптовая скидка» | Асинхронная обработка от OpenAI со скидкой 50% — отправьте до 50 000 запросов, получите результаты в течение 24 часов |
| Сжатие промптов | «Токен-диета» | Переписывание системных промптов и контекста для использования меньшего числа токенов при сохранении смысла — более короткие промпты стоят дешевле и часто работают лучше |
| Доля попаданий в кэш | «Эффективность кэша» | Процент запросов, обслуженных из кэша вместо вызова LLM — 40-60% типично для продакшн-чат-ботов, экономит пропорционально на стоимости |

## Дополнительное чтение

- [Руководство по кэшированию промптов Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — официальная документация по явным маркерам cache_control Anthropic, ценам и поведению времени жизни кэша
- [Кэширование промптов OpenAI](https://platform.openai.com/docs/guides/prompt-caching) — автоматическое кэширование OpenAI, как проверить попадания в кэш через поля usage, и минимальная длина префикса
- [Batch API OpenAI](https://platform.openai.com/docs/guides/batch) — скидка 50% на асинхронную обработку, формат JSONL, окно выполнения 24 часа и лимит 50 тыс. запросов
- [GPTCache](https://github.com/zilliztech/GPTCache) — библиотека семантического кэширования с открытым исходным кодом, поддерживающая несколько бэкендов эмбеддингов, векторных хранилищ и политик вытеснения
- [Martian Model Router](https://docs.withmartian.com) — продуктивная маршрутизация моделей, автоматически выбирающая самую дешевую модель, способную обработать каждый запрос
- [Not Diamond](https://www.notdiamond.ai) — маршрутизатор моделей на основе машинного обучения, который учится на ваших паттернах трафика для оптимизации компромиссов между стоимостью и качеством у разных провайдеров
- [Helicone](https://www.helicone.ai) — платформа наблюдаемости LLM с отслеживанием затрат, кэшированием, ограничением скорости запросов и бюджетными оповещениями в качестве прокси-слоя
- [Dean & Barroso, "The Tail at Scale" (CACM 2013)](https://research.google/pubs/the-tail-at-scale/) — задержка, пропускная способность, перцентили TTFT/TPOT и hedged-запросы; модель затрат, лежащая в основе принципа «выбирайте самую дешевую модель, которая все еще укладывается в P95»
- [Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023)](https://arxiv.org/abs/2309.06180) — статья о vLLM; почему страничный KV-кэш и непрерывная пакетная обработка превосходят наивные серверы в 24 раза по пропускной способности — инфраструктурный слой под «кэшированием и затратами»
- [Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (ICLR 2024)](https://arxiv.org/abs/2307.08691) — снижение затрат на уровне ядер, ортогональное