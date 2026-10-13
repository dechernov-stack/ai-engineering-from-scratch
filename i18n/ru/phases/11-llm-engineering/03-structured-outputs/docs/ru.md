# Структурированный вывод: JSON, валидация схем, ограниченное декодирование

> Ваша LLM возвращает строку. Вашему приложению нужен JSON. Этот разрыв обрушил больше продакшн-систем, чем любые галлюцинации модели. Структурированный вывод — это мост между естественным языком и типизированными данными. Сделайте это правильно — и ваша LLM станет надёжным API. Сделайте неправильно — и вы будете парсить свободный текст регулярками в 3 часа ночи.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lessons 01-05 (LLMs from Scratch)
**Time:** ~90 минут
**Related:** Phase 5 · 20 (Structured Outputs & Constrained Decoding) рассматривает теорию на уровне декодера (FSM/CFG logit processors, Outlines, XGrammar). Этот урок фокусируется на прикладном SDK-уровне (OpenAI `response_format`, Anthropic tool use, Instructor) — сначала прочитайте Phase 5 · 20, если хотите понять, что происходит под капотом API.

## Цели обучения

- Реализовать JSON-режим и вывод с ограничением по схеме с использованием параметров API OpenAI и Anthropic
- Построить слой валидации на Pydantic, который отклоняет некорректный вывод LLM и повторяет запрос с обратной связью об ошибке
- Объяснить, как ограниченное декодирование принудительно формирует валидный JSON на уровне токенов без постобработки
- Спроектировать надёжные промпты для извлечения данных, которые стабильно преобразуют неструктурированный текст в типизированные структуры данных

## Проблема

Вы спрашиваете LLM: "Извлеки название продукта, цену и доступность из этого текста." Она отвечает:

```
The product is the Sony WH-1000XM5 headphones, which cost $348.00 and are currently in stock.
```

Это совершенно верный ответ. Он также совершенно бесполезен для вашего приложения. Вашей системе учёта запасов нужен `{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}`. Вам нужен объект JSON с конкретными ключами, конкретными типами и конкретными ограничениями на значения. Вам не нужно предложение.

Наивное решение: добавить в промпт "Отвечай в формате JSON". Это работает в 90% случаев. В оставшихся 10% модель оборачивает JSON в markdown-блоки кода, или добавляет преамбулу вроде "Вот JSON:", или выдаёт синтаксически некорректный JSON, потому что закрыла скобку слишком рано. Ваш JSON-парсер падает. Ваш пайплайн ломается. Вы добавляете try/except и цикл повтора. Повтор иногда выдаёт другие данные. Теперь у вас проблема с согласованностью поверх проблемы с парсингом.

Это не проблема prompt-инжиниринга. Это проблема декодирования. Модель генерирует токены слева направо. На каждой позиции она выбирает наиболее вероятный следующий токен из словаря размером 100K+ вариантов. Большинство этих вариантов дали бы невалидный JSON в данной позиции. Если модель только что выдала `{"price":`, следующий токен должен быть цифрой, кавычкой (для строки), `null`, `true`, `false` или знаком минус. Всё остальное даёт невалидный JSON. Без ограничений модель может выбрать совершенно разумное английское слово, которое катастрофически неверно синтаксически.

## Концепция

### Спектр структурированного вывода

Существует четыре уровня контроля над структурированным выводом, каждый надёжнее предыдущего.

```mermaid
graph LR
    subgraph Spectrum["Structured Output Spectrum"]
        direction LR
        A["Prompt-based\n'Return JSON'\n~90% valid"] --> B["JSON Mode\nGuaranteed valid JSON\nNo schema guarantee"]
        B --> C["Schema Mode\nJSON + matches schema\nGuaranteed compliance"]
        C --> D["Constrained Decoding\nToken-level enforcement\n100% compliance"]
    end

    style A fill:#1a1a2e,stroke:#ff6b6b,color:#fff
    style B fill:#1a1a2e,stroke:#ffa500,color:#fff
    style C fill:#1a1a2e,stroke:#51cf66,color:#fff
    style D fill:#1a1a2e,stroke:#0f3460,color:#fff
```

**На основе промпта** ("Отвечай в валидном JSON"): нет принуждения. Модель обычно соблюдает требование, но не всегда. Надёжность: ~90%. Режим отказа: markdown-блоки, текст преамбулы, обрезанный вывод, неверная структура.

**JSON-режим**: API гарантирует, что вывод — валидный JSON. `response_format: { type: "json_object" }` от OpenAI включает это. Вывод распарсится без ошибок. Но он может не соответствовать ожидаемой схеме — лишние ключи, неверные типы, отсутствующие поля.

**Режим схемы**: API принимает JSON Schema и гарантирует, что вывод ей соответствует. В 2026 году каждый крупный провайдер поддерживает это нативно: `response_format: { type: "json_schema", json_schema: {...} }` от OpenAI (а также как `tool_choice="required"`), tool use от Anthropic с `input_schema`, и `response_schema` + `response_mime_type: "application/json"` от Gemini. Вывод имеет точно те ключи, типы и ограничения, которые вы указали.

**Ограниченное декодирование**: на каждой позиции токена во время генерации декодер маскирует все токены, которые дали бы невалидный вывод. Если схема требует число, а модель собирается выдать букву, вероятность этого токена устанавливается в ноль. Модель может производить только токены, ведущие к валидному выводу. Именно это реализуют режим структурированного вывода OpenAI и библиотеки вроде Outlines и Guidance под капотом.

### JSON Schema: язык контракта

JSON Schema — это способ сообщить модели (или слою валидации), какую форму должен иметь вывод. Каждая крупная система структурированного вывода использует её.

```json
{
  "type": "object",
  "properties": {
    "product": { "type": "string" },
    "price": { "type": "number", "minimum": 0 },
    "in_stock": { "type": "boolean" },
    "categories": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["product", "price", "in_stock"]
}
```

Эта схема говорит: вывод должен быть объектом со строковым полем `product`, неотрицательным числом `price`, булевым значением `in_stock` и необязательным массивом строк `categories`. Любой вывод, не соответствующий этому, отклоняется.

Схемы обрабатывают сложные случаи: вложенные объекты, массивы с типизированными элементами, перечисления (ограничение строки конкретными значениями), сопоставление по шаблону (регулярные выражения для строк) и комбинаторы (oneOf, anyOf, allOf для полиморфного вывода).

### Паттерн Pydantic

В Python вы не пишете JSON Schema вручную. Вы определяете модель Pydantic, и она генерирует схему за вас.

```python
from pydantic import BaseModel

class Product(BaseModel):
    product: str
    price: float
    in_stock: bool
    categories: list[str] = []
```

Это создаёт ту же JSON Schema, что и выше. Библиотека Instructor (и SDK OpenAI) принимают модели Pydantic напрямую: передайте класс модели, получите обратно валидированный экземпляр. Если вывод LLM не соответствует, Instructor автоматически повторяет запрос.

### Function Calling / Tool Use

Альтернативный интерфейс для той же проблемы. Вместо того чтобы просить модель напрямую производить JSON, вы определяете "инструменты" (функции) с типизированными параметрами. Модель выдаёт вызов функции со структурированными аргументами. OpenAI называет это "function calling". Anthropic называет это "tool use". Результат тот же: структурированные данные.

```mermaid
graph TD
    subgraph ToolUse["Tool Use Flow"]
        U["User: Extract product info\nfrom this review text"] --> M["Model processes input"]
        M --> TC["Tool Call:\nextract_product(\n  product='Sony WH-1000XM5',\n  price=348.00,\n  in_stock=true\n)"]
        TC --> V["Validate against\nfunction schema"]
        V --> R["Structured Result:\n{product, price, in_stock}"]
    end

    style U fill:#1a1a2e,stroke:#0f3460,color:#fff
    style TC fill:#1a1a2e,stroke:#e94560,color:#fff
    style V fill:#1a1a2e,stroke:#ffa500,color:#fff
    style R fill:#1a1a2e,stroke:#51cf66,color:#fff
```

Tool use предпочтителен, когда модели нужно выбрать, какую функцию вызвать, а не просто заполнить параметры. Если у вас 10 разных схем извлечения, и модель должна выбрать правильную на основе входных данных, tool use даёт вам одновременно выбор схемы и структурированный вывод.

### Распространённые режимы отказа

Даже при принудительном соблюдении схемы структурированный вывод может ломаться тонкими способами.

**Галлюцинированные значения**: вывод соответствует схеме, но содержит выдуманные данные. Модель выдаёт `{"price": 299.99}`, когда в тексте указано $348. Валидация схемы не может это поймать — тип верный, значение неверное.

**Путаница с перечислениями**: вы ограничиваете поле значениями `["in_stock", "out_of_stock", "preorder"]`. Модель выдаёт `"available"` — семантически верно, но не входит в допустимый набор. Хорошее ограниченное декодирование предотвращает это. Подходы на основе промпта — нет.

**Глубина вложенных объектов**: глубоко вложенные схемы (4+ уровня) порождают больше ошибок. Каждый уровень вложенности — ещё одно место, где модель может потерять отслеживание структуры.

**Длина массива**: модель может выдать слишком много или слишком мало элементов в массиве. Схемы поддерживают `minItems` и `maxItems`, но не все провайдеры применяют их на уровне декодирования.

**Пропуск необязательных полей**: модель опускает поля, которые технически необязательны, но семантически важны для вашего варианта использования. Делайте их обязательными в схеме, даже если данные иногда отсутствуют — заставляйте модель явно выдавать `null`.

```figure
mx-schema-funnel
```

## Создаём это

### Шаг 1: Валидатор JSON Schema

Постройте с нуля валидатор, который проверяет, соответствует ли объект Python схеме JSON Schema. Это то, что выполняется на стороне вывода для проверки соответствия.

```python
import json

def validate_schema(data, schema):
    errors = []
    _validate(data, schema, "", errors)
    return errors

def _validate(data, schema, path, errors):
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            errors.append(f"{path}: expected object, got {type(data).__name__}")
            return
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}.{key}: required field missing")
        properties = schema.get("properties", {})
        for key, value in data.items():
            if key in properties:
                _validate(value, properties[key], f"{path}.{key}", errors)

    elif schema_type == "array":
        if not isinstance(data, list):
            errors.append(f"{path}: expected array, got {type(data).__name__}")
            return
        min_items = schema.get("minItems", 0)
        max_items = schema.get("maxItems", float("inf"))
        if len(data) < min_items:
            errors.append(f"{path}: array has {len(data)} items, minimum is {min_items}")
        if len(data) > max_items:
            errors.append(f"{path}: array has {len(data)} items, maximum is {max_items}")
        items_schema = schema.get("items", {})
        for i, item in enumerate(data):
            _validate(item, items_schema, f"{path}[{i}]", errors)

    elif schema_type == "string":
        if not isinstance(data, str):
            errors.append(f"{path}: expected string, got {type(data).__name__}")
            return
        enum_values = schema.get("enum")
        if enum_values and data not in enum_values:
            errors.append(f"{path}: '{data}' not in allowed values {enum_values}")

    elif schema_type == "number":
        if not isinstance(data, (int, float)):
            errors.append(f"{path}: expected number, got {type(data).__name__}")
            return
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and data < minimum:
            errors.append(f"{path}: {data} is less than minimum {minimum}")
        if maximum is not None and data > maximum:
            errors.append(f"{path}: {data} is greater than maximum {maximum}")

    elif schema_type == "boolean":
        if not isinstance(data, bool):
            errors.append(f"{path}: expected boolean, got {type(data).__name__}")

    elif schema_type == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            errors.append(f"{path}: expected integer, got {type(data).__name__}")
```

### Шаг 2: Модель в стиле Pydantic в схему

Постройте минимальный конвертер класса в схему. Определите класс Python и автоматически сгенерируйте его JSON Schema.

```python
class SchemaField:
    def __init__(self, field_type, required=True, default=None, enum=None, minimum=None, maximum=None):
        self.field_type = field_type
        self.required = required
        self.default = default
        self.enum = enum
        self.minimum = minimum
        self.maximum = maximum

def python_type_to_schema(field):
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    schema = {}

    if field.field_type in type_map:
        schema["type"] = type_map[field.field_type]
    elif field.field_type == list:
        schema["type"] = "array"
        schema["items"] = {"type": "string"}
    elif isinstance(field.field_type, dict):
        schema = field.field_type

    if field.enum:
        schema["enum"] = field.enum
    if field.minimum is not None:
        schema["minimum"] = field.minimum
    if field.maximum is not None:
        schema["maximum"] = field.maximum

    return schema

def model_to_schema(name, fields):
    properties = {}
    required = []

    for field_name, field in fields.items():
        properties[field_name] = python_type_to_schema(field)
        if field.required:
            required.append(field_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }
```

### Шаг 3: Фильтр ограниченных токенов

Смоделируйте ограниченное декодирование. По частичной JSON-строке и схеме определите, какие категории токенов допустимы в текущей позиции.

```python
def next_valid_tokens(partial_json, schema):
    stripped = partial_json.strip()

    if not stripped:
        return ["{"]

    try:
        json.loads(stripped)
        return ["<EOS>"]
    except json.JSONDecodeError:
        pass

    last_char = stripped[-1] if stripped else ""

    if last_char == "{":
        return ['"', "}"]
    elif last_char == '"':
        if stripped.endswith('":'):
            return ['"', "0-9", "true", "false", "null", "[", "{"]
        return ["a-z", '"']
    elif last_char == ":":
        return [" ", '"', "0-9", "true", "false", "null", "[", "{"]
    elif last_char == ",":
        return [" ", '"', "{", "["]
    elif last_char in "0123456789":
        return ["0-9", ".", ",", "}", "]"]
    elif last_char == "}":
        return [",", "}", "]", "<EOS>"]
    elif last_char == "]":
        return [",", "}", "<EOS>"]
    elif last_char == "[":
        return ['"', "0-9", "true", "false", "null", "{", "[", "]"]
    else:
        return ["any"]

def demonstrate_constrained_decoding():
    partial_states = [
        '',
        '{',
        '{"product"',
        '{"product":',
        '{"product": "Sony"',
        '{"product": "Sony",',
        '{"product": "Sony", "price":',
        '{"product": "Sony", "price": 348',
        '{"product": "Sony", "price": 348}',
    ]

    print(f"{'Partial JSON':<45} {'Valid Next Tokens'}")
    print("-" * 80)
    for state in partial_states:
        valid = next_valid_tokens(state, {})
        display = state if state else "(empty)"
        print(f"{display:<45} {valid}")
```

### Шаг 4: Пайплайн извлечения

Объедините всё в пайплайн извлечения: определите схему, смоделируйте LLM, производящую структурированный вывод, провалидируйте вывод и обработайте повторы.

```python
def simulate_llm_extraction(text, schema, attempt=0):
    if "headphones" in text.lower() or "sony" in text.lower():
        if attempt == 0:
            return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true, "categories": ["audio", "headphones"]}'
        return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}'

    if "laptop" in text.lower():
        return '{"product": "MacBook Pro 16", "price": 2499.00, "in_stock": false, "categories": ["computers"]}'

    return '{"product": "Unknown", "price": 0, "in_stock": false}'

def extract_with_retry(text, schema, max_retries=3):
    for attempt in range(max_retries):
        raw = simulate_llm_extraction(text, schema, attempt)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt + 1}: JSON parse error -- {e}")
            continue

        errors = validate_schema(data, schema)
        if not errors:
            return data

        print(f"  Attempt {attempt + 1}: Schema validation errors -- {errors}")

    return None

product_schema = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "price": {"type": "number", "minimum": 0},
        "in_stock": {"type": "boolean"},
        "categories": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["product", "price", "in_stock"],
}
```

### Шаг 5: Запустите весь пайплайн

```python
def run_demo():
    print("=" * 60)
    print("  Structured Output Pipeline Demo")
    print("=" * 60)

    print("\n--- Schema Definition ---")
    product_fields = {
        "product": SchemaField(str),
        "price": SchemaField(float, minimum=0),
        "in_stock": SchemaField(bool),
        "categories": SchemaField(list, required=False),
    }
    generated_schema = model_to_schema("Product", product_fields)
    print(json.dumps(generated_schema, indent=2))

    print("\n--- Schema Validation ---")
    test_cases = [
        ({"product": "Test", "price": 10.0, "in_stock": True}, "Valid object"),
        ({"product": "Test", "price": -5.0, "in_stock": True}, "Negative price"),
        ({"product": "Test", "in_stock": True}, "Missing price"),
        ({"product": "Test", "price": "ten", "in_stock": True}, "String as price"),
        ("not an object", "String instead of object"),
    ]

    for data, label in test_cases:
        errors = validate_schema(data, product_schema)
        status = "PASS" if not errors else f"FAIL: {errors}"
        print(f"  {label}: {status}")

    print("\n--- Constrained Decoding Simulation ---")
    demonstrate_constrained_decoding()

    print("\n--- Extraction Pipeline ---")
    texts = [
        "The Sony WH-1000XM5 headphones are priced at $348 and currently available.",
        "The new MacBook Pro 16-inch laptop costs $2499 but is sold out.",
        "This is a random sentence with no product info.",
    ]

    for text in texts:
        print(f"\n  Input: {text[:60]}...")
        result = extract_with_retry(text, product_schema)
        if result:
            print(f"  Output: {json.dumps(result)}")
        else:
            print(f"  Output: FAILED after retries")
```

## Используем это

### Structured Outputs OpenAI

```python
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = OpenAI()
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# response = client.beta.chat.completions.parse(
#     model="gpt-5-mini",
#     messages=[
#         {"role": "system", "content": "Extract product information."},
#         {"role": "user", "content": "Sony WH-1000XM5, $348, in stock"},
#     ],
#     response_format=Product,
# )
#
# product = response.choices[0].message.parsed
# print(product.product, product.price, product.in_stock)
```

Режим структурированного вывода OpenAI внутренне использует ограниченное декодирование. Каждый токен, который генерирует модель, гарантированно даёт вывод, соответствующий схеме Pydantic. Повторы не нужны. Валидация не нужна. Ограничение встроено в сам процесс декодирования.

### Tool Use Anthropic

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-opus-4-7",
#     max_tokens=1024,
#     tools=[{
#         "name": "extract_product",
#         "description": "Extract product information from text",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "product": {"type": "string"},
#                 "price": {"type": "number"},
#                 "in_stock": {"type": "boolean"},
#             },
#             "required": ["product", "price", "in_stock"],
#         },
#     }],
#     messages=[{"role": "user", "content": "Extract: Sony WH-1000XM5, $348, in stock"}],
# )
```

Anthropic достигает структурированного вывода через tool use. Модель выдаёт вызов инструмента со структурированными аргументами, соответствующими input_schema. Тот же результат, другая поверхность API.

### Библиотека Instructor

```python
# pip install instructor
# import instructor
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = instructor.from_openai(OpenAI())
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# product = client.chat.completions.create(
#     model="gpt-5-mini",
#     response_model=Product,
#     messages=[{"role": "user", "content": "Sony WH-1000XM5, $348, in stock"}],
# )
```

Instructor оборачивает любой клиент LLM и добавляет автоматические повторы с валидацией. Если первая попытка не проходит валидацию, он отправляет ошибки обратно модели как контекст и просит её исправить вывод. Это работает с любым провайдером, а не только с OpenAI.

## Публикуем это

Этот урок производит `outputs/prompt-structured-extractor.md` — многоразовый шаблон промпта, который извлекает структурированные данные из любого текста по заданному определению схемы. Подайте ему JSON Schema и неструктурированный текст, и он вернёт валидированный JSON.

Он также производит `outputs/skill-structured-outputs.md` — фреймворк для принятия решений по выбору правильной стратегии структурированного вывода в зависимости от вашего провайдера, требований к надёжности и сложности схемы.

## Упражнения

1. Расширьте валидатор схем для поддержки `oneOf` (данные должны соответствовать ровно одной из нескольких схем). Это обрабатывает полиморфный вывод — например, поле, которое может быть либо объектом `Product`, либо объектом `Service` с разными формами.

2. Постройте инструмент "diff схем", который сравнивает две схемы и определяет критичные изменения (удалённые обязательные поля, изменённые типы) в сравнении с некритичными изменениями (добавленные необязательные поля, ослабленные ограничения). Это необходимо для версионирования ваших схем извлечения в продакшне.

3. Реализуйте более реалистичный симулятор ограниченного декодирования. По JSON Schema и словарю из 100 токенов (буквы, цифры, пунктуация, ключевые слова) пройдите через генерацию шаг за шагом, маскируя недопустимые токены на каждой позиции. Измерьте, какой процент словаря допустим на каждом шаге.

4. Постройте набор оценки для извлечения. Создайте 50 описаний продуктов с вручную размеченными JSON-выводами. Запустите ваш пайплайн извлечения на всех 50 и измерьте точное совпадение, точность на уровне полей и соответствие типов. Определите, какие поля сложнее всего извлечь корректно.

5. Добавьте "оценки уверенности" в ваш пайплайн извлечения. Для каждого извлечённого поля оцените, насколько уверена модель (на основе вероятностей токенов или путём запуска извлечения 3 раза и измерения согласованности). Помечайте поля с низкой уверенностью для проверки человеком.

## Ключевые термины

| Термин | Что говорят люди | Что это на самом деле означает |
|------|----------------|----------------------|
| JSON mode | "Возвращает JSON" | Флаг API, гарантирующий синтаксически валидный вывод JSON, но не обеспечивающий соответствие какой-либо конкретной схеме |
| Structured output | "Типизированный JSON" | Вывод, соответствующий конкретной JSON Schema с правильными ключами, типами и ограничениями |
| Constrained decoding | "Направляемая генерация" | На каждой позиции токена маскируются токены, которые дали бы невалидный вывод — гарантирует 100%-ное соответствие схеме |
| JSON Schema | "Шаблон JSON" | Декларативный язык для описания структуры, типов и ограничений данных JSON (используется OpenAPI, JSON Forms и др.) |
| Pydantic | "Python dataclasses+" | Библиотека Python, определяющая модели данных с валидацией типов, используется FastAPI и Instructor для генерации JSON Schema |
| Function calling | "Tool use" | LLM выдаёт структурированный вызов функции (имя + типизированные аргументы) вместо свободного текста — это поддерживают и OpenAI, и Anthropic |
| Instructor | "Pydantic для LLM" | Библиотека Python, оборачивающая клиенты LLM для возврата валидированных экземпляров Pydantic, с автоматическим повтором при неудачной валидации |
| Token masking | "Фильтрация словаря" | Установка вероятностей конкретных токенов в ноль во время генерации, чтобы модель не могла их производить |
| Schema compliance | "Соответствует форме" | Вывод содержит все обязательные поля, правильные типы, значения в пределах ограничений и не содержит лишних недопустимых полей |
| Retry loop | "Попробовать снова, пока не сработает" | Отправка ошибок валидации обратно модели с просьбой исправить вывод — Instructor делает это автоматически, до настраиваемого максимума |

## Дополнительное чтение

- [Руководство по Structured Outputs OpenAI](https://platform.openai.com/docs/guides/structured-outputs) — официальная документация по ограниченному декодированию на основе JSON Schema в API OpenAI
- [Willard & Louf, 2023 — "Efficient Guided Generation for Large Language Models"](https://arxiv.org/abs/2307.09702) — статья Outlines, описывающая, как компилировать JSON Schema в конечные автоматы для ограничений на уровне токенов
- [Документация Instructor](https://python.useinstructor.com/) — стандартная библиотека для получения структурированного вывода от любой LLM с валидацией Pydantic и повторами
- [Руководство по Tool Use Anthropic](https://docs.anthropic.com/en/docs/tool-use) — как Claude реализует структурированный вывод через tool use с входной схемой JSON Schema input_schema
- [Спецификация JSON Schema](https://json-schema.org/) — полная спецификация языка схем, используемого каждой крупной системой структурированного вывода
- [Библиотека Outlines](https://github.com/outlines-dev/outlines) — генерация с открытым исходным кодом с ограничениями, использующая регулярные выражения и JSON Schema, скомпилированные в конечные автоматы
- [Dong et al., "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models" (MLSys 2025)](https://arxiv.org/abs/2411.15100) — современный передовой грамматический движок; компиляция в автомат с магазинной памятью, маскирующая токены за ~100 нс/токен.
- [Beurer-Kellner et al., "Prompting Is Programming: A Query Language for Large Language Models" (LMQL)](https://arxiv.org/abs/2212.06094) — статья LMQL, представляющая ограниченное декодирование как язык запросов с ограничениями на типы и значения.
- [Microsoft Guidance (документация фреймворка)](https://github.com/guidance-ai/guidance) — генерация с ограничениями на основе шаблонов; независимое от поставщика дополнение к Outlines и XGrammar.