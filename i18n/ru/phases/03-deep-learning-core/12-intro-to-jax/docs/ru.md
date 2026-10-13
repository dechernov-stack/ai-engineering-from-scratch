# Введение в JAX

> PyTorch мутирует тенсоры, TensorFlow создает графики, JAX сочиняет чистые функции, а последний изменяет то, как мы думаем о глубоком обучении.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 Lessons 01-10, basic NumPy
**Time:** ~90 minutes

## Цели обучения

- Напишите чистый функциональный нейронный сетевой код с использованием функционального API JAX (jax.numpy, jax.grad, jax.jit, jax.vmap)
- Объясните ключевую разницу в дизайне между стремленной мутацией PyTorch и функциональной моделью компиляции JAX
- Применить компиляцию jit и векторизацию vmap для ускорения тренировочных петлей по сравнению с наивным Python
- Обучить простую сеть в JAX и противопоставить явное управление состоянием с объективно-ориентированным подходом PyTorch

## Проблема

Вы знаете, как создавать нейронные сети в PyTorch.`nn.Module`, звоните`.backward()`Это работает, миллионы людей используют его.

Но у PyTorch есть ограничение, которое входит в его ДНК: он с нетерпением отслеживает операции, одна за другой, в Python.`tensor + tensor`Каждый шаг обучения пересматривает тот же код Python. Это работает хорошо, пока вам не нужно обучить модель с параметрами 540 миллиардов через 2048 ТПУ.

Google DeepMind обучает Gemini на JAX. Anthropic обучил Клода на JAX. Это не маленькие операции - это крупнейшие тренировочные операции в нейронной сети на Земле. Они выбрали JAX, потому что он рассматривает тренировочный цикл как программу, которую можно будет компилировать, а не последовательность звонков Python.

JAX - это NumPy с тремя суперсилами: автоматическая дифференциация, компиляция JIT на XLA и автоматическая векторизация. Вы пишете функцию, которая обрабатывает один пример. JAX дает вам функцию, которая обрабатывает партию, вычисляет градиенты, компилирует на машинный код и работает на нескольких устройствах. Все это без изменения первоначальной функции.

## Концепция

### Философия JAX

JAX - это функциональная система, без классов, без изменяющегося состояния, без`.backward()`Вместо этого:

| PyTorch | JAX |
|---------|-----|
| `nn.Module` class with state | Pure function: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager execution | JIT compilation via XLA |
| `for x in batch:` manual loop | `jax.vmap(f)` auto-vectorization |
| `DataParallel` / `FSDP` | `jax.pmap(f)` auto-parallelism |
| Mutable `model.parameters()` | Immutable pytree of arrays |

Это не стиль предпочтения. Это ограничение компилятора. Сборка JIT требует чистых функций - одни и те же вводы всегда производят те же выводы, без побочных эффектов. Это ограничение делает возможными 100x скорости.

### Jax.numpy: Знакомая поверхность

JAX вновь внедряет API NumPy на ускорителях:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

Те же названия функций, те же правила трансляции, те же семантики на срезках, но массивы работают на GPU/TPU, и каждая операция отслеживается компилятором.

Одно важное отличие: JAX массивы неизменны.`a[0] = 5`Вместо этого:`a = a.at[0].set(5)`Это кажется неловким на неделю, а потом нажимается - неизменность - это то, что делает преобразования такими, как`grad`- Я не знаю .`jit`, и `vmap`- Совместимость.

### jax.grad: Функциональный автодифф

PyTorch прикрепляет градиенты к тенсорам (`.grad`JAX прикрепляет градиенты к функциям.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad`принимает функцию и возвращает новую функцию, которая вычисляет градиент.`.backward()`Нет вычислительной графики, хранящейся на тензорах. Грейдент - это просто другая функция, которую можно вызвать, составить или составить JIT.

Это произвольно составляет:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

Второй производный, третий производный, якобийский, гессианский, все это сочиняется`grad`PyTorch может сделать это тоже (`torch.autograd.functional.hessian`В JAX это основа.

Ограничение: `grad`Не используется только на чистых функциях. Нет печатных заявлений внутри (они выполняются во время отслеживания, а не выполнения). Нет мутации внешнего состояния. Нет генерации случайных чисел без четкого управления ключами.

### jit: Компиляция на XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

На первом вызове JAX отслеживает функцию - записывает, какие операции происходят, не выполняя их. Затем передает этот след в XLA (Accelerated Linear Algebra), компилятор Google для TPU и GPU. XLA объединяет операции, устраняет избыточные копии памяти и генерирует оптимизированный машинный код.

Последующие звонки полностью перестают использовать Python.

Когда JIT помогает:
- Шаги обучения (одно и то же вычисление повторяется тысячи раз)
- Инференция (одинаковая модель, разные входы)
- Любая функция, называемая более одного раза с вводами с аналогичной формой

Когда JIT болит:
- Функции с потоком управления Python, которые зависят от значений (`if x > 0`где x - отслеживаемая массива)
- Вычисления в однократном режиме (общие затраты на составление превышают время выполнения)
- Дебгунирование (отслеживание скрывает фактическое выполнение)

Ограничение потока контроля реально. `jax.lax.cond`Заменяет`if/else`- Я не знаю .`jax.lax.scan`Заменяет`for`Это не выборная цена - это цена на сборку.

### vmap: Автоматическая векторизация

Вы пишете функцию , которая обрабатывает один пример:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap`поднимает его для обработки партии:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)`Средства: не перегружайте `params`(распределенный), партия по оси 0 `x`- Нет руководства .`for`Якс вычисляет размер партии и векторизирует все вычисления.

Это не синтаксический сахар.`vmap`генерирует слияние векторизированного кода, который работает в 10-100 раз быстрее, чем петля Python.`jit`и `grad`В следующем случае:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

Это почти невозможно в PyTorch без взломов.

### pmap: Паралелизм данных между устройствами

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap`репликирует функцию на всех доступных устройствах (GPU/TPU) и делится партией.`jax.lax.pmean`и `jax.lax.psum`синхронизировать градиенты между устройствами.

Google обучает Gemini через тысячи чипов TPU v5e с помощью `pmap`(и его преемник)`shard_map`). Модель программирования: напишите версию для одного устройства, завершите с `pmap`- Я закончил.

### Питр: Всеобщая структура данных

JAX работает на "питреях" - внушенных комбинациях списков, туплей, диктов и массивов.

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

Каждое преобразование JAX ...`grad`- Я не знаю .`jit`- Я не знаю .`vmap`- знает, как пересечь питрь.`jax.tree.map(f, tree)`применяется `f`Вот как оптимизаторы обновляют все параметры сразу:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

Нет , нет .`.parameters()`Нет регистрации параметров, структура дерева - это модель.

### Функциональный против объектно-ориентированный

В магазинах PyTorch указано внутри объектов:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX использует чистые функции с ясным состоянием:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

Параметры передаются. Ничто не хранится. Ничто не мутируется. Это делает каждую функцию проверяемой, совместимой и компиляемой. Это также означает, что вы сами управляете параметрами - или используете библиотеку, такую как Флакс или Эквинокс.

### Экосистема JAX

JAX дает примитивные, библиотеки дают эргономику:

| Library | Role | Style |
|---------|------|-------|
| **Flax** (Google) | Neural network layers | `nn.Module` with explicit state |
| **Equinox** (Patrick Kidger) | Neural network layers | Pytree-based, Pythonic |
| **Optax** (DeepMind) | Optimizers + LR schedules | Composable gradient transforms |
| **Orbax** (Google) | Checkpointing | Save/restore pytrees |
| **CLU** (Google) | Metrics + logging | Training loop utilities |

Optax - это стандартная библиотека оптимизатора. Она отделяет преобразование градиента (Адам, SGD, клипинг) от обновления параметров, что делает его тривиальным для составления:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### Когда использовать JAX против PyTorch

| Factor | JAX | PyTorch |
|--------|-----|---------|
| TPU support | First-class (Google built both) | Community-maintained (torch_xla) |
| GPU support | Good (CUDA via XLA) | Best-in-class (native CUDA) |
| Debugging | Hard (tracing + compilation) | Easy (eager, line-by-line) |
| Ecosystem | Research-focused (Flax, Equinox) | Massive (HuggingFace, torchvision, etc.) |
| Hiring | Niche (Google/DeepMind/Anthropic) | Mainstream (everywhere) |
| Large-scale training | Superior (XLA, pmap, mesh) | Good (FSDP, DeepSpeed) |
| Prototyping speed | Slower (functional overhead) | Faster (mutate and go) |
| Production inference | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| Who uses it | DeepMind (Gemini), Anthropic (Claude) | Meta (Llama), OpenAI (GPT), Stability AI |

Честный ответ: используйте PyTorch, если у вас нет конкретной причины использовать JAX. Эти причины - доступ к TPU, необходимость в градиентах на пример, многоустройство обучения в масштабах, или работа в Google / DeepMind / Anthropic.

### Случайные числа в JAX

JAX не имеет глобального случайного состояния.

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

Сначала это раздражает, но гарантирует воспроизводимость на всех устройствах и компиляциях - свойство, которое PyTorch`torch.manual_seed`не может гарантировать в установках с несколькими GPU.

```figure
batchnorm-effect
```

## Создайте его

### Шаг 1: Настройка и данные

Мы обучим 3-слойный MLP на МНИСТ с использованием JAX и Optax. 784 ввода, два скрытых слоя 256 и 128 нейронов, 10 классов вывода.

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### Шаг 2: Запустить параметры

Нет класса, просто функция, которая возвращает питрь:

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

Он-инициализация, выполненная вручную. три ключа PRNG разделены из одного семена. Каждый вес является неизменной массивой в заложенном дикторе.

### Третий шаг: проход вперед

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

Чистые функции, параметры в, прогноз.`self`, нет хранимого состояния. `loss_fn`вычисляет кросс-энтропию с нуля - мягкий максимум, лог, отрицательная средняя.

### Шаг 4: Сборник подготовки

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad`возвращает как потерю, так и градиенты в одном проходе.`@jax.jit`После первого вызова каждый тренировочный шаг выполняется без прикосновения к Python.

### Шаг 5: Обучение

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 эпох. ~ 97 процентная точность испытаний. Первая эпоха медленная (компиляция JIT).

Заметьте , что не хватает: нет .`.zero_grad()`Нет , нет .`.backward()`Нет , нет .`.step()`Все обновление - это один совокупный вызов функции. Градиенты вычисляются, трансформируются Адамом и применяются к параметрам - все внутри`train_step`- Я не знаю .

## Используйте его

### Лень: Стандарт Google

Флакс - наиболее распространенная библиотека нейронной сети JAX.`nn.Module`В том числе и в случае, если это не так.

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

То же самое, что и PyTorch, но `params`является отдельным от модели. `model.init()`создает парамы. `model.apply(params, x)`Пропускается вперед.

### Равноденствие: Питоновская альтернатива

Эквинокс (патрика Кидгера) представляет модели как питрь:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

Сама модель - питрь.`.apply()`Параметры - это только листья модели.

### Оптимас: Композиционные оптимизаторы

Optax расчленяет преобразование градиента от обновления:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

Снижение градиентов, повышение скорости обучения, снижение веса - все это составлено как цепочка трансформаций. Каждая трансформация видит градиенты, изменяет их и передает их следующему. Нет монолитного класса оптимизатора.

## Погрузи его

**Installation:**

```bash
pip install jax jaxlib optax flax
```

Для поддержки GPU:

```bash
pip install jax[cuda12]
```

Для TPU (Google Cloud):

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Performance gotchas:**

- Первый звонок JIT медленный (компиляция).
- Избегайте петли Python над JAX массивами внутри JIT. Используйте `jax.lax.scan`или `jax.lax.fori_loop`- Я не знаю .
- `jax.debug.print()`работает внутри JIT.`print()`- Нет, нет.
- Профиль с `jax.profiler`Сборник XLA может скрыть узлы.
- JAX предопределяет 75% памяти GPU по умолчанию.`XLA_PYTHON_CLIENT_PREALLOCATE=false`отключить.

**Checkpointing:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**This lesson produces:**
- `outputs/prompt-jax-optimizer.md`-- прописка для выбора правильной конфигурации JAX оптимизатора
- `outputs/skill-jax-patterns.md`-- умение охватывать функциональные модели в JAX

## Упражнения

1. Добавьте выпуск в MLP. В JAX выпуск требует PRNG-ключа - протягивайте ключ через пропуск вперед и разделите его на каждый слой выпуск. Сравните точность теста с и без.

2. Использование`jax.vmap`Вычислить градиенты на пример для партии из 32 изображений MNIST. Вычислить норму градиента для каждого примера.

3. Заменить ручную функцию вперед с помощью общего `mlp_forward(params, x)`Это работает на любой слой.`jax.tree.leaves`для автоматического определения глубины.

4. Сравните уровень обучения с и без `@jax.jit`Сколько ускорения на вашем аппаратном обеспечении?

5. Реализуйте нарезку градиента путем составления `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`Тренируйтесь с и без резки.

## Ключевые условия

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| XLA | "The thing that makes JAX fast" | Accelerated Linear Algebra -- a compiler that fuses operations and generates optimized GPU/TPU kernels from a computation graph |
| JIT | "Just-in-time compilation" | JAX traces the function on first call, compiles to XLA, then runs the compiled version on subsequent calls |
| Pure function | "No side effects" | A function where the output depends only on inputs -- no global state, no mutation, no randomness without explicit keys |
| vmap | "Auto-batching" | Transforms a function that processes one example into one that processes a batch, without rewriting |
| pmap | "Auto-parallelism" | Replicates a function across multiple devices and splits the input batch |
| Pytree | "Nested dict of arrays" | Any nested structure of lists, tuples, dicts, and arrays that JAX can traverse and transform |
| Tracing | "Recording the computation" | JAX executes the function with abstract values to build a computation graph, without computing real results |
| Functional autodiff | "grad of a function" | Computing derivatives by transforming functions, not by attaching gradient storage to tensors |
| Optax | "JAX's optimizer library" | A composable library of gradient transformations -- Adam, SGD, clipping, scheduling -- that chain together |
| Flax | "JAX's nn.Module" | Google's neural network library for JAX, adding layer abstractions while keeping state explicit |

## Дальнейшее чтение

- Документация JAX: https://jax.readthedocs.io/-- официальные документы, с отличными учебниками по граду, JIT и Vmap
- "JAX: композиционные преобразования программ Python+NumPy" (Bradbury et al., 2018) - оригинальная статья, объясняющая философию дизайна
- Документация из льняного: https://flax.readthedocs.io/-- библиотека нейронных сетей Google для JAX
- Патрик Кидгер, "Эквинокс: нейронные сети в JAX через звонки PyTrees и фильтрованные трансформации" (2021) -- питонистская альтернатива Ланну
- DeepMind, "Оптакс: композиционная трансформация и оптимизация градиента" - стандартная библиотека оптимизаторов
- "Ты не знаешь JAX" (Колин Раффелл, 2020) - практический справочник по JAX-готекам и шаблонам, от одного из авторов T5
