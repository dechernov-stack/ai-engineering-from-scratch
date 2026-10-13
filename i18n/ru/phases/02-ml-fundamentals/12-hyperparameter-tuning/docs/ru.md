# Настройка гиперпараметров

> Гиперпараметры — это рычаги, которые вы настраиваете до начала обучения. От того, насколько хорошо вы их настроите, зависит разница между посредственной моделью и отличной.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 2, Lesson 11 (Ensemble Methods)
**Time:** ~90 minutes

## Цели обучения

- Реализовать grid search, random search и байесовскую оптимизацию с нуля и сравнить их эффективность по числу проб
- Объяснить, почему random search превосходит grid search, когда большинство гиперпараметров имеют низкую эффективную размерность
- Построить цикл байесовской оптимизации с использованием суррогатной модели и функции приобретения для направления поиска
- Спроектировать стратегию настройки гиперпараметров, которая избегает переобучения на валидационном наборе за счёт правильной кросс-валидации

## Проблема

Ваша модель градиентного бустинга имеет скорость обучения, число деревьев, максимальную глубину, минимальное число объектов на лист, долю подвыборки и долю отбираемых столбцов. Это шесть гиперпараметров. Если у каждого по 5 разумных значений, сетка содержит 5^6 = 15 625 комбинаций. Обучение каждой занимает 10 секунд. Это 43 часа вычислений, чтобы перебрать их все.

Grid search — очевидный подход и худший при больших масштабах. Random search справляется лучше при меньших затратах вычислений. Байесовская оптимизация справляется ещё лучше, обучаясь на прошлых оценках. Знание того, какую стратегию использовать и какие гиперпараметры действительно важны, экономит дни впустую потраченного времени GPU.

## Концепция

### Параметры против гиперпараметров

Параметры обучаются в процессе тренировки (веса, смещения, пороги разбиения). Гиперпараметры задаются до начала обучения и управляют тем, как происходит обучение.

| Гиперпараметр | Что контролирует | Типичный диапазон |
|---------------|-----------------|---------------|
| Скорость обучения | Размер шага на каждое обновление | 0.001 до 1.0 |
| Число деревьев/эпох | Как долго обучать | 10 до 10 000 |
| Максимальная глубина | Сложность модели | 1 до 30 |
| Регуляризация (лямбда) | Предотвращение переобучения | 0.0001 до 100 |
| Размер батча | Шум оценки градиента | 16 до 512 |
| Коэффициент dropout | Доля отключаемых нейронов | 0.0 до 0.5 |

### Grid Search

Grid search оценивает каждую комбинацию заданных значений. Это исчерпывающий и простой для понимания метод, но его сложность растёт экспоненциально с числом гиперпараметров.

```
Grid for 2 hyperparameters:

  learning_rate: [0.01, 0.1, 1.0]
  max_depth:     [3, 5, 7]

  Evaluations: 3 x 3 = 9 combinations

  (0.01, 3)  (0.01, 5)  (0.01, 7)
  (0.1,  3)  (0.1,  5)  (0.1,  7)
  (1.0,  3)  (1.0,  5)  (1.0,  7)
```

У grid search есть фундаментальный недостаток: если один гиперпараметр важен, а другой нет, большинство оценок тратятся впустую. Из 9 проверок вы получаете всего 3 уникальных значения важного параметра.

### Random Search

Random search выбирает значения гиперпараметров из распределений, а не из сетки. При том же бюджете в 9 проверок вы получаете 9 уникальных значений каждого гиперпараметра.

```mermaid
flowchart LR
    subgraph Grid Search
        G1[3 unique learning rates]
        G2[3 unique max depths]
        G3[9 total evaluations]
    end

    subgraph Random Search
        R1[9 unique learning rates]
        R2[9 unique max depths]
        R3[9 total evaluations]
    end
```

Почему random search превосходит grid search (Bergstra & Bengio, 2012):

- У большинства гиперпараметров низкая эффективная размерность. Обычно для конкретной задачи важны только 1-2 из 6 гиперпараметров.
- Grid search тратит оценки на неважные измерения.
- Random search плотнее покрывает важные измерения при том же бюджете.
- При 60 случайных пробах у вас 95%-ная вероятность найти точку в пределах 5% от оптимума (если он существует в пространстве поиска).

### Байесовская оптимизация

Random search игнорирует результаты. Он не учитывает, что высокая скорость обучения вызывает расхождение или что глубина 3 стабильно превосходит глубину 10. Байесовская оптимизация использует прошлые оценки, чтобы решить, где искать дальше.

```mermaid
flowchart TD
    A[Define search space] --> B[Evaluate initial random points]
    B --> C[Fit surrogate model to results]
    C --> D[Use acquisition function to pick next point]
    D --> E[Evaluate the model at that point]
    E --> F{Budget exhausted?}
    F -->|No| C
    F -->|Yes| G[Return best hyperparameters found]
```

Два ключевых компонента:

**Суррогатная модель:** дешёвая в оценке модель (обычно гауссовский процесс), которая аппроксимирует дорогую целевую функцию. Она даёт как прогноз, так и оценку неопределённости в любой точке пространства поиска.

**Функция приобретения:** решает, где оценивать дальше, балансируя эксплуатацию (поиск рядом с известными хорошими точками) и исследование (поиск там, где неопределённость высока). Распространённые варианты:

- **Expected Improvement (EI):** насколько сильно мы ожидаем улучшения относительно текущего лучшего результата в этой точке?
- **Upper Confidence Bound (UCB):** прогноз плюс кратная неопределённость. Более высокий UCB означает либо перспективность, либо неисследованность.
- **Probability of Improvement (PI):** какова вероятность, что эта точка превзойдёт текущий лучший результат?

Байесовская оптимизация обычно находит лучшие гиперпараметры, чем random search, при в 2-5 раз меньшем числе оценок. Накладные расходы на подгонку суррогатной модели пренебрежимо малы по сравнению с обучением самой модели.

### Ранняя остановка

Не каждый запуск обучения нужно доводить до конца. Если конфигурация явно плохая после 10 эпох, остановите её и переходите дальше. Это ранняя остановка в контексте поиска гиперпараметров.

Стратегии:
- **На основе терпения (patience-based):** остановка, если валидационные потери не улучшаются N эпох подряд
- **Медианное отсечение (median pruning):** остановка, если промежуточный результат пробы хуже медианы завершённых проб на том же шаге
- **Hyperband:** выделение малых бюджетов многим конфигурациям, затем постепенное увеличение бюджета для лучших из них

Hyperband особенно эффективен. Он запускает 81 конфигурацию по 1 эпохе каждую, оставляет лучшую треть, даёт им 3 эпохи, снова оставляет лучшую треть и так далее. Это позволяет найти хорошие конфигурации в 10-50 раз быстрее, чем оценка всех конфигураций при полном бюджете.

### Планировщики скорости обучения

Скорость обучения почти всегда является самым важным гиперпараметром. Вместо того чтобы держать её постоянной, планировщики корректируют её в процессе обучения.

| Планировщик | Формула | Когда использовать |
|-----------|---------|--------------|
| Ступенчатое затухание | Умножение на 0.1 каждые N эпох | Классическое обучение CNN |
| Косинусное отжигание | lr * 0.5 * (1 + cos(pi * t / T)) | Современный вариант по умолчанию |
| Разогрев + затухание | Линейное увеличение, затем косинусное затухание | Transformers |
| One-cycle | Увеличение, затем уменьшение в течение одного цикла | Быстрая сходимость |
| Уменьшение на плато | Снижение в N раз при застое метрики | Безопасный вариант по умолчанию |

### Важность гиперпараметров

Не все гиперпараметры одинаково важны. Исследования случайных лесов (Probst et al., 2019) и градиентного бустинга показывают устойчивые закономерности:

**Высокая важность:**
- Скорость обучения (всегда настраивать первой)
- Число оценщиков / эпох (использовать раннюю остановку вместо настройки)
- Сила регуляризации

**Средняя важность:**
- Максимальная глубина / число слоёв
- Минимальное число объектов на лист / weight decay
- Доля подвыборки

**Низкая важность:**
- Максимальное число признаков (для случайных лесов)
- Конкретный выбор функции активации
- Размер батча (в разумном диапазоне)

Сначала настраивайте важные параметры, остальные оставляйте по умолчанию.

### Практическая стратегия

```mermaid
flowchart TD
    A[Start with defaults] --> B[Coarse random search: 20-50 trials]
    B --> C[Identify important hyperparameters]
    C --> D[Fine random or Bayesian search: 50-100 trials in narrowed space]
    D --> E[Final model with best hyperparameters]
    E --> F[Retrain on full training data]
```

Конкретный рабочий процесс:

1. **Начните со значений по умолчанию из библиотеки.** Они выбраны опытными практиками и часто дают 80% результата.
2. **Грубый random search.** Широкие диапазоны, 20-50 проб. Используйте раннюю остановку, чтобы быстро прерывать неудачные запуски.
3. **Проанализируйте результаты.** Какие гиперпараметры коррелируют с производительностью? Сузьте пространство поиска.
4. **Точный поиск.** Байесовская оптимизация или сфокусированный random search в суженном пространстве. 50-100 проб.
5. **Переобучите на всех обучающих данных** с найденными лучшими гиперпараметрами.

### Интеграция с кросс-валидацией

Настройка гиперпараметров на одном валидационном разбиении рискованна. Лучшие гиперпараметры могут переобучиться под конкретный валидационный фолд. Вложенная кросс-валидация решает эту проблему с помощью двух циклов:

- **Внешний цикл** (оценка): разбивает данные на train+val и test. Сообщает несмещённую производительность.
- **Внутренний цикл** (настройка): разбивает train+val на train и val. Находит лучшие гиперпараметры.

```mermaid
flowchart TD
    D[Full Dataset] --> O1[Outer Fold 1: Test]
    D --> O2[Outer Fold 2: Test]
    D --> O3[Outer Fold 3: Test]
    D --> O4[Outer Fold 4: Test]
    D --> O5[Outer Fold 5: Test]

    O1 --> I1[Inner 5-fold CV on remaining data]
    I1 --> T1[Best hyperparams for fold 1]
    T1 --> E1[Evaluate on outer test fold 1]

    O2 --> I2[Inner 5-fold CV on remaining data]
    I2 --> T2[Best hyperparams for fold 2]
    T2 --> E2[Evaluate on outer test fold 2]
```

Каждый внешний фолд независимо находит свои собственные лучшие гиперпараметры. Внешние оценки являются несмещённой оценкой способности к обобщению.

С sklearn:

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

inner_cv = GridSearchCV(
    GradientBoostingRegressor(),
    param_grid={
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 5],
        "n_estimators": [50, 100, 200],
    },
    cv=5,
    scoring="neg_mean_squared_error",
)

outer_scores = cross_val_score(
    inner_cv, X, y, cv=5, scoring="neg_mean_squared_error"
)

print(f"Nested CV MSE: {-outer_scores.mean():.4f} +/- {outer_scores.std():.4f}")
```

Это дорого (5 внешних фолдов x 5 внутренних фолдов x 27 точек сетки = 675 обучений модели), но даёт надёжную оценку производительности. Используйте это при публикации итоговых результатов в статьях или когда цена решения высока.

### Практические советы

**Начните со скорости обучения.** Это всегда самый важный гиперпараметр для градиентных методов. Плохая скорость обучения делает всё остальное неважным. Зафиксируйте остальные гиперпараметры по умолчанию и сначала переберите скорость обучения.

**Используйте лог-равномерные распределения для скорости обучения и регуляризации.** Разница между 0.001 и 0.01 так же важна, как разница между 0.1 и 1.0. Линейный поиск тратит бюджет впустую на большом конце диапазона.

**Используйте раннюю остановку вместо настройки n_estimators.** Для бустинга и нейронных сетей задайте n_estimators или epochs высоким значением и позвольте ранней остановке решать, когда остановиться. Это убирает один гиперпараметр из поиска.

**Распределение бюджета.** Тратьте 60% бюджета настройки на 2 самых важных гиперпараметра. Оставшиеся 40% — на всё остальное. На эти 2 параметра приходится большая часть вариации производительности.

**Масштаб имеет значение.** Никогда не ищите размер батча по логарифмической шкале (16, 32, 64 вполне подходят). Всегда ищите скорость обучения по логарифмической шкале. Соотносите распределение поиска с тем, как гиперпараметр влияет на модель.

| Тип модели | Главные гиперпараметры | Рекомендуемый поиск | Бюджет |
|-----------|--------------------|--------------------|--------|
| Random Forest | n_estimators, max_depth, min_samples_leaf | Random search, 50 проб | Низкий (быстрое обучение) |
| Gradient Boosting | learning_rate, n_estimators, max_depth | Байесовский, 100 проб + ранняя остановка | Средний |
| Neural Network | learning_rate, weight_decay, batch_size | Байесовский или random, 100+ проб | Высокий (медленное обучение) |
| SVM | C, gamma (RBF kernel) | Grid по лог. шкале, 25-50 проб | Низкий (2 параметра) |
| Lasso/Ridge | alpha | 1D поиск по лог. шкале, 20 проб | Очень низкий |
| XGBoost | learning_rate, max_depth, subsample, colsample | Байесовский, 100-200 проб + ранняя остановка | Средний |

**Если сомневаетесь:** random search с числом проб в 2 раза больше числа гиперпараметров (например, 6 гиперпараметров = минимум 12+ проб). Вас удивит, как часто random search с 50 пробами обходит тщательно спроектированный grid search.

```figure
k-fold-cv
```

## Реализация

### Шаг 1: Grid Search с нуля

Код в `code/tuning.py` реализует grid search, random search и простой байесовский оптимизатор с нуля.

```python
def grid_search(model_fn, param_grid, X_train, y_train, X_val, y_val):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    best_score = -float("inf")
    best_params = None
    n_evals = 0

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        model = model_fn(**params)
        model.fit(X_train, y_train)
        score = evaluate(model, X_val, y_val)
        n_evals += 1

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score, n_evals
```

### Шаг 2: Random Search с нуля

```python
def random_search(model_fn, param_distributions, X_train, y_train,
                  X_val, y_val, n_iter=50, seed=42):
    rng = np.random.RandomState(seed)
    best_score = -float("inf")
    best_params = None

    for _ in range(n_iter):
        params = {k: sample(v, rng) for k, v in param_distributions.items()}
        model = model_fn(**params)
        model.fit(X_train, y_train)
        score = evaluate(model, X_val, y_val)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score, n_iter
```

### Шаг 3: Байесовская оптимизация (упрощённая)

Основная идея: подогнать гауссовский процесс к наблюдаемым парам (гиперпараметр, оценка), затем использовать функцию приобретения, чтобы решить, куда смотреть дальше.

```python
class SimpleBayesianOptimizer:
    def __init__(self, search_space, n_initial=5):
        self.search_space = search_space
        self.n_initial = n_initial
        self.X_observed = []
        self.y_observed = []

    def _kernel(self, x1, x2, length_scale=1.0):
        dists = np.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=2)
        return np.exp(-0.5 * dists / length_scale ** 2)

    def _fit_gp(self, X_new):
        X_obs = np.array(self.X_observed)
        y_obs = np.array(self.y_observed)
        y_mean = y_obs.mean()
        y_centered = y_obs - y_mean

        K = self._kernel(X_obs, X_obs) + 1e-4 * np.eye(len(X_obs))
        K_star = self._kernel(X_new, X_obs)

        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_centered))
        mu = K_star @ alpha + y_mean

        v = np.linalg.solve(L, K_star.T)
        var = 1.0 - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-6)

        return mu, var

    def _expected_improvement(self, mu, var, best_y):
        sigma = np.sqrt(var)
        z = (mu - best_y) / (sigma + 1e-10)
        ei = sigma * (z * norm_cdf(z) + norm_pdf(z))
        return ei

    def suggest(self):
        if len(self.X_observed) < self.n_initial:
            return sample_random(self.search_space)

        candidates = [sample_random(self.search_space) for _ in range(500)]
        X_cand = np.array([to_vector(c) for c in candidates])
        mu, var = self._fit_gp(X_cand)
        ei = self._expected_improvement(mu, var, max(self.y_observed))
        return candidates[np.argmax(ei)]

    def observe(self, params, score):
        self.X_observed.append(to_vector(params))
        self.y_observed.append(score)
```

Суррогат GP даёт две вещи в каждой кандидатной точке: прогнозируемую оценку (mu) и неопределённость (var). Expected Improvement балансирует их: он отдаёт предпочтение точкам, где модель прогнозирует высокие оценки ИЛИ где неопределённость высока. На раннем этапе у большинства точек высокая неопределённость, поэтому оптимизатор исследует. Позже он фокусируется на наиболее перспективной области.

### Шаг 4: Сравнение всех методов

Запустите все три метода на одной и той же синтетической целевой функции и сравните. Это сравнение использует упрощённую обёртку, вызывающую каждый оптимизатор с прямой целевой функцией (без обучения модели), поэтому API отличается от реализаций выше, основанных на моделях:

```python
def synthetic_objective(params):
    lr = params["learning_rate"]
    depth = params["max_depth"]
    return -(np.log10(lr) + 2) ** 2 - (depth - 4) ** 2 + 10

param_grid = {
    "learning_rate": [0.001, 0.01, 0.1, 1.0],
    "max_depth": [2, 3, 4, 5, 6, 7, 8],
}

grid_best = None
grid_score = -float("inf")
grid_history = []
for combo in itertools.product(*param_grid.values()):
    params = dict(zip(param_grid.keys(), combo))
    score = synthetic_objective(params)
    grid_history.append((params, score))
    if score > grid_score:
        grid_score = score
        grid_best = params

param_dist = {
    "learning_rate": ("log_float", 0.001, 1.0),
    "max_depth": ("int", 2, 8),
}

rand_best = None
rand_score = -float("inf")
rand_history = []
rng = np.random.RandomState(42)
for _ in range(28):
    params = {k: sample(v, rng) for k, v in param_dist.items()}
    score = synthetic_objective(params)
    rand_history.append((params, score))
    if score > rand_score:
        rand_score = score
        rand_best = params

optimizer = SimpleBayesianOptimizer(param_dist, n_initial=5)
bayes_history = []
for _ in range(28):
    params = optimizer.suggest()
    score = synthetic_objective(params)
    optimizer.observe(params, score)
    bayes_history.append((params, score))
bayes_score = max(s for _, s in bayes_history)

print(f"{'Method':<20} {'Best Score':>12} {'Evaluations':>12}")
print("-" * 50)
print(f"{'Grid Search':<20} {grid_score:>12.4f} {len(grid_history):>12}")
print(f"{'Random Search':<20} {rand_score:>12.4f} {len(rand_history):>12}")
print(f"{'Bayesian Opt':<20} {bayes_score:>12.4f} {len(bayes_history):>12}")
```

При одинаковом бюджете байесовская оптимизация обычно находит лучший результат быстрее всех, потому что не тратит оценки впустую в явно плохих областях. Random search покрывает больше пространства, чем grid search. Grid search выигрывает только тогда, когда у вас очень мало гиперпараметров и вы можете позволить себе исчерпывающий перебор.

## Применение

### Optuna на практике

Optuna — рекомендуемая библиотека для серьёзной настройки гиперпараметров. Она поддерживает отсечение, распределённый поиск и визуализацию «из коробки».

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    n_est = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 2, 10)

    model = GradientBoostingRegressor(
        learning_rate=lr,
        n_estimators=n_est,
        max_depth=max_depth,
    )
    model.fit(X_train, y_train)
    return mean_squared_error(y_val, model.predict(X_val))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print(f"Best params: {study.best_params}")
print(f"Best MSE: {study.best_value:.4f}")
```

Ключевые возможности Optuna:
- `suggest_float(..., log=True)` для параметров, которые лучше искать по логарифмической шкале (скорость обучения, регуляризация)
- `suggest_int` для целочисленных параметров
- `suggest_categorical` для дискретных вариантов
- Встроенный MedianPruner для ранней остановки неудачных проб
- `study.trials_dataframe()` для анализа

### Optuna с отсечением

Отсечение (pruning) останавливает бесперспективные пробы досрочно, экономя огромные вычислительные ресурсы. Вот шаблон:

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("lr", 1e-4, 0.5, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }

    model = GradientBoostingRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3,
                             scoring="neg_mean_squared_error")
    mean_score = -scores.mean()

    trial.report(mean_score, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return mean_score

pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
study = optuna.create_study(direction="minimize", pruner=pruner)
study.optimize(objective, n_trials=200)
```

`MedianPruner` останавливает пробу, если её промежуточное значение хуже медианы всех завершённых проб на том же шаге. Отсечение требует вызова `trial.report()` для сообщения промежуточных метрик и `trial.should_prune()` для проверки, следует ли остановить пробу. `n_startup_trials=10` гарантирует, что по крайней мере 10 проб завершатся полностью, прежде чем отсечение начнёт действовать. Обычно это экономит 40-60% общих вычислений.

### Встроенные механизмы настройки sklearn

Для быстрых экспериментов sklearn предоставляет `GridSearchCV`, `RandomizedSearchCV` и `HalvingRandomSearchCV`:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint

param_dist = {
    "learning_rate": loguniform(1e-4, 0.5),
    "max_depth": randint(2, 10),
    "n_estimators": randint(50, 500),
}

search = RandomizedSearchCV(
    GradientBoostingRegressor(),
    param_dist,
    n_iter=100,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
    n_jobs=-1,
)
search.fit(X_train, y_train)
print(f"Best params: {search.best_params_}")
print(f"Best CV MSE: {-search.best_score_:.4f}")
```

Используйте `loguniform` из scipy для скорости обучения и регуляризации. Используйте `randint` для целочисленных гиперпараметров. Флаг `n_jobs=-1` распараллеливает вычисления по всем ядрам CPU.

### Типичные ошибки при настройке гиперпараметров

**Утечка данных через предобработку.** Если вы подгоняете scaler на всём наборе данных до кросс-валидации, информация из валидационного фолда просачивается в обучение. Всегда помещайте предобработку внутрь `Pipeline`, чтобы она подгонялась только на обучающем фолде.

**Переобучение на валидационном наборе.** Запуск тысяч проб фактически означает обучение на валидационном наборе. Используйте вложенную кросс-валидацию для итоговых оценок производительности или выделите отдельный тестовый набор, к которому вы не прикасаетесь во время настройки.

**Слишком узкий диапазон поиска.** Если ваше лучшее значение находится на границе пространства поиска, значит, вы искали недостаточно широко. Оптимальное значение может быть за пределами вашего диапазона. Всегда проверяйте, не находятся ли лучшие параметры на краях.

**Игнорирование эффектов взаимодействия.** Скорость обучения и число оценщиков сильно взаимодействуют в бустинге. Низкой скорости обучения требуется больше оценщиков. Настройка их по отдельности даёт худшие результаты, чем настройка вместе.

**Отказ от ранней остановки для итеративных моделей.** Для градиентного бустинга и нейронных сетей задайте n_estimators или epochs высоким значением и используйте раннюю остановку. Это строго лучше, чем настраивать число итераций как гиперпараметр.

## Упражнения

1. Запустите grid search и random search с одинаковым общим бюджетом (например, 50 проверок). Сравните найденные лучшие оценки. Повторите эксперимент 10 раз с разными сидами. Как часто побеждает random search?

2. Реализуйте Hyperband с нуля. Начните со 81 конфигурации, каждая обучается 1 эпоху. Оставляйте лучшую 1/3 на каждом раунде и утраивайте их бюджет. Сравните общие вычисления (сумму всех эпох по всем конфигурациям) с запуском 81 конфигурации при полном бюджете.

3. Добавьте планировщик скорости обучения (косинусное отжигание) к реализации градиентного бустинга из Lesson 11. Помогает ли это по сравнению с фиксированной скоростью обучения?

4. Используйте Optuna для настройки RandomForestClassifier на реальном наборе данных (например, наборе данных breast cancer из sklearn). Используйте `optuna.visualization.plot_param_importances(study)`, чтобы увидеть, какие гиперпараметры наиболее важны. Совпадает ли это с рейтингом важности из этого урока?

5. Реализуйте простую функцию приобретения (Expected Improvement) и продемонстрируйте баланс исследования и эксплуатации. Постройте график среднего значения и неопределённости суррогатной модели и покажите, где EI выбирает следующую точку для оценки.

## Ключевые термины

| Термин | Что говорят люди | Что это на самом деле означает |
|------|----------------|----------------------|
| Гиперпараметр | «Настройка, которую вы выбираете» | Значение, задаваемое до обучения, которое управляет процессом обучения, а не выучивается из данных |
| Grid search | «Перебрать каждую комбинацию» | Исчерпывающий поиск по заданной сетке параметров. Экспоненциальная стоимость. |
| Random search | «Просто случайная выборка» | Выборка гиперпараметров из распределений. Лучше покрывает важные измерения, чем grid search. |
| Байесовская оптимизация | «Умный поиск» | Использует суррогатную модель целевой функции, чтобы решить, где оценивать дальше, балансируя исследование и эксплуатацию |
| Суррогатная модель | «Дешёвая аппроксимация» | Модель (обычно гауссовский процесс), аппроксимирующая дорогую целевую функцию на основе наблюдаемых оценок |
| Функция приобретения | «Куда смотреть дальше» | Оценивает кандидатные точки, балансируя ожидаемое улучшение с неопределённостью. EI и UCB — распространённые варианты. |
| Ранняя остановка | «Перестать тратить время впустую» | Прекращение обучения досрочно, когда валидационная производительность перестаёт улучшаться |
| Hyperband | «Турнирная сетка для конфигураций» | Адаптивное распределение ресурсов: запуск множества конфигураций с малыми бюджетами, сохранение лучших и увеличение их бюджетов |
| Планировщик скорости обучения | «Изменение lr во время обучения» | Функция, которая корректирует скорость обучения на протяжении обучения для лучшей сходимости |

## Дополнительное чтение

- [Bergstra & Bengio: Random Search for Hyper-Parameter Optimization (2012)](https://jmlr.org/papers/v13/bergstra12a.html) -- статья, показавшая, что random search превосходит grid search
- [Snoek et al., Practical Bayesian Optimization of Machine Learning Algorithms (2012)](https://arxiv.org/abs/1206.2944) -- байесовская оптимизация для ML
- [Li et al., Hyperband: A Novel Bandit-Based Approach (2018)](https://jmlr.org/papers/v18/16-558.html) -- статья о Hyperband
- [Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902) -- статья об Optuna
- [Probst et al., Tunability: Importance of Hyperparameters (2019)](https://jmlr.org/papers/v20/18-444.html) -- какие гиперпараметры важны