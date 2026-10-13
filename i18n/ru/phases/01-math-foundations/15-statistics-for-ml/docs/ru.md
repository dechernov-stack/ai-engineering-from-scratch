# Статистика для машинного обучения

> Статистика — это способ узнать, действительно ли ваша модель работает, или просто повезло.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06 (Probability and Distributions), 07 (Bayes' Theorem)
**Time:** ~120 minutes

## Learning Objectives

- Вычислять описательную статистику, корреляцию Пирсона/Спирмена и матрицы ковариации с нуля
- Проводить тесты гипотез (t-тест, хи-квадрат) и правильно интерпретировать p-значения и доверительные интервалы
- Использовать бутстреп-ресэмплирование для построения доверительных интервалов для любой метрики без предположений о распределении
- Различать статистическую значимость и практическую значимость, используя меры размера эффекта

## The Problem

Вы обучили две модели. Модель A набирает 0,87 на тестовом наборе. Модель B набирает 0,89. Вы разворачиваете модель B. Через три недели производственные метрики хуже, чем раньше. Что произошло?

Модель B не превзошла модель A на самом деле. Разница 0,02 — это шум. Ваш тестовый набор был слишком маленьким, или дисперсия слишком высокой, или обе проблемы сразу. Вы развернули случайность, переодетую в улучшение.

Это происходит постоянно. Колебания лидерборда Kaggle. Статьи, которые не воспроизводятся. A/B-тесты, объявляющие победителей на основе нескольких сотен образцов. Причина всегда одна и та же: кто-то пропустил статистику.

Статистика дает вам инструменты для различения сигнала от шума. Она говорит вам, когда разница реальна, насколько уверенными вы должны быть, и сколько данных вам нужно, прежде чем вы сможете доверять результату. Каждый ML-конвейер, каждое сравнение моделей, каждый эксперимент нуждается в статистике. Без нее вы просто угадываете.

## The Concept

### Descriptive Statistics: Summarizing Your Data

Прежде чем моделировать что-либо, вам нужно знать, как выглядят ваши данные. Описательная статистика сжимает набор данных в несколько чисел, которые захватывают его форму.

**Меры центральной тенденции** отвечают на вопрос «где середина?»

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

Среднее — это точка равновесия. Медиана — это отметка половины. Когда они расходятся, ваше распределение асимметрично. Распределения доходов имеют среднее >> медиану (правая асимметрия из-за миллиардеров). Распределения потерь во время обучения часто имеют среднее << медиану (левая асимметрия из-за легких образцов).

**Меры разброса** отвечают на вопрос «насколько дисперсны данные?»

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**Процентили** делят отсортированные данные на 100 равных частей. 25-й процентиль (Q1) означает, что 25% значений ниже этой точки. 50-й процентиль — это медиана. 75-й процентиль — это Q3.

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

В ML вы используете процентили для задержки вывода, распределений уверенности предсказаний и понимания распределений ошибок. Модель с низкой средней ошибкой, но ужасной ошибкой P99 может быть бесполезна для критичных по безопасности приложений.

**Выборочная статистика и статистика населения.** При вычислении дисперсии из выборки делите на (n-1) вместо n. Это коррекция Бесселя. Она компенсирует тот факт, что ваше выборочное среднее не является истинным средним генеральной совокупности. С n в знаменателе вы систематически недооцениваете истинную дисперсию. С (n-1), оценка несмещена.

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

На практике: если n большое (тысячи образцов), разница незначительна. Если n маленькое (десятки образцов), это имеет значение.

### Correlation: How Variables Move Together

Корреляция измеряет силу и направление линейной связи между двумя переменными.

**Коэффициент корреляции Пирсона** измеряет линейную ассоциацию:

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

Пирсон предполагает, что связь линейна и обе переменные примерно нормально распределены. Он чувствителен к выбросам. Одна экстремальная точка может изменить r с 0,1 на 0,9.

**Ранговая корреляция Спирмена** измеряет монотонную ассоциацию:

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**Когда использовать каждый:**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**Золотое правило:** корреляция не подразумевает причинность. Продажи мороженого и смертность от утопления коррелируют, потому что обе увеличиваются летом. Точность вашей модели и количество параметров коррелируют, но добавление параметров не автоматически улучшает точность (см.: переобучение).

### Covariance Matrix

Ковариация между двумя переменными измеряет, как они варьируются вместе:

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

Для d признаков матрица ковариации C — это матрица d x d, где C[i][j] = Cov(feature_i, feature_j). Диагональные элементы C[i][i] — это дисперсии каждого признака.

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**Связь с PCA.** PCA разлагает матрицу ковариации на собственные значения. Собственные векторы — это главные компоненты (направления максимальной дисперсии). Собственные значения говорят вам, сколько дисперсии захватывает каждая компонента. Это ровно то, что охватывается на Уроке 10, но теперь вы видите, почему матрица ковариации — это правильная вещь для разложения: она кодирует все парные линейные связи в ваших данных.

**Связь с корреляцией.** Матрица корреляции — это матрица ковариации стандартизированных переменных (каждая разделена на свое стандартное отклонение). Корреляция нормализует ковариацию, чтобы все значения попадали в [-1, 1].

### Hypothesis Testing

Проверка гипотез — это основа для принятия решений в условиях неопределенности. Вы начинаете с предположения, собираете данные и определяете, согласуются ли данные с предположением.

**Установка:**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**p-значение** — это вероятность наблюдения данных, которые так же экстремальны, как наблюденные, при условии, что H0 верна. Это НЕ вероятность того, что H0 верна. Это самое распространенное неправильное понимание статистики.

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**Доверительные интервалы** дают диапазон вероятных значений для параметра:

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

Ширина доверительного интервала говорит вам о точности. Широкие интервалы означают высокую неопределенность. Узкие интервалы означают, что ваша оценка точна (но не обязательно корректна, если ваши данные смещены).

### The t-test

t-тест сравнивает средние значения. Существует несколько вариантов.

**Одновыборочный t-тест:** отличается ли среднее значение генеральной совокупности от гипотетического значения?

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**Двухвыборочный t-тест (независимый):** отличаются ли средние значения двух групп?

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**Парный t-тест:** когда измерения предоставляются парами (одна модель оценивается на одних и тех же разбиениях данных):

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

В ML парный t-тест является распространенным: вы запускаете обе модели на одних и тех же 10 складках кросс-валидации и сравниваете их оценки попарно.

### Chi-squared Test

Тест хи-квадрат проверяет, совпадают ли наблюдаемые частоты с ожидаемыми. Полезен для категориальных данных.

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### A/B Testing for ML Models

A/B-тестирование в ML отличается от A/B-тестирования в интернете. Сравнение моделей имеет конкретные проблемы:

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**Процедура:**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### Statistical Significance vs Practical Significance

Результат может быть статистически значимым, но практически бессмысленным. При достаточном объеме данных даже тривиальная разница становится статистически значимой.

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**Размер эффекта** количественно определяет, насколько велика разница, независимо от размера выборки:

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

Всегда сообщайте и p-значение, и размер эффекта. p-значение говорит вам, реальна ли разница. Размер эффекта говорит вам, имеет ли это значение.

### Multiple Comparison Problem

Когда вы проверяете множество гипотез, некоторые будут «значимыми» по случайности. Если вы проверяете 20 вещей при alpha = 0,05, вы ожидаете 1 ложный положительный результат, даже когда ничего не реально.

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**Коррекция Бонферрони:** разделите alpha на количество тестов.

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

В ML это имеет значение, когда вы сравниваете модель по нескольким метрикам, тестируете много конфигураций гиперпараметров или оцениваете несколько наборов данных.

### Bootstrap Methods

Бутстреп оценивает распределение выборки статистики путем ресэмплирования ваших данных с возвращением. Никаких предположений об основном распределении не требуется.

**Алгоритм:**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**Доверительный интервал бутстрэпа (метод процентилей):**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**Почему бутстреп важен для ML:**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**Бутстреп для сравнения моделей:**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

Это более надежно, чем парный t-тест, потому что не делает никаких предположений о распределении.

### Parametric vs Non-parametric Tests

**Параметрические тесты** предполагают конкретное распределение (обычно нормальное):

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**Непараметрические тесты** не делают предположений о распределении:

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**Когда использовать непараметрический:**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**Когда использовать параметрический:**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

В ML-экспериментах вы обычно имеете малое n (5 или 10 складок кросс-валидации), поэтому непараметрические тесты, такие как тест знаков Уилкоксона, часто более подходящи, чем t-тесты.

### Central Limit Theorem: Practical Implications

CLT говорит, что распределение выборочных средних приближается к нормальному распределению по мере увеличения n, независимо от основного распределения генеральной совокупности.

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**Почему это имеет значение для ML:**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**Что CLT НЕ делает:**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### Common Statistical Mistakes in ML Papers

1. **Тестирование на обучающем наборе.** Гарантирует переобучение. Всегда отложите данные, которые модель никогда не видела во время обучения.

2. **Отсутствие доверительных интервалов.** Сообщение одного числа точности без неопределенности делает результаты невоспроизводимыми и непроверяемыми.

3. **Игнорирование множественных сравнений.** Тестирование 50 конфигураций и сообщение о лучшей без коррекции завышает коэффициент ложных положительных результатов.

4. **Путаница статистической и практической значимости.** p-значение 0,001 на улучшение точности на 0,01% не имеет смысла.

5. **Использование точности на несбалансированных данных.** 99% точности на наборе данных с 99% отрицательного класса означает, что модель ничему не научилась. Используйте precision, recall, F1 или AUC.

6. **Выбор вишневых метрик.** Сообщение только той метрики, где ваша модель побеждает. Честная оценка сообщает все релевантные метрики.

7. **Утечка информации через границы train/test.** Нормализация перед разделением или использование будущих данных для предсказания прошлого.

8. **Маленькие тестовые наборы без оценок дисперсии.** Оценка на 100 образцах и утверждение об улучшении на 2% — это шум, а не сигнал.

9. **Предположение независимости, когда данные не являются независимыми.** Медицинские изображения одного пациента, несколько предложений из одного документа. Наблюдения внутри группы коррелированы.

10. **P-хакинг.** Попытка разных тестов, подмножеств или критериев исключения, пока вы не получите p < 0,05. Результат — артефакт поиска.

## Build It

Вы будете реализовывать:

1. **Описательную статистику с нуля** (среднее, медиану, моду, стандартное отклонение, процентили, IQR)
2. **Функции корреляции** (Пирсон и Спирмен с матрицей ковариации)
3. **Тесты гипотез** (одновыборочный t-тест, двухвыборочный t-тест, тест хи-квадрат)
4. **Доверительные интервалы бутстрэпа** (для любой статистики, без предположений)
5. **Симулятор A/B-теста** (генерировать данные, тестировать, проверять ошибки типа I и типа II)
6. **Демо статистической и практической значимости** (показывающее, что большое n делает все «значимым»)

Все с нуля, используя только `math` и `random`. Без numpy, без scipy.

```figure
f3-bootstrap-resample
```

## Key Terms

| Термин | Определение |
|---|---|
| Mean | Сумма значений, разделенная на счет. Чувствителен к выбросам. |
| Median | Среднее значение отсортированных данных. Устойчив к выбросам. |
| Standard deviation | Квадратный корень из дисперсии. Измеряет разброс в исходных единицах. |
| Percentile | Значение, ниже которого падает заданный процент данных. |
| IQR | Интерквартильный диапазон. Q3 минус Q1. Разброс средних 50%. |
| Pearson correlation | Измеряет линейную ассоциацию между двумя переменными. Диапазон [-1, 1]. |
| Spearman correlation | Измеряет монотонную ассоциацию, используя ранги. |
| Covariance matrix | Матрица парных ковариаций между всеми признаками. |
| Null hypothesis | Предположение по умолчанию об отсутствии эффекта или разницы. |
| p-value | Вероятность данных такой же экстремальности при условии истинности нулевой гипотезы. |
| Confidence interval | Диапазон вероятных значений для параметра на заданном уровне доверия. |
| t-test | Проверяет, отличаются ли средние значения существенно. Использует t-распределение. |
| Chi-squared test | Проверяет, отличаются ли наблюдаемые частоты от ожидаемых. |
| Effect size | Величина разницы, независимо от размера выборки. Cohen's d является общим. |
| Bonferroni correction | Делит порог значимости на количество тестов для контроля ложных положительных результатов. |
| Bootstrap | Ресэмплирование с возвращением для оценки распределений выборки. |
| Type I error | Ложный положительный результат. Отклонение H0, когда оно верно. |
| Type II error | Ложный отрицательный результат. Неудача отклонить H0, когда оно ложно. |
| Statistical power | Вероятность правильного отклонения ложной H0. Power = 1 минус коэффициент ошибки типа II. |
| Central limit theorem | Выборочные средние сходятся к нормальному распределению по мере увеличения размера выборки. |
| Parametric test | Предполагает конкретное распределение данных (обычно нормальное). |
| Non-parametric test | Не делает предположений о распределении. Работает с рангами или знаками. |