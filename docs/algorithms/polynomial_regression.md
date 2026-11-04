# Polynomial regression

## Intuition

A straight line can't follow a curve. But if we add the features' powers ($x^2$, $x^3$, ...) and
their products ($x \cdot z$) to the features, a **linear** regression on those new features can
draw curves. The model is still linear in the weights; only the features became curved.

It's the classic demonstration of **overfitting**: with a low degree, the curve is too simple
(underfitting); with a high degree, it goes through every training point and wiggles wildly
between them.

## Essential maths

With one feature and degree $d$:

$$\hat y = b + w_1 x + w_2 x^2 + \dots + w_d x^d$$

With $p$ features, the polynomial of degree $d$ has every product up to degree $d$: that's
$\binom{p + d}{d} - 1$ terms. With 10 features and degree 3, that's already 285. The lab's model
is a *pipeline* of three pieces:

1. create the powers and the products (`PolynomialFeatures`);
2. standardise them ($x^{15}$ and $x$ have incomparable scales);
3. fit a Ridge, with the chosen $\alpha$, which holds back the terms' weights.

## Hyperparameters

| Hyperparameter | What it controls | Default |
|---|---|---|
| `degree` | The highest power. Degree 1 is the straight line; high degrees make curves that go through every training point | 2 |
| `alpha` | The Ridge regularisation. Near zero shows overfitting in all its glory; raising it tames the curves of the high degrees | $10^{-6}$ |

## When to use

- When the relationship is **curved and smooth** and there are few features (one or two).
- To **see** the balance between bias and variance (the Lab screen's "Complexity and
  overfitting" demo).

## Pitfalls

- **The number of terms explodes** with the degree and the number of features. The model refuses
  more than 5,000 terms.
- **Overfitting with high degrees:** the training error always falls; the test error falls and
  then rises again. Pick the degree by the test (or with cross-validation), never by the training
  data.
- **Disastrous extrapolation:** outside the range of the data, a high-degree polynomial shoots off
  towards plus or minus infinity.

## What to watch in training

It trains in one go. With a single feature, the **Train & evaluate** screen draws the curve over
the points. Try degree 1, 3 and 15. In the evaluation, compare the error on the training data and
on the test: the gap grows with the degree.
