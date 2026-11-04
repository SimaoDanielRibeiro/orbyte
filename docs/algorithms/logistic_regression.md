# Logistic regression

## Intuition

Despite its name, logistic regression **classifies**. For each example, it adds up the
features multiplied by weights (a line, or a plane, separating the classes) and turns that sum
into a probability between 0 and 1. If the probability of "survived" goes over 50%, the
prediction is "survived".

Learning is choosing the weights that give high probabilities to the examples of the right
class and low ones to those of the wrong class. Each weight says how much a feature pushes
towards a class: on the Titanic, the weight of `Sex_female` is positive (being a woman raised
the chance of surviving) and the weight of `Pclass` is negative (lower classes, 3rd instead of
1st, lowered it).

## Essential maths

With two classes, for an example with features $x$:

$$z = w \cdot x + b \qquad p(\text{class } 1 \mid x) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

The logistic function $\sigma$ (or sigmoid) turns any number into a probability:
$\sigma(0) = 0.5$, large values give almost 1, very negative values give almost 0. The
**decision boundary** is where $p = 0.5$, that is, where $w \cdot x + b = 0$: a line (or a
plane), which is why the model is **linear**.

The weights are chosen by minimising the **log-loss** (cross-entropy) on the training data,
plus a penalty on large weights (L2 regularisation):

$$\text{cost}(w, b) = \sum_i -\big[y_i \log p_i + (1 - y_i)\log(1 - p_i)\big] + \frac{1}{2C}\lVert w \rVert^2$$

The log-loss punishes a confident wrong prediction hard (saying 99% and missing). The term with
$C$ stops the weights from growing without limit: with a small $C$ the penalty is strong and
the boundary is more "cautious".

With more than two classes, there's one weight vector per class, and the **softmax** function
turns the sums into probabilities that add up to 1: $p_k = e^{z_k} / \sum_j e^{z_j}$.

There's no closed formula for the best weights: they have to be improved step by step until
the cost stops falling. The lab has two versions that minimise **the same cost** (and so reach
the same weights) by different paths:

- `logistic_regression` (scikit-learn) uses the L-BFGS optimiser, which takes advantage of the
  cost's curvature and gets there in a few dozen steps;
- `logistic_regression_gd` (teaching version, in NumPy) uses **gradient descent**: it starts
  with the weights at zero and, in each **epoch** (one pass over all the training examples),
  takes a small step against the gradient of the cost.

The gradient has a surprisingly simple form: the error of each prediction, $p_i - y_i$,
multiplied by the example's features, on average, plus the regularisation part. Dividing the
cost by $n$ (which doesn't move the minimum):

$$\nabla_w = \frac{1}{n}\sum_i (p_i - y_i)\, x_i + \frac{w}{C\,n} \qquad \nabla_b = \frac{1}{n}\sum_i (p_i - y_i)$$

$$w \leftarrow w - \eta\, \nabla_w \qquad b \leftarrow b - \eta\, \nabla_b$$

where $\eta$ is the **learning rate**. Examples classified right and with confidence barely
contribute ($p_i \approx y_i$); it's the misclassified ones that push the boundary. Training
stops when the gradient is practically zero (below `tol`): it's at the bottom of the cost.

## Hyperparameters

In both versions:

| Hyperparameter | What it controls | Default |
|---|---|---|
| `C` | Inverse of the regularisation strength: smaller = smaller weights and a simpler boundary | 1.0 |
| `fit_intercept` | Whether it learns the intercept $b$ (without it, the boundary goes through the origin) | yes |

Only in scikit-learn:

| Hyperparameter | What it controls | Default |
|---|---|---|
| `max_iter` | The most steps the optimiser can take to converge | 100 |
| `class_weight` | With `balanced`, the errors on rare classes count more | `none` |

Only in gradient descent:

| Hyperparameter | What it controls | Default |
|---|---|---|
| `learning_rate` | The size of each step: too small drags on, too large jumps from side to side | 1.0 |
| `max_epochs` | The most passes over the training data | 2000 |
| `tol` | Stops when the gradient falls below this: it reached the minimum | 1e-6 |

## When to use

- As the **first model** for any classification problem: it's fast, rarely memorises the
  training data and gives the baseline that more complex models have to beat.
- When the decision has to be **explained**: each weight reads directly.
- When you need reasonable **probabilities** (to decide with your own threshold, for example).

## Pitfalls

- **It's linear.** It can't separate classes that need a curved boundary (the synthetic
  "moons" or "circles" data). Polynomial features help, but raise the risk of overfitting.
- **Feature scale.** Without standardisation, the optimiser takes much longer to converge (the
  convergence warning may appear) and the regularisation punishes features with small numbers
  more. Standardise in the preparation.
- **Strongly correlated features** split the weight between them in an unstable way: the
  weights can no longer be read one by one, even if the predictions stay good.
- **Imbalanced classes:** with 95% of one class, the model may almost never predict the other.
  Look at the balanced accuracy and the recall, and try `class_weight = balanced`.
- **"The weights don't match scikit-learn's."** By default, scikit-learn stops early (steps that
  improve less than 1e-4, at most 100 iterations), so the weights of the two versions differ
  around the third decimal place. With tight tolerances on both, they match to the fifth: the
  cost is the same, only the path changes.
- **Too large a learning rate** (gradient descent only): the loss oscillates instead of falling,
  or blows up (the training stops with the "loss blew up" warning). With standardised features,
  1 usually works; without standardisation, you may need 100 times less.

## What to watch in training

The **scikit-learn** version trains in one go: on the **Train & evaluate** screen you only see
the final boundary and, further down, the learned weights, the metrics on the training and test
data and the confusion matrix. Try lowering `C` and watch the weights shrink and the gap between
training and test change.

The **teaching** version trains epoch by epoch, and the **Train & evaluate** screen takes
snapshots along the way:

- the **loss curve** falls fast in the first epochs and then slowly, and the training accuracy
  rises in jumps (each jump is an example that crossed to the right side of the boundary);
- the **decision boundary** (where the probability is 50%) starts in the wrong place, turns and
  shifts until it separates the classes; the colours show the probability, sharper away from
  the boundary as the weights grow.

At the end, compare with the scikit-learn version on the same data: the final boundary and the
metrics come out practically the same. Try the learning rate on the moons: at 0.05 the training
drags on and doesn't reach the minimum in 2000 epochs; at 20, the loss goes up and down without
settling; at 50, the boundary jumps from side to side between snapshots.
