# Gradient boosting

## Intuition

Boosting combines weak models (small trees) **in sequence**, each correcting the errors of the
previous ones. It starts with a simple prediction (the mean, in regression). The first tree
learns what that prediction got wrong, the second what was still left to get right, and so on.
It's like a student who, after each test, studies only the topics they got wrong.

Unlike the random forest (independent trees, in parallel, voting), here each tree depends on
the previous ones.

## Essential maths

The model is a sum of trees:

$$F_M(x) = F_0 + \nu \sum_{m=1}^{M} h_m(x)$$

At each stage $m$, it computes the **negative gradient** of the loss with respect to the current
prediction, for each example: the generalised "residual". With the squared error, it's the
actual residual, $r_i = y_i - F_{m-1}(x_i)$. The tree $h_m$ fits those residuals and joins the
model, multiplied by the learning rate $\nu$. It's gradient descent done in the space of
functions: each tree is a step in the direction that brings the loss down the most.

In classification, $F$ gives the *log-odds* (passed through the sigmoid, or the softmax), and
the loss is the log-loss.

## Hyperparameters

| Hyperparameter | What it controls | Default |
|---|---|---|
| `n_estimators` | How many trees (stages). Unlike the forest, too many trees end up memorising the training data | 100 |
| `learning_rate` | How much each tree counts ($\nu$). Small: it needs more trees, but generalises better | 0.1 |
| `max_depth` | The depth of each tree. Small trees (2 to 5 levels) are enough: the power comes from adding them up | 3 |
| `subsample` | The share of the rows each tree sees. Below 1, it's *stochastic gradient boosting*: less overfitting | 1.0 |

## When to use

- When you want the **best accuracy** on tabular data: well tuned, gradient boosting (and its
  fast versions, like XGBoost and LightGBM) wins most competitions.
- In classification and in regression, with features on different scales (they're trees).

## Pitfalls

- **Overfitting with too many stages:** keep an eye on the validation error, or lower the
  `learning_rate` and add stages.
- **`learning_rate` and `n_estimators` go together:** half the rate asks for roughly twice the
  trees.
- **Sequential:** each tree waits for the previous one, so it trains more slowly than a forest
  of the same size.
- **Sensitive to noise in the target:** it keeps chasing residuals that are just noise.

## What to watch in training

On the **Train & evaluate** screen, each step is a stage (one more tree). The loss is the
training one. It always falls, because each tree corrects the training data, which is why it's
the test that tells you when to stop. The boundary (or the prediction) starts coarse and gains
detail stage by stage. In regression with one feature, you see the step function getting
closer to the points.
