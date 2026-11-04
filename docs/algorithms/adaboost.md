# AdaBoost

## Intuition

AdaBoost (*adaptive boosting*) was the first boosting algorithm to work well in practice (Freund
and Schapire, 1997). It trains very weak models, almost always trees with a single question (the
*stumps*), one at a time. After each one, it **raises the weight** of the examples that model
got wrong, so the next one focuses on them. At the end, each model votes, with more weight for
those that got more right.

## Essential maths

In classification (the SAMME algorithm), every example starts with the same weight $w_i$. At
each stage $m$:

1. train the weak model $h_m$ with the current weights;
2. compute the weighted error $\varepsilon_m$ (the sum of the weights of the misclassified
   examples);
3. give it the voting weight $\alpha_m = \nu \left(\log\frac{1 - \varepsilon_m}{\varepsilon_m} + \log(K - 1)\right)$,
   with $K$ classes: the smaller the error, the more it counts;
4. multiply the weight of each misclassified example by $e^{\alpha_m}$ and normalise.

The prediction is the class with the most weighted votes, $\arg\max_k \sum_m \alpha_m\, [h_m(x) = k]$.

In regression (AdaBoost.R2), each example's error (proportional, squared or exponential,
according to `loss`) decides its weight in the next stage, and the prediction is the **weighted
median** of the trees' predictions.

## Hyperparameters

| Hyperparameter | What it controls | Default |
|---|---|---|
| `n_estimators` | How many weak models, at most. It stops earlier if one of them gets everything right | 50 |
| `learning_rate` | How much each new model counts ($\nu$): smaller needs more models, but is steadier | 1.0 |
| `loss` | In regression, how each example's error becomes its weight: `linear`, `square` or `exponential` | `linear` |

## When to use

- To **understand boosting**: the example weights make the idea very visible.
- With clean data, where the classes separate with several simple rules.

## Pitfalls

- **Sensitive to noise and outliers:** an example with the wrong label is missed again and again
  and its weight grows without end, pulling the next models towards it.
- **Weak models that are too strong** (deep trees) memorise the training data in the very first
  stage, and boosting has nothing left to correct.
- Gradient boosting is usually more accurate and more flexible (it lets you choose the loss).

## What to watch in training

On the **Train & evaluate** screen, each step is a stage. With *stumps*, the first boundary is a
single line parallel to an axis. Each stage adds one more cut, and the boundary is built bit by
bit, like a staircase. The loss is the share of errors (or the squared error) on the training
data.
