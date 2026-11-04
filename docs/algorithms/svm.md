# Support vector machines (SVM and SVR)

## Intuition

Of all the lines that separate two classes, the SVM picks the one that stays **furthest from
both**: the one in the middle of the widest empty strip (the **margin**). A boundary with room to
spare generalises better than one pressed against the points of one of the classes.

Only the points on the edge of the margin (or on the wrong side) decide where the boundary goes:
they're the **support vectors**. The others could disappear without changing anything.

With a **kernel**, the boundary can curve: it's as if the points were taken to a space with more
dimensions, where a straight line separates them, without ever computing those dimensions.

## Essential maths

The SVM looks for the $w$ and $b$ that minimise

$$\frac{1}{2}\lVert w \rVert^2 + C \sum_i \max\big(0,\ 1 - y_i (w \cdot x_i + b)\big)$$

The first term widens the margin (whose width is $2/\lVert w \rVert$); the second punishes the
points inside the margin or on the wrong side (the *hinge loss*), weighted by $C$.

The **kernel trick** swaps each product $x \cdot z$ for a function $K(x, z)$. The RBF kernel,
$K(x, z) = e^{-\gamma \lVert x - z \rVert^2}$, measures how similar two points are: each support
vector creates a "hill" around it, with a width of $1/\sqrt{\gamma}$.

The SVM gives distances to the boundary, not probabilities. To get them, the lab **calibrates**
the distances with a logistic regression fitted by cross-validation (Platt's method).

**SVR** applies the same idea to regression: it looks for a function as flat as possible that
passes within $\varepsilon$ of each point. The errors inside that "tube" don't count.

## Hyperparameters

| Hyperparameter | What it controls | Default |
|---|---|---|
| `C` | How much each point on the wrong side of the margin costs. Large: a narrow margin that tries to get everything right (overfitting); small: a wide, tolerant margin | 1.0 |
| `kernel` | The shape of the boundary: `linear`, `rbf` (curves around the points) or `poly` (a polynomial of degree 3) | `rbf` |
| `gamma` | In the RBF and polynomial kernels, each point's reach. Large: each point makes its own island (overfitting); small: a smooth boundary | 0.5 |
| `epsilon` | In SVR, the width of the tube where errors don't count (on the target's scale) | 0.1 |

## When to use

- With **medium-sized data** (up to a few tens of thousands of rows) and many features, even
  more features than rows.
- When the boundary is curved but smooth: the RBF kernel adapts well.

## Pitfalls

- **Feature scale:** the margin and the RBF kernel use distances. Always standardise.
- **`C` and `gamma` go together:** both large memorise the training data; both small give an
  almost constant model. Tune them together, with cross-validation.
- **Slow with many rows:** training grows between the square and the cube of the number of rows.
- **The probabilities are an extra calibration:** they cost one more training per fold and, with
  little data, can be unreliable.
- **In SVR, `epsilon` is on the target's scale:** with a target in the thousands, 0.1 is
  practically zero.

## What to watch in training

It trains in one go. On the **Train & evaluate** screen, see how the boundary changes with
`gamma`: small gives a smooth curve, large gives islands around the points. **What it learned**
shows the number of support vectors. Many vectors (close to the number of rows) point to a
margin that barely separates anything.
