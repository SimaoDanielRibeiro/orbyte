# DBSCAN

## Intuition

DBSCAN (*density-based spatial clustering of applications with noise*) sees groups as **dense
areas** of points, separated by empty areas. You don't have to say how many groups there are, the
groups can have any shape (moons, rings), and isolated points are left out, as **noise**.

## Essential maths

With a radius $\varepsilon$ and a minimum number of points $m$:

- a point is a **core** point if it has at least $m$ points (itself included) within
  $\varepsilon$;
- two core points within $\varepsilon$ of each other go in the same group, and the groups grow
  from core to core;
- a point that isn't a core point, but is within $\varepsilon$ of one, sits on the **border** of
  that group;
- the rest are **noise** (group $-1$).

scikit-learn's DBSCAN doesn't predict new points. The lab uses a simple rule: a new point joins
the group of the closest core point, if it's within $\varepsilon$ of it; otherwise, it's noise.
The training points keep exactly the group DBSCAN gave them.

## Hyperparameters

| Hyperparameter | What it controls | Default |
|---|---|---|
| `eps` | The neighbourhood radius ($\varepsilon$). Small: almost everything is noise; large: everything merges into one group | 0.5 |
| `min_samples` | How many points there must be within the radius for a point to be a core point ($m$). Larger demands denser areas | 5 |

## When to use

- When the groups have **irregular shapes** or you want to set the noise apart.
- When you don't know how many groups there are.

## Pitfalls

- **`eps` depends on the scale:** standardise the features. To choose it, sort the distance from
  each point to its `min_samples`-th neighbour and look for the "elbow" in that curve.
- **Different densities:** a single `eps` can't suit both a tight group and a spread-out one. A
  radius that's good for one is bad for the other.
- **Many dimensions:** distances lose their meaning, and so does density.
- **Shape metrics** (silhouette, Davies-Bouldin) ignore the noise. Also look at the share of
  noise: more than half means an `eps` that's too small.

## What to watch in training

It trains in one go. On the **Train & evaluate** screen, the coloured areas are the groups (with
the new-points rule) and the grey ones are the noise. Try `eps`: you see the groups break into
pieces (small) or merge into one (large). In the evaluation, the "Noise" metric says what share of
the points was left out.
