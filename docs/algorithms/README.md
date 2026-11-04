# Fichas didáticas dos algoritmos

Uma ficha por algoritmo, em `docs/algorithms/<identificador>.md` (por exemplo `knn.md`). Lê-a
antes de implementar o algoritmo; a interface mostra-a ao lado do modelo. A ficha técnica de
cada modelo aponta para o seu ficheiro, e um teste de contrato confirma que ele existe (a
partir da fase 2).

As fichas escrevem-se em **inglês**, como todo o site (ADR 0043): são textos que a interface
mostra. Os títulos das secções são os do modelo abaixo, por esta ordem, e dão as chaves estáveis
de cada secção (`intuition`, `essential-maths`…), que um teste de contrato confirma.

## As fichas

Uma ficha serve às duas versões de um algoritmo (a didática, em NumPy, e a do
scikit-learn) e à classificação e à regressão, quando o algoritmo faz as duas.

**Classificação**

- [Logistic regression](logistic_regression.md): scikit-learn e didática (gradiente
  descendente)
- [Perceptron](perceptron.md): didática
- [K-nearest neighbours (KNN)](knn.md): scikit-learn e didática
- [Gaussian naive Bayes](gaussian_nb.md)
- [Decision tree](decision_tree.md): também de regressão
- [Random forest](random_forest.md): também de regressão
- [Gradient boosting](gradient_boosting.md): também de regressão
- [AdaBoost](adaboost.md): também de regressão
- [SVM and SVR](svm.md): o SVR é a versão de regressão

**Regressão**

- [Linear regression](linear_regression.md): mínimos quadrados (scikit-learn) e didática
  (gradiente descendente)
- [Ridge and Lasso](ridge_lasso.md)
- [Polynomial regression](polynomial_regression.md)
- E as versões de regressão da árvore, da Random Forest, do boosting e do SVM (acima).

**Clustering**

- [K-means](kmeans.md): scikit-learn e didática
- [DBSCAN](dbscan.md)
- [Hierarchical clustering](agglomerative.md)
- [Gaussian mixture](gaussian_mixture.md)

**Redução de dimensionalidade**

- [PCA](pca.md): scikit-learn e didática
- [t-SNE](tsne.md)

**Deteção de anomalias**

- [Isolation Forest](isolation_forest.md)
- [One-Class SVM](one_class_svm.md)
- [Local Outlier Factor (LOF)](lof.md)

**Redes neuronais**

- [Neural network (MLP)](mlp.md): PyTorch, de classificação e de regressão

**Reinforcement learning** (agentes, não modelos: aprendem num ambiente, sem dataset)

- [Q-learning](q_learning.md): didático (NumPy), com uma tabela
- [DQN (Deep Q-Network)](dqn.md): PyTorch, com uma rede no lugar da tabela

**Escolher o modelo** (os agentes do modo automático, fase 9)

- [AutoML agents](automl.md): a triagem e a procura aleatória de hiperparâmetros,
  medidas com validação cruzada só no treino

## Modelo

````markdown
# <Algorithm name>

## Intuition

The idea in a few sentences, with no formulas.

## Essential maths

The least maths needed to understand what the training does.

## Hyperparameters

| Hyperparameter | What it controls | Default |
|---|---|---|

## When to use

## Pitfalls

## What to watch in training

Which snapshots the algorithm shows and what to look for in them.
````
