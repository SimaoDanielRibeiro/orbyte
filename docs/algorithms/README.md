# Fichas didáticas dos algoritmos

Uma ficha por algoritmo, em `docs/algorithms/<identificador>.md` (por exemplo `knn.md`). Lê-a
antes de implementar o algoritmo; a interface mostra-a ao lado do modelo. A ficha técnica de
cada modelo aponta para o seu ficheiro, e um teste de contrato confirma que ele existe (a
partir da fase 2).

## As fichas

Uma ficha serve às duas versões de um algoritmo (a didática, em NumPy, e a do
scikit-learn) e à classificação e à regressão, quando o algoritmo faz as duas.

**Classificação**

- [Regressão logística](logistic_regression.md): scikit-learn e didática (gradiente
  descendente)
- [Perceptrão](perceptron.md): didática
- [K vizinhos mais próximos (KNN)](knn.md): scikit-learn e didática
- [Naive Bayes gaussiano](gaussian_nb.md)
- [Árvore de decisão](decision_tree.md): também de regressão
- [Random Forest](random_forest.md): também de regressão
- [Gradient boosting](gradient_boosting.md): também de regressão
- [AdaBoost](adaboost.md): também de regressão
- [SVM e SVR](svm.md): o SVR é a versão de regressão

**Regressão**

- [Regressão linear](linear_regression.md): mínimos quadrados (scikit-learn) e didática
  (gradiente descendente)
- [Ridge e Lasso](ridge_lasso.md)
- [Regressão polinomial](polynomial_regression.md)
- E as versões de regressão da árvore, da Random Forest, do boosting e do SVM (acima).

**Clustering**

- [K-means](kmeans.md): scikit-learn e didática
- [DBSCAN](dbscan.md)
- [Clustering hierárquico](agglomerative.md)
- [Mistura de gaussianas](gaussian_mixture.md)

**Redução de dimensionalidade**

- [PCA](pca.md): scikit-learn e didática
- [t-SNE](tsne.md)

**Deteção de anomalias**

- [Isolation Forest](isolation_forest.md)
- [One-Class SVM](one_class_svm.md)
- [Local Outlier Factor (LOF)](lof.md)

## Modelo

````markdown
# <Nome do algoritmo>

## Intuição

A ideia em poucas frases, sem fórmulas.

## Matemática essencial

O mínimo de matemática para perceber o que o treino faz.

## Hiperparâmetros

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|

## Quando usar

## Armadilhas

## O que observar no treino

Que snapshots o algoritmo mostra e o que procurar neles.
````
