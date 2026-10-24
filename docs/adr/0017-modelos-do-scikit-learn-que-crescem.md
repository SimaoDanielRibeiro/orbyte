# 0017 — Modelos do scikit-learn que se mostram a crescer, com o mesmo resultado final

- **Estado:** Proposta
- **Data:** 2026-10-23

## Contexto

Na fase 3, só os modelos didáticos mostravam o treino passo a passo; os do scikit-learn
treinavam de uma vez e davam uma só fotografia. Vários modelos do catálogo da fase 4 têm
passos naturais, e vê-los crescer ensina muito: a profundidade de uma árvore, as árvores de
uma floresta, as etapas do boosting, as iterações do EM. Duas condições:

1. o modelo final tem de ser igual ao que o scikit-learn daria a treinar normalmente (os
   testes comparam-nos);
2. quando ninguém está a ver (validação cruzada, importância por permutação,
   demonstrações do Laboratório, testes), o treino não pode ficar mais lento.

## Decisão

- **`GrowingSklearnModel`** (`models/adapters/growth.py`) treina aos bocados, e cada bocado
  é um passo do ML Lab, com a sua fotografia. Há três maneiras de crescer:
  - **por profundidade** (árvores de decisão): treina com profundidade 1, 2, 3…, até ao
    limite ou até a árvore deixar de crescer. Cada divisão escolhida não depende do limite
    de profundidade, por isso a árvore de profundidade d é o início da árvore completa.
    A loss é o erro no treino;
  - **por árvores** (Random Forest): o `warm_start` junta árvores às que já existem, em 20
    grupos. A loss é 1 − a pontuação "fora do saco" (OOB), em que cada linha é prevista
    só pelas árvores que não a viram no treino. O scikit-learn garante que o resultado é
    igual ao de treinar todas de uma vez;
  - **por etapas** (Gradient Boosting, AdaBoost): treina tudo e percorre as etapas com
    `staged_predict` e `staged_predict_proba`. A etapa m é exatamente o modelo com as
    primeiras m árvores. Se o treino for parado na etapa m, volta a treinar com m árvores,
    que são as mesmas, porque as seeds são tiradas pela mesma ordem.
- **A mistura de gaussianas** usa a mesma ideia com o `warm_start`: uma iteração do EM por
  passo, a continuar de onde estava.
- **`TrainingContext.observed`:** o contexto diz se alguém acompanha o treino. Sem ninguém
  a ver (o contexto nulo), estes modelos treinam de uma vez.
- **`step_limit`** na ficha técnica dá a barra de progresso: `max_depth` nas árvores,
  `n_estimators` no boosting e `max_iter` na mistura de gaussianas. A floresta cresce
  sempre em 20 grupos e não tem um hiperparâmetro que diga os passos.

## Consequências

- No ecrã Treinar, as árvores, as florestas, o boosting e a mistura de gaussianas mostram
  a fronteira a refinar-se e uma curva da loss, como os modelos didáticos.
- Um teste de contrato treina cada modelo com e sem observador e exige as mesmas
  previsões. Um modelo que cresça de outra maneira tem de passar o mesmo teste.
- Crescer custa mais do que treinar de uma vez. Uma árvore de profundidade d treina d
  árvores, e a floresta calcula a pontuação OOB até 20 vezes. Só se paga quando alguém
  está a ver.
- Os modelos sem passos naturais (SVM, KNN, Naive Bayes, t-SNE, DBSCAN, clustering
  hierárquico, Isolation Forest, One-Class SVM, LOF) continuam a treinar de uma vez. O
  ecrã Treinar diz que não há passos para rever e mostra o modelo final.
