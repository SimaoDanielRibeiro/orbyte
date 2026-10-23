# 0012 — Contrato LabModel completo desde a fase 2, e avaliação por registos

- **Estado:** Proposta
- **Data:** 2026-10-23

## Contexto

A fase 2 traz o primeiro modelo (regressão logística do scikit-learn), mas o contrato
`LabModel` tem de servir também a fase 3 (modelos didáticos que treinam passo a passo e
emitem snapshots), a fase 4 (todo o catálogo clássico) e a fase 6 (redes em PyTorch). Mudar o
contrato depois obrigaria a mexer em todos os modelos já escritos.

Do lado da avaliação, o plano pede um avaliador por tipo de tarefa que recolha sozinho as
métricas registadas, e métricas que sejam funções puras. Há ainda um problema prático: os
testes tratam os avisos como erros, e o scikit-learn avisa com frequência (por exemplo,
quando o otimizador não converge).

## Decisão

- **`LabModel[P]`** é genérico na classe dos hiperparâmetros: cada modelo tem
  `self.hyperparameters` com o seu tipo exato. A ficha técnica (`ModelInfo`) é um atributo de
  classe, e os modelos registam-se em `MODELS` com descoberta automática (ADR 0007).
- **`fit(X, y, context)`** recebe desde já um `TrainingContext` (um `Protocol` em
  `core/training/context.py`). Na fase 2 é um contexto que não faz nada; na fase 3 chega o que
  publica os passos no `EventBus` e grava os snapshots, sem mudar a assinatura.
- As verificações comuns (formas dos arrays, modelo já treinado, número de features,
  capacidades declaradas) vivem na classe base, que chama `_fit`, `_predict` e
  `_predict_proba`. O adaptador `SklearnModel` implementa-os uma vez para qualquer estimador.
- Na classificação, o alvo são os números das classes (0, 1, 2…); `classes()` diz que classes
  o modelo viu no treino, e o avaliador alinha as probabilidades com todas as classes (zero
  para as que não viu).
- Os **avisos das bibliotecas** durante o treino são capturados e transformados em avisos
  didáticos do modelo (`Notice`), com uma explicação do que fazer.
- Cada **métrica** é uma função pura registada em `METRICS` com os tipos de tarefa, se maior é
  melhor, se precisa de probabilidades e se se mostra em percentagem; a explicação vive em
  `evaluation/explanations.py` e um teste de contrato exige-a. Cada **avaliador** regista-se
  em `EVALUATORS` com os seus tipos de tarefa; `evaluator_for(tarefa)` escolhe-o.

## Consequências

- Acrescentar um modelo é escrever a ficha, os hiperparâmetros e dois ou três métodos (só
  `build_estimator`, se for do scikit-learn); os testes de contrato testam-no sozinhos.
- A fase 3 acrescenta o contexto real e os modelos iterativos sem mudar os que já existem.
- Um aviso do scikit-learn nunca se perde nem faz falhar o treino: aparece no ecrã Avaliar.
- Uma métrica nova aparece em todos os relatórios do seu tipo de tarefa, com explicação, sem
  mexer no avaliador nem na interface.
- O genérico com `ClassVar` acrescenta alguma complexidade de tipos; em troca, o mypy confirma
  que cada modelo usa os seus próprios hiperparâmetros.
