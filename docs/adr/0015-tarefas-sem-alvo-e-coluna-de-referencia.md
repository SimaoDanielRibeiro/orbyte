# 0015 — Tarefas sem alvo no mesmo contrato, com uma coluna de referência para comparar

- **Estado:** Proposta
- **Data:** 2026-10-23

## Contexto

Até à fase 3, todos os modelos eram supervisionados: `fit(X, y)`, `predict` e, às vezes,
`predict_proba`. A fase 4 traz clustering, redução de dimensionalidade e deteção de
anomalias, e com elas três perguntas:

1. Como é que um modelo sem alvo, que não prevê nada (a PCA) ou que dá uma pontuação (um
   detetor de anomalias), cabe no contrato `LabModel` sem o partir em vários?
2. Como se avalia aprendizagem não supervisionada de forma honesta, se não há respostas
   certas?
3. As métricas passam a precisar de coisas diferentes: os valores reais e os previstos, as
   probabilidades, as features (a silhueta), as coordenadas novas (a confiança dos
   vizinhos) ou as pontuações (a ROC AUC das anomalias).

## Decisão

- **Um só contrato.** `fit(X, None)` nas tarefas sem alvo. `predict` devolve o grupo no
  clustering (-1 para o ruído) e 1/0 na deteção de anomalias. Duas capacidades novas:
  `transform(X)`, na redução de dimensionalidade (que não prevê), e `anomaly_scores(X)`
  (maior = mais anómalo). A base confirma as capacidades e responde com `CapabilityError`.
- **`ModelInfo.supports_new_data`:** o t-SNE só coloca no mapa os pontos com que treinou.
  Esse modelo só tem relatório do treino e não tem sonda.
- **A sonda** (ADR 0013) mostra o que cada tarefa tem: as probabilidades ou a classe, o
  grupo, a pontuação de anomalia. A redução de dimensionalidade não tem campo para
  desenhar: mostra o mapa das coordenadas novas.
- **Coluna de referência.** No ecrã Preparar, um alvo categórico escolhido com uma tarefa
  sem alvo:
  - não entra no treino: não é feature, e o serviço passa `None` ao modelo;
  - fica no `Dataset` como `y`, com as suas classes, só para os avaliadores. No clustering,
    dá a tabela das classes contra os grupos e o Rand ajustado. Na redução de
    dimensionalidade, dá as cores do mapa. Nas anomalias, a classe mais rara conta como
    anomalia e dá a precisão, o recall e a ROC AUC. É a preparação que lhe dá o número 1,
    contando a tabela toda, para "1 = anomalia" valer igual no treino, no teste e em
    todos os folds.
- **A referência pode estratificar a divisão.** O modelo nunca a vê; ela só reparte as
  linhas. Com anomalias raras, uma divisão aleatória pode deixar o teste sem nenhuma (com
  10 anomalias em 200 linhas, acontece em cerca de uma seed em cada dez), e sem elas não
  há precisão nem recall. Por isso, no ecrã Preparar, a estratificada vem escolhida
  sempre que há classes (no núcleo, o `SplitConfig` continua a ter a aleatória por
  omissão).
- **As métricas declaram o que precisam** (`needs`: `y_true`, `y_pred`, `y_proba`, `X`,
  `embedding`, `scores`). Quando falta alguma coisa (um modelo sem probabilidades, um
  clustering sem referência), a métrica fica de fora do relatório. As métricas que só
  descrevem (o número de grupos, a proporção de ruído, a proporção de anomalias marcadas)
  têm `higher_is_better = None`. Não há um sentido "melhor" nelas, por isso o diagnóstico
  treino vs. teste ignora-as, e as contagens aparecem sem casas decimais.

## Consequências

- Os serviços e a interface continuam a tratar todos os modelos da mesma maneira; o que
  muda é o avaliador, escolhido pelo tipo de tarefa.
- A avaliação não supervisionada fica com dois níveis. O interno vem sempre: grupos
  compactos e separados (silhueta, Davies-Bouldin, Calinski-Harabasz) e vizinhos
  preservados (confiança dos vizinhos). O externo só com referência: o acordo com as
  classes e as anomalias verdadeiras encontradas.
- Quem treina tem de lembrar-se de que, nas tarefas sem alvo, o `y` do `Dataset` é a
  referência e não vai para o modelo. O serviço de experiências trata disso, e um teste
  de integração confirma que o modelo nunca o recebe.
- A interface tem de explicar a ideia de referência: um aviso no ecrã Preparar e o tema
  "Coluna de referência" no painel "Porquê?".
