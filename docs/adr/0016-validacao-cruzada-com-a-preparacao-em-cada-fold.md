# 0016 — Validação cruzada com as transformações aprendidas de novo em cada fold

- **Estado:** Proposta
- **Data:** 2026-10-23

## Contexto

A fase 4 traz a validação cruzada e a curva de aprendizagem. A forma mais curta seria
preparar os dados uma vez e partir as matrizes do treino em k folds. Só que a imputação, a
escala e a codificação já teriam aprendido com o treino todo, incluindo o fold que depois
serve para avaliar. É uma fuga de dados pequena, mas contradiz a regra que a camada de
dados ensina ("as transformações aprendem só com o treino").

O scikit-learn resolve isto com um `Pipeline` dentro do `cross_val_score`. As nossas
transformações são da camada de dados, em pandas e NumPy (ADR 0008), e não transformadores
do scikit-learn.

## Decisão

- **`prepare_folds(tabela, configuração, seeds, k)`** devolve um `Dataset` por fold:
  1. faz a divisão como `prepare`, com a mesma seed, por isso o teste é o mesmo e fica de
     fora;
  2. parte as linhas de treino em k folds com a seed `"folds"`. Se houver classes (o alvo
     ou a referência), a partição é estratificada: as linhas de cada classe são baralhadas
     e distribuídas à vez pelos folds;
  3. em cada fold, corre as etapas 7 a 9 (transformação, projeção, saída) só com as linhas
     de treino desse fold. A avaliação é o fold que ficou de fora, e a linhagem regista
     "Validação cruzada, fold i de k".
- **`preparation.py`** foi partido em `_select` (etapas 4 e 5), `_split` (6) e `_assemble`
  (7 a 9), para `prepare` e `prepare_folds` usarem as mesmas peças.
- **`cross_validate`** treina um modelo novo por fold, com os mesmos hiperparâmetros e a
  mesma seed, e mede a métrica principal no treino e no fold de fora. **`learning_curve`**
  treina com frações crescentes do treino de cada fold (subconjuntos estratificados) e
  mede o mesmo.
- **Leituras didáticas:** variância alta (treino e avaliação a mais de 10 pontos), mais
  dados ainda ajudariam, patamar, e resultados instáveis entre folds.

## Consequências

- O fold de avaliação nunca influencia a mediana, as categorias, a média ou o desvio das
  transformações, e o teste nunca entra na validação cruzada: continua a ser a última
  verificação.
- Custa k preparações e k treinos (mais as frações da curva). Por isso o ecrã Avaliar só os
  calcula quando se carrega em "Calcular", e o serviço de análise guarda o resultado de
  cada run.
- Só faz sentido nas tarefas com alvo. Nas outras, o ecrã Avaliar não mostra este
  separador.
