# 0018 — Explicações independentes do modelo, escritas em NumPy

- **Estado:** Proposta
- **Data:** 2026-10-23

## Contexto

O ecrã Explicar da fase 4 responde a três perguntas: que features o modelo usa (importância
por permutação), como a previsão muda com uma feature (dependência parcial e curvas ICE) e
porque é que uma previsão saiu assim. As alternativas:

- **`sklearn.inspection`** (`permutation_importance`, `partial_dependence`) espera
  estimadores do scikit-learn. Os nossos modelos são `LabModel`, e alguns são didáticos, em
  NumPy; na fase 6 virão os do PyTorch.
- **SHAP ou LIME** dão explicações locais mais completas, mas são dependências grandes,
  com conceitos que ainda não fazem falta (ADR 0003: stack enxuta).

## Decisão

Implementações próprias em `core/explainability/`, que só usam o que todos os modelos têm
(`predict` e, se houver, `predict_proba`):

- **Importância por permutação:** baralha cada feature no teste, 5 vezes (no máximo 2000
  linhas), e mede quanto a métrica principal piora. Usa a seed do run, por isso o mesmo
  run dá sempre a mesma importância.
- **Dependência parcial:** uma grelha de 30 valores, do 5.º ao 95.º percentil da feature;
  em cada valor, a média das previsões (no máximo 500 linhas), mais as curvas ICE de 20
  linhas.
- **Uma previsão:** nos modelos lineares, o contributo é peso × valor (método
  `"weights"`, exato). Nos outros, troca-se a feature pela média do treino e vê-se quanto
  a previsão muda (método `"replace"`, aproximado). O resultado diz que método usou.

O serviço de análise calcula-as quando o ecrã as pede e guarda-as por run.

## Consequências

- As explicações funcionam com todos os modelos do catálogo, com os didáticos e com os
  futuros do PyTorch, e o código é curto o suficiente para ler e aprender.
- Os limites conhecidos ficam escritos no painel "Porquê?":
  - com features correlacionadas, baralhar uma custa pouco, porque a outra a substitui;
  - a grelha da dependência parcial cria combinações que não existem nos dados;
  - as features são as colunas depois da preparação (as colunas 0/1 de uma categoria
    contam em separado);
  - o método `"replace"` não reparte as interações de forma justa, como faria o SHAP.
- Se um dia fizer falta o SHAP, entra atrás do mesmo serviço, como uma explicação a mais,
  sem mudar o ecrã.
