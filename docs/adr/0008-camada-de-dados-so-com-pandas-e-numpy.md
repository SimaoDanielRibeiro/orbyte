# 0008 — Camada de dados só com pandas e NumPy

- **Estado:** Proposta
- **Data:** 2026-10-23

## Contexto

A fase 1 entrega as nove etapas da camada de dados, incluindo a divisão, as
transformações (imputar, codificar, escalar), os dados sintéticos e a projeção 2D. O
scikit-learn faz tudo isto (`train_test_split`, `SimpleImputer`, `OneHotEncoder`,
`StandardScaler`, `make_moons`, `PCA`), mas o roadmap só o traz na fase 2: cada fase
introduz poucas tecnologias, para se aprenderem uma de cada vez. As tecnologias da fase 1
são o pandas, o NumPy e o Plotly.

## Decisão

A camada de dados implementa as suas etapas com pandas e NumPy, sem scikit-learn:

- **transformações** com o contrato `Transform` (`fit` só no treino, `transform` em
  qualquer parte), escritas em pandas: imputação, one-hot, booleanos, padronização,
  min-max, partes das datas, polinomiais e intervalos;
- **divisão** aleatória, estratificada (método do maior resto) e temporal em NumPy, sempre
  com a seed da experiência;
- **dados sintéticos** com a mesma geometria dos `make_*` do scikit-learn;
- **PCA didática** por decomposição em valores singulares, com o mesmo critério de sinal do
  scikit-learn.

Na fase 2, quando o scikit-learn entra, testes comparam estas implementações com as dele no
mesmo dataset ("As implementações didáticas têm um teste extra" em `docs/architecture.md`).

## Consequências

- A fase 1 respeita o roadmap e ensina o pandas a sério: `fillna`, `value_counts`, `groupby`,
  `category`, cópias e Copy-on-Write aparecem no código da própria camada.
- Cada transformação é curta e lê-se de uma vez; a linhagem descreve o que aprendeu ("Age
  preenchida com a mediana do treino, 28,0").
- Os nomes das colunas ficam preservados até ao fim sem configuração extra (regra 6 da camada),
  porque as transformações trabalham com DataFrames.
- Há código que o scikit-learn já tem. Se um dia for preciso algo que só ele faz bem (por
  exemplo, `TargetEncoder`), basta uma nova `Transform` que o envolva: o contrato não muda.
- O desempenho é o do pandas, mais do que suficiente para datasets até ao limite de 200 MB.
