# 0010 — Datasets de exemplo versionados, com descarga verificada

- **Estado:** Proposta
- **Data:** 2026-10-23

## Contexto

O plano prevê quatro datasets de exemplo (Iris, Wine, Titanic e California Housing) em
`data/samples/` ("datasets de exemplo pequenos, versionados") e um script para os descarregar.
O critério da fase 1 usa o Titanic e o da fase 2 usa o Iris.

O scikit-learn traz o Iris e o Wine, mas só entra na fase 2; não traz o Titanic; e o
California Housing descarrega-se da internet. Depender da rede para abrir a app ou correr os
testes tornava-os frágeis.

## Decisão

- Os quatro ficheiros ficam no Git, em `data/samples/` (cerca de 470 KB no total; o California
  Housing, com 20 640 linhas, vai comprimido em `.csv.gz`).
- `scripts/download_samples.py` volta a criá-los a partir das origens públicas (repositório do
  pandas, seaborn-data, UCI e o repositório do livro de A. Géron) e confirma o SHA-256 de cada
  descarga: se a origem mudar, o script pára.
- Cada exemplo declara regras por coluna (`validation/schemas.py`) e o número de linhas; a
  leitura confirma-as, por isso um ficheiro alterado por engano dá um erro claro.
- `.gitattributes` marca `data/samples/` como binário, para o Git não trocar os fins de linha
  no Windows.

## Consequências

- A app e os testes funcionam sem rede, e os resultados dos testes não dependem de versões
  externas dos dados.
- O California Housing é a versão do livro de Géron (com 207 nulos e a coluna
  `ocean_proximity`), mais rica para aprender a preparar dados do que a do scikit-learn.
- O Iris vem do seaborn-data, que tem os valores corrigidos de Fisher (a cópia da UCI tem dois
  erros conhecidos).
- Acrescentar um exemplo exige três passos: o ficheiro, a entrada no script e o `SampleInfo`.
