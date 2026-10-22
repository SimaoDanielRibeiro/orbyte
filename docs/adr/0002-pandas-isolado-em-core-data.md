# 0002 — pandas isolado em `core/data/`

- **Estado:** Aceite
- **Data:** 2026-10-22

## Contexto

Os dados reais chegam como tabelas mistas (texto, números, datas, valores em falta), e o
pandas é a ferramenta certa para os ler, explorar e limpar. Os modelos, o treino e a
avaliação precisam de matrizes numéricas, que são o terreno do NumPy.

Se o pandas se espalhar pelo projeto, cada camada passa a lidar com DataFrames, índices e
tipos de coluna; a fronteira entre "preparar dados" e "calcular" desaparece, e trocar de
biblioteca de tabelas fica impossível.

## Decisão

O pandas só é importado em `core/data/`. A camada de dados termina em `boundary.py`, que
entrega um `Dataset` com arrays NumPy e metadados: nomes das features, esquema das colunas e
linhagem das operações. Um teste de arquitetura falha se algum ficheiro fora de
`core/data/` (incluindo a interface) importar o pandas.

## Consequências

- Fronteira clara: sabes sempre onde procurar qualquer operação sobre dados, e o resto do
  sistema recebe sempre o mesmo objeto, venha o dado de onde vier.
- Substituível: trocar o pandas pelo Polars afeta só esta pasta.
- Cada etapa da camada testa-se sozinha com tabelas pequenas feitas à mão.
- Os nomes das colunas têm de viajar explicitamente nos metadados do `Dataset`, para
  chegarem à explicabilidade e à interface.
- A interface não pode mostrar DataFrames vindos do núcleo: recebe estruturas simples
  (listas, dicionários, arrays) dos serviços.
