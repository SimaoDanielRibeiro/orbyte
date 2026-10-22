# Decisões de arquitetura (ADR)

Cada decisão importante fica num ficheiro curto, com contexto, decisão e consequências.
Daqui a meses vais saber porque é que algo foi feito assim.

| N.º | Decisão | Estado |
|---|---|---|
| [0001](0001-pytorch-em-vez-de-tensorflow.md) | PyTorch em vez de TensorFlow | Aceite |
| [0002](0002-pandas-isolado-em-core-data.md) | pandas isolado em `core/data/` | Aceite |
| [0003](0003-stack-enxuta-extensoes-atras-de-contratos.md) | Stack enxuta, extensões atrás de contratos | Aceite |
| [0004](0004-sqlite-com-sqlite3.md) | SQLite com `sqlite3` | Aceite |
| [0005](0005-interface-simples-primeiro-react-no-fim.md) | Interface simples (Streamlit) primeiro, React no fim | Aceite |
| [0006](0006-seeds-derivadas-por-nome.md) | Seeds derivadas por nome a partir de uma seed raiz | Proposta |
| [0007](0007-descoberta-automatica-de-componentes.md) | Descoberta automática dos componentes registados | Proposta |

## Como escrever uma ADR

1. Copia o modelo abaixo para `NNNN-titulo-curto.md`, com o próximo número livre.
2. Começa com o estado **Proposta**; passa a **Aceite** ou **Rejeitada** depois de decidir.
   Uma ADR aceite não se reescreve: se a decisão mudar, escreve uma nova e marca a antiga
   como **Substituída por NNNN**.
3. Acrescenta-a à tabela acima.

````markdown
# NNNN — Título da decisão

- **Estado:** Proposta
- **Data:** AAAA-MM-DD

## Contexto

Que problema há para resolver e que forças estão em jogo.

## Decisão

O que se decidiu, numa ou duas frases claras.

## Consequências

O que fica mais fácil, o que fica mais difícil e o que passa a ser preciso fazer.
````
