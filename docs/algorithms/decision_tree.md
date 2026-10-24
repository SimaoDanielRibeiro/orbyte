# Árvore de decisão

## Intuição

Uma árvore de decisão é um fluxograma de perguntas de sim ou não sobre as features: "a
pétala tem menos de 2,45 cm?" Se sim, é *setosa*; se não, pergunta outra coisa. Cada
caminho acaba numa **folha**, que dá a resposta: a classe mais frequente dos exemplos de
treino que lá caíram (classificação) ou a sua média (regressão).

É dos modelos mais fáceis de explicar: o caminho de uma previsão lê-se como uma regra.

## Matemática essencial

A árvore constrói-se de cima para baixo. Em cada nó, experimenta todas as perguntas
"$x_j \le t$" (cada feature $j$, cada limiar $t$ entre dois valores) e escolhe a que deixa
os dois lados mais **puros**:

- na classificação, com o índice de **Gini**, $G = 1 - \sum_k p_k^2$ (0 quando só há uma
  classe), ou com a **entropia**, $H = -\sum_k p_k \log_2 p_k$;
- na regressão, com o **erro quadrático** à volta da média de cada lado (ou o erro
  absoluto à volta da mediana).

A escolha é **gananciosa**: a melhor pergunta agora, sem olhar para as seguintes. Repete-se
em cada lado até a folha ficar pura, até chegar à profundidade máxima ou até haver poucos
exemplos.

## Hiperparâmetros

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `max_depth` | Quantas perguntas seguidas (0 = sem limite). Árvores fundas decoram o treino | 5 |
| `min_samples_leaf` | O mínimo de exemplos em cada folha: valores maiores evitam folhas feitas para um só ponto | 1 |
| `criterion` | A medida de impureza: `gini` ou `entropy` (classificação); `squared_error` ou `absolute_error` (regressão) | `gini` / `squared_error` |

## Quando usar

- Quando é preciso **explicar** a decisão como regras.
- Com features de tipos e escalas misturados: as perguntas são sobre uma feature de cada
  vez, por isso a escala não importa.
- Como peça de modelos mais fortes: a Random Forest e o boosting são feitos de árvores.

## Armadilhas

- **Overfitting:** sem limite de profundidade, a árvore cresce até cada folha ter um só
  exemplo e acerta em 100% do treino. Limita `max_depth` ou aumenta `min_samples_leaf`.
- **Instável:** mudar umas linhas do treino pode mudar a primeira pergunta e, com ela, a
  árvore toda. As florestas resolvem isto com a média de muitas árvores.
- **Fronteiras em escada:** as perguntas são sempre sobre uma feature de cada vez, por isso
  a fronteira é feita de segmentos paralelos aos eixos. Uma fronteira diagonal precisa de
  muitos degraus.
- **Na regressão, não extrapola:** fora dos dados de treino, prevê o valor da folha da
  ponta, sempre o mesmo.

## O que observar no treino

No ecrã **Treinar**, cada passo é a árvore com mais um nível de profundidade. Vês a fronteira
a partir-se em retângulos cada vez mais pequenos, e a loss (a proporção de erros no treino,
ou o erro quadrático médio) a descer. Repara quando a fronteira começa a fazer retângulos à
volta de pontos isolados: é o overfitting a chegar. Em **O que o modelo aprendeu** (ecrã
Avaliar) estão as regras da árvore.
