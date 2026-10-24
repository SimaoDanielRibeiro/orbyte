# Isolation Forest

## Intuição

Para isolar um ponto, faz-se cortes ao acaso: escolhe-se uma feature e um valor qualquer entre
o mínimo e o máximo, e divide-se. Repete-se em cada lado até o ponto ficar sozinho. Um ponto
**anómalo**, longe dos outros, fica isolado com **poucos cortes**. Um ponto normal, no meio de
uma nuvem, precisa de muitos. A Isolation Forest faz isto com muitas árvores e usa a média do
número de cortes.

É a ideia ao contrário da maioria dos detetores: em vez de descrever os pontos normais,
procura os que são fáceis de separar.

## Matemática essencial

Cada árvore de isolamento é construída com uma amostra das linhas (256, por omissão), com
cortes ao acaso. A **profundidade** $h(x)$ a que o ponto $x$ fica sozinho, em média nas
árvores, dá a pontuação:

$$s(x) = 2^{-\overline{h(x)} / c(n)}$$

onde $c(n)$ é a profundidade média esperada numa árvore de $n$ pontos (para normalizar).
Perto de 1 é uma anomalia; bem abaixo de 0,5, um ponto normal. O limiar escolhe-se para
marcar a fração de anomalias esperada (`contamination`).

## Hiperparâmetros

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `n_estimators` | Quantas árvores de isolamento fazem a média | 100 |
| `contamination` | Que fração do treino se espera que seja anómala: põe o limiar de forma a marcar essa fração | 0,05 |

## Quando usar

- Como **primeiro detetor** de anomalias: rápido, funciona com muitas features e muitos dados,
  e não precisa de padronização (os cortes são numa feature de cada vez).

## Armadilhas

- **A `contamination` decide quantos são marcados:** se a fração verdadeira for outra, o
  limiar fica no sítio errado. A pontuação é a mesma; só o corte muda.
- **Anomalias em grupo** (várias anomalias juntas) já não são tão fáceis de isolar.
- **Anomalias que só se veem numa combinação de features** (um valor normal de cada feature,
  mas uma combinação rara) são difíceis com cortes paralelos aos eixos.

## O que observar no treino

Treina de uma vez. No ecrã **Treinar**, as cores mostram a pontuação de anomalia em cada ponto
do plano: as zonas longe dos dados ficam mais "anómalas". No ecrã **Avaliar** estão a
distribuição das pontuações com o limiar e, se escolheste a coluna de referência (por exemplo,
no gerador "Anomalias"), a precisão, a recall e a ROC AUC.
