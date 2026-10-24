# Local Outlier Factor (LOF)

## Intuição

O LOF compara a **densidade à volta de cada ponto** com a densidade à volta dos seus vizinhos.
Um ponto numa zona muito mais vazia do que a dos vizinhos é anómalo. Por ser **local**,
encontra anomalias mesmo quando os grupos têm densidades muito diferentes. Um ponto um
bocadinho afastado de um grupo apertado é suspeito; o mesmo afastamento num grupo espalhado
é normal.

## Matemática essencial

Para cada ponto $x$, com os seus $k$ vizinhos mais próximos $N_k(x)$:

1. a **densidade de alcançabilidade local** é o inverso da distância média aos vizinhos
   (usando a "distância de alcançabilidade", que suaviza as distâncias muito pequenas):
   $\mathrm{lrd}(x) = 1 / \overline{d_{\mathrm{reach}}(x, N_k(x))}$;
2. o **LOF** é a razão entre a densidade dos vizinhos e a do próprio ponto:
   $$\mathrm{LOF}(x) = \frac{1}{k}\sum_{o \in N_k(x)} \frac{\mathrm{lrd}(o)}{\mathrm{lrd}(x)}$$

Perto de 1, o ponto é tão denso como os vizinhos; bem acima de 1, está numa zona mais vazia,
e é anómalo. O laboratório usa o modo de "deteção de novidades" do scikit-learn
(`novelty=True`), que sabe pontuar pontos novos. Os pontos de treino ficam com a pontuação
calculada no treino, porque pontuá-los de novo contaria cada ponto como vizinho de si próprio.

## Hiperparâmetros

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `n_neighbors` | Quantos vizinhos definem a densidade local. Poucos: sensível ao ruído; muitos: perde as anomalias em grupos pequenos | 20 |
| `contamination` | Que fração do treino se espera que seja anómala: põe o limiar de forma a marcar essa fração | 0,05 |

## Quando usar

- Quando os dados têm **grupos de densidades diferentes** e um limiar global (como o da
  Isolation Forest) não serve a todos.
- Com dados de tamanho médio e poucas features.

## Armadilhas

- **Escala das features:** usa distâncias. Padroniza.
- **O `n_neighbors` muda tudo:** tem de ser maior do que o tamanho dos pequenos grupos de
  anomalias que se querem encontrar (senão, elas são vizinhas umas das outras e parecem
  densas).
- **Lento com muitos dados:** cada pontuação procura os vizinhos de um ponto.
- **Muitas dimensões:** as distâncias, e a densidade com elas, perdem significado.

## O que observar no treino

Treina de uma vez. No ecrã **Treinar**, as cores mostram a pontuação de anomalia no plano. Ao
contrário da Isolation Forest, a pontuação sobe logo à volta dos grupos apertados e mais
devagar à volta dos espalhados. No ecrã **Avaliar**, compara a distribuição das pontuações e o
limiar com os dos outros detetores.
