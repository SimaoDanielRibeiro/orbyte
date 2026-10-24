# t-SNE

## Intuição

O t-SNE (*t-distributed stochastic neighbor embedding*) desenha um **mapa 2D** em que pontos
parecidos ficam juntos. Não procura direções, como a PCA: procura posições no plano que
respeitem quem é vizinho de quem. É ótimo para ver grupos em dados com muitas features. Mas
o mapa engana em tudo o que não seja "estes estão perto": as distâncias entre grupos e o
tamanho de cada grupo não querem dizer nada.

## Matemática essencial

1. Para cada par de pontos nos dados originais, a probabilidade de serem vizinhos, $p_{ij}$:
   uma gaussiana à volta de cada ponto, com a largura escolhida para que o "número efetivo de
   vizinhos" seja a **perplexidade**.
2. No mapa, a mesma ideia com uma distribuição t de Student (de caudas longas), $q_{ij}$.
3. Procura posições no mapa que minimizam a divergência de Kullback-Leibler,
   $\mathrm{KL}(P \,\|\, Q) = \sum_{ij} p_{ij} \log \frac{p_{ij}}{q_{ij}}$, por gradiente
   descendente.

As caudas longas da distribuição t empurram os grupos diferentes para longe uns dos outros
no mapa. É por isso que os grupos aparecem tão bem separados, e também por isso que as
distâncias entre eles não são de confiança.

## Hiperparâmetros

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `perplexity` | Mais ou menos quantos vizinhos cada ponto tenta manter perto. Pequena: muitos grupinhos; grande: a estrutura global, mais esborratada. Tem de ser menor do que o número de linhas | 30 |
| `max_iter` | Máximo de passos de gradiente para arrumar o mapa | 1000 |

## Quando usar

- Para **explorar visualmente** dados com muitas features e ver se há grupos.
- Depois de uma PCA para umas dezenas de componentes, em dados muito grandes.

## Armadilhas

- **Não serve para pontos novos:** não aprende uma função, só coloca no mapa os pontos com
  que treinou. Por isso o ecrã Avaliar só tem relatório do treino.
- **Não leias distâncias nem tamanhos:** só a vizinhança é fiável.
- **Lento com muitos dados** (dezenas de milhares de linhas levam minutos).
- **A perplexidade muda o mapa todo:** experimenta vários valores antes de tirar conclusões.
- **Seeds diferentes dão mapas diferentes** (rodados, espelhados, com os grupos noutros
  sítios).

## O que observar no treino

Treina de uma vez (o scikit-learn não mostra as iterações). No ecrã **Treinar** vês o mapa,
pintado com a coluna de referência, se a escolheste. Compara com o mapa da PCA dos mesmos
dados: no t-SNE os grupos ficam muito mais separados, e o ecrã Avaliar mostra se os vizinhos
no mapa eram vizinhos nos dados (a "confiança dos vizinhos").
