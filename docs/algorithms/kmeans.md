# K-means

## Intuição

O k-means divide os dados em k grupos, cada um representado pelo seu **centróide** (o ponto
médio do grupo). Cada ponto pertence ao grupo do centróide mais próximo. O algoritmo começa
com k centróides em sítios escolhidos e repete dois passos simples até deixarem de mexer:
atribuir cada ponto ao centróide mais próximo e mover cada centróide para o meio do seu grupo.

## Matemática essencial

O k-means procura os centróides $\mu_1, \dots, \mu_k$ que minimizam a **inércia**, a soma das
distâncias ao quadrado de cada ponto ao centróide do seu grupo:

$$J = \sum_i \lVert x_i - \mu_{c(i)} \rVert^2$$

O algoritmo de Lloyd alterna:

1. **atribuir**: $c(i) = \arg\min_j \lVert x_i - \mu_j \rVert^2$;
2. **atualizar**: $\mu_j = $ média dos pontos com $c(i) = j$.

Cada passo nunca aumenta $J$, por isso o algoritmo pára sempre. Mas pára num **mínimo local**,
que depende de onde os centróides começaram. O **k-means++** escolhe os pontos de partida
afastados uns dos outros: o primeiro ao acaso, e cada um dos seguintes com probabilidade
proporcional ao quadrado da distância ao centróide mais próximo já escolhido.

A versão do scikit-learn repete tudo `n_init` vezes, com partidas diferentes, e fica com a de
menor inércia. A versão didática (`kmeans_numpy`) corre uma só vez, para se ver o caminho.

## Hiperparâmetros

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `n_clusters` | Quantos grupos (o $k$). O k-means encontra sempre k grupos, mesmo que os dados não os tenham | 3 |
| `init` | Onde começam os centróides: `k-means++` (espalhados) ou `random` (pontos ao acaso) | `k-means++` |
| `max_iter` | Máximo de repetições de atribuir e atualizar | 300 |
| `tol` | Pára quando os centróides quase não mexem (uma fração da variância dos dados) | $10^{-4}$ |
| `n_init` | No scikit-learn, quantas partidas diferentes experimentar | 10 |

## Quando usar

- Para grupos **redondos e de tamanhos parecidos**, como as nuvens (blobs).
- Como primeiro clustering a experimentar: é rápido e fácil de explicar.

## Armadilhas

- **É preciso escolher o k.** Compara a silhueta de vários valores, ou procura o "cotovelo"
  na curva da inércia (que desce sempre com mais grupos).
- **Grupos alongados ou encaixados** (as luas, os círculos) ficam mal divididos: o k-means
  corta tudo em regiões convexas à volta dos centróides. O DBSCAN ou a mistura de gaussianas
  lidam melhor.
- **Escala das features:** as distâncias misturam todas as features. Padroniza.
- **Más partidas:** com `random`, dois centróides podem começar no mesmo grupo. A versão
  didática fica com o resultado dessa partida (por isso a do scikit-learn repete várias vezes).
- **Grupos vazios:** se um centróide ficar sem pontos, a versão didática mantém-no onde
  estava e avisa. O scikit-learn muda-o para longe.

## O que observar no treino

Na versão didática, o ecrã **Treinar e avaliar** mostra cada iteração: as cruzes (os centróides) andam
até ao meio dos seus grupos, e as zonas coloridas (o grupo que cada ponto do plano receberia)
mudam com elas. A loss é a inércia, que só desce, e em poucas iterações estabiliza. Com
`random`, outra seed pode acabar noutra divisão.
