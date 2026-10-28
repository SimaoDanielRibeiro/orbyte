# AdaBoost

## Intuição

O AdaBoost (*adaptive boosting*) foi o primeiro algoritmo de boosting a funcionar bem na
prática (Freund e Schapire, 1997). Treina modelos muito fracos, quase sempre árvores de uma
só pergunta (os *stumps*), um de cada vez. Depois de cada um, **aumenta o peso** dos exemplos
que ele falhou, para o seguinte se concentrar neles. No fim, cada modelo vota, com mais peso
para os que acertaram mais.

## Matemática essencial

Na classificação (algoritmo SAMME), todos os exemplos começam com o mesmo peso $w_i$. Em cada
etapa $m$:

1. treina o modelo fraco $h_m$ com os pesos atuais;
2. calcula o erro pesado $\varepsilon_m$ (a soma dos pesos dos exemplos falhados);
3. dá-lhe o peso de voto $\alpha_m = \nu \left(\log\frac{1 - \varepsilon_m}{\varepsilon_m} + \log(K - 1)\right)$,
   com $K$ classes: quanto menor o erro, mais conta;
4. multiplica o peso de cada exemplo falhado por $e^{\alpha_m}$ e normaliza.

A previsão é a classe com mais votos pesados, $\arg\max_k \sum_m \alpha_m\, [h_m(x) = k]$.

Na regressão (AdaBoost.R2), o erro de cada exemplo (proporcional, ao quadrado ou exponencial,
segundo `loss`) decide o seu peso na etapa seguinte, e a previsão é a **mediana pesada** das
previsões das árvores.

## Hiperparâmetros

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `n_estimators` | Quantos modelos fracos, no máximo. Pára antes se um deles acertar em tudo | 50 |
| `learning_rate` | Quanto conta cada modelo novo ($\nu$): menor exige mais modelos, mas é mais estável | 1,0 |
| `loss` | Na regressão, como o erro de cada exemplo vira o seu peso: `linear`, `square` ou `exponential` | `linear` |

## Quando usar

- Para **perceber o boosting**: os pesos dos exemplos tornam a ideia muito visível.
- Com dados limpos, onde as classes se separam com várias regras simples.

## Armadilhas

- **Sensível ao ruído e aos outliers:** um exemplo com o rótulo errado é falhado vezes sem
  conta e o seu peso cresce sem parar, a puxar os modelos seguintes.
- **Modelos fracos fortes demais** (árvores fundas) decoram o treino logo na primeira etapa,
  e o boosting não tem nada para corrigir.
- O gradient boosting costuma ser mais preciso e mais flexível (escolhe a loss).

## O que observar no treino

No ecrã **Treinar e avaliar**, cada passo é uma etapa. Com *stumps*, a primeira fronteira é uma só reta
paralela a um eixo. A cada etapa junta-se mais um corte, e a fronteira constrói-se aos
bocadinhos, como uma escada. A loss é a proporção de erros (ou o erro quadrático) no treino.
