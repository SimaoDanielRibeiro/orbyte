# Random Forest

## Intuição

Uma árvore de decisão sozinha decora o treino e muda muito com pequenas alterações dos dados.
Mas muitas árvores **diferentes**, cada uma com os seus erros, a votar em conjunto, erram
muito menos: os erros de umas compensam os das outras. É a "sabedoria das multidões".

Para as árvores serem diferentes, a Random Forest usa dois sorteios: cada árvore treina com
uma amostra das linhas, e cada nó só pode escolher entre algumas features sorteadas.

## Matemática essencial

Para cada uma das $B$ árvores:

1. sorteia $n$ linhas **com reposição** (o *bootstrap*): umas aparecem várias vezes e cerca
   de um terço ($e^{-1} \approx 37\%$) fica de fora;
2. constrói uma árvore funda; em cada nó, só considera um subconjunto sorteado das features
   (por exemplo, $\sqrt{p}$ de $p$ features).

A previsão é o voto da maioria (classificação) ou a média (regressão). Fazer a média de
modelos com pouca correlação entre si reduz a variância sem aumentar o viés: é o *bagging*.

**Erro fora do saco (OOB):** cada linha ficou de fora de cerca de um terço das árvores. As
previsões dessas árvores para essa linha dão uma estimativa do erro em dados novos, sem
gastar um conjunto de validação.

## Hiperparâmetros

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `n_estimators` | Quantas árvores. Mais árvores não causam overfitting: só tornam o treino mais lento, e a partir de certo ponto deixam de ajudar | 100 |
| `max_depth` | A profundidade de cada árvore (0 = sem limite). Árvores fundas são normais numa floresta | 0 |
| `max_features` | Quantas features cada nó pode experimentar: `sqrt`, `log2` ou `all`. Menos features dão árvores mais diferentes | `sqrt` (classificação) / `all` (regressão) |

## Quando usar

- Como **primeiro modelo forte** para dados em tabela: funciona bem com quase nenhuma
  afinação, não precisa de padronização e mede a importância das features.
- Quando há interações e não linearidades que um modelo linear não apanha.

## Armadilhas

- **Mais lenta e mais pesada** do que uma árvore: guarda todas as árvores.
- **Difícil de explicar:** já não é uma regra legível; usa a importância das features e a
  dependência parcial (ecrã Explicar).
- **A importância "das impurezas"** (a do scikit-learn) favorece features com muitos
  valores. A importância por permutação, medida no teste, é mais honesta.
- **Na regressão, não extrapola:** a média de árvores fica presa ao intervalo do treino.

## O que observar no treino

No ecrã **Treinar e avaliar**, cada passo junta mais árvores. A loss é o **erro OOB**: 1 − exatidão OOB
(classificação) ou 1 − R² OOB (regressão). Desce depressa com as primeiras dezenas de árvores
e depois estabiliza: é por isso que mais árvores não fazem mal, só deixam de ajudar. A
fronteira começa irregular (poucas árvores) e fica mais suave à medida que a média se
estabiliza.
