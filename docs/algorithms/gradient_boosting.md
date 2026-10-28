# Gradient boosting

## Intuição

O boosting junta modelos fracos (árvores pequenas) **em sequência**, cada um a corrigir os
erros dos anteriores. Começa com uma previsão simples (a média, na regressão). A primeira
árvore aprende o que essa previsão falhou, a segunda o que ficou por acertar, e assim por
diante. É como um aluno que, a cada teste, estuda só a matéria em que errou.

Ao contrário da Random Forest (árvores independentes, em paralelo, a votar), aqui cada árvore
depende das anteriores.

## Matemática essencial

O modelo é uma soma de árvores:

$$F_M(x) = F_0 + \nu \sum_{m=1}^{M} h_m(x)$$

Em cada etapa $m$, calcula-se o **gradiente negativo** da loss em relação à previsão atual,
para cada exemplo: o "resíduo" generalizado. Com o erro quadrático, é o resíduo de verdade,
$r_i = y_i - F_{m-1}(x_i)$. A árvore $h_m$ ajusta-se a esses resíduos e junta-se ao
modelo, multiplicada pela taxa de aprendizagem $\nu$. É o gradiente descendente feito
no espaço das funções: cada árvore é um passo na direção que mais faz descer a loss.

Na classificação, $F$ dá o *log-odds* (passado pela sigmoide, ou pela softmax), e a loss é a
log-loss.

## Hiperparâmetros

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `n_estimators` | Quantas árvores (etapas). Ao contrário da floresta, árvores a mais acabam por decorar o treino | 100 |
| `learning_rate` | Quanto conta cada árvore ($\nu$). Pequena: precisa de mais árvores, mas generaliza melhor | 0,1 |
| `max_depth` | A profundidade de cada árvore. Árvores pequenas (2 a 5 níveis) bastam: o poder vem de as somar | 3 |
| `subsample` | A fração das linhas que cada árvore vê. Abaixo de 1, é o *stochastic gradient boosting*: menos overfitting | 1,0 |

## Quando usar

- Quando se quer a **melhor precisão** em dados em tabela: com boa afinação, o gradient
  boosting (e as suas versões rápidas, como XGBoost e LightGBM) ganha a maioria das
  competições.
- Em classificação e em regressão, com features de escalas diferentes (são árvores).

## Armadilhas

- **Overfitting com etapas a mais:** acompanha o erro de validação, ou baixa a
  `learning_rate` e aumenta as etapas.
- **`learning_rate` e `n_estimators` andam juntos:** metade da taxa pede mais ou menos o
  dobro das árvores.
- **Sequencial:** cada árvore espera pela anterior, por isso treina mais devagar do que uma
  floresta do mesmo tamanho.
- **Sensível ao ruído do alvo:** continua a perseguir resíduos que são só ruído.

## O que observar no treino

No ecrã **Treinar e avaliar**, cada passo é uma etapa (uma árvore a mais). A loss é a do treino. Desce
sempre, porque cada árvore corrige o treino, e é por isso que o teste é que diz quando parar.
A fronteira (ou a previsão) começa grosseira e vai ganhando pormenor etapa a etapa. Na
regressão com uma feature, vês a função em degraus a aproximar-se dos pontos.
