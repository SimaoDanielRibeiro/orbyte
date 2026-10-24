# Regressão linear

## Intuição

A regressão linear prevê um **número** (o preço de uma casa, a nota de um exame) como uma soma
pesada das features mais uma constante: "cada quarto a mais acrescenta tanto ao preço, cada
quilómetro até ao centro tira tanto". Com uma só feature, o modelo é uma reta; com duas, um
plano; com mais, um hiperplano, a mesma ideia em mais dimensões.

Aprender é escolher os pesos que põem a reta o mais perto possível dos pontos de treino. "Perto"
mede-se com o **erro quadrático**: a distância vertical de cada ponto à reta, ao quadrado.

## Matemática essencial

Para um exemplo com features $x$, a previsão é

$$\hat y = w \cdot x + b$$

e o treino procura os $w$ e $b$ que minimizam o **erro quadrático médio** (EQM; no ecrã
Avaliar aparece com a sigla inglesa, MSE) no treino:

$$\text{EQM}(w, b) = \frac{1}{n} \sum_i (\hat y_i - y_i)^2$$

O quadrado faz os erros grandes contarem muito mais: errar por 10 custa 100 vezes mais do que
errar por 1. É por isso que um único ponto muito fora da tendência consegue puxar a reta.

Há duas maneiras de chegar aos melhores pesos.

**De uma vez, pela fórmula dos mínimos quadrados** (`linear_regression`, do scikit-learn). O
EQM é uma "taça" (uma função quadrática dos pesos), e o fundo da taça tem fórmula:
$w = (X^\top X)^{-1} X^\top y$. O scikit-learn calcula-a de forma numericamente mais estável,
sem inverter a matriz.

**Aos poucos, por gradiente descendente** (`linear_regression_gd`, a versão didática do
laboratório). Começa com
$w = 0$ e $b = 0$ e, em cada **época** (uma passagem por todos os exemplos de treino):

1. calcula os resíduos com os pesos atuais: $r_i = \hat y_i - y_i$;
2. calcula o gradiente, a direção em que o erro mais sobe:
   $\nabla_w = \frac{1}{n} X^\top r$ e $\nabla_b = \frac{1}{n} \sum_i r_i$ (é o gradiente de
   $\text{EQM}/2$, que tem o mesmo mínimo e uma fórmula mais limpa);
3. dá um passo no sentido contrário: $w \leftarrow w - \eta \, \nabla_w$, onde $\eta$ é a
   **taxa de aprendizagem**.

Como a taça só tem um fundo, o gradiente descendente chega sempre lá, desde que os passos não
sejam grandes demais. O limite depende da inclinação da taça na direção mais íngreme: com
features padronizadas, taxas até cerca de 1 costumam funcionar; acima disso, cada passo salta
por cima do fundo e cai mais alto do outro lado, e o erro **explode**.

No fim, o **R²** resume a qualidade: $R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i -
\bar y)^2}$, a fração da variação do alvo que o modelo explica. Prever sempre a média dá
$R^2 = 0$; um modelo pior do que isso tem $R^2$ negativo.

## Hiperparâmetros

Nas duas versões:

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `fit_intercept` | Se aprende a ordenada $b$ (sem ela, a reta é obrigada a passar pela origem) | sim |

Só no gradiente descendente:

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `learning_rate` | Tamanho de cada passo do gradiente descendente: pequeno demais arrasta-se, grande demais explode | 0,1 |
| `max_epochs` | Máximo de passagens pelos dados de treino | 1000 |
| `tol` | Pára quando o gradiente (medido na escala do alvo) fica abaixo disto: chegou ao fundo | 1e-6 |

## Quando usar

- Como **primeiro modelo** de qualquer regressão: é rápido e dá a referência que os modelos mais
  complexos têm de bater.
- Quando a relação entre as features e o alvo é aproximadamente uma reta, ou quando é preciso
  **explicar** a previsão: cada peso diz quanto o alvo muda por cada unidade da feature.

## Armadilhas

- **É linear.** Uma relação curva (o preço que sobe cada vez mais depressa com a área) fica mal
  descrita: os resíduos mostram um padrão em vez de ruído. Features polinomiais ajudam.
- **Outliers.** O erro ao quadrado dá a um único ponto extremo um peso enorme: vê os outliers na
  limpeza antes de treinar.
- **Escala das features.** O gradiente descendente sofre muito sem padronização: a taça fica um
  vale estreito e comprido, onde a taxa que serve a uma feature é grande demais para outra. O
  treino explode (aviso "a loss explodiu") ou demora milhares de épocas. Padroniza na
  preparação.
- **Features redundantes.** Se uma feature é combinação de outras (o one-hot com todas as
  categorias mais a ordenada, por exemplo), há infinitas soluções igualmente boas: os pesos
  deixam de se poder ler um a um e o gradiente descendente avança muito devagar ao longo do
  "fundo plano" da taça. As previsões continuam boas; a regularização (Ridge) resolve.
- **Extrapolar.** Fora da gama dos dados de treino, a reta continua a subir (ou a descer) sem
  limite, mesmo quando a realidade não o faz.

## O que observar no treino

No ecrã **Treinar**, a versão didática mostra, época a época:

- a **curva da loss** (o EQM no treino): desce depressa no início e depois cada vez mais devagar,
  à medida que se aproxima do fundo da taça;
- com uma só feature, a **reta a rodar e a deslocar-se** por cima da nuvem de pontos até se
  ajustar a ela; com mais features, o valor previsto em cada ponto do plano da projeção 2D.

Experimenta a taxa de aprendizagem nos dados sintéticos de regressão: com 0,01 o treino
arrasta-se e não chega ao fundo nas 1000 épocas; com 0,5 chega lá em pouco mais de 20; com 3
explode logo nas primeiras épocas, e o treino pára com o aviso "a loss explodiu". A
demonstração "Taxa de aprendizagem" do ecrã Laboratório desenha o caminho dos pesos na própria
taça.

A versão do scikit-learn treina de uma vez: compara os pesos das duas no ecrã **Avaliar**. Quando
o gradiente descendente converge, coincidem.
