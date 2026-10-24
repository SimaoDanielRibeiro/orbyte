# Regressão logística

## Intuição

Apesar do nome, a regressão logística **classifica**. Para cada exemplo, soma as features
multiplicadas por pesos (uma reta, ou um plano, a separar as classes) e transforma essa soma
numa probabilidade entre 0 e 1. Se a probabilidade de "sobreviveu" passar de 50%, a previsão é
"sobreviveu".

Aprender é escolher os pesos que dão probabilidades altas aos exemplos da classe certa e baixas
aos da classe errada. Cada peso diz quanto uma feature empurra para uma classe: no Titanic, o
peso de `Sex_female` é positivo (ser mulher aumentava a probabilidade de sobreviver) e o de
`Pclass` é negativo (classes mais baixas, 3.ª em vez de 1.ª, diminuíam-na).

## Matemática essencial

Com duas classes, para um exemplo com features $x$:

$$z = w \cdot x + b \qquad p(\text{classe } 1 \mid x) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

A função logística $\sigma$ (ou sigmoide) transforma qualquer número numa probabilidade:
$\sigma(0) = 0{,}5$, valores grandes dão quase 1, valores muito negativos dão quase 0. A
**fronteira de decisão** é onde $p = 0{,}5$, isto é, onde $w \cdot x + b = 0$: uma reta (ou um
plano), por isso o modelo é **linear**.

Os pesos escolhem-se a minimizar a **log-loss** (entropia cruzada) no treino, mais uma
penalização dos pesos grandes (regularização L2):

$$\text{custo}(w, b) = \sum_i -\big[y_i \log p_i + (1 - y_i)\log(1 - p_i)\big] + \frac{1}{2C}\lVert w \rVert^2$$

A log-loss castiga muito uma previsão errada feita com confiança (dizer 99% e falhar). O termo
com $C$ impede os pesos de crescerem sem limite: com $C$ pequeno a penalização é forte e a
fronteira fica mais "cautelosa".

Com mais de duas classes, há um vetor de pesos por classe, e a função **softmax** transforma as
somas em probabilidades que somam 1: $p_k = e^{z_k} / \sum_j e^{z_j}$.

Não há uma fórmula fechada para os melhores pesos: é preciso ir melhorando-os passo a passo até
o custo deixar de descer. O laboratório tem duas versões que minimizam **o mesmo custo** (e por
isso chegam aos mesmos pesos) por caminhos diferentes:

- `logistic_regression` (scikit-learn) usa o otimizador L-BFGS, que aproveita a curvatura do
  custo e chega lá em poucas dezenas de passos;
- `logistic_regression_gd` (didática, em NumPy) usa **gradiente descendente**: começa com os
  pesos a zero e, em cada **época** (uma passagem por todos os exemplos de treino), dá um passo
  pequeno contra o gradiente do custo.

O gradiente tem uma forma surpreendentemente simples: o erro de cada previsão, $p_i - y_i$,
multiplicado pelas features do exemplo, em média, mais a parte da regularização. Dividindo o
custo por $n$ (o que não muda o mínimo):

$$\nabla_w = \frac{1}{n}\sum_i (p_i - y_i)\, x_i + \frac{w}{C\,n} \qquad \nabla_b = \frac{1}{n}\sum_i (p_i - y_i)$$

$$w \leftarrow w - \eta\, \nabla_w \qquad b \leftarrow b - \eta\, \nabla_b$$

onde $\eta$ é a **taxa de aprendizagem**. Os exemplos bem classificados com confiança quase não
contribuem ($p_i \approx y_i$); são os mal classificados que empurram a fronteira. O treino pára
quando o gradiente fica praticamente nulo (abaixo de `tol`): está no fundo do custo.

## Hiperparâmetros

Nas duas versões:

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `C` | Inverso da força da regularização: menor = pesos mais pequenos e fronteira mais simples | 1,0 |
| `fit_intercept` | Se aprende a ordenada $b$ (sem ela, a fronteira passa pela origem) | sim |

Só no scikit-learn:

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `max_iter` | Máximo de passos do otimizador até convergir | 100 |
| `class_weight` | Com `balanced`, os erros nas classes raras contam mais | `none` |

Só no gradiente descendente:

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `learning_rate` | Tamanho de cada passo: pequeno demais arrasta-se, grande demais salta de um lado para o outro | 1,0 |
| `max_epochs` | Máximo de passagens pelos dados de treino | 2000 |
| `tol` | Pára quando o gradiente fica abaixo disto: chegou ao mínimo | 1e-6 |

## Quando usar

- Como **primeiro modelo** de qualquer problema de classificação: é rápido, raramente decora o
  treino e dá a referência que os modelos mais complexos têm de bater.
- Quando é preciso **explicar** a decisão: cada peso lê-se diretamente.
- Quando são precisas **probabilidades** razoáveis (para decidir com um limiar próprio, por
  exemplo).

## Armadilhas

- **É linear.** Não separa classes que precisam de uma fronteira curva (os dados sintéticos
  "luas" ou "círculos"). Features polinomiais ajudam, mas aumentam o risco de overfitting.
- **Escala das features.** Sem padronização, o otimizador demora muito mais a convergir (pode
  aparecer o aviso de convergência) e a regularização castiga mais as features com números
  pequenos. Padroniza na preparação.
- **Features muito correlacionadas** dividem o peso entre si de forma instável: os pesos deixam
  de se poder ler um a um, mesmo que as previsões continuem boas.
- **Classes desequilibradas:** com 95% de uma classe, o modelo pode quase nunca prever a outra.
  Vê a exatidão equilibrada e a recall, e experimenta `class_weight = balanced`.
- **"Os pesos não batem certo com o scikit-learn."** Por omissão, o scikit-learn pára cedo (passos
  que melhoram menos de 1e-4, no máximo 100 iterações), por isso os pesos das duas versões
  diferem lá para a terceira casa decimal. Com tolerâncias apertadas nas duas, coincidem até à
  quinta casa: o custo é o mesmo, só o caminho muda.
- **Taxa de aprendizagem grande demais** (só no gradiente descendente): a loss oscila em vez de
  descer, ou explode (o treino pára com o aviso "a loss explodiu"). Com features padronizadas, 1
  costuma servir; sem padronização, pode ser preciso 100 vezes menos.

## O que observar no treino

A versão **scikit-learn** treina de uma vez: no ecrã **Treinar** vês só a fronteira final e, no
**Avaliar**, os pesos aprendidos, as métricas no treino e no teste e a matriz de confusão.
Experimenta baixar o `C` e ver os pesos a encolher e a diferença entre treino e teste a mudar.

A versão **didática** treina época a época, e o ecrã **Treinar** fotografa-a pelo caminho:

- a **curva da loss** desce depressa nas primeiras épocas e depois devagar, e a exatidão no
  treino sobe aos saltos (cada salto é um exemplo que passou para o lado certo da fronteira);
- a **fronteira de decisão** (onde a probabilidade é 50%) começa no sítio errado, roda e
  desloca-se até separar as classes; as cores mostram a probabilidade, mais nítidas longe da
  fronteira à medida que os pesos crescem.

No fim, compara com a versão scikit-learn nos mesmos dados: a fronteira final e as métricas
ficam praticamente iguais. Experimenta a taxa de aprendizagem nas luas: com 0,05 o treino
arrasta-se e não chega ao mínimo nas 2000 épocas; com 20, a loss sobe e desce sem assentar;
com 50, a fronteira salta de um lado para o outro entre fotografias.
