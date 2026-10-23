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

Não há uma fórmula fechada para os melhores pesos: o scikit-learn usa o otimizador L-BFGS, que
melhora os pesos passo a passo até o custo deixar de descer.

## Hiperparâmetros

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `C` | Inverso da força da regularização: menor = pesos mais pequenos e fronteira mais simples | 1,0 |
| `max_iter` | Máximo de passos do otimizador até convergir | 100 |
| `class_weight` | Com `balanced`, os erros nas classes raras contam mais | `none` |
| `fit_intercept` | Se aprende a ordenada $b$ (sem ela, a fronteira passa pela origem) | sim |

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

## O que observar no treino

Esta versão (scikit-learn) treina de uma vez: no ecrã **Avaliar** vês os pesos aprendidos, as
métricas no treino e no teste e a matriz de confusão. Experimenta baixar o `C` e ver os pesos a
encolher e a diferença entre treino e teste a mudar.

Na fase 3 chega a versão didática em NumPy, que treina por gradiente descendente e mostra, época
a época, a loss a descer e a fronteira de decisão a mover-se até ficar parecida com a do
scikit-learn.
