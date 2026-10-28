# Perceptrão

## Intuição

O perceptrão (Rosenblatt, 1958) é o classificador linear mais antigo que aprende sozinho. Tem
uma reta (ou um plano) a separar as classes e só a mexe quando erra: se um exemplo cai do
lado errado, a reta roda e desloca-se um bocadinho na direção dele. Se acerta, nada muda.

É o antepassado direto das redes neuronais: um neurónio artificial é um perceptrão com uma
função mais suave no lugar do "sim ou não".

## Matemática essencial

Com as classes codificadas como $y = +1$ e $y = -1$, o perceptrão prevê o sinal de
$w \cdot x + b$. Em cada época percorre os exemplos e, para cada um:

$$\text{se } y\,(w \cdot x + b) \le 0: \qquad w \leftarrow w + \eta\, y\, x \qquad b \leftarrow b + \eta\, y$$

onde $\eta$ é a taxa de aprendizagem. A correção empurra a pontuação do exemplo no sentido
certo: $y\,(w + \eta y x) \cdot x = y\, w \cdot x + \eta \lVert x \rVert^2$, maior do que antes.

**Teorema de convergência:** se as classes se separarem com uma reta, o perceptrão encontra
uma fronteira sem erros ao fim de um número finito de correções. Se não se separarem, nunca
pára de corrigir.

Com mais de duas classes, treina-se um perceptrão por classe ("esta classe contra as
outras") e prevê-se a classe com a maior pontuação $w_k \cdot x + b_k$.

## Hiperparâmetros

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `learning_rate` | O tamanho de cada correção ($\eta$). Com os pesos a começar em zero, só muda a escala dos pesos e da ordenada | 1,0 |
| `max_epochs` | Máximo de passagens pelos dados de treino | 100 |
| `shuffle` | Percorrer os exemplos por uma ordem diferente em cada época (com a seed do modelo) | sim |
| `fit_intercept` | Se aprende a ordenada $b$ (sem ela, a fronteira passa pela origem) | sim |

## Quando usar

- Para **perceber** como um classificador aprende com os erros, e a origem das redes
  neuronais.
- Em dados que se separam com uma reta, onde encontra uma fronteira perfeita no treino.

Na prática, a regressão logística ou o SVM linear são melhores: dão probabilidades ou a
melhor margem, e lidam bem com classes que se sobrepõem.

## Armadilhas

- **Classes que não se separam com uma reta** (as luas, o Iris com versicolor e virginica):
  o perceptrão nunca pára de corrigir, e a fronteira final depende da época em que parou. O
  modelo avisa ("ainda errava no treino").
- **Não dá probabilidades:** só a classe. Por isso não há curva ROC nem log-loss.
- **Qualquer fronteira sem erros serve:** entre várias retas possíveis, fica com a primeira
  que encontrou, muitas vezes encostada a uma das classes. O SVM, pelo contrário, procura a
  que fica mais longe das duas.
- **A ordem dos exemplos conta:** com `shuffle`, outra seed dá outra fronteira.

## O que observar no treino

No ecrã **Treinar e avaliar**, cada passo é uma época. A curva da loss mostra a proporção de exemplos
corrigidos nessa época. Em dados separáveis chega a zero e o treino pára; nos outros
oscila sem nunca chegar lá. A fronteira salta de época para época, porque só se mexe com os
erros: é o contraste com a descida suave do gradiente descendente da regressão logística.
