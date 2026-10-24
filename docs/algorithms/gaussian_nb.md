# Naive Bayes gaussiano

## Intuição

Para cada classe, o Naive Bayes aprende como são os seus exemplos: a média e a variância de
cada feature. Uma flor *setosa* tem pétalas curtas, com pouca variação; uma *virginica* tem
pétalas compridas. Para classificar uma flor nova, pergunta: "com estas medidas, de que
classe é mais provável que venha?"

É **ingénuo** (*naive*) porque supõe que, dentro de cada classe, as features não têm nada a
ver umas com as outras. Quase nunca é verdade, mas funciona surpreendentemente bem.

## Matemática essencial

A regra de Bayes dá a probabilidade de cada classe $c$ dado o ponto $x$:

$$p(c \mid x) \propto p(c)\, p(x \mid c)$$

$p(c)$ é a frequência da classe no treino. Com a hipótese ingénua, $p(x \mid c)$ é o produto
das probabilidades de cada feature, cada uma com a sua curva normal (gaussiana):

$$p(x \mid c) = \prod_j \frac{1}{\sqrt{2\pi\sigma_{cj}^2}} \exp\!\left(-\frac{(x_j - \mu_{cj})^2}{2\sigma_{cj}^2}\right)$$

Treinar é só calcular $\mu_{cj}$ e $\sigma_{cj}^2$, a média e a variância da feature $j$ na
classe $c$: uma passagem pelos dados, sem otimizador. As contas fazem-se com logaritmos,
para os produtos de números pequenos não darem zero.

## Hiperparâmetros

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `var_smoothing` | Uma fração da maior variância somada a todas, para nenhuma ser zero. Uma feature constante numa classe daria probabilidades de 0 ou 1 | $10^{-9}$ |

## Quando usar

- Com **poucos dados**: aprende só médias e variâncias, por isso não precisa de muitos
  exemplos.
- Como referência rapidíssima, e em texto (com a variante multinomial, que conta palavras).

## Armadilhas

- **Features correlacionadas contam a dobrar:** se duas features dizem a mesma coisa, o
  modelo trata-as como duas provas independentes e fica confiante demais. As probabilidades
  tendem a ser extremas (quase 0 ou quase 1), mesmo quando a classe prevista está certa.
- **Supõe curvas normais:** uma feature com duas "corcovas" na mesma classe fica mal descrita.
- **A fronteira é quadrática:** curva, mas sempre com a forma de uma elipse ou de uma
  parábola, nunca arbitrária.

## O que observar no treino

Treina de uma vez. No ecrã **Treinar**, as cruzes são as médias de cada classe no plano da
projeção, e a fronteira passa entre elas, mais perto da classe com menor variância. Em
**O que o modelo aprendeu** (ecrã Avaliar) estão as médias e as variâncias de cada classe.
