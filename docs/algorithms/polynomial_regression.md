# Regressão polinomial

## Intuição

Uma reta não segue uma curva. Mas se juntarmos às features as suas potências ($x^2$, $x^3$,
...) e os seus produtos ($x \cdot z$), uma regressão **linear** nessas novas features já
desenha curvas. O modelo continua linear nos pesos; só as features é que ficaram curvas.

É a demonstração clássica do **overfitting**: com grau baixo, a curva é simples demais
(underfitting); com grau alto, passa por todos os pontos de treino e ondula loucamente entre
eles.

## Matemática essencial

Com uma feature e grau $d$:

$$\hat y = b + w_1 x + w_2 x^2 + \dots + w_d x^d$$

Com $p$ features, o polinómio de grau $d$ tem todos os produtos até grau $d$: são
$\binom{p + d}{d} - 1$ termos. Com 10 features e grau 3, já são 285. O modelo do laboratório
é um *pipeline* de três peças:

1. cria as potências e os produtos (`PolynomialFeatures`);
2. padroniza-os ($x^{15}$ e $x$ têm escalas incomparáveis);
3. ajusta uma Ridge, com o $\alpha$ escolhido, que trava os pesos dos termos.

## Hiperparâmetros

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `degree` | A maior potência. Grau 1 é a reta; graus altos fazem curvas que passam por todos os pontos de treino | 2 |
| `alpha` | A regularização da Ridge. Quase zero mostra o overfitting em toda a sua glória; aumentar amansa as curvas dos graus altos | $10^{-6}$ |

## Quando usar

- Quando a relação é **curva e suave** e há poucas features (uma ou duas).
- Para **ver** o equilíbrio entre viés e variância (a demonstração "Overfitting" do ecrã
  Laboratório).

## Armadilhas

- **O número de termos explode** com o grau e o número de features. O modelo recusa mais de
  5000 termos.
- **Overfitting com graus altos:** o erro de treino desce sempre; o de teste desce e volta a
  subir. Escolhe o grau pelo teste (ou com validação cruzada), nunca pelo treino.
- **Extrapolação desastrosa:** fora do intervalo dos dados, um polinómio de grau alto dispara
  para mais ou menos infinito.

## O que observar no treino

Treina de uma vez. Com uma só feature, o ecrã **Treinar** desenha a curva por cima dos pontos.
Experimenta o grau 1, o 3 e o 15. No ecrã **Avaliar**, compara o erro no treino e no teste: a
diferença cresce com o grau.
