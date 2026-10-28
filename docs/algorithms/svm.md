# Máquinas de vetores de suporte (SVM e SVR)

## Intuição

Entre todas as retas que separam duas classes, o SVM escolhe a que fica **mais longe das
duas**: a do meio da faixa mais larga sem pontos (a **margem**). Uma fronteira com folga
generaliza melhor do que uma encostada aos pontos de uma das classes.

Só os pontos na borda da margem (ou do lado errado) decidem onde fica a fronteira: são os
**vetores de suporte**. Os outros podiam desaparecer sem mudar nada.

Com um **kernel**, a fronteira pode curvar-se: é como se os pontos fossem levados para um
espaço com mais dimensões, onde uma reta os separa, sem nunca calcular essas dimensões.

## Matemática essencial

O SVM procura $w$ e $b$ que minimizam

$$\frac{1}{2}\lVert w \rVert^2 + C \sum_i \max\big(0,\ 1 - y_i (w \cdot x_i + b)\big)$$

O primeiro termo alarga a margem (que tem largura $2/\lVert w \rVert$); o segundo castiga os
pontos dentro da margem ou do lado errado (a *hinge loss*), pesado por $C$.

O **truque do kernel** troca cada produto $x \cdot z$ por uma função $K(x, z)$. O kernel RBF,
$K(x, z) = e^{-\gamma \lVert x - z \rVert^2}$, mede a semelhança entre dois pontos: cada
vetor de suporte cria uma "colina" à sua volta, com largura $1/\sqrt{\gamma}$.

O SVM dá distâncias à fronteira, não probabilidades. Para as ter, o laboratório **calibra**
as distâncias com uma regressão logística ajustada por validação cruzada (o método de Platt).

O **SVR** aplica a mesma ideia à regressão: procura uma função tão plana quanto possível que
passe a menos de $\varepsilon$ de cada ponto. Os erros dentro desse "tubo" não contam.

## Hiperparâmetros

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `C` | Quanto custa cada ponto do lado errado da margem. Grande: margem estreita, que tenta acertar em tudo (overfitting); pequeno: margem larga e tolerante | 1,0 |
| `kernel` | A forma da fronteira: `linear`, `rbf` (curvas à volta dos pontos) ou `poly` (polinómio de grau 3) | `rbf` |
| `gamma` | Nos kernels RBF e polinomial, o alcance de cada ponto. Grande: cada ponto cria a sua ilha (overfitting); pequeno: fronteira suave | 0,5 |
| `epsilon` | No SVR, a largura do tubo onde os erros não contam (na escala do alvo) | 0,1 |

## Quando usar

- Com **dados de tamanho médio** (até umas dezenas de milhares de linhas) e muitas features,
  incluindo mais features do que linhas.
- Quando a fronteira é curva, mas suave: o kernel RBF adapta-se bem.

## Armadilhas

- **Escala das features:** a margem e o kernel RBF usam distâncias. Padroniza sempre.
- **`C` e `gamma` andam juntos:** os dois grandes decoram o treino; os dois pequenos dão um
  modelo quase constante. Afina-os em conjunto, com validação cruzada.
- **Lento com muitas linhas:** o treino cresce entre o quadrado e o cubo do número de linhas.
- **As probabilidades são uma calibração extra:** custam mais um treino por fold e, com
  poucos dados, podem ser pouco fiáveis.
- **No SVR, `epsilon` está na escala do alvo:** com um alvo em milhares, 0,1 é praticamente
  zero.

## O que observar no treino

Treina de uma vez. No ecrã **Treinar e avaliar**, vê como a fronteira muda com o `gamma`:
pequeno dá uma curva suave, grande dá ilhas à volta dos pontos. Em **O que aprendeu** está o
número de vetores de suporte. Muitos vetores (perto do número de linhas)
indicam uma margem que quase não separa nada.
