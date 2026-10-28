# Mistura de gaussianas

## Intuição

Supõe que os dados vêm de k "nuvens" em forma de sino (gaussianas), cada uma com o seu centro,
a sua forma (redonda, alongada, inclinada) e o seu peso (que fração dos pontos gera). O
modelo descobre essas nuvens. Ao contrário do k-means, cada ponto não pertence a um só
grupo: pertence a cada nuvem com uma **probabilidade**. Um ponto entre duas nuvens fica meio
de cada uma.

## Matemática essencial

A densidade dos dados é uma soma pesada de gaussianas:

$$p(x) = \sum_{k=1}^{K} \pi_k\, \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

Os parâmetros encontram-se com o algoritmo **EM** (*expectation-maximization*), que repete:

1. **E**: a responsabilidade de cada nuvem por cada ponto,
   $r_{ik} = \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k) / p(x_i)$;
2. **M**: as médias, as covariâncias e os pesos que melhor explicam essas responsabilidades
   (médias pesadas por $r_{ik}$).

Cada iteração nunca baixa a **log-verosimilhança** média, $\frac{1}{n}\sum_i \log p(x_i)$. O
algoritmo pára quando ela quase deixa de subir. Como o k-means, encontra um máximo local,
que depende do ponto de partida (o scikit-learn começa pelo resultado de um k-means).

## Hiperparâmetros

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `n_components` | Quantas nuvens (grupos) | 3 |
| `covariance_type` | A forma das nuvens: `full` (cada uma a sua elipse, inclinada à vontade), `tied` (a mesma elipse para todas), `diag` (elipses alinhadas com os eixos) ou `spherical` (círculos) | `full` |
| `max_iter` | Máximo de iterações do EM | 100 |
| `tol` | Pára quando a log-verosimilhança média sobe menos do que isto | $10^{-3}$ |

## Quando usar

- Quando os grupos são **elipses** de tamanhos e inclinações diferentes.
- Quando interessa a **incerteza**: a probabilidade de cada ponto pertencer a cada grupo.

## Armadilhas

- **Muitos parâmetros:** com `full`, cada nuvem tem uma matriz de covariância completa
  ($d(d+1)/2$ números). Com muitas features e poucos dados, usa `diag` ou `spherical`.
- **Nuvens que encolhem até um ponto** (a variância vai para zero e a verosimilhança para o
  infinito): o scikit-learn soma uma pequena regularização à diagonal para o evitar.
- **Escolher o k:** compara a silhueta, ou critérios como o BIC.
- **Os grupos não têm de ser gaussianos:** uma lua não é uma elipse.

## O que observar no treino

No ecrã **Treinar e avaliar**, cada passo é uma iteração do EM. As **elipses** mostram cada nuvem (a dois
desvios padrão do centro), e as cruzes os seus centros. Vês as elipses a deslocarem-se, a
esticarem e a rodarem até se ajustarem aos dados. A loss é o negativo da log-verosimilhança
média: desce até estabilizar.
