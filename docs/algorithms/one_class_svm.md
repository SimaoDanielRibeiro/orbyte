# One-Class SVM

## Intuição

O One-Class SVM aprende só com dados (quase todos) normais e desenha a **fronteira mais justa
à volta deles**. Na previsão, o que fica de fora da fronteira é anómalo. É a ideia do SVM (uma
fronteira com margem, desenhada pelos vetores de suporte), com uma só classe: separar os
dados da origem, no espaço do kernel.

## Matemática essencial

O One-Class SVM procura $w$ e $\rho$ que minimizam

$$\frac{1}{2}\lVert w \rVert^2 + \frac{1}{\nu n} \sum_i \xi_i - \rho \qquad \text{com } w \cdot \phi(x_i) \ge \rho - \xi_i,\ \xi_i \ge 0$$

onde $\phi$ é o mapeamento do kernel (com o RBF, uma "colina" à volta de cada vetor de
suporte). A função de decisão $w \cdot \phi(x) - \rho$ é positiva dentro da fronteira. O
parâmetro $\nu$ é, ao mesmo tempo, um limite máximo para a fração de pontos de treino fora da
fronteira e um limite mínimo para a fração de vetores de suporte.

## Hiperparâmetros

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `nu` | O máximo de pontos de treino fora da fronteira: na prática, a proporção de anomalias esperada | 0,05 |
| `kernel` | A forma da fronteira à volta dos dados normais: `rbf`, `linear` ou `poly` | `rbf` |
| `gamma` | No kernel RBF, o alcance de cada ponto: grande, a fronteira abraça cada ponto; pequeno, é suave | 0,5 |

## Quando usar

- Quando os dados normais formam uma ou poucas zonas compactas e se quer uma fronteira
  suave à volta delas.
- Com conjuntos de tamanho médio (como qualquer SVM, fica lento com muitas linhas).

## Armadilhas

- **Escala das features:** o kernel RBF usa distâncias. Padroniza sempre.
- **`gamma` grande** desenha uma ilha à volta de cada ponto de treino: qualquer ponto novo
  parece anómalo.
- **Anomalias no treino** puxam a fronteira para fora. O `nu` dá-lhes alguma folga.
- **Muitas linhas:** o treino cresce mais depressa do que o número de linhas.

## O que observar no treino

Treina de uma vez. No ecrã **Treinar**, as cores mostram a pontuação de anomalia no plano: vês
a fronteira a abraçar as nuvens de pontos. Experimenta o `gamma`: pequeno dá uma forma lisa;
grande, ilhas à volta de cada ponto.
