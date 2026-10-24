# PCA (análise de componentes principais)

## Intuição

Dados com muitas features não cabem num gráfico. A PCA procura as **direções em que os dados
variam mais** e reescreve cada ponto nessas direções. A primeira componente é a direção de
maior variação; a segunda, a de maior variação entre as perpendiculares à primeira; e assim
por diante. Ficar com as duas primeiras dá um gráfico 2D que guarda o máximo possível da
"forma" dos dados.

É o que o ecrã Preparar já usa para a projeção 2D, aqui como modelo, com o número de
componentes à escolha.

## Matemática essencial

1. Centra os dados: $X_c = X - \bar{x}$.
2. Calcula a matriz de covariância: $\Sigma = \frac{1}{n-1} X_c^\top X_c$.
3. Os **vetores próprios** de $\Sigma$ são as componentes principais; os **valores
   próprios** $\lambda_j$ são a variância dos dados ao longo de cada uma.
4. Projeta: as novas coordenadas são $Z = X_c W_k$, com as $k$ componentes de maior
   $\lambda_j$ nas colunas de $W_k$.

A **variância explicada** de cada componente é $\lambda_j / \sum_l \lambda_l$: a fração da
variação total que ela guarda. A versão didática (`pca_numpy`) faz exatamente estas contas
com `np.linalg.eigh`; a do scikit-learn usa a decomposição em valores singulares (SVD) dos
dados centrados, que dá as mesmas componentes com mais estabilidade numérica. O sinal de cada
componente é arbitrário; as duas escolhem o que torna positivo o maior valor da componente.

## Hiperparâmetros

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `n_components` | Quantas direções guardar. Com 2, os dados cabem num gráfico | 2 |
| `whiten` | No scikit-learn, dividir cada componente pelo seu desvio padrão (todas ficam com variância 1) | não |

## Quando usar

- Para **ver** dados com muitas features num gráfico 2D.
- Para **simplificar** antes de outro modelo: menos features, menos ruído, treino mais rápido.
- Para perceber que features andam juntas (os pesos de cada componente).

## Armadilhas

- **Escala das features:** sem padronização, a feature com números maiores domina a primeira
  componente só por causa da unidade. Padroniza na preparação.
- **É linear:** estruturas curvas (um rolo, um anel) ficam esmagadas. O t-SNE mostra melhor os
  vizinhos, mas não é linear nem reversível.
- **Variância não é importância:** a direção que mais varia pode não ser a que separa as
  classes.
- **As componentes são misturas** de todas as features: dar-lhes nome nem sempre é fácil.

## O que observar no treino

Treina de uma vez. No ecrã **Treinar** vês o **mapa**: os pontos de treino nas duas primeiras
componentes, pintados com a coluna de referência, se a escolheste no ecrã Preparar. No ecrã
**Avaliar**, a variância explicada de cada componente e a "confiança dos vizinhos" dizem quanto
o mapa guardou dos dados originais.
