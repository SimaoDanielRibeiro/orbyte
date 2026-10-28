# Ridge e Lasso

## Intuição

São a regressão linear com os pesos **travados**. Com muitas features, poucas linhas ou
features muito parecidas entre si, a regressão linear dá pesos enormes, que se anulam uns aos
outros e mudam muito de uma amostra para outra. A regularização acrescenta um custo aos pesos
grandes: o modelo só os usa se valer a pena.

- A **Ridge** encolhe todos os pesos, mas nenhum chega a zero.
- O **Lasso** põe a zero os pesos das features menos úteis, por isso também **escolhe** as
  features.

## Matemática essencial

As duas minimizam o erro quadrático mais uma penalização, com força $\alpha$:

$$\text{Ridge (L2):}\quad \frac{1}{n}\lVert y - Xw \rVert^2 + \alpha \lVert w \rVert_2^2 \qquad\qquad \text{Lasso (L1):}\quad \frac{1}{2n}\lVert y - Xw \rVert^2 + \alpha \lVert w \rVert_1$$

(o scikit-learn escreve a Ridge sem o $\frac{1}{n}$, o que só muda a escala do $\alpha$). Com
$\alpha = 0$ voltam a ser a regressão linear; com $\alpha$ enorme, os pesos vão para zero e o
modelo prevê a média.

A Ridge tem fórmula fechada, $w = (X^\top X + \alpha I)^{-1} X^\top y$: somar $\alpha$ à
diagonal torna a matriz sempre invertível, mesmo com features redundantes. O Lasso não tem.
O scikit-learn usa a descida por coordenadas, que otimiza um peso de cada vez. O $\lVert w
\rVert_1$ tem um "bico" em zero, e é por isso que os pesos do Lasso ficam *exatamente* a zero:
o bico prende-os lá.

## Hiperparâmetros

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `alpha` | A força da regularização: maior = pesos mais pequenos (e, no Lasso, mais pesos a zero) | 1,0 (Ridge) / 0,1 (Lasso) |
| `max_iter` | No Lasso, o máximo de iterações do otimizador | 1000 |
| `fit_intercept` | Se aprende a ordenada $b$ (a ordenada nunca é penalizada) | sim |

## Quando usar

- **Ridge:** features correlacionadas (as várias medidas de tamanho de uma casa), ou quase
  tantas features como linhas. É a escolha segura quando se quer usar todas as features.
- **Lasso:** quando se suspeita que só algumas features importam e se quer saber quais.

## Armadilhas

- **Escala das features:** a penalização trata todos os pesos por igual, mas o peso de uma
  feature em milhares é naturalmente pequeno. Padroniza na preparação.
- **O Lasso com features correlacionadas** escolhe uma quase ao acaso e põe as outras a zero:
  não leias "peso zero" como "não tem nada a ver".
- **O `alpha` certo depende dos dados:** escolhe-o com validação cruzada. O ecrã Laboratório
  mostra o R² no teste a subir e a voltar a descer à medida que o alpha cresce.
- **Aviso de convergência no Lasso:** aumenta `max_iter` ou padroniza as features.

## O que observar no treino

As duas treinam de uma vez. Em **O que aprendeu**, compara os pesos com
os da regressão linear: na Ridge encolhem todos; no Lasso alguns desaparecem. A demonstração
"Regularização" do ecrã Laboratório mostra o caminho de cada peso à medida que o alpha cresce.
