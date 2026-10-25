# Agentes de AutoML (triagem e procura aleatória)

## Intuição

Escolher um modelo à mão é sempre o mesmo ciclo: preparar os dados, experimentar um modelo,
medir, mudar um hiperparâmetro, medir outra vez. O **AutoML** (machine learning automático)
faz esse ciclo sozinho, com um orçamento. No ML Lab, o trabalho está dividido como numa
equipa:

- o **agente de dados** prepara a tabela a partir do perfil: remove identificadores e texto
  livre, confirma a tarefa, divide (20% para o teste), preenche os nulos, passa as
  categorias a one-hot e padroniza os números. Cada decisão vem com a razão;
- um **agente por família de modelos** (os lineares, as árvores, os ensembles, as redes…)
  conhece os modelos da sua família e sabe o que vale a pena experimentar em cada um;
- o **coordenador** gere o orçamento e escolhe o vencedor.

A procura tem duas rondas. Na **triagem**, cada modelo é experimentado uma vez, com os
valores por omissão: é uma primeira impressão barata, que mostra que famílias se dão bem
com estes dados. Na **afinação**, só os melhores (os **finalistas**) recebem o resto do
orçamento, e os seus agentes sorteiam outras combinações de hiperparâmetros.

Não há magia nem um modelo de linguagem: são regras e medições. Com a mesma seed, a procura
repete-se tal e qual, e a conversa dos agentes diz porque é que o vencedor ganhou.

## Matemática essencial

Cada candidato (um modelo com uns hiperparâmetros) é medido com **validação cruzada em
$k$ folds**, só no treino: o treino parte-se em $k$ partes, e o candidato treina $k$
vezes, de cada vez com $k - 1$ partes, e é avaliado na que ficou de fora. A nota é a média,
e o desvio padrão diz quanto ela varia de fold para fold:

$$\bar{m} = \frac{1}{k} \sum_{i=1}^{k} m_i \qquad s = \sqrt{\frac{1}{k} \sum_{i=1}^{k} (m_i - \bar{m})^2}$$

O vencedor é o candidato com a melhor média (a maior, na exatidão; a menor, na RMSE). Num
empate, ganha o de menor desvio, o mais estável.

Na afinação, os hiperparâmetros sorteiam-se no **espaço de procura** que cada modelo declara
(o `x-search` dos seus hiperparâmetros). Um número em escala logarítmica, como o $C$ do SVM
entre $0{,}1$ e $100$, sorteia-se nos expoentes:

$$C = 10^{u}, \qquad u \sim \text{Uniforme}(\log_{10} 0{,}1,\ \log_{10} 100)$$

Assim, sair entre $0{,}1$ e $1$ é tão provável como sair entre $10$ e $100$. É a **procura
aleatória** (random search): com o mesmo número de tentativas que uma grelha, experimenta
mais valores diferentes de cada hiperparâmetro, e isso conta porque, em cada problema, só
alguns deles importam a sério.

O **teste** só se usa no fim, uma vez, para o vencedor. Se se escolhesse pelo teste, o
teste passava a ser mais um treino: o vencedor seria o que teve mais sorte nele.

## Hiperparâmetros

Os "hiperparâmetros" da própria procura, no ecrã Agentes:

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `budget` | Quantos candidatos medir, contando a triagem e a afinação. Cada um treina uma vez por fold | 24 |
| `folds` | Em quantas partes se divide o treino para medir cada candidato | 5 |
| `finalists` | Quantos modelos da triagem passam à afinação | 3 |
| `max_rows` | Com mais linhas de treino do que isto, a validação cruzada usa uma amostra (o vencedor treina no fim com todas) | 2000 |

E o pedido: o **alvo**, a **métrica a otimizar** (a exatidão, a exatidão equilibrada, a F1,
a ROC AUC ou a log-loss na classificação; o R², a RMSE ou o MAE na regressão) e os
**agentes chamados**. Com classes desequilibradas, o agente de dados sugere a exatidão
equilibrada: a exatidão ganha-se com um modelo que diz sempre a classe mais frequente.

## Quando usar

- Para responder depressa a «que modelo serve melhor para isto?», com um ponto de partida
  honesto e explicado.
- Para comparar famílias de modelos nos mesmos dados, com a mesma preparação e os mesmos
  folds: a comparação é justa.
- Como primeiro passo: o vencedor abre-se no Avaliar e no Explicar, e qualquer candidato se
  pode treinar à mão, para se ver por dentro.
- O modo manual continua a ser onde se aprende cada escolha. Os agentes só trabalham com um
  alvo (classificação e regressão): no clustering não há previsões certas para comparar.

## Armadilhas

- **O vencedor é um pouco otimista.** Entre muitos candidatos parecidos, o que ganha a
  validação cruzada também teve alguma sorte nos folds. A nota do teste costuma ficar um
  pouco abaixo, e quanto mais candidatos, mais.
- **Diferenças dentro do ruído.** Se o segundo fica a menos de um desvio padrão do primeiro,
  na prática empatam. Nesses casos, o modelo mais simples é uma escolha tão boa como o
  vencedor (a referência, como a regressão logística, está no quadro para isso).
- **Orçamento curto.** Com poucos candidatos, a afinação quase não acontece: a triagem
  gasta tudo. Com muitos, a procura demora mais e o otimismo do vencedor cresce.
- **Poucos dados.** Com poucas linhas, os folds são pequenos e o desvio grande: a escolha
  muda com a seed. Experimenta duas seeds e compara.
- **A preparação é uma só.** Todos os candidatos usam a mesma preparação (é o que torna a
  comparação justa). Se os dados precisam de outra coisa, prepara-os à mão e pede aos
  agentes que usem a tua preparação.

## O que observar na procura

No ecrã **Agentes**:

- a **conversa**: o que o agente de dados decidiu e porquê, o que cada agente de modelo
  experimentou e o que o coordenador escolheu;
- o gráfico **A procura**: cada candidato pela ordem em que foi medido, com o desvio entre
  folds, e a linha do melhor até aí. A linha sobe aos degraus: na triagem, quando aparece
  uma família melhor; na afinação, quando um sorteio acerta;
- **O melhor de cada agente**: que família se deu melhor com estes dados, comparada com a
  referência;
- o **quadro**: a média no treino ao lado da média na validação mostra quanto cada
  candidato decora; um candidato que falhou fica lá, com a razão;
- no fim, o **teste** do vencedor, comparado com o que a validação cruzada previa.
