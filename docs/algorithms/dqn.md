# DQN (Deep Q-Network)

## Intuição

O DQN aprende o mesmo que o **Q-learning**, os valores **Q(s, a)**, mas guarda-os
numa **rede neuronal** em vez de uma tabela. A rede recebe o estado e devolve um Q por ação. Numa
grelha pequena, uma tabela chegava; a rede é o que permite problemas com estados demais para
uma tabela, como as imagens de um jogo. Foi assim que, em 2015, um só algoritmo aprendeu a jogar
dezenas de jogos da Atari a partir dos píxeis.

Treinar uma rede com a surpresa de cada passo é instável: a rede persegue um alvo que ela
própria calcula, e esquece o que aprendeu com passos antigos. O DQN junta duas peças para o
acalmar:

- uma **memória de repetição**: cada passo vai para uma memória, e a rede aprende com **lotes
  tirados dela ao acaso**. Passos seguidos são muito parecidos entre si (a mesma zona da grelha);
  misturá-los quebra essa correlação e deixa reaproveitar cada passo várias vezes;
- uma **rede-alvo**: uma cópia da rede, congelada, que calcula os alvos. Só se atualiza de vez
  em quando. Sem ela, cada passo de treino mudava também o alvo, como perseguir a própria sombra.

## Matemática essencial

No ML Lab, o estado entra na rede em **one-hot**: um vetor de zeros com um 1 na posição do
estado. A rede tem duas camadas escondidas com ReLU e uma saída por ação:

$$\text{one-hot}(s) \rightarrow \text{ReLU} \rightarrow \text{ReLU} \rightarrow \big(Q(s, a_1), \dots, Q(s, a_4)\big)$$

Em cada passo, depois de guardar o passo na memória, tira-se um lote ao acaso e calcula-se, para
cada passo $(s, a, r, s')$ do lote, o alvo com a **rede-alvo** $\hat{Q}$:

$$y = r + \gamma \max_{a'} \hat{Q}(s', a') \qquad (y = r \text{ se o episódio acabou})$$

A loss é a de **Huber** entre $Q(s, a)$ e $y$: quadrática para erros pequenos e linear para os
grandes, para uma surpresa enorme (uma armadilha inesperada) não dar um gradiente enorme. O
otimizador é o Adam. A cada `target_update` passos de treino, a rede-alvo passa a ser uma cópia
da rede.

A escolha das ações é a mesma do Q-learning: **ε-greedy**, com o $\varepsilon$ a descer de
`epsilon_start` até `epsilon_end`, ao longo de `exploration_fraction` dos episódios.

## Hiperparâmetros

O que tem em comum com o Q-learning:

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `episodes` | Quantas vezes o agente atravessa o ambiente | 300 |
| `discount` | O desconto $\gamma$: quanto vale o futuro | 0,95 |
| `epsilon_start` | A exploração no primeiro episódio | 1 |
| `epsilon_end` | A exploração depois de descer | 0,05 |
| `exploration_fraction` | Em que parte dos episódios o $\varepsilon$ desce | 0,5 |

O que é só do DQN:

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `learning_rate` | O passo do Adam: grande demais, a rede oscila; pequeno demais, demora | 0,001 |
| `hidden_units` | Neurónios em cada uma das duas camadas escondidas | 64 |
| `batch_size` | Quantos passos da memória entram em cada passo de treino | 32 |
| `replay_size` | Quantos passos a memória guarda (os mais antigos saem primeiro) | 5000 |
| `target_update` | De quantos em quantos passos de treino a rede-alvo é atualizada | 100 |

## Quando usar

- Quando há estados demais para uma tabela: posições contínuas, imagens, muitas variáveis.
- Com ações discretas e poucas (para ações contínuas, como o ângulo de um volante, há outros
  métodos, como o PPO).
- Numa grelha pequena, o Q-learning chega e é muito mais rápido: o DQN está aqui para se ver
  o que muda quando a tabela passa a ser uma rede.

## Armadilhas

- **Instabilidade.** Sem a memória de repetição e a rede-alvo, o treino oscila ou diverge.
  Experimenta `target_update` a 1 (sem rede-alvo) e compara a recompensa por episódio.
- **Lento.** Cada passo do agente é também um passo de treino da rede: o DQN demora segundos
  onde o Q-learning demora décimas.
- **Memória pequena.** Com `replay_size` perto do `batch_size`, os lotes voltam a ser passos
  seguidos e correlacionados, e perde-se a razão de ter uma memória.
- **Taxa de aprendizagem alta.** A rede persegue alvos que mudam; um passo grande demais faz os
  Q explodirem.
- **Aleatoriedade.** Os pesos iniciais, a exploração e os lotes vêm da seed: outra seed dá
  outro treino. Compara várias seeds no ecrã Experiências.

## O que observar no treino

No ecrã **Reinforcement learning**, como no Q-learning:

- a **recompensa por episódio**, que costuma ser mais ruidosa do que a do Q-learning, porque a
  rede muda os Q de todos os estados ao mesmo tempo;
- a **grelha**, com o valor e a política em cada célula: os valores espalham-se a partir do
  objetivo, mas de forma menos arrumada do que na tabela, porque a rede generaliza (mudar um
  peso mexe em vários estados);
- a **loss** de cada episódio, a de Huber: desce quando os Q estabilizam.
