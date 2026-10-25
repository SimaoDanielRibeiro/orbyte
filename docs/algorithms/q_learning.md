# Q-learning

## Intuição

Não há dataset nem respostas certas. Há um **agente** num **ambiente**: em cada passo vê onde
está (o **estado**), escolhe uma **ação** e recebe uma **recompensa**. Na grelha do ML Lab, o
agente ganha +1 ao chegar ao objetivo, perde 1 numa armadilha e paga 0,01 por cada passo.
Ninguém lhe diz que ação era a boa: tem de descobrir, jogando muitas vezes (os **episódios**).

O Q-learning guarda uma **tabela**: para cada estado e cada ação, um número **Q(s, a)**, a
recompensa que espera ganhar daqui até ao fim se fizer aquela ação ali e depois jogar o
melhor que sabe. A **política** é escolher, em cada estado, a ação de maior Q.

No início a tabela está toda a zeros: o agente não sabe nada e anda ao acaso. Quando chega ao
objetivo pela primeira vez, a ação que o levou lá ganha valor. No episódio seguinte, a célula
antes dessa também ganha, porque leva a uma célula que já vale alguma coisa. O +1 do objetivo
**espalha-se para trás**, célula a célula, até à partida. Na grelha do ecrã Reinforcement
learning vê-se isso a acontecer, episódio a episódio.

## Matemática essencial

Depois de cada passo (estado $s$, ação $a$, recompensa $r$, estado seguinte $s'$), o Q-learning
corrige um único número da tabela:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \big[\, r + \gamma \max_{a'} Q(s', a') - Q(s, a) \,\big]$$

- $r + \gamma \max_{a'} Q(s', a')$ é o **alvo**: a recompensa que se viu, mais o melhor que se
  espera a partir do estado seguinte. Se o episódio acabou ali (objetivo ou armadilha), o alvo
  é só $r$: não há futuro;
- a diferença entre o alvo e o $Q(s, a)$ é o **erro de diferença temporal**: a surpresa;
- $\alpha$ (`learning_rate`) é a fração da surpresa que se corrige;
- $\gamma$ (`discount`) é o **desconto**: uma recompensa a $k$ passos de distância vale
  $\gamma^k$. Com $\gamma = 0{,}95$, o objetivo a 10 passos vale $0{,}95^{10} \approx 0{,}6$, e
  é por isso que o caminho curto vale mais do que o longo.

Para escolher a ação, o agente usa **ε-greedy**: com probabilidade $\varepsilon$, uma ação ao
acaso (**explorar**); senão, a de maior Q (**aproveitar**). O $\varepsilon$ começa em
`epsilon_start` e desce em linha reta até `epsilon_end`, ao longo da fração `exploration_fraction`
dos episódios. Sem explorar, o agente repetia o primeiro caminho que encontrou, mesmo que
houvesse um melhor.

O $\max$ no alvo faz do Q-learning um método **off-policy**: aprende o valor da melhor
política, mesmo enquanto se porta mal (a explorar). É a diferença para o SARSA, que usa no alvo
a ação que vai mesmo fazer.

Esgotar os passos de um episódio não é um estado final: o agente só parou de jogar. Por isso,
nesse último passo, o alvo continua a contar com o futuro.

## Hiperparâmetros

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `episodes` | Quantas vezes o agente atravessa o ambiente: é o limite do treino | 500 |
| `learning_rate` | O $\alpha$: quanto cada passo corrige o Q. Com 1, o Q passa a ser só o que se viu agora; com 0,1, muda devagar | 0,5 |
| `discount` | O $\gamma$: quanto vale o futuro. Perto de 1, o agente pensa a longo prazo; perto de 0, é míope | 0,95 |
| `epsilon_start` | A exploração no primeiro episódio (1 = sempre ao acaso) | 1 |
| `epsilon_end` | A exploração depois de descer: nunca a zero, para continuar a experimentar | 0,05 |
| `exploration_fraction` | Em que parte dos episódios o $\varepsilon$ desce do início ao fim | 0,5 |

## Quando usar

- Quando os estados e as ações se contam e cabem numa tabela: grelhas, jogos de tabuleiro
  pequenos, problemas de inventário com poucos níveis.
- Para aprender reinforcement learning: tudo se vê e tudo se explica, número a número.
- Quando há estados demais para uma tabela (imagens, posições contínuas), a tabela é
  substituída por uma rede neuronal: é o **DQN**.

## Armadilhas

- **Explorar pouco.** Com `epsilon_end` a zero e `exploration_fraction` curta, o agente fica
  com o primeiro caminho que encontrou. No labirinto, pode nunca descobrir o caminho.
- **Ambientes com acaso.** Com escorregar, o mesmo passo dá recompensas diferentes, e uma
  `learning_rate` alta faz o Q saltar com cada surpresa. Com escorregar, experimenta 0,1.
- **O preço de explorar.** No penhasco, o Q-learning aprende o caminho colado às armadilhas, o
  mais curto. Durante o treino, o $\varepsilon$ ainda escolhe ao acaso e o agente cai muitas
  vezes: a recompensa do treino fica abaixo da do teste. É o que o relatório compara.
- **Desconto baixo.** Com $\gamma$ perto de 0, o objetivo distante quase não vale nada, e um
  agente longe dele não tem por onde se guiar.
- **Poucos episódios.** O valor tem de se espalhar do objetivo até à partida; num mapa grande,
  isso leva muitos episódios.

## O que observar no treino

No ecrã **Reinforcement learning**:

- a **recompensa por episódio** começa baixa (o agente anda às voltas e paga cada passo) e
  sobe quando aprende o caminho. A linha é a média móvel, porque cada episódio é ruidoso;
- a **grelha** pinta cada célula com o seu valor e mostra a política (as setas) e o caminho que
  o agente faria. Percorre os episódios e vê o valor a espalhar-se a partir do objetivo;
- a **tabela Q**, ação a ação: ao lado das armadilhas, a ação que lá entra fica negativa.

Experimenta o penhasco com escorregar a 0 e a 0,1: com acaso, o caminho colado às armadilhas
deixa de ser o melhor.
