# 0048 — Os modelos arrumados por problema, a letra maior e um movimento com peso

- **Estado:** Proposta
- **Data:** 2026-11-03
- **Muda:** a [0037](0037-cores-por-familia-barra-que-desliza-e-licao-do-rl.md) (no ecrã Modelos,
  a cor deixa de ser a da família), a [0044](0044-um-botao-de-prosseguir-letra-maior-barra-alinhada.md)
  e a [0045](0045-identidade-visual-letra-propria-e-barra-mais-estreita.md) (a escala da letra e a
  largura da barra) e a [0047](0047-a-historia-do-treino-no-ecra-explicar.md) (a disposição dos
  capítulos)

## Contexto

O pedido:

- tirar o "API online" da barra lateral e da entrada;
- pôr o botão de abrir e fechar a barra por cima do Histórico, e o Histórico no fundo;
- tirar o "suggested to start" da lista dos modelos;
- arrumar melhor o Explicar;
- transições mais suaves e trabalhadas, em todo o lado;
- a letra maior em todo o lado e um ar mais premium, só com o texto necessário;
- as cores do ecrã Modelos com sentido, e outra arrumação dos modelos.

O diagnóstico:

- **As cores dos modelos não diziam nada.** Cada família tinha uma cor (ADR 0037), mas eram
  onze famílias para oito cores, três repetidas, e nada na página dizia o que a cor queria
  dizer: lia-se como cores ao acaso.
- **O Explicar desalinhava.** Os capítulos eram uma linha do tempo (o número num círculo, ligado
  por um traço), com o texto e o visual lado a lado dentro de cada um. Um texto curto ao lado de
  um visual alto (uma frase ao lado da matriz de confusão) deixava um buraco por baixo, e os
  capítulos sem visual empurravam o seguinte para longe do seu.
- **O movimento era curto e seco.** As transições duravam 150 a 200 ms, a barra abria em
  `ease-linear`, os menus abriam em 100 ms, o destaque da barra chegava com ressalto e as páginas
  apareciam de uma vez.
- **O "API online" estava em todas as páginas**, mas é o normal: só a falha precisa de texto.

## Decisão

**Os modelos arrumam-se pelo problema que resolvem, e a cor é o problema.** O backend ganha o
enum `Problem` (`core/foundation/types.py`: classificação, regressão, clustering, redução de
dimensionalidade, deteção de anomalias e controlo), com `TaskType.problem` e o campo `problem` no
`TaskInfo` do vocabulário. Juntar a classificação binária e a multiclasse num só problema é uma
regra do domínio, por isso fica no backend, e o nome de cada problema vem no vocabulário.

O ecrã Modelos tem dois andares: o paradigma (supervisionado, não supervisionado) e, dentro dele,
o problema, com um ponto da cor dele e quantos modelos o resolvem. Cada modelo é um cartão numa
grelha: o desenho em cima, o nome e a família por baixo. A cor diz o que o modelo resolve; o nome
da família diz como aprende.

A cor lê-se em dois tempos (`lib/problems.ts`):

| Paradigma | Temperatura | Problema e cor |
|---|---|---|
| Supervisionado | fria | classificação azul, regressão ciano |
| Não supervisionado | quente | clustering amarelo, redução de dimensionalidade rosa, anomalias laranja |
| Reinforcement learning | a meio | controlo verde |

Os agentes continuam com a cor da família (`lib/families.ts`): um agente é uma família, não um
problema. A página de cada modelo põe a cor do problema no chip das tarefas, o primeiro. O
"suggested to start" saiu da lista (a sugestão continua no backend, para quem treina).

**A escala sobe mais um degrau:** 15, 17, 18, 20, 24 e 30 px, e o título de cada página entre 36
e 44 px (cresce com o ecrã). Os textos dos gráficos sobem 1 px (de 12 para 13 px nos eixos), e a
margem dos eixos conta com a letra nova (`AXIS_CHAR` passa de 7,2 para 7,8). Os botões, os campos
e os seletores passam a 44 px de altura. A barra passa a 19,5rem, para "Reinforcement learning"
caber à letra de 18 px.

**O movimento tem uma curva, com peso:** `--ease-out-expo` no que entra (rápida a partir, longa a
assentar) e `--ease-out-quart` no que só muda de cor; são a omissão do Tailwind (220 ms) e do
Motion (550 ms). Os menus, as dicas e as gavetas entram em 320 ms e saem em 180 ms; a barra abre e
fecha em 500 ms; o destaque da barra desliza sem ressalto; os separadores entram a subir um nada;
e cada página entra em cascata (`page-in-stagger`: os blocos de cima sobem 12 px e acendem, 50 ms
um depois do outro). Só se animam a opacidade e a posição, com `backwards`: no fim, o conteúdo
fica sem `transform`, e o que estiver fixo lá dentro não perde a referência.

**O Explicar arruma-se em linhas.** Cada capítulo é uma linha da página entre filetes: o número em
mono, o título e o texto numa coluna, o visual na outra, com o topo à altura do título. A história
abre com o título da secção e, por baixo, a frase do resultado, maior. As três perguntas das
previsões são linhas com o mesmo filete, e a explicação da secção passa para o "?".

**A API só se mostra quando falha**, com o que fazer: na barra, por cima do botão de recolher; na
entrada, no canto. **O botão de recolher a barra** desce para o fundo, por cima do Histórico, como
uma linha da barra, alinhada pelos mesmos ícones (recolhida, fica só o ícone).

## Consequências

- Uma família deixa de ter a mesma cor no ecrã Modelos e no dos Agentes: nos Modelos, a família
  lê-se pelo nome, e a cor é a do problema.
- Um problema novo precisa de uma cor em `PROBLEM_COLORS` (sem ela, fica cinzento) e de um nome
  no vocabulário (o teste do vocabulário falha sem ele).
- A letra maior ocupa mais espaço: uma regra de largura nova tem de contar com a barra de 19,5rem.
- Um bloco que chega mais tarde como filho direto de uma página (um erro, um resultado) também
  entra com a animação. É de propósito: vê-se o que mudou.
- Os testes de ponta a ponta deixam de procurar o "API online" e passam a procurar o botão de
  recolher e o Histórico no fundo da barra.
