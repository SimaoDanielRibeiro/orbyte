# 0037 — Cores fixas por família, a barra que desliza e a lição do reinforcement learning

- **Estado:** Proposta
- **Data:** 2026-10-29

## Contexto

Depois de ver a interface a correr, o pedido seguinte foi:

- o ecrã Modelos mais organizado e bonito, com a cor de cada tipo de modelo, como nos agentes,
  e pequenas animações em cada modelo;
- nos Dados, os exemplos e, logo por baixo, o carregamento do ficheiro, em grande;
- o Explorar bem organizado, para se perceber como cada parte se enquadra no todo;
- o reinforcement learning explicado melhor, mais visual e com animações;
- na barra lateral, a secção aberta mais clara, com um movimento que pareça arrastá-la até à
  secção nova;
- tirar o "Procurar" da barra;
- o painel que abre ao clicar num agente do fluxograma maior e mais arrumado.

Ao olhar para o código apareceram três problemas por baixo do pedido:

- **A cor dos agentes seguia a posição, não a família.** Cada agente tinha a cor da sua posição
  no plano. As árvores eram amarelas numa procura de regressão e verdes numa de classificação, e
  tirar um agente pintava os outros de novo. A entrada (o cartão dos Agentes) tinha outra ordem
  ainda.
- **Os painéis laterais não tinham a largura pedida.** O `Sheet` do shadcn tem
  `data-[side=right]:sm:max-w-sm`, e uma classe com essa variante ganha à `sm:max-w-2xl` que o
  painel dos agentes passava. O painel ficava com 24rem e partia as linhas.
- **O RL não tinha nada para mostrar antes do treino.** Para animar o ciclo de um agente era
  preciso cada passo de um treino (o estado, a ação, a recompensa, a correção). O treino só
  devolve uma fotografia por snapshot e a recompensa por episódio, e a regra 1 da interface não
  deixa calcular nada de ML do lado dela.

## Decisão

**Uma cor fixa por família (`lib/families.ts`), a mesma em todo o lado.** São onze famílias e oito
cores, por isso as cores escolhem-se pelo paradigma:

- as sete famílias dos modelos supervisionados (as dos agentes) têm sete cores diferentes;
- as famílias com modelos não supervisionados são oito, também diferentes entre si;
- as quatro que só têm modelos não supervisionados repetem a cor de uma família que só tem
  supervisionados:
  - os centróides, o azul dos lineares;
  - os hierárquicos, o amarelo das árvores;
  - as projeções, o roxo das redes;
  - a densidade fica com a oitava cor, o vermelho.

As duas sequências passam no validador do skill de visualização de dados, na ordem do catálogo:

- **na visão normal**, as cores vizinhas distinguem-se nas duas sequências;
- **com daltonismo**, distinguem-se todas menos o rosa e o ciano, nas não supervisionadas, que
  ficam na faixa aceite quando o nome está escrito ao lado (e está sempre);
- **a faixa de luminosidade** falha, como já falhava na paleta: é a exceção registada em
  `colors.ts`.

O fluxograma, a conversa, os gráficos da procura, o painel de cada agente e a entrada passam a
ler a cor daqui.

**O ecrã Modelos em dois andares.**

- **Os paradigmas:** primeiro a aprendizagem supervisionada, depois a não supervisionada. O
  paradigma de um modelo é o das suas tarefas, que vem no vocabulário.
- **As famílias:** em cada paradigma, um cartão por família, com a cor dela (um fio de luz no
  topo e um brilho fraco). Uma família com modelos dos dois paradigmas aparece nos dois.
- **Os modelos:** cada linha tem um desenho pequeno, a mexer, do que o modelo faz. A reta
  ajusta-se aos pontos, o KNN liga o ponto novo aos vizinhos, a árvore cresce, os centróides
  andam até aos grupos, a luz passa pela rede…
  - O desenho escolhe-se pela família e pelas tarefas (`glyphs.ts`), nunca pelo nome: um modelo
    novo de uma família conhecida tem desenho sem mudar a interface.
  - O movimento é CSS, em ciclos lentos e desencontrados.
  - Com "reduzir movimento", cada desenho fica no seu estado de base, que é o desenho completo.
- **Os textos:** o título e o "?" de cada paradigma são temas novos do backend (`supervised` e
  `unsupervised`).

**A barra lateral.**

- **A secção aberta** tem um cartão claro por trás, e ao abrir outra o cartão desliza até ela:
  uma mola curta, com um ressalto pequeno (Motion, o mesmo `layoutId` em todos os itens). O ponto
  aceso do trilho corre pelo trilho até ao passo novo. O cartão passa por baixo da troca entre o
  Manual e os Agentes.
- **Fora dos passos**, um traço de luz à esquerda do cartão marca a secção; nos passos, quem
  marca é o ponto do trilho.
- **A pesquisa** perdeu o botão da barra. Continua no `Ctrl K` (ou `⌘ K`) e no ícone do
  cabeçalho do telemóvel, onde não há teclado.

**Os Dados sem separadores.** As fontes ficam umas por baixo das outras:

1. os exemplos, em cartões;
2. logo a seguir, o ficheiro, numa zona grande: larga-se o ficheiro ou clica-se em qualquer sítio
   dela, e com o ficheiro escolhido aparece o botão de o carregar;
3. no fim, os dados sintéticos.

**O Explorar por partes numeradas.** Os quatro separadores (colunas, correlações, valores em
falta e alvo) saíram. O perfil lê-se em seis partes, todas à vista e pela ordem em que se
percorrem:

1. o resumo;
2. os avisos;
3. as colunas;
4. as correlações;
5. os valores em falta;
6. o alvo.

Por cima delas, um índice fica preso ao topo ao descer. Mostra em que parte se está (o destaque
desliza, como na barra) e salta para qualquer outra.

**O painel de cada agente**, que antes empilhava tudo numa coluna:

- **a largura:** passou a 46rem, pela variante certa do `Sheet` (`data-[side=right]:sm:max-w-*`);
  no telemóvel ocupa o ecrã;
- **o cabeçalho, sempre à vista:**
  - a cor e o nome do agente;
  - o que ele faz agora, numa frase;
  - uma fila com todos os agentes, para saltar de um para outro sem voltar ao fluxograma (que o
    painel tapa em parte);
- **os separadores:**
  - o resumo: os números numa fila de quatro e os modelos em cartões;
  - os candidatos;
  - o que o agente disse;
- **ao mudar de agente**, o separador fica, e comparam-se os candidatos de dois agentes num
  clique.

**A lição do reinforcement learning (`ReinforcementService.lesson`).**

- **O que o backend calcula:** um Q-learning no penhasco, com uma seed fixa e 160 episódios.
  Guarda cinco episódios inteiros:
  1. o primeiro;
  2. uma queda dos primeiros;
  3. o primeiro êxito;
  4. um do meio, já a chegar ao objetivo;
  5. o último.
- **Cada passo** diz o estado, a ação, se foi ao acaso, a recompensa, a surpresa (o erro de
  diferença temporal) e o valor novo da célula.
- **Cada episódio** traz o valor de cada célula no início e uma frase sobre o que nele se vê.
- **Não é um run:** não vai para o histórico. É sempre a mesma, por isso calcula-se uma vez e
  fica guardada.
- **O ε-greedy** da base dos agentes passou a ter um `_decide`, que diz se a ação foi ao acaso. O
  `_choose` usa-o, com o mesmo uso do acaso, e o treino não muda.
- **O que a interface anima:** o agente anda na grelha, a recompensa salta da célula onde ele
  chega, e a cor de cada célula muda quando ele aprende com ela.
- **O ciclo de cada passo** fica em quatro cartões que acendem à vez: o estado, a ação, a
  recompensa e a correção. Os textos são temas novos do backend (`rl_state`, `rl_action`,
  `rl_reward` e `rl_update`).
- **Os controlos:** a lição começa sozinha quando aparece no ecrã, pára quando sai dele, e
  tem pausa, velocidade e um separador por episódio.

## Consequências

- **O que substitui:**
  - a cor dos agentes da ADR 0034 (a cor pela posição no plano);
  - os separadores dos Dados e do Explorar.
- **Uma exceção à ADR 0032** ("as explicações não ficam à vista"): os textos dos quatro cartões
  da lição estão à vista. A lição é uma explicação; escondê-los num "?" tirava-lhe o sentido.
- **Uma família nova no backend** precisa de uma cor em `lib/families.ts`: o tipo
  `Record<ModelFamily, string>` falha no `typecheck` até ela existir (sem cor, a família ficava
  cinzenta).
- **O contrato** ganhou `GET /reinforcement/lesson` e os tipos da lição (`ReinforcementLesson`,
  `LessonEpisode`, `LessonStep`), no `openapi.json`.
- **Os testes de ponta a ponta:** o caminho à mão salta para as correlações e para o alvo pelo
  índice do Explorar, em vez dos separadores.
- **Ficou de fora:** os outros painéis laterais (o "?", o de um run e o de uma métrica) têm o
  mesmo limite escondido de 24rem do `Sheet`. Só se alargou o dos agentes, que era o pedido.
