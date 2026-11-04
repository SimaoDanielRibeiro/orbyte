# 0049 — Os agentes com todos os modelos, escolhidos à mão, a falar por palavras; os ícones fixos

- **Estado:** Proposta
- **Data:** 2026-11-04
- **Muda:** a [0025](0025-agentes-de-automl-deterministas.md) (as implementações didáticas deixam
  de ficar de fora), a [0037](0037-cores-por-familia-barra-que-desliza-e-licao-do-rl.md) (o trilho
  dos passos passa pelos ícones) e a [0048](0048-modelos-por-problema-letra-maior-e-movimento.md)
  (a largura e a letra da barra)

## Contexto

O pedido:

- que os agentes procurem com todos os modelos disponíveis;
- poder escolher que modelos entram na procura, com o ecrã arrumado à volta dessa escolha;
- que a conversa dos agentes tenha texto que se perceba, e não só valores;
- a barra lateral maior, com a letra maior, e os ícones na mesma posição com a barra aberta e
  recolhida.

O diagnóstico:

- **Faltavam modelos.** As implementações didáticas com uma versão das bibliotecas na família (a
  regressão logística e a linear por descida do gradiente, o perceptrão, o KNN em NumPy) ficavam de
  fora, por serem mais lentas. Para quem aprende, ver se dão o mesmo que as das bibliotecas também
  é uma lição.
- **Só se escolhia a família.** As fichas "The agents called" ligavam e desligavam famílias
  inteiras: não havia como tirar um modelo e deixar os outros.
- **A conversa eram números.** "Random forest (regression) with max_features log2: 0.725 ± 0.020,
  better than the 0.707 it already had": os nomes do código, sem o valor de partida, sem dizer o
  que 0,725 quer dizer. Os pedidos do coordenador eram só "it's your turn to tune X", cortados numa
  linha.
- **Os ícones mudavam de sítio.** Aberta a barra, os ícones dos passos ficavam 24 px mais à direita
  do que os do "Learn" (os pontos do trilho estavam à esquerda); recolhida, iam todos para o meio,
  e o rótulo do "Learn" e a troca de interface encolhiam. Tudo se mexia, na horizontal e na
  vertical.

## Decisão

**Entram todos os modelos da tarefa**, menos os que não se podem medir com o objetivo: os que não
preveem dados novos e, para a ROC AUC e a log-loss, os que não dão probabilidades. O
`include_didactic` sai do pedido.

**O pedido escolhe os modelos** (`AutoMLRequest.models`, pelo nome no registo; vazio = todos). As
famílias continuam (`families`), e entra o que estiver nas duas escolhas. Um nome que não é da
tarefa é recusado, com o que fazer. `AutoMLOptions.agents` lista todos os modelos que podem entrar
com a métrica escolhida (a opção `objective` muda-os), `skipped` os que não podem, com a razão, e
`SkippedModel.chosen_out` distingue quem ficou de fora por escolha de quem ficou por não se poder
medir.

**O ecrã Agentes arruma-se à volta da escolha:**

- à esquerda, **o pedido**: o alvo, a métrica, o tamanho da procura e as opções;
- à direita, **os modelos**, por família: a caixa da família marca os dela todos e fica a meio
  quando só alguns estão escolhidos; "All", "None" e a contagem ("4 of 12 models") por cima;
- por baixo, **quem vai trabalhar** (o fluxograma, que segue a escolha) e **o plano** do agente de
  dados.

"Ask the agents" passa para o cabeçalho, como o passo seguinte de todos os ecrãs (ADR 0044).

**A conversa diz por palavras o que se faz**; os números ficam no cartão do candidato, ao lado:

- o pedido do coordenador diz porquê: a referência ("the result the other models must beat"), a
  primeira impressão ou, na afinação, onde o modelo está ("1.1 points behind the leader, Decision
  tree"). E lê-se inteiro, sem se cortar numa linha;
- a resposta do agente diz que hiperparâmetros mudou, pelo nome que se mostra e com o valor de
  omissão ao lado ("maximum depth 10 (default 5)"), o resultado e o que ele quer dizer
  (`Objective.reading`: "It gets about 78 in every 100 rows right", "It explains about 75% of the
  variation in the target"), se a mudança ajudou, se o modelo passou a liderar e se decora o treino
  (`Objective.overfits`: o treino mais de dez pontos acima, numa métrica de 0 a 1, ou o erro da
  validação 50% maior);
- um empate à vista (96,6% contra 96,6%) diz-se empate, e não "ganha";
- os valores especiais têm nome (`x-labels`: a profundidade 0 é "no limit", na conversa e no
  formulário).

**A barra lateral cresce** para 21rem (a letra a 19 px e os ícones a 22 px), e para 4.25rem
recolhida. **Os ícones não se mexem:** com a barra aberta ou recolhida, o centro de cada um fica a
35 px da borda, à mesma altura. Para isso:

- o ícone fica sempre a 11 px do contorno do botão. Centrá-lo só na barra recolhida fazia-o saltar
  durante a animação;
- recolher só estreita os botões e apaga os nomes (uma transição de opacidade);
- o rótulo do "Learn" apaga-se mas guarda o lugar, e a troca de interface tem 48 px de altura nas
  duas formas.

O trilho dos passos passa pelo centro dos ícones, como as paragens de uma linha (um disco da cor da
barra tapa-o à volta de cada ícone). Acende do primeiro passo até ao aberto e fica também com a
barra recolhida.

## Consequências

- Uma procura com tudo é mais lenta (há mais modelos na triagem); quem tem pressa desmarca.
- Os testes que contavam com os didáticos de fora escolhem agora os modelos (`models=FAST`).
- As mensagens são mais compridas: quem as lê vê o porquê, e o histórico das trocas tem mais texto
  para mostrar.
- Uma métrica nova dos agentes precisa da sua leitura em `_READINGS` (`objective.py`); o teste
  falha sem ela.
- Uma regra de largura nova tem de contar com a barra de 21rem.
