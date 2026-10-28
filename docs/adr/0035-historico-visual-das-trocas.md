# 0035 — O histórico visual das trocas: cada mensagem diz a quem se dirige

- **Estado:** Proposta
- **Data:** 2026-10-28

## Contexto

O pedido: um histórico visual das trocas entre os agentes. Uma animação mostra o que o
coordenador mandou a um agente, o que esse agente lhe respondeu, e assim por diante, até tudo
chegar ao fim para se decidir o vencedor. Assim vê-se como tudo aconteceu.

A conversa já tinha as mensagens por ordem, mas só dizia quem falava. Faltavam duas coisas:

- **A quem se dirige cada mensagem.** Um anúncio vai para os agentes, uma resposta vai para o
  coordenador, a decisão vai para o vencedor. Sem isso, a interface teria de o adivinhar pelo
  papel e pelo texto, e decidir isso é uma regra do domínio (regra 4 do `CONTRIBUTING.md`).
- **O pedido de cada candidato.** O coordenador dava a vez a um agente sem o dizer. Na conversa
  só aparecia a resposta, e a ida não se via.

## Decisão

**No backend** (`core/automl/search_log.py`), cada `AgentMessage` ganhou dois campos:

- `kind` (`MessageKind`), o que a mensagem é na troca:
  - `report`: o agente de dados entrega o plano ao coordenador;
  - `announcement`: o coordenador anuncia aos agentes o plano, as rondas e os finalistas;
  - `request`: o coordenador dá a vez a um agente;
  - `reply`: o agente responde ao coordenador com o resultado;
  - `decision`: o coordenador passa o melhor ao vencedor;
  - `test`: o vencedor vai ao teste, e a leitura do teste fecha a conversa.
- `recipients` (`Participant`), a quem se dirige: o agente de dados, o coordenador, o agente de
  uma família, o vencedor ou o teste. Vazio quer dizer "a quem acompanha a procura" (é o caso da
  leitura do teste).

**O pedido de cada candidato** passou a ser uma mensagem, dita antes de cada medição:

- na triagem: "Agente · Árvores de decisão, mede Árvore de decisão com os valores por omissão";
- na afinação: "…, é a tua vez de afinar…".

O pedido traz o número do candidato que vai sair dele (`candidate`). A decisão e o teste trazem o
número do vencedor.

**Os nomes a mostrar.** Os dois enums novos estão no vocabulário, e o tema `exchange_history`
explica como se lê o histórico.

**Na interface**, o ecrã de uma procura tem um interruptor por cima do fluxograma: "Ao vivo" (ou
"No fim") e "Histórico das trocas". O histórico (`exchange-history.tsx`) repete a conversa:

- cada mensagem é uma luz que parte de quem fala, segue pelas ligações do fluxograma e acende
  quem ouve. Numa resposta, a luz faz o caminho ao contrário;
- a decisão e o teste vão pelo agente do vencedor: vê-se o vencedor a sair do agente que o
  propôs, a chegar ao vencedor e a seguir para o teste;
- o fluxograma mostra a procura como estava nesse momento: os candidatos já medidos, quem
  liderava e o agente a quem o coordenador acabou de dar a vez;
- por baixo ficam o que se disse (de quem, para quem, que tipo de mensagem) e a linha do tempo,
  com uma marca por mensagem na cor de quem a disse, mais alta quando decide mais, e as fases
  por baixo. A linha do tempo arrasta-se com o rato e funciona com as setas, o Home e o End;
- a reprodução começa sozinha, pode pausar-se, avançar ou recuar, e tem três velocidades;
- a conversa ao lado acompanha o histórico: marca a mensagem à vista e diz a quem cada uma se
  dirige. Os pedidos ficam numa linha curta.

A luz é um `animateMotion` do SVG, que segue exatamente o caminho desenhado. Com "reduzir
movimento" não há luz, só o rasto.

## Consequências

- **O contrato mudou** (dois campos e dois enums novos), por isso o `openapi.json` foi
  regenerado e o cliente da interface voltou a gerar-se. No Python, os campos têm valores por
  omissão (um anúncio para quem acompanha), por isso quem chama `SearchLog.say` sem os dizer
  continua a funcionar.
- **A conversa ficou mais longa:** há um pedido por candidato, mais 24 mensagens numa procura de
  24. Os pedidos mostram-se numa linha só, para a leitura não pesar.
- **Quem liderava num momento passado** calcula-se na interface, com a mesma regra do
  coordenador (`bestCandidate`, que espelha `rank`). Ao vivo, o líder continua a vir da API.
- **Os testes** verificam que cada candidato tem um pedido seguido da resposta, que os anúncios
  vão para os agentes, que a decisão vai para o vencedor, e o fim da conversa no serviço. Há
  também um passo novo no teste de ponta a ponta dos agentes.
