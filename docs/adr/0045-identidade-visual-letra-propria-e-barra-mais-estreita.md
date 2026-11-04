# 0045 — Identidade visual: uma letra própria nos títulos, rótulos sem maiúsculas e a barra a 18rem

- **Estado:** Proposta
- **Data:** 2026-11-03

## Contexto

A revisão do desenho (ADR 0044) deixou três marcas muito comuns nas interfaces geradas:

- o rótulo em maiúsculas espaçadas por cima do título ("MACHINE LEARNING LAB");
- a letra monoespaçada nos rótulos pequenos (o `mono-label`);
- a seta "→" no fim dos botões de prosseguir.

Nenhuma delas dizia nada sobre o conteúdo: estavam lá por hábito. E toda a página usava a mesma
Geist, sem nada que fosse do Orbyte.

A barra lateral, a 20rem, deixava pouco espaço num portátil: num ecrã de 1440 px sobravam uns
880 px para o conteúdo, e o fluxograma dos agentes e as tabelas largas ficavam apertados.

## Decisão

**Uma letra própria, só nos títulos.** A Unbounded (variável, num pacote `@fontsource`, servida
pela própria app) é larga e geométrica, como uma órbita. Usa-se no nome da entrada, na marca da
barra e nos títulos de página (`h1`, pelo token `--font-display`). O resto continua na Geist: a
letra de leitura não muda, e o contraste entre as duas é o que dá personalidade.

**Rótulos pequenos em sans** (a utilidade `small-label`): a Geist a 14 px, peso médio, cor
apagada, sem maiúsculas nem espaçamento. A monoespaçada fica só onde os algarismos têm de se
alinhar (os valores das tabelas, o tipo de cada coluna, o código).

**Sem setas nos botões.** O botão de prosseguir diz o que faz ("Choose the model"); a seta
repetia a ideia de "ir para a frente", que o lugar do botão (à direita, no cabeçalho) já diz.

**Sem o rótulo por cima do título** na entrada: o que é o Orbyte diz-se na frase de baixo.

**A barra a 18rem** (`SIDEBAR_WIDTH`): ganha 32 px para o conteúdo, com a mesma letra. Os nomes
mais compridos ("Reinforcement learning") continuam a caber numa linha.

**Abaixo de 1024 px a barra é uma gaveta, e os 3 cm de margem só valem a partir de 1280 px.** A
varredura das resoluções (de 375 a 2560 px, em todos os ecrãs) mostrou que, num tablet ao alto
(768 px), a barra fixa e as margens de 3 cm deixavam 243 px ao conteúdo: menos do que no
telemóvel. As margens ficam 16 px no telemóvel, 32 px no tablet e 3 cm num computador, onde o
pedido as pôs. O ponto da gaveta é o mesmo no JavaScript (`useIsMobile`) e nas classes da barra
(`lg:`).

## Consequências

- A página fica com uma identidade (o nome e os títulos) sem perder a legibilidade da Geist.
- Um pacote novo (`@fontsource-variable/unbounded`): pesa uns 30 kB, carrega-se com a app e não
  depende de serviços de fora.
- Quem escrever um ecrã novo usa o `small-label` para os rótulos e deixa a monoespaçada para os
  números alinhados; os botões não levam seta.
- Os gráficos têm o tamanho da letra fixo (13 px) nas etiquetas: a letra maior do resto da página
  (ADR 0044) não os desfaz.
- Entre 768 e 1023 px aparece o cabeçalho do telemóvel (a gaveta e a pesquisa); o teste de ponta a
  ponta do tablet ao alto confirma que o conteúdo fica com a largura toda.
- O fluxograma dos agentes encolhe até 0,8 (antes 0,85), com 24 px entre colunas (antes 32): a
  1280 px, com a barra aberta, cabe inteiro. Mais estreito, rola.
