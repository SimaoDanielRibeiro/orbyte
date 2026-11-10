# 0057 — O analista em ecrã inteiro, as respostas sem caixa e os passos pelo que ele fez

- **Estado:** Proposta
- **Data:** 2026-11-09
- **Muda:** a [0054](0054-o-analista-numa-conversa-o-quadro-inteiro-e-pronto-para-outros-computadores.md)
  (o painel do analista ganha o ecrã inteiro, a resposta sai da bolha e os passos arrumam-se pelo
  que ele fez)

## Contexto

A ADR 0054 pôs o analista numa conversa: um cartão de altura fixa ao centro da página, em que só a
conversa desliza. Ao usá-lo, ficaram três coisas por resolver:

1. **Num portátil, a conversa tem pouco espaço.** O cartão tem 78% da altura da janela (entre
   32rem e 52rem), e o cabeçalho, as sugestões e a caixa da pergunta levam mais de 200 píxeis:
   numa janela de 1536 × 730 (um portátil Full HD com a escala a 125%, menos as barras do
   browser), a conversa fica com 353 píxeis de altura, menos de metade. Uma resposta com os passos
   não cabe, e lê-se sempre a deslizar numa janela pequena. A conversa dos agentes já tinha a
   saída para isto: o "Expand", em ecrã inteiro (`ConversationRoom`).
2. **A resposta numa caixa, e a confirmação por extenso.** A bolha com borda à volta de uma
   resposta de vários parágrafos pesa mais do que o texto. A nota "Every number in the answer is
   in what the tools returned." ocupava uma linha inteira, com o mesmo peso de um aviso, quando é
   uma boa notícia. E não havia como levar a resposta para outro lado: selecioná-la à mão
   apanhava também os passos.
3. **Os passos eram difíceis de ler.** A lista seguia o rasto pela ordem: pensamento, ferramenta,
   pensamento, ferramenta, pensamento. Cada pensamento fechado mostrava o princípio do texto, em
   letra de máquina, e os do `qwen3:8b` começam quase todos da mesma maneira ("Okay, let's tackle
   the user's question…"): três linhas iguais a separar as duas que dizem o que ele fez.

## Decisão

**O "Expand" abre o analista em ecrã inteiro**, como a conversa dos agentes: um diálogo do Radix
que ocupa a janela, com o mesmo cabeçalho (o nome, o modelo e o "Collapse" para voltar), a conversa
ao centro, numa coluna de 48rem, e a caixa da pergunta em baixo, com a dica do Enter e do
Shift+Enter (só nos ecrãs com teclado), como num chat. Fecha-se no "Collapse" ou com o Esc, e o
foco volta ao "Expand".

- A conversa é a mesma nos dois sítios. O estado (as respostas, a pergunta a decorrer, o que se
  está a escrever) vive num hook do painel (`useAnalystChat`), e o cartão e o ecrã inteiro
  desenham-no com o mesmo componente (`Chat`).
- Mostra-a um sítio de cada vez: com o ecrã inteiro aberto, o cartão fica na página com a mesma
  altura e um aviso no meio, para nada saltar ao fechar. Assim, a pergunta a decorrer tem sempre um
  só a segui-la pelo WebSocket; mudar de sítio volta a pedir os passos, como sair do ecrã e voltar
  (o `useQuestion` já o sabia fazer).
- Ao abrir, o foco vai para a caixa da pergunta, se der para perguntar (sem o Ollama, fica no
  diálogo).
- O cartão continua com a altura fixa da 0054: na página, o analista é uma parte entre outras, e o
  ecrã inteiro é para quando se quer conversar.
- Nos dois sítios, a caixa da pergunta tem a largura das mensagens (era 2rem mais larga de cada
  lado), como nos chats: a conversa e a pergunta ficam na mesma coluna.
- A conversa é o bloco de contenção (`relative`) do que tem lá dentro com posição absoluta. Os
  textos só para os leitores de ecrã (`sr-only`, como o "The answer was copied." do "Copy" e a
  mensagem do selo) têm posição absoluta; sem isso, escapavam ao recorte da conversa e ficavam
  abaixo do cartão. O cartão, com `overflow-hidden`, não desliza com a roda do rato, mas desliza
  quando o foco ou um leitor de ecrã lá chega: o cabeçalho desaparecia e ficava um vazio em baixo
  (mediu-se um cartão de 567 píxeis com 908 de conteúdo).

**A resposta fica em texto corrido, sem caixa**, alinhada com o nome do analista, como nos chats.
Por baixo, numa linha:

- **"Copy"**, que copia a resposta como o analista a escreveu (em Markdown) e diz "Copied" durante
  dois segundos, também aos leitores de ecrã;
- **o selo "Numbers checked"**, quando o backend dá o aviso `numbers_checked`: a mensagem dele, que
  diz o que se verificou, aparece na dica e vai para os leitores de ecrã;
- o modelo, as ferramentas e o tempo, como antes.

Os outros avisos da resposta (um número que nenhuma ferramenta devolveu, uma resposta sem
ferramentas, uma resposta forçada) continuam por extenso, entre a resposta e essa linha: são o que
é preciso ver.

**Os passos contam-se pelo que o analista fez** (`analyst-steps.ts`). Cada vez que o modelo
responde, pensa e depois pede ferramentas, ou escreve a resposta (`Analyst.answer`): junta-se cada
pensamento ao que ele fez a seguir, numa volta. A lista mostra cada ferramenta (o nome, a chamada e
o que devolveu numa frase) e, no fim, a resposta. Por baixo de cada passo, fechados, ficam o
pensamento que o pediu ("Thought · 52 words": o tamanho, em vez do princípio) e o que a ferramenta
devolveu. O pensamento abre na letra do resto, com a entrelinha de ler, e o que cresce ao vivo
também.

## Consequências

- Num portátil, o analista pode ocupar a janela toda: em 1536 × 730, a conversa passa de 353 para
  493 píxeis de altura, e num telemóvel (390 × 844), de 441 para 607. A coluna da conversa fica
  com a largura de um chat.
- A resposta lê-se como num chat, e a confirmação dos números fica à vista sem pesar. Um aviso a
  sério continua a destacar-se: é o único texto por extenso por baixo da resposta.
- A lista dos passos fica mais curta e diz o que ele fez: na pergunta dos testes de ponta a ponta,
  três passos em vez de cinco. Os pensamentos continuam lá, inteiros, a um clique.
- "Numbers checked" é um texto da interface: o nome curto do aviso `numbers_checked`, que se mostra
  como selo. Os avisos não têm nome curto no backend, e para um só não se justificava acrescentá-lo
  ao `Notice`; a mensagem, que explica o que se verificou, continua a vir do backend.
- Os testes de ponta a ponta do analista acompanham: os passos pelo que ele fez, o pensamento
  fechado com o tamanho, o selo com a dica, o "Copy" (com a área de transferência do browser) e um
  teste novo para o ecrã inteiro (abrir com o rascunho, perguntar lá dentro, a acessibilidade e
  voltar ao cartão com a resposta nova).
- A fotografia do analista no README (`docs/images/analyst.png`) voltou a tirar-se com o script
  das imagens (`e2e/readme.shots.ts`, só o teste dos agentes), como a 0054 pede quando a interface
  muda.
