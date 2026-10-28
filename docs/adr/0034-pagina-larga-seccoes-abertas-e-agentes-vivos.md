# 0034 — A página larga, as secções abertas e o fluxograma vivo dos agentes

- **Estado:** Proposta
- **Data:** 2026-10-28

## Contexto

Depois do redesenho (ADR 0032 e 0033), o pedido seguinte trouxe quatro pontos:

- **O espaço:** ocupar mais a página, com só 2 cm de margem de cada lado, de forma simétrica e
  bem arquitetada.
- **As caixas:** evitar os cartões retangulares, e usá-los só onde fizerem mesmo sentido.
- **O fluxograma:** os agentes a trabalhar, com o movimento de um vídeo de referência (um mapa de
  nós ligados por curvas, em que os nós entram um a um e uma luz percorre as ligações).
- **O detalhe:** ao clicar num agente do fluxograma, ver ao lado tudo o que ele fez, com os
  valores de cada treino, para se perceber como os agentes chegaram à conclusão.
- **A conversa:** uma cor por agente, para se ir identificando quem fala.

## Decisão

**A página.**

- As páginas largas ocupam o ecrã todo, com `2cm` de margem de cada lado (só 16 px no telemóvel).
  As de leitura continuam numa coluna centrada.
- Entre colunas fica um espaço largo e igual (`gap-x-12 gap-y-10`), e é ele que separa as
  secções.

**As secções abertas.**

- O `Panel` deixou de ser uma caixa: é uma secção com um filete por cima, o nome e o "?".
- Os números grandes passaram para uma faixa (`StatStrip`), lado a lado e separados por filetes,
  como o mostrador de um instrumento. A versão `compact` mostra dois por linha, para colunas
  estreitas e para o painel lateral.
- As mensagens vazias passaram a texto simples.

A caixa (a classe `panel`) ficou só onde faz sentido:

- o que se escolhe (os cartões da entrada e a lista dos modelos);
- o aviso de um passo que falta;
- os diálogos e a pesquisa.

**O fluxograma dos agentes.**

- Os nós são HTML, por cima das ligações em SVG. Quando há painel para abrir, são botões, com
  foco de teclado e o nome cortado com reticências quando não cabe.
- As colunas laterais têm largura fixa. É o espaço das ligações que estica com a página, e é aí
  que cabem as etiquetas.
- O movimento:
  - ao abrir, os nós entram coluna a coluna e as ligações desenham-se atrás deles;
  - na ligação ativa corre uma luz, com a cor do agente, na direção do trabalho;
  - nas ligações dos agentes que já trabalharam passa, de vez em quando, uma luz lenta.
- As ligações têm etiquetas em pílula: as linhas que os dados levam ao coordenador, o fold que o
  agente ativo mede e a nota do teste.
- Com "reduzir movimento" não há entradas nem luzes; o estado lê-se nas cores.

**O painel de um agente.** Abre-se ao lado e não é modal: o fluxograma continua à vista e a
mexer, e clicar noutro nó troca o painel em vez de o fechar. O conteúdo depende do nó:

- **Um agente de família:**
  - o que mede agora, com o fold;
  - os números do seu melhor candidato (validação, treino, quantos mediu, o lugar no quadro ou a
    distância ao líder);
  - os modelos que pode experimentar e o que afina em cada um;
  - cada candidato, com a nota em cada fold (um ponto por fold e a média a branco);
  - o que disse na conversa.
- **O agente de dados:** o plano e o que disse.
- **O coordenador:** o progresso, como repartiu os candidatos pelos agentes e o que disse.
- **O líder e o teste:** o melhor candidato, com todos os hiperparâmetros, cada fold, a nota no
  teste e a ligação para o abrir.

Tudo vem do que a procura já devolve (os candidatos com os folds, a conversa e o plano), por
isso o backend não muda. O painel só junta os dados por agente.

**A cor de cada agente** (`agent-colors.ts`) é a mesma no fluxograma, na conversa e no painel:

- cada família tem a cor da sua posição no plano;
- o agente de dados é branco;
- o coordenador é cinzento.

Na conversa, a cor está no ícone e num filete à esquerda da mensagem. O texto fica nas cores do
texto.

**A disposição dos agentes.**

- **No pedido:** o fluxograma fica em cima, a toda a largura, e por baixo o pedido e o plano, em
  duas colunas iguais.
- **Na procura:** a conversa fica por baixo do fluxograma. Só se põe ao lado em ecrãs muito
  largos (`@8xl`, 88rem), para o fluxograma nunca ficar apertado.

## Consequências

- A página aproveita a largura, e a estrutura vem do espaço e dos filetes. Isto substitui a
  parte da ADR 0032 em que "cada coisa vive num cartão".
- O fluxograma precisa de 868 px. Abaixo disso, rola na horizontal dentro da sua área (nunca a
  página).
- Os testes de ponta a ponta não mudaram: os títulos e os papéis que procuram ficaram iguais (a
  conversa continua a ser a secção do título "A conversa dos agentes").
