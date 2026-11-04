# 0043 — Orbyte: o nome novo e o site todo em inglês

- **Estado:** Proposta
- **Data:** 2026-11-02

## Contexto

O pedido:

- um nome novo para o projeto, "Orbyte", e um ar mais moderno e profissional;
- o site todo em inglês;
- uma entrada mais simples e futurista, com duas animações, uma para o Manual e outra para os
  Agentes;
- a barra lateral um pouco maior, com o Histórico em baixo;
- a conversa dos agentes em ecrã inteiro, como um chat, com uma linha do tempo das trocas ao
  lado, para se perceber como a procura aconteceu.

O que pesa na decisão:

- os textos didáticos vivem no backend (ADR 0030) e os nomes dos valores vêm do vocabulário
  (ADR 0026): traduzir o site é traduzir os dois lados;
- o projeto continua a ser para aprender em português: a documentação, as docstrings e os
  comentários são em português;
- os números escrevem-se igual nos dois lados (o `formatting.py` e o `format.ts`).

## Decisão

**O nome é Orbyte** em tudo o que se vê: o título, a marca e o logótipo (uma órbita com um byte).
O pacote Python continua `ml_lab`. Mudar o nome do pacote mexia em todos os imports e nos ficheiros
guardados, sem ganho para quem usa o site.

**Tudo o que se lê no site fica em inglês:**

- os textos da interface;
- os do backend: as mensagens de erro, os avisos, os temas do "Porquê?", os nomes do
  vocabulário e a conversa dos agentes;
- os números: ponto decimal e vírgula nos milhares (1,234.5);
- os caminhos: `/data`, `/explore`, `/prepare`, `/models`, `/train`, `/explain`, `/save`,
  `/agents`, `/history`, `/lab`, `/metrics` e `/rl`.

**O que não se vê fica em português:** a documentação, as ADRs, as docstrings, os comentários,
as mensagens dos commits e os nomes internos (os ids do DOM, a chave da sessão no navegador). O
projeto aprende-se em português e usa-se em inglês.

**A entrada:** o nome grande, uma frase e dois cartões. Cada cartão tem uma animação em SVG que
mostra a interface em ciclo:

- o Manual: os passos e um modelo a ajustar a fronteira aos pontos;
- os Agentes: o coordenador a dar a vez aos agentes de cada família, numa órbita.

As animações são gerais (não dependem da API) e ficam paradas com "reduzir movimento".

**A barra:** mais larga (20rem), com o Histórico no rodapé, por cima do estado da API. É para
onde se volta, e fica igual nas duas interfaces.

**A conversa em ecrã inteiro** (`ConversationRoom`) abre-se no botão "Expand" e pode ir para o
ecrã inteiro do browser. É um chat:

- o coordenador fala à direita; os agentes à esquerda, cada um com a cor e o ícone do seu nó;
- à esquerda, a linha do tempo vertical das trocas: uma marca por mensagem, as fases e a janela
  do que se está a ler;
- por baixo, "quem falou", para seguir um agente pela conversa.

## Consequências

- Os testes que comparam textos (os do Python e os de ponta a ponta) mudam com a tradução.
- O `CONTRIBUTING.md` passa a dizer que a interface e os textos do backend são em inglês.
- Os valores dos enums não mudam, só os nomes que o vocabulário lhes dá.
- Uma segunda língua de uso precisaria de traduções por chave. Fica de fora: o site tem uma só
  língua.
