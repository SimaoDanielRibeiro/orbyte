# 0044 — Um só botão de prosseguir, a letra maior e a barra alinhada

- **Estado:** Proposta
- **Data:** 2026-11-02
- **Muda:** a [0036](0036-barra-pelos-passos-espaco-e-prosseguir.md) (o botão de prosseguir
  deixa de aparecer outra vez no fim da página)

## Contexto

O pedido:

- os botões de prosseguir estavam repetidos em todas as secções;
- tudo mais organizado, com os espaçamentos e as posições certos;
- a letra maior na barra lateral e no site todo, mais premium;
- a barra lateral ocupada como deve ser;
- uma análise ao código, à procura de bugs.

O diagnóstico:

- **O botão repetido:** o `NextStep` no fim de cada página repetia o botão do topo (ADR 0036).
  No Explorar, o do topo nem estava no sítio dos outros ecrãs: ficava uma linha abaixo do título.
- **As larguras desfasadas:** com a barra a 20rem e as margens de 3 cm, as regras de largura
  (as consultas ao contentor `@container/main`) já não batiam certo:
  - o Preparar só punha o resultado ao lado dos controlos a partir de 1472 px de ecrã;
  - o fluxograma dos agentes precisava de 960 px, tinha 851 e o nó do teste ficava cortado;
  - a tabela do Histórico deslizava para o lado.
- **A barra:** recolhida, os ícones ficavam encostados à direita e cortados (as margens novas
  não tinham versão recolhida). Aberta, os blocos começavam a 12, 16 ou 20 px da borda.

## Decisão

**Um só botão de prosseguir por ecrã, sempre no mesmo sítio: à direita do título.** Fica no
`PageHeader`, ou no cabeçalho próprio dos ecrãs com um run. O `NextStep` saiu.

**A escala de letra sobe um degrau**, nos tokens do Tailwind (`globals.css`):

| Token | Antes | Agora |
|---|---|---|
| `text-xs` | 13 px | 14 px |
| `text-sm` | 15 px | 16 px |
| `text-base` | 16 px | 17 px |
| `text-lg` | 18 px | 19 px |
| `text-xl` | 20 px | 22 px |
| `text-2xl` | 24 px | 26 px |
| `text-title` (novo) | 32 px | 34 px, para o título de cada página |

Os textos dos gráficos em SVG ficam com os tamanhos fixos que já tinham (11 a 13 px). As margens
e os cortes dos gráficos calculam-se para essa letra: a 14 px, as etiquetas compridas saíam do
desenho.

**A barra alinha por duas linhas verticais.** Os itens começam a 20 px de uma borda e acabam a
20 px da outra. O logótipo, o trilho dos passos, os ícones do "Learn", o Histórico e o ponto da
API caem na mesma vertical. Recolhida (56 px), os botões de 36 px ficam ao centro, e o nome de
cada um fica só para os leitores de ecrã.

**As larguras acompanham a barra nova:**

- o Preparar passa a duas colunas a partir de `@5xl` (um portátil de 1440 px); numa coluna, os
  controlos não passam de `max-w-2xl`, porque um interruptor longe do seu nome lê-se mal;
- o fluxograma dos agentes aperta as colunas (de 960 para 922 px) e, quando falta pouco espaço
  (até 15 %), encolhe por igual em vez de rolar; mais estreito do que isso, rola;
- o Histórico põe a tarefa por baixo do dataset e o nome da métrica por baixo do valor; a seed
  só aparece num ecrã largo (também está na gaveta do run);
- a barra de deslocamento tem sempre o lugar guardado (`scrollbar-gutter: stable`), e nada salta
  para o lado entre uma página curta e uma comprida.

## Consequências

- Os testes de ponta a ponta deixam de ter dois botões iguais por página para escolher.
- Uma regra de largura nova tem de contar com a barra de 20rem e as margens de 3 cm: num ecrã de
  1440 px, o conteúdo tem cerca de 880 px.
- Um gráfico novo em SVG usa tamanhos fixos (`text-[12px]`, `text-[13px]`), não os tokens que
  crescem com a escala.
