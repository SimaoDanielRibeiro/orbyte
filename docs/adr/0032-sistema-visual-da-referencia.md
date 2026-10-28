# 0032 — O sistema visual da referência: preto, cinzentos e os dados a cores

- **Estado:** Proposta
- **Data:** 2026-10-28

## Contexto

O pedido do redesenho trouxe uma imagem de referência (um painel escuro) e pediu as mesmas
cores, a mesma letra e o mesmo contraste. O fundo, a moldura e o texto ficam em preto, cinzentos
escuros e branco, e há muita cor nos gráficos. Pediu também menos texto à vista: as explicações
passam para trás de um botão discreto, para as secções ficarem limpas e arrumadas, com o
espaçamento devido entre cartões, ícones e tabelas.

A "ardósia e giz" (ADR 0029) partia de outra ideia: o quadro da sala de aula, a serifa no que se
lê e o amarelo para o agora. Por isso é substituída, e não afinada.

## Decisão

**As cores** (`ui/web/src/styles/globals.css`), medidas na referência:

| Nome | Hex | Papel |
|---|---|---|
| `--ink` | `#0F0F0F` | A barra lateral e a página de entrada |
| `--ground` | `#121212` | O fundo das páginas |
| `--surface` | `#191919` | Os cartões, os campos e as gavetas |
| `--surface-high` | `#232323` | O que está escolhido, ou por baixo do rato |
| `--line` / `--line-strong` | `#262626` / `#383838` | Os contornos dos cartões e dos controlos |
| `--text-strong` | `#FAFAFA` | Os títulos e os números grandes (16,8:1 no cartão) |
| `--text` | `#EDEDED` | O texto (15:1) |
| `--text-soft` | `#A0A0A0` | O texto secundário, os nomes dos cartões (6,7:1) |
| `--text-dim` | `#8A8A8A` | Os rótulos em mono, os eixos, as contagens (5,1:1) |
| `--text-faint` | `#757575` | Só em texto grande (3,8:1) |

O cinzento dos rótulos pequenos da referência (`#6B6B6B`) tem 3,3:1 no cartão. Ficou um degrau
mais claro (`#8A8A8A`) para chegar aos 4,5:1 que o texto pequeno pede.

Há só uma cor de interface além dos cinzentos: o branco do agora (o passo atual, o agente que
trabalha). Os estados têm as suas cores: bom `#62C68C`, aviso `#EAB34D` e erro `#F07070`. Vêm
sempre com texto ou com um ícone, nunca sozinhos.

**Os dados.** A cor é dos gráficos, como na referência. A paleta categórica tem oito cores, por
esta ordem:

1. azul `#6996D4`;
2. âmbar `#E2B94E`;
3. rosa `#D26DA1`;
4. verde `#62C68C`;
5. violeta `#8E7DD8`;
6. laranja `#E0884F`;
7. ciano `#3DB6D1`;
8. vermelho `#E36B6B`.

Foi validada com o script do skill `dataviz` contra o cartão: passa a separação para daltónicos,
a separação em visão normal e o contraste (todas com 5:1 ou mais). Para isso, o verde passou de
`#64C097` para `#62C68C`, para se separar do rosa.

A verificação da banda de luminosidade não passa, de propósito: a referência tem cores vivas, de
luminosidades diferentes, e igualá-las apagava os gráficos. Isto é aceitável porque a identidade
de uma série nunca depende só da cor: há sempre a legenda, os rótulos diretos ou a forma.

As escalas seguem a mesma lógica:

- Sequenciais: uma rampa azul de sete degraus e uma âmbar de cinco, validadas como ordinais.
- Divergente: laranja, cinzento neutro e azul.
- O campo de decisão de duas classes vai do azul ao âmbar: as cores das duas primeiras classes.

As linhas brilham, como na referência: por baixo de cada uma há uma cópia desfocada (um filtro
SVG, `GlowFilter`). O véu do campo de decisão ficou mais leve, para o fundo continuar escuro e os
dados acenderem.

**A letra.** Geist no texto e Geist Mono nos números pequenos, nos eixos e nos rótulos, como na
referência. Servem-se do próprio projeto (`@fontsource-variable`). A Literata e a Atkinson saem.
Os nomes das categorias dentro dos gráficos ficam na letra do texto: a mono é mais larga e não
cabia nas margens.

**As peças** (`components/shell/blocks.tsx`):

- `panel` é a classe do cartão: raio de 16 px, contorno `#262626` e fundo `#191919`.
- `Panel` é um cartão com o nome pequeno e cinzento em cima e o "?" ao lado.
- `Stat` é um número grande com o nome por cima.
- `mono-label` é o rótulo pequeno em mono.
- As tabelas têm o cabeçalho em mono pequeno e cinzento e divisórias subtis. A linha escolhida
  fica um degrau acima do cartão.

**As explicações atrás do "?"** (`components/explain/explain.tsx`):

- `Explain` abre uma caixa ao lado do título de um cartão ou de uma secção. Também guarda o "Como
  se lê" de cada gráfico, que deixou de ficar escrito por baixo.
- `PageExplain` é o "?" do título da página. Abre uma gaveta com o que se faz no ecrã e as ideias
  por trás dele (o antigo painel "Porquê?").
- Os textos continuam a vir do backend, pedidos pelo nome (`topic="confusion_matrix"`), por isso
  `tests/contract/test_topics_in_use.py` continua a encontrá-los (ADR 0030).

## Consequências

- Os tokens continuam a ser variáveis CSS, e os componentes do shadcn leem-nos pelos seus papéis:
  mudar uma cor continua a ser mudar uma linha.
- O texto didático deixa de estar à vista, mas fica a um clique, e o "?" está sempre no mesmo
  sítio, ao lado do título que explica. O preço é que quem aprende tem de dar pelo "?". Por isso
  ele acende ao passar o rato e tem sempre um nome para os leitores de ecrã ("Explicar: …",
  "Como se lê este gráfico").
- O verde do "bom" é o mesmo da quarta série dos gráficos. Não se confundem porque os estados
  vêm sempre com texto ou com um ícone.
- Os gráficos afinam-se só para o tema escuro; um tema claro obrigaria a rever as cores dos
  dados, não só o fundo.
- A ADR 0029 fica substituída por esta.
