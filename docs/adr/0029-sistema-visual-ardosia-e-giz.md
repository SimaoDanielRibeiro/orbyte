# 0029 — O sistema visual: ardósia e giz

- **Estado:** Substituída por 0032
- **Data:** 2026-10-25

## Contexto

O pedido da interface aprimorada: tons escuros, uma barra lateral que se recolhe em vez da
barra no topo, um painel inicial com os dois caminhos (à mão ou com os agentes), os agentes a
trabalhar à vista, uma página por modelo e as métricas com explicações visuais. O desenho
seguiu o processo do skill `frontend-design`: primeiro um sistema de tokens, depois uma
revisão contra o pedido, para não cair no que qualquer interface escura de ML teria (fundo
quase preto, um acento néon, cartões iguais, números em monoespaçada).

## Decisão

**A ideia.** O ML Lab é um laboratório para aprender; o objeto que melhor o representa é o
quadro onde se dá a matéria. A interface é esse quadro: a moldura é ardósia e giz, a cor fica
para os dados e para o que está a acontecer agora.

**As cores** (`src/styles/globals.css`):

| Nome | Hex | Papel |
|---|---|---|
| Ardósia | `#141A21` | O fundo, onde se desenham os campos e os gráficos |
| Ardósia clara | `#1C242D` | A barra lateral, os painéis, as gavetas |
| Grafite | `#2A3440` | As divisórias (os contornos dos controlos usam `#5B6876`, para terem 3:1) |
| Giz | `#ECE7DD` | O texto (14:1 sobre a ardósia) |
| Giz apagado | `#9AA3AD` | O texto secundário (6,8:1) |
| Giz amarelo | `#F2C94C` | Só o agora: o passo atual, o agente que trabalha, o foco do teclado, os avisos (11:1) |

Os dados usam a paleta categórica validada (azul `#3987E5`, laranja `#D95926`, água
`#199E70`, …); os erros, um giz vermelho (`#F0806E`). Há só o tema escuro.

**Os tipos.** O que se lê é serifado: Literata, com tamanho ótico, nas fichas, explicações,
títulos e legendas. O que se usa é sem serifa: Atkinson Hyperlegible Next, desenhada para
distinguir `I`, `l`, `1`, `0` e `O`, o que importa ao ler `l1_ratio` ou `0,912`. As duas têm
algarismos tabulares, por isso as tabelas alinham os números sem uma monoespaçada. As
fórmulas são do KaTeX. A escala é a de *The Elements of Typographic Style* (12, 14, 16, 18,
21, 24, 36, 48 e 60 px); o texto de leitura tem 17 px, entrelinha 1,65 e até 68 caracteres
por linha. Não há rótulos em maiúsculas.

**A disposição.** A barra lateral tem 256 px aberta e 48 px recolhida (só ícones, com o nome
numa dica); abre e fecha com o botão ou `Ctrl+B`, lembra-se do estado e, no telemóvel, passa
a gaveta. As páginas reorganizam-se pela largura que têm (*container queries*), não pela do
ecrã. Tudo alinha à esquerda; os números à direita nas tabelas; as fórmulas ao centro. Os
raios têm hierarquia: 6 px nos controlos, 11 px nas superfícies.

**Os princípios.**

- O que se lê é serifado; o que se usa é sem serifa.
- O amarelo marca o agora. Nada decorativo é amarelo.
- A cor é dos dados. A moldura é ardósia e giz.
- O movimento mostra o que mudou: responde a ações e mostra a aprendizagem. Há um só momento
  orquestrado, o campo do início a formar-se. Com "reduzir movimento", tudo aparece já no fim.
- O espaço segue a barra.

## Consequências

- Os tokens são variáveis CSS com nomes em inglês (`--slate`, `--chalk-yellow`), e os
  componentes do shadcn leem-nos pelos seus papéis (`--background`, `--ring`…): mudar uma cor
  é mudar uma linha.
- O amarelo é escasso de propósito. Usá-lo para decorar apagava o que ele quer dizer.
- Sem tema claro, os gráficos afinam-se para um só fundo; acrescentar um tema claro mais tarde
  obriga a rever as cores dos dados, não só a trocar o fundo.
- As fontes servem-se do próprio projeto (`@fontsource-variable`), sem pedidos ao Google.
