# 0033 — Duas interfaces e uma entrada: o Manual e os Agentes, cada um com a sua barra

- **Estado:** Proposta
- **Data:** 2026-10-28

## Contexto

Até aqui, a interface web era uma só. Tinha dois caminhos, escolhidos no início (à mão ou com os
agentes), e a barra lateral mostrava os passos do caminho escolhido. O pedido do redesenho foi
separar as duas formas de trabalhar:

- uma página de entrada, sem barra lateral, com o nome da app e as duas escolhas;
- depois, duas interfaces semelhantes, cada uma com as suas secções na barra.

## Decisão

**A entrada** (`/`) fica fora da moldura. Mostra o nome, uma frase e dois cartões, um por
interface. Cada cartão tem:

- um desenho vivo: a rede neuronal a aprender as luas, ou o coordenador a distribuir o trabalho
  pelas famílias de modelos;
- os passos da interface;
- o botão que entra nela.

**Cada interface tem a sua barra.** `NAVIGATION` (`components/shell/screens.ts`) diz os grupos
e os ecrãs de cada uma:

| Interface | Grupos e ecrãs |
|---|---|
| Manual | Fluxo: Dados, Explorar, Preparar, Modelos, Treinar e avaliar, Explicar. Aprender: Laboratório, Métricas, Reinforcement learning. Histórico: Experiências |
| Agentes | Fluxo: Dados, Agentes, Treinar e avaliar, Explicar. Consultar: Modelos, Métricas. Histórico: Experiências |

Os ecrãs são os mesmos nas duas interfaces (as mesmas rotas e os mesmos serviços); só a barra
muda. A interface escolhida fica guardada no browser, com o resto da sessão do laboratório (no
`localStorage`).

Alguns ecrãs só existem numa interface (`ONLY_IN`): o Explorar, o Preparar, o Laboratório e o
RL são do Manual, e os Agentes são da interface com o mesmo nome. Quando um deles se abre por
um endereço, a moldura passa para a interface dele. Assim, uma ligação partilhada mostra sempre
a barra certa.

**A troca** fica no topo da barra, num grupo de alternância ("Interface"). Quem troca fica no
mesmo ecrã, se ele existir na outra interface; se não existir, vai para os Dados, onde as duas
começam.

**A pesquisa** abre com `Ctrl K` (`⌘ K` no Mac) ou com o botão no topo da barra. Salta para um
ecrã, um modelo ou uma métrica sem passar pela barra (`cmdk`, dentro de um diálogo do Radix).

**O backend não muda.** Que barra se mostra é uma decisão de apresentação, por isso fica na
interface (regra 4 do `CONTRIBUTING.md`).

## Consequências

- Cada interface mostra menos coisas, e só as dela. O Manual ensina passo a passo; os Agentes
  vão direto do pedido ao resultado.
- Os ecrãs comuns (Dados, Treinar e avaliar, Explicar, Modelos, Métricas e Experiências) não se
  duplicam. Quando um ecrã mostra uma coisa só de uma interface, a escolha vem da sessão: o
  passo seguinte dos Dados, por exemplo, é "Explorar este dataset" ou "Pedir aos agentes".
- Um ecrã novo tem de entrar em `NAVIGATION` (numa interface ou nas duas) e, se for só de uma,
  em `ONLY_IN`. Os testes de `screens.test.ts` verificam que todos os ecrãs estão numa das
  barras e que um ecrã de uma só interface está só na barra dela.
- Os testes de ponta a ponta entram pela página de entrada ou diretamente pelo endereço de um
  ecrã. A barra já não existe em `/`.
