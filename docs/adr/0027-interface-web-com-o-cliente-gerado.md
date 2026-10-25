# 0027 — A interface web: React, com o cliente gerado do contrato

- **Estado:** Proposta
- **Data:** 2026-10-25

## Contexto

A fase 10 refaz a interface em React, por cima da API (ADRs 0005, 0022 e 0026). O pedido
fixou o núcleo: React, TypeScript, Vite, Tailwind CSS, shadcn/ui e Motion, nas versões mais
recentes. Ficavam por decidir as peças à volta, com quatro forças em jogo:

1. **O contrato já existe** (`ui/web/openapi.json`, ADR 0026), com os nomes das operações
   iguais aos métodos dos serviços (`experiments_start`). Escrever à mão os tipos e os pedidos
   seria duplicar o backend, e deixá-los ficar para trás.
2. **O mais recente nem sempre encaixa.** O TypeScript 7.0 (o compilador nativo) é o mais
   recente, mas o `typescript-eslint` 8.72 só aceita versões abaixo da 6.1, e o
   `openapi-typescript` 7.13 só aceita a 5.
3. **O treino é em tempo real**, por WebSocket, e as respostas de um modelo iterativo rondam os
   2 MB: os dados do servidor precisam de uma cache que se saiba invalidar.
4. **O projeto é para aprender**: cada ferramenta deve ter um papel claro, de preferência com
   um paralelo no lado do Python.

## Decisão

| Peça | Escolha | Porquê |
|---|---|---|
| Linguagem | TypeScript 6.0 (`strict`, `noUncheckedIndexedAccess`) | O 6.0 é a ponte para o 7: aceita as ferramentas de hoje, e com uma configuração limpa no 6 a mudança para o 7 fica quase reduzida a trocar a versão |
| Cliente da API | `@hey-api/openapi-ts`, a partir do `openapi.json` | Gera os tipos, uma função por operação (`experimentsStart`) e as opções do TanStack Query. Aceita o TypeScript 6 |
| Dados do servidor | TanStack Query | A cache dos pedidos: o vocabulário pede-se uma vez, um resultado de 2 MB também |
| Rotas | TanStack Router, com rotas em ficheiros | Parâmetros tipados (`/modelos/$name`), carregamento antes de mostrar e código dividido por rota |
| Estado do cliente | Zustand, no `localStorage` | Só o pouco que é do cliente: o caminho escolhido e a barra lateral aberta ou fechada |
| Componentes | shadcn/ui, com o Radix | O código fica no projeto (`components/ui/`), para o adaptar ao desenho (ADR 0029) |
| Animação | Motion | As transições que respondem a uma ação e o fluxograma dos agentes |
| Gráficos | Módulos do D3 e componentes próprios, em SVG e canvas | O Plotly pesa vários MB e o seu aspeto choca com o desenho; os campos, as matrizes e a rede desenham-se melhor à medida |
| Fórmulas | KaTeX | É o que as fichas já pedem |
| Verificação | ESLint (`strictTypeChecked`) e Prettier; Vitest com a Testing Library; Playwright com o axe | O paralelo do ruff, do mypy e do pytest; o Playwright e o axe testam a interface inteira, com a acessibilidade |
| Pacotes | npm | Já vem com o Node 24; não é preciso instalar mais nada |

E três regras de funcionamento:

1. **O cliente gera-se, não se escreve.** `npm run api` gera `src/api/generated/` a partir do
   `openapi.json`; não vai para o Git e corre sozinho antes de `dev`, `build`, `typecheck`,
   `lint` e `test`. A árvore das rotas (`routeTree.gen.ts`) vai para o Git, para os tipos
   funcionarem logo depois de clonar.
2. **A interface e a API ficam na mesma origem.** Os pedidos vão para `/api`, e o servidor do
   Vite passa-os à API (os WebSockets incluídos). O endereço lê-se das mesmas variáveis que a
   API usa (`ML_LAB_API_HOST` e `ML_LAB_API_PORT`).
3. **Os números escrevem-se como no Python.** `lib/format.ts` segue `ml_lab/formatting.py`,
   com o arredondamento metade para o par; os testes usam os mesmos exemplos.

## Consequências

- Uma mudança no backend chega ao frontend como erro de tipos: regenerar o `openapi.json` e o
  cliente mostra tudo o que deixou de bater certo.
- Os erros da API chegam sempre como `ApiRequestError`, com o `code` estável e a `message` em
  português; um pedido sem resposta tem o código `api_unreachable`.
- Ficamos no TypeScript 6 até o `typescript-eslint` aceitar o 7.
- Os componentes do shadcn são código nosso: atualizam-se à mão, quando valer a pena.
- Há duas linguagens no projeto, cada uma com as suas verificações: a fase só acaba quando
  passam as do Python e as da web.
