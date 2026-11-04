# 0046 — O cliente da API gera-se só no `dev`, no `build` e na verificação

- **Estado:** Proposta
- **Data:** 2026-11-03

## Contexto

O cliente TypeScript da API gera-se do contrato guardado (`ui/web/openapi.json`, ADR 0027). Até
aqui, o `npm run api` corria antes de quase tudo: do `lint`, do `typecheck` e dos testes, além do
`dev` e do `build`.

Isso tinha dois custos:

- cada verificação demorava mais, a gerar outra vez o mesmo cliente;
- a geração reescreve os ficheiros de `src/api/generated`; com o servidor do Vite a correr, cada
  reescrita disparava um recarregamento, e uma geração a meio de uma edição chegou a derrubar o
  servidor.

## Decisão

O cliente gera-se só onde faz falta:

- **`npm run dev`** e **`npm run build`** (pelos `predev` e `prebuild`): quem arranca a app ou a
  compila parte sempre do contrato atual;
- **`npm run check`**: a verificação completa (gera o cliente, depois o lint, os tipos e os
  testes), para usar antes de um commit.

O `lint`, o `typecheck` e o `test` sozinhos já não geram nada: usam o cliente que lá está.

## Consequências

- O ciclo rápido (lint, tipos, testes) fica mais rápido, e não mexe nos ficheiros que o Vite vê.
- Depois de mudar o contrato (`scripts/export_openapi.py`), é preciso gerar o cliente: o `dev`, o
  `build` ou o `check` fazem-no; à mão, `npm run api`.
- O `CONTRIBUTING.md` e o `ui/web/README.md` passam a dizer `npm run check` como a verificação.
