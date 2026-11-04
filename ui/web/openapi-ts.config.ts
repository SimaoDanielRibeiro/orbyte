/**
 * Gera o cliente da API a partir do contrato guardado (`openapi.json`).
 *
 * O contrato é a fonte da verdade: gera-se do lado do Python (`scripts/export_openapi.py`) e
 * um teste avisa quando fica desatualizado. Daqui saem os tipos, uma função por operação
 * (`experimentsStart`, `automlUpdates`…) e as opções do TanStack Query. O resultado fica em
 * `src/api/generated/`, fora do Git, e volta a gerar-se antes de `dev` e de `build`, e na
 * verificação (`npm run check`). O lint, os tipos e os testes usam o que já está gerado: gerar
 * enquanto o `dev` corre reescrevia os ficheiros que ele vigia (ADR 0027 e 0046).
 */
import { defineConfig } from "@hey-api/openapi-ts";

export default defineConfig({
  input: "./openapi.json",
  output: { path: "./src/api/generated" },
  plugins: [
    { name: "@hey-api/client-fetch", runtimeConfigPath: "./src/api/client-config.ts" },
    "@hey-api/typescript",
    "@hey-api/sdk",
    "@tanstack/react-query",
  ],
});
