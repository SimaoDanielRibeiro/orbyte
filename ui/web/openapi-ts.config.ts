/**
 * Gera o cliente da API a partir do contrato guardado (`openapi.json`).
 *
 * O contrato é a fonte da verdade: gera-se do lado do Python (`scripts/export_openapi.py`) e
 * um teste avisa quando fica desatualizado. Daqui saem os tipos, uma função por operação
 * (`experimentsStart`, `automlUpdates`…) e as opções do TanStack Query. O resultado fica em
 * `src/api/generated/`, fora do Git, e volta a gerar-se antes de `dev`, `build`, `typecheck`,
 * `lint` e `test` (ADR 0027).
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
