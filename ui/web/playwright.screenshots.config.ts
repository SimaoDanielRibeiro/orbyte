/**
 * As imagens do README e das publicações (`npm run screenshots`): a interface a sério, no
 * Chromium, a falar com uma API a sério, como nos testes de ponta a ponta.
 *
 * O arranque é o do `playwright.config.ts`, noutras portas (8002 e 5175) e noutra pasta
 * temporária: as imagens mostram um laboratório acabado de abrir, sem os datasets nem os runs de
 * quem o usa. Ao contrário dos testes, o Ollama é o deste computador: o agente de IA e o analista
 * respondem a sério, com o `qwen3:8b` (sem ele, as imagens dos agentes não se tiram).
 *
 * O ecrã é um monitor Full HD (1920 × 1080): é a largura a partir da qual o ecrã dos agentes põe
 * o fluxograma e a conversa lado a lado. Cada imagem sai duas vezes (`e2e/readme.shots.ts`).
 */
import { mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { defineConfig, devices } from "@playwright/test";

const API_PORT = "8002";
const WEB_PORT = "5175";
const DATA = join(tmpdir(), "orbyte-screenshots");

// Só o processo principal arranca os servidores; os outros só leem a configuração.
if (!process.env.TEST_WORKER_INDEX) {
  rmSync(DATA, { recursive: true, force: true });
  mkdirSync(DATA, { recursive: true });
}

export default defineConfig({
  testDir: "./e2e",
  // As fotografias não são testes: o `npm run e2e` só apanha os `*.spec.ts`.
  testMatch: /\.shots\.ts$/,
  // Uma de cada vez, pela ordem do ficheiro: a procura dos agentes e o modelo de linguagem
  // querem o processador todo, e o History, no fim, mostra os runs das outras.
  workers: 1,
  fullyParallel: false,
  retries: 0,
  reporter: [["list"]],
  outputDir: "test-results/screenshots",
  use: {
    ...devices["Desktop Chrome"],
    baseURL: `http://localhost:${WEB_PORT}`,
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 2,
  },
  webServer: [
    {
      command: "uv run python -m ml_lab.api",
      cwd: "../..",
      url: `http://127.0.0.1:${API_PORT}/health`,
      timeout: 120_000,
      reuseExistingServer: false,
      env: {
        ML_LAB_API_PORT: API_PORT,
        ML_LAB_UPLOADS_DIR: join(DATA, "uploads"),
        ML_LAB_ARTIFACTS_DIR: join(DATA, "artifacts"),
        ML_LAB_DATABASE_PATH: join(DATA, "ml_lab.db"),
        ML_LAB_LOG_LEVEL: "WARNING",
      },
    },
    {
      command: `npm run build && npx vite preview --port ${WEB_PORT} --strictPort`,
      url: `http://localhost:${WEB_PORT}`,
      timeout: 180_000,
      reuseExistingServer: false,
      env: { ML_LAB_API_PORT: API_PORT },
    },
  ],
});
