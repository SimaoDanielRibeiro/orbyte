/**
 * Os testes de ponta a ponta: a interface a sério, no Chromium, a falar com a API a sério.
 *
 * A API destes testes corre noutra porta (8001) e guarda tudo numa pasta temporária, apagada
 * no início de cada corrida: os testes nunca tocam nos datasets, nos artefactos nem no registo
 * de quem usa o laboratório. A interface (o build final) corre na porta 5174, para não chocar com
 * a do dia a dia.
 */
import { mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { defineConfig, devices } from "@playwright/test";

const API_PORT = "8001";
const WEB_PORT = "5174";
const DATA = join(tmpdir(), "ml-lab-e2e");

// Só o processo principal arranca os servidores; os outros só leem a configuração.
if (!process.env.TEST_WORKER_INDEX) {
  rmSync(DATA, { recursive: true, force: true });
  mkdirSync(DATA, { recursive: true });
}

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: `http://localhost:${WEB_PORT}`,
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
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
      // O build final, servido pelo `vite preview` (com o mesmo proxy para a API): testa-se o
      // que se entrega, e sem as recargas do servidor de desenvolvimento.
      command: `npm run build && npx vite preview --port ${WEB_PORT} --strictPort`,
      url: `http://localhost:${WEB_PORT}`,
      timeout: 180_000,
      reuseExistingServer: false,
      env: { ML_LAB_API_PORT: API_PORT },
    },
  ],
});
