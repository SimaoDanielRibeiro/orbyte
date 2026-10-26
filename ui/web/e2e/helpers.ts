/**
 * O que os testes de ponta a ponta fazem muitas vezes: abrir um dataset de exemplo, prepará-lo e
 * esperar por um treino lançado diretamente na API.
 */
import { expect, type APIRequestContext, type Page } from "@playwright/test";

/** Os campos de um pedido de treino que os testes usam (o resto fica por omissão). */
export interface ExperimentRequest {
  dataset: { kind: string; params: Record<string, unknown> };
  preparation: Record<string, unknown>;
  model: { name: string; hyperparameters?: Record<string, unknown> };
  seed?: number;
}

/** Carrega um dataset de exemplo no ecrã Dados e espera que fique aberto. */
export async function loadSample(page: Page, title: string): Promise<void> {
  await page.goto("/dados");
  await page.getByRole("button", { name: `Carregar ${title}` }).click();
  await expect(page.getByRole("heading", { level: 2, name: title })).toBeVisible();
}

/** Prepara o dataset aberto com as sugestões do backend e espera que a preparação sirva. */
export async function prepareWithSuggestions(page: Page): Promise<void> {
  await page.goto("/preparar");
  await expect(page.getByRole("link", { name: "Escolher o modelo" })).toBeVisible({
    timeout: 30_000,
  });
}

/** Começa um treino pela API (pelo mesmo proxy da interface) e espera que acabe. */
export async function trainThroughApi(
  request: APIRequestContext,
  body: ExperimentRequest,
): Promise<string> {
  const started = await request.post("/api/experiments", { data: body });
  expect(started.status()).toBe(202);
  const { run_id: runId } = (await started.json()) as { run_id: string };
  await expect
    .poll(
      async () => {
        const job = await request.get(`/api/experiments/${runId}/job`);
        return ((await job.json()) as { status: string }).status;
      },
      { timeout: 60_000 },
    )
    .toBe("finished");
  return runId;
}
