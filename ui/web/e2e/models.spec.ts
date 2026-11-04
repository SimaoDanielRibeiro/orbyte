/**
 * Os modelos (etapa 10.5): cada um abre com o exemplo ao vivo e a ficha, e mudar um
 * hiperparâmetro muda o exemplo.
 */
import { expect, test } from "@playwright/test";

interface Model {
  name: string;
  display_name: string;
}

test("os 34 modelos abrem com o exemplo e a ficha", async ({ page, request }) => {
  test.setTimeout(600_000);
  const models = (await (await request.get("/api/catalog/models")).json()) as Model[];
  expect(models).toHaveLength(34);

  for (const model of models) {
    await page.goto(`/models/${model.name}`);
    await expect(page.getByRole("heading", { level: 1, name: model.display_name })).toBeVisible();
    // A ficha abre na intuição; as outras secções estão nos separadores (o nome de cada uma é o
    // do separador, que dá nome ao painel).
    await expect(page.getByRole("tabpanel", { name: "Intuition" })).toBeVisible();
    await page.getByRole("tab", { name: "Essential maths" }).click();
    await expect(page.getByRole("tabpanel", { name: "Essential maths" })).toBeVisible();
    // O exemplo: o campo de decisão (ou, na redução, o mapa dos pontos) desenhado.
    await expect(page.locator("#live [role='img']").first()).toBeVisible({ timeout: 60_000 });
    await expect(page.getByRole("heading", { level: 2, name: "What it learned" })).toBeVisible();
  }
});

test("mudar um hiperparâmetro volta a treinar o exemplo", async ({ page }) => {
  test.setTimeout(120_000);
  await page.goto("/models/decision_tree");
  const live = page.locator("#live");
  await expect(live.getByText(/^Step 5,/)).toBeVisible({ timeout: 30_000 });

  const demo = page.waitForRequest(
    (sent) =>
      sent.url().includes("/catalog/models/decision_tree/demo") &&
      (sent.postDataJSON() as { hyperparameters: { max_depth?: number } }).hyperparameters
        .max_depth === 2,
  );
  await page.getByRole("slider", { name: "Maximum depth" }).focus();
  await page.keyboard.press("ArrowLeft");
  await page.keyboard.press("ArrowLeft");
  await page.keyboard.press("ArrowLeft");
  await demo;

  // A árvore mais baixa dá uma animação mais curta: o exemplo é outro.
  await expect(live.getByText(/^Step 2,/)).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole("button", { name: "Reset to defaults" })).toBeVisible();
});

test("o nome de um modelo da lista é o título da página dele", async ({ page }) => {
  await page.goto("/models?task=all");
  await page.getByRole("link", { name: /^Neural network \(MLP\)$/ }).click();

  await expect(page).toHaveURL(/\/models\/mlp$/);
  await expect(page.getByRole("heading", { level: 1, name: "Neural network (MLP)" })).toBeVisible();
});
