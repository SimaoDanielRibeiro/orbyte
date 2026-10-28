/**
 * O glossário das métricas (etapa 10.8): todas as métricas, cada uma com a fórmula, e o filtro
 * pela tarefa.
 */
import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

interface Card {
  display_name: string;
}

test("o glossário tem todas as métricas, cada uma com a fórmula", async ({ page, request }) => {
  const cards = (await (await request.get("/api/catalog/metrics")).json()) as Card[];
  const regression = (await (
    await request.get("/api/catalog/metrics", { params: { task_type: "regression" } })
  ).json()) as Card[];

  await page.goto("/metricas");

  await expect(page.getByRole("article")).toHaveCount(cards.length);
  for (const card of cards) {
    const article = page.getByRole("article", { name: card.display_name, exact: true });
    await expect(article.locator(".katex").first()).toBeVisible();
  }
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations.map((violation) => `${violation.id}: ${violation.help}`)).toEqual([]);

  await page.getByRole("combobox", { name: "Tarefa" }).click();
  await page.getByRole("option", { name: "Regressão", exact: true }).click();
  await expect(page).toHaveURL(/tarefa=regression/);
  await expect(page.getByRole("article")).toHaveCount(regression.length);
});
