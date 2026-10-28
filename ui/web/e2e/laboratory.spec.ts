/**
 * Laboratório (etapa 10.8): as quatro demonstrações, e um controlo que muda o resultado.
 */
import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test("as quatro demonstrações desenham-se, e o controlo muda o resultado", async ({ page }) => {
  test.setTimeout(120_000);
  await page.goto("/laboratorio");
  await expect(page.getByRole("img", { name: /polinómio de grau 3/ })).toBeVisible({
    timeout: 30_000,
  });

  const asked = page.waitForRequest(
    (sent) => sent.url().includes("/laboratory/overfitting") && sent.url().includes("degree=4"),
  );
  await page.getByRole("slider", { name: "Grau do polinómio" }).focus();
  await page.keyboard.press("ArrowRight");
  await asked;
  await expect(page.getByText(/Com grau 4,/)).toBeVisible();

  const demos: [string, RegExp][] = [
    ["Taxa de aprendizagem", /taça do erro/],
    ["O k do KNN", /fronteira do KNN/],
    ["Regularização", /peso de cada feature/],
  ];
  for (const [tab, chart] of demos) {
    await page.getByRole("tab", { name: tab }).click();
    await expect(page.getByRole("img", { name: chart })).toBeVisible({ timeout: 30_000 });
  }
  await expect(page).toHaveURL(/demo=regularization/);

  // A Ridge anda noutra gama de alphas: o controlo é outro.
  await page.getByRole("radio", { name: "Ridge (L2)" }).click();
  await expect(page.getByText(/Com alpha = 10,/)).toBeVisible({ timeout: 30_000 });

  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations.map((violation) => `${violation.id}: ${violation.help}`)).toEqual([]);
});

test("a página de um modelo abre a demonstração que o explica", async ({ page }) => {
  await page.goto("/modelos/knn");
  await page.getByRole("link", { name: "O k do KNN" }).click();

  await expect(page).toHaveURL(/\/laboratorio\?demo=knn/);
  await expect(page.getByRole("tab", { name: "O k do KNN" })).toHaveAttribute(
    "aria-selected",
    "true",
  );
});
