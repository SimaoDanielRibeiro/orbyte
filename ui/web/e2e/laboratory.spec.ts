/**
 * Laboratório (etapa 10.8, ADR 0051): as seis demonstrações, com o guia à vista, e um controlo
 * que muda o resultado.
 */
import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

import { animationsSettled } from "./helpers";

test("as seis demonstrações desenham-se, e o controlo muda o resultado", async ({ page }) => {
  test.setTimeout(120_000);
  await page.goto("/lab");
  await expect(page.getByRole("img", { name: /degree 3 polynomial/ })).toBeVisible({
    timeout: 30_000,
  });
  // O guia fica à vista: o que experimentar, e o que fica no fim.
  await expect(page.getByRole("heading", { name: "Try this" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "What to take away" })).toBeVisible();

  const asked = page.waitForRequest(
    (sent) => sent.url().includes("/laboratory/overfitting") && sent.url().includes("degree=4"),
  );
  await page.getByRole("slider", { name: "Polynomial degree" }).focus();
  await page.keyboard.press("ArrowRight");
  await asked;
  await expect(page.getByText(/With degree 4,/)).toBeVisible();

  const demos: [string, RegExp][] = [
    ["Learning rate", /error bowl/],
    ["The k in KNN", /KNN boundary/],
    ["Regularisation", /Each feature's weight/],
  ];
  for (const [tab, chart] of demos) {
    await page.getByRole("tab", { name: tab }).click();
    await expect(page.getByRole("img", { name: chart })).toBeVisible({ timeout: 30_000 });
  }
  await expect(page).toHaveURL(/demo=regularization/);

  // A Ridge anda noutra gama de alphas: o controlo é outro.
  await page.getByRole("radio", { name: "Ridge (L2)" }).click();
  await expect(page.getByText(/With alpha = 10,/)).toBeVisible({ timeout: 30_000 });

  // O limiar: descê-lo encontra mais positivos, à custa de mais alarmes falsos.
  await page.getByRole("tab", { name: "Decision threshold" }).click();
  await expect(page.getByRole("img", { name: /Each test row as a dot/ })).toBeVisible({
    timeout: 30_000,
  });
  await expect(page.getByText(/With a threshold of 50%,/)).toBeVisible();
  await page.getByRole("slider", { name: "Decision threshold" }).focus();
  await page.keyboard.press("Home");
  await expect(page.getByText(/With a threshold of 5%,/)).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText(/A low threshold catches almost every case/)).toBeVisible();

  await page.getByRole("tab", { name: "Data size" }).click();
  await expect(page.getByRole("img", { name: /boundary of a decision tree/ })).toBeVisible({
    timeout: 30_000,
  });
  await expect(page.getByText(/the simpler model generalises better/)).toBeVisible();
  await expect(page).toHaveURL(/demo=data_size/);

  await animationsSettled(page);
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations.map((violation) => `${violation.id}: ${violation.help}`)).toEqual([]);
});

test("a página de um modelo abre a demonstração que o explica", async ({ page }) => {
  await page.goto("/models/knn");
  await page.getByRole("link", { name: "The k in KNN" }).click();

  await expect(page).toHaveURL(/\/lab\?demo=knn/);
  await expect(page.getByRole("tab", { name: "The k in KNN" })).toHaveAttribute(
    "aria-selected",
    "true",
  );
});
