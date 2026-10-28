/**
 * Explicar (etapa 10.8): o que pesa nas previsões de um run, no conjunto e numa previsão.
 */
import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

import { trainThroughApi, type ExperimentRequest } from "./helpers";

const TITANIC: ExperimentRequest = {
  dataset: { kind: "sample", params: { name: "titanic" } },
  preparation: {
    target: "Survived",
    task_type: "binary_classification",
    cleaning: [
      {
        operation: "drop_columns",
        params: { columns: ["PassengerId", "Name", "Ticket", "Cabin"] },
      },
    ],
  },
  model: { name: "logistic_regression" },
  seed: 7,
};

test("explica um run: a importância, a dependência parcial e uma previsão", async ({
  page,
  request,
}) => {
  test.setTimeout(120_000);
  const runId = await trainThroughApi(request, TITANIC);
  // Guardado, o run volta à memória da API mesmo que os outros testes o tirem de lá (só ficam
  // os últimos): o ecrã reabre-o pelo histórico, com o modelo.
  const saved = await request.post(`/api/history/runs/${runId}/save`, { data: { name: null } });
  expect(saved.status()).toBe(200);

  await page.goto(`/explicar/${runId}`);
  await expect(page.getByRole("heading", { level: 1, name: "Explicar" })).toBeVisible();

  await page.getByRole("button", { name: "Calcular a importância" }).click();
  await expect(page.getByRole("img", { name: /quando se baralha cada feature/ })).toBeVisible({
    timeout: 60_000,
  });

  await page.getByRole("combobox", { name: "Feature" }).click();
  await page.getByRole("option", { name: "Age", exact: true }).click();
  await expect(page.getByRole("img", { name: /quando Age muda/ })).toBeVisible();

  await page.getByRole("button", { name: "Linha seguinte" }).click();
  await expect(page.getByRole("spinbutton", { name: /Linha do teste/ })).toHaveValue("2");
  await expect(page.getByRole("img", { name: /O contributo de cada feature/ })).toBeVisible();

  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations.map((violation) => `${violation.id}: ${violation.help}`)).toEqual([]);
});

test("sem um run, o Explicar diz onde se treina um", async ({ page }) => {
  await page.goto("/explicar");

  await expect(page.getByRole("link", { name: "Escolher o modelo" }).first()).toBeVisible();
});
