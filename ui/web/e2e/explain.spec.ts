/**
 * Explicar (etapas 10.8 e 13): a história do treino, com texto e visuais, e o que pesa nas
 * previsões de um run, no conjunto e numa previsão (ADR 0047).
 */
import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

import { animationsSettled, trainThroughApi, type ExperimentRequest } from "./helpers";

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

test("explica um run: a história do treino, a importância, a dependência e uma previsão", async ({
  page,
  request,
}) => {
  test.setTimeout(120_000);
  const runId = await trainThroughApi(request, TITANIC);
  // Guardado, o run volta à memória da API mesmo que os outros testes o tirem de lá (só ficam
  // os últimos): o ecrã reabre-o pelo histórico, com o modelo.
  const saved = await request.post(`/api/history/runs/${runId}/save`, { data: { name: null } });
  expect(saved.status()).toBe(200);

  await page.goto(`/explain/${runId}`);
  await expect(page.getByRole("heading", { level: 1, name: "Explain" })).toBeVisible();

  // A história: os seis capítulos, pela ordem em que se leem, cada um com o seu visual.
  const story = page.getByRole("region", { name: "What happened in the training" });
  await expect(story.getByText(/Logistic regression learned to predict Survived/)).toBeVisible();
  await expect(story.getByRole("heading", { level: 3 })).toHaveText([
    "What it learned from",
    "How it learned",
    "What it learned",
    "How well it did",
    "Where it goes wrong",
    "What to try next",
  ]);
  await expect(story.getByRole("img", { name: /The 891 rows/ })).toBeVisible();
  await expect(story.getByRole("img", { name: /Weight on the chance of “1”/ })).toBeVisible();
  await expect(story.getByRole("img", { name: /^Accuracy:/ })).toBeVisible();
  await expect(story.getByText("Rows: actual class. Columns: predicted class.")).toBeVisible();

  await page.getByRole("button", { name: "Compute the importance" }).click();
  await expect(page.getByRole("img", { name: /when each feature is shuffled/ })).toBeVisible({
    timeout: 60_000,
  });

  await page.getByRole("combobox", { name: "Feature" }).click();
  await page.getByRole("option", { name: "Age", exact: true }).click();
  await expect(page.getByRole("img", { name: /changes when Age changes/ })).toBeVisible();

  await page.getByRole("button", { name: "Next row" }).click();
  await expect(page.getByRole("spinbutton", { name: /Test row/ })).toHaveValue("2");
  await expect(page.getByRole("img", { name: /Each feature's contribution/ })).toBeVisible();

  await animationsSettled(page);
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations.map((violation) => `${violation.id}: ${violation.help}`)).toEqual([]);
});

test("um run de clustering conta a história, e as previsões dizem porque não se explicam", async ({
  page,
  request,
}) => {
  test.setTimeout(120_000);
  const runId = await trainThroughApi(request, {
    dataset: { kind: "sample", params: { name: "iris" } },
    preparation: { target: "species", task_type: "clustering" },
    model: { name: "kmeans" },
    seed: 7,
  });

  await page.goto(`/explain/${runId}`);

  const story = page.getByRole("region", { name: "What happened in the training" });
  await expect(story.getByText(/K-means split the rows into 3 groups/)).toBeVisible();
  await expect(story.getByText(/stays out of the training/)).toBeVisible();
  await expect(page.getByText(/Explaining predictions needs a target/)).toBeVisible();
});

test("sem um run, o Explicar diz onde se treina um", async ({ page }) => {
  await page.goto("/explain");

  await expect(page.getByRole("link", { name: "Choose the model" }).first()).toBeVisible();
});
