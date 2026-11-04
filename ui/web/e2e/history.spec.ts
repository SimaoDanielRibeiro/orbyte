/**
 * O ecrã Histórico (etapa 10.8; antes, Experiências): os runs, a comparação e a gaveta de cada
 * run (mudar o nome, guardar, apagar e reproduzir).
 *
 * Os outros testes também registam runs ao mesmo tempo: estes têm nomes próprios, e é por eles
 * que se procuram.
 */
import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

import { animationsSettled, trainThroughApi, type ExperimentRequest } from "./helpers";

function iris(c: number): ExperimentRequest {
  return {
    dataset: { kind: "sample", params: { name: "iris" } },
    preparation: { target: "species" },
    model: { name: "logistic_regression", hyperparameters: { C: c } },
    seed: 3,
  };
}

test("compara dois runs, muda o nome, guarda, apaga e reproduz", async ({ page, request }) => {
  test.setTimeout(180_000);
  const strong = await trainThroughApi(request, iris(10));
  const weak = await trainThroughApi(request, iris(0.01));
  await request.patch(`/api/history/runs/${strong}`, { data: { name: "History strong C" } });
  await request.patch(`/api/history/runs/${weak}`, { data: { name: "History weak C" } });

  await page.goto("/history");
  await page.getByRole("checkbox", { name: "Compare “History strong C”" }).click();
  await page.getByRole("checkbox", { name: "Compare “History weak C”" }).click();
  await expect(page.getByRole("heading", { name: "What changes between the runs" })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "The hyperparameters" })).toBeVisible();
  await expect(
    page.getByRole("rowheader", { name: "C (inverse of regularisation)" }),
  ).toBeVisible();

  await animationsSettled(page);
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations.map((violation) => `${violation.id}: ${violation.help}`)).toEqual([]);

  // A gaveta de um run: mudar o nome e guardar.
  await page.getByRole("button", { name: "History strong C", exact: true }).click();
  const sheet = page.getByRole("dialog");
  await sheet.getByRole("textbox", { name: "Name" }).fill("History strong C, saved");
  await sheet.getByRole("button", { name: "Rename" }).click();
  await expect(sheet.getByRole("heading", { name: "History strong C, saved" })).toBeVisible();
  await sheet.getByRole("button", { name: "Save", exact: true }).click();
  await expect(sheet.getByRole("button", { name: "Saved", exact: true })).toBeDisabled();
  await page.keyboard.press("Escape");

  // Apagar o outro, com a confirmação.
  await page.getByRole("button", { name: "History weak C", exact: true }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Delete the run" }).click();
  await page.getByRole("alertdialog").getByRole("button", { name: "Delete" }).click();
  await expect(page.getByRole("button", { name: "History weak C", exact: true })).toHaveCount(0);

  // Reproduzir: o run novo treina e abre no ecrã Treinar e avaliar.
  await page.getByRole("button", { name: "History strong C, saved", exact: true }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Replay", exact: true }).click();
  await expect(page).toHaveURL(/\/train\/run-/);
  await expect(page.getByRole("heading", { level: 2, name: "The evaluation" })).toBeVisible({
    timeout: 60_000,
  });
});
