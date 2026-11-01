/**
 * O ecrã Histórico (etapa 10.8; antes, Experiências): os runs, a comparação e a gaveta de cada
 * run (mudar o nome, guardar, apagar e reproduzir).
 *
 * Os outros testes também registam runs ao mesmo tempo: estes têm nomes próprios, e é por eles
 * que se procuram.
 */
import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

import { trainThroughApi, type ExperimentRequest } from "./helpers";

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
  await request.patch(`/api/history/runs/${strong}`, { data: { name: "Histórico C forte" } });
  await request.patch(`/api/history/runs/${weak}`, { data: { name: "Histórico C fraco" } });

  await page.goto("/experiencias");
  await page.getByRole("checkbox", { name: "Comparar «Histórico C forte»" }).click();
  await page.getByRole("checkbox", { name: "Comparar «Histórico C fraco»" }).click();
  await expect(page.getByRole("heading", { name: "O que muda entre os runs" })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "Os hiperparâmetros" })).toBeVisible();
  await expect(page.getByRole("rowheader", { name: "C (inverso da regularização)" })).toBeVisible();

  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations.map((violation) => `${violation.id}: ${violation.help}`)).toEqual([]);

  // A gaveta de um run: mudar o nome e guardar.
  await page.getByRole("button", { name: "Histórico C forte", exact: true }).click();
  const sheet = page.getByRole("dialog");
  await sheet.getByRole("textbox", { name: "Nome" }).fill("Histórico C forte, guardado");
  await sheet.getByRole("button", { name: "Mudar o nome" }).click();
  await expect(sheet.getByRole("heading", { name: "Histórico C forte, guardado" })).toBeVisible();
  await sheet.getByRole("button", { name: "Guardar", exact: true }).click();
  await expect(sheet.getByRole("button", { name: "Guardado", exact: true })).toBeDisabled();
  await page.keyboard.press("Escape");

  // Apagar o outro, com a confirmação.
  await page.getByRole("button", { name: "Histórico C fraco", exact: true }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Apagar o run" }).click();
  await page.getByRole("alertdialog").getByRole("button", { name: "Apagar" }).click();
  await expect(page.getByRole("button", { name: "Histórico C fraco", exact: true })).toHaveCount(0);

  // Reproduzir: o run novo treina e abre no ecrã Treinar e avaliar.
  await page.getByRole("button", { name: "Histórico C forte, guardado", exact: true }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Reproduzir", exact: true }).click();
  await expect(page).toHaveURL(/\/treinar\/run-/);
  await expect(page.getByRole("heading", { level: 2, name: "A avaliação" })).toBeVisible({
    timeout: 60_000,
  });
});
