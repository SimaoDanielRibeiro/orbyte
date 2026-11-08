/**
 * Apagar no History, na própria tabela (ADR 0054): o caixote do lixo de cada linha e "Delete the
 * selected", os dois com confirmação.
 *
 * Corre no projeto `no-fim` (`playwright.config.ts`), depois de todos os outros: treina três runs,
 * e a API dos testes, partilhada por todos, guarda só os cinco mais recentes em memória. A correr
 * ao mesmo tempo que os outros, empurrava para fora da memória os runs que eles ainda iam guardar
 * ou explicar, e esses falhavam com "is no longer in memory".
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

test("apaga na própria tabela: um run pelo caixote do lixo, e os escolhidos de uma vez", async ({
  page,
  request,
}) => {
  test.setTimeout(180_000);
  const names = ["History bin one", "History bin two", "History bin three"];
  for (const [index, name] of names.entries()) {
    const runId = await trainThroughApi(request, iris(0.1 * (index + 1)));
    await request.patch(`/api/history/runs/${runId}`, { data: { name } });
  }
  const [one, two, three] = names as [string, string, string];
  const run = (name: string) => page.getByRole("button", { name, exact: true });

  await page.goto("/history");
  await expect(run(one)).toBeVisible();

  // Um run: o caixote do lixo da linha, e a confirmação (que se pode cancelar).
  await page.getByRole("button", { name: `Delete “${one}”` }).click();
  await page.getByRole("alertdialog").getByRole("button", { name: "Cancel" }).click();
  await expect(run(one)).toBeVisible();
  await page.getByRole("button", { name: `Delete “${one}”` }).click();
  await expect(page.getByRole("alertdialog")).toContainText(`Delete “${one}”?`);
  await page.getByRole("alertdialog").getByRole("button", { name: "Delete" }).click();
  await expect(run(one)).toHaveCount(0);

  // Os escolhidos: a confirmação diz quais são.
  await page.getByRole("checkbox", { name: `Compare “${two}”` }).click();
  await page.getByRole("checkbox", { name: `Compare “${three}”` }).click();
  await page.getByRole("button", { name: "Delete the selected (2)" }).click();
  const confirm = page.getByRole("alertdialog");
  await expect(confirm).toContainText("Delete 2 runs?");
  await expect(confirm).toContainText(two);
  await expect(confirm).toContainText(three);

  await animationsSettled(page);
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations.map((violation) => `${violation.id}: ${violation.help}`)).toEqual([]);

  await confirm.getByRole("button", { name: "Delete" }).click();
  await expect(run(two)).toHaveCount(0);
  await expect(run(three)).toHaveCount(0);
  // Sem nada escolhido, a ação sai da página.
  await expect(page.getByRole("button", { name: /Delete the selected/ })).toHaveCount(0);
});
