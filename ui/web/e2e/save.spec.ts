/**
 * Guardar (fase 11): o pacote de um run visto antes de existir, gravado, descarregado e carregado
 * outra vez; e um pacote de outro ML Lab, cujo modelo só abre depois do aviso.
 */
import { readFile } from "node:fs/promises";

import AxeBuilder from "@axe-core/playwright";
import { expect, test, type APIRequestContext } from "@playwright/test";

import { trainThroughApi, withoutSignature, type ExperimentRequest } from "./helpers";

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
  seed: 11,
};

interface Package {
  package_id: string;
  name: string;
}

/**
 * Treina no Titanic e guarda o run no histórico: assim volta com o modelo mesmo que os outros
 * testes o tirem da memória da API (só ficam os últimos).
 */
async function savedRun(request: APIRequestContext): Promise<string> {
  const runId = await trainThroughApi(request, TITANIC);
  const saved = await request.post(`/api/history/runs/${runId}/save`, { data: { name: null } });
  expect(saved.status()).toBe(200);
  return runId;
}

/** O pacote da biblioteca com este nome (os testes correm em paralelo, cada um com o seu). */
async function packageNamed(request: APIRequestContext, name: string): Promise<Package> {
  const library = (await (await request.get("/api/packages")).json()) as Package[];
  const found = library.find((pkg) => pkg.name === name);
  if (found === undefined) {
    throw new Error(`Não há nenhum pacote «${name}» na biblioteca.`);
  }
  return found;
}

test("guarda um run: vê o ficheiro antes, grava-o, descarrega-o e carrega-o outra vez", async ({
  page,
  request,
}) => {
  test.setTimeout(120_000);
  const runId = await savedRun(request);
  const name = `Titanic ${runId}`;
  const fileName = `titanic-${runId}.mllab`;

  await page.goto(`/guardar/${runId}`);
  await expect(page.getByRole("heading", { level: 1, name: "Guardar" })).toBeVisible();
  const parts = page.getByRole("list", { name: "As partes do ficheiro" });
  await expect(parts.getByRole("button", { name: /model\.joblib/ })).toBeVisible();

  // A ficha abre primeiro, desenhada.
  await expect(
    page
      .getByRole("region", { name: "O conteúdo de README.md" })
      .getByRole("heading", { name: "Como foi feito" }),
  ).toBeVisible();

  // Mudar uma opção muda o ficheiro: sem o modelo, sai o model.joblib e chega o aviso.
  const model = page.getByRole("switch", { name: "O modelo, para prever" });
  await model.click();
  await expect(parts.getByRole("button", { name: /model\.joblib/ })).toHaveCount(0);
  await expect(page.getByText(/Sem o modelo, o pacote lê-se todo/)).toBeVisible();
  await model.click();
  await expect(parts.getByRole("button", { name: /model\.joblib/ })).toBeVisible();

  // Os JSON numa árvore, o CSV numa tabela.
  await parts.getByRole("button", { name: /input\.json/ }).click();
  await expect(
    page.getByRole("region", { name: "O conteúdo de input.json" }).getByText("target:"),
  ).toBeVisible();
  await parts.getByRole("button", { name: /examples\.csv/ }).click();
  await expect(
    page
      .getByRole("table", { name: "As linhas de examples.csv" })
      .getByRole("columnheader", { name: "previsto" }),
  ).toBeVisible();

  // O nome dá o nome do ficheiro.
  await page.getByLabel("Nome").fill(name);
  await expect(page.getByText(fileName, { exact: true })).toBeVisible();

  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Guardar o modelo" }).click();
  const file = await download;
  expect(file.suggestedFilename()).toBe(fileName);
  await expect(page.getByRole("button", { name: "Guardado" })).toBeDisabled();
  const card = page.getByRole("group", { name });
  await expect(card).toBeVisible();

  // O ficheiro vem com os cabeçalhos certos (descarrega-se com uma ligação normal).
  const saved = await packageNamed(request, name);
  const response = await request.get(`/api/packages/${saved.package_id}/file`);
  expect(response.headers()["content-type"]).toBe("application/zip");
  expect(response.headers()["content-disposition"]).toBe(`attachment; filename="${fileName}"`);

  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations.map((violation) => `${violation.id}: ${violation.help}`)).toEqual([]);

  // Apagado da biblioteca, volta com o ficheiro descarregado: feito aqui, abre logo no leitor.
  await card.getByRole("button", { name: `Apagar ${name}` }).click();
  await page.getByRole("alertdialog").getByRole("button", { name: "Apagar" }).click();
  await expect(card).toHaveCount(0);
  // O ficheiro vai pelo conteúdo, e não pelo caminho: o Chromium perde um caminho com acentos.
  await page.locator("#pacote-ficheiro").setInputFiles({
    name: fileName,
    mimeType: "application/zip",
    buffer: await readFile(await file.path()),
  });
  await page.getByRole("button", { name: "Carregar o modelo" }).click();
  const reader = page.getByRole("dialog", { name });
  await expect(reader.getByRole("link", { name: "Abrir no Treinar e avaliar" })).toBeVisible();
  await expect(reader.getByRole("button", { name: /run\.json/ })).toBeVisible();
});

test("o modelo de um pacote de outro ML Lab só abre depois do aviso", async ({ page, request }) => {
  test.setTimeout(120_000);
  const runId = await savedRun(request);
  const name = `De fora ${runId}`;
  const made = await request.post(`/api/packages/runs/${runId}`, { data: { name } });
  expect(made.status()).toBe(201);
  const { package_id: packageId } = (await made.json()) as Package;
  const data = await (await request.get(`/api/packages/${packageId}/file`)).body();

  await page.goto("/guardar");
  await expect(page.getByRole("heading", { level: 1, name: "Guardar" })).toBeVisible();
  await page.locator("#pacote-ficheiro").setInputFiles({
    name: "de-fora.mllab",
    mimeType: "application/zip",
    buffer: withoutSignature(data),
  });
  await page.getByRole("button", { name: "Carregar o modelo" }).click();

  const warning = page.getByRole("alertdialog", { name: "Abrir um modelo feito noutro ML Lab?" });
  await expect(warning).toContainText("noutro ML Lab");
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations.map((violation) => `${violation.id}: ${violation.help}`)).toEqual([]);
  await warning.getByRole("button", { name: "Confio, abrir o modelo" }).click();

  await expect(page.getByText(`O modelo de «${name}» abriu`)).toBeVisible();
  const reader = page.getByRole("dialog", { name });
  await expect(reader.getByRole("link", { name: "Abrir no Treinar e avaliar" })).toBeVisible();
});
