/**
 * Treinar e avaliar (etapa 10.6): treina-se um modelo e abre-se a explicação de cada métrica,
 * com a fórmula e a conta do run, das 22 que o laboratório tem. Tudo com "reduzir movimento"
 * ligado: as animações não podem esconder nada.
 */
import { expect, test, type Page } from "@playwright/test";

import {
  loadSample,
  prepareWithSuggestions,
  trainThroughApi,
  type ExperimentRequest,
} from "./helpers";

test.use({ reducedMotion: "reduce" });

/** Abre a gaveta de cada métrica da tabela e devolve os nomes das que abriu. */
async function openEveryMetric(page: Page): Promise<string[]> {
  const table = page.getByRole("table", { name: "As métricas" });
  await expect(table).toBeVisible({ timeout: 60_000 });
  // Os nomes leem-se antes de abrir a primeira gaveta: com uma aberta, o resto da página fica
  // escondido dos leitores de ecrã (e das procuras por papel).
  const names = (await table.getByRole("button").allTextContents()).map((name) => name.trim());
  for (const name of names) {
    await table.getByRole("button", { name, exact: true }).click();
    const drawer = page.getByRole("dialog", { name });
    await expect(drawer).toBeVisible();
    await expect(drawer.locator(".katex").first()).toBeVisible();
    await expect(drawer.getByRole("heading", { name: "A conta neste run" })).toBeVisible();
    await expect(drawer.getByRole("heading", { name: "Como se lê" })).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(drawer).toBeHidden();
  }
  return names;
}

test("treina um modelo à mão e abre a explicação de cada métrica", async ({ page }) => {
  test.setTimeout(180_000);
  await loadSample(page, "Titanic");
  await prepareWithSuggestions(page);
  await page.goto("/modelos/logistic_regression");
  await page.getByRole("button", { name: "Treinar com estes valores" }).click();

  await expect(page).toHaveURL(/\/treinar\/run-/);
  await expect(page.getByRole("heading", { level: 1, name: "Regressão logística" })).toBeVisible();
  const opened = await openEveryMetric(page);

  expect(opened).toEqual(
    expect.arrayContaining(["Exatidão", "Precisão", "Recall (sensibilidade)", "F1", "ROC AUC"]),
  );
  await expect(page.getByRole("heading", { level: 3, name: "Matriz de confusão" })).toBeVisible();
  await expect(page.getByRole("heading", { level: 2, name: "O que aprendeu" })).toBeVisible();
});

test("as 22 métricas abrem com a fórmula e a conta de um run", async ({ page, request }) => {
  test.setTimeout(300_000);
  const synthetic = (generator: string) => ({
    kind: "synthetic",
    params: { generator, seed: 7, parameters: {} },
  });
  // Uma tarefa de cada: juntas, têm as 22 métricas.
  const requests: ExperimentRequest[] = [
    {
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
    },
    {
      dataset: synthetic("regression"),
      preparation: { target: "target", task_type: "regression" },
      model: { name: "linear_regression" },
      seed: 7,
    },
    {
      dataset: synthetic("blobs"),
      preparation: { target: "target", task_type: "clustering" },
      model: { name: "kmeans" },
      seed: 7,
    },
    {
      dataset: synthetic("anomalies"),
      preparation: { target: "target", task_type: "anomaly_detection" },
      model: { name: "isolation_forest" },
      seed: 7,
    },
    {
      dataset: { kind: "sample", params: { name: "iris" } },
      preparation: { target: "species", task_type: "dimensionality_reduction" },
      model: { name: "pca" },
      seed: 7,
    },
  ];

  // Um de cada vez: a API guarda os últimos runs em memória, e os outros testes também treinam.
  const opened = new Set<string>();
  for (const body of requests) {
    const runId = await trainThroughApi(request, body);
    await page.goto(`/treinar/${runId}`);
    for (const name of await openEveryMetric(page)) {
      opened.add(name);
    }
  }

  const cards = (await (await request.get("/api/catalog/metrics")).json()) as {
    display_name: string;
  }[];
  expect(cards).toHaveLength(22);
  expect([...opened].sort()).toEqual(cards.map((card) => card.display_name).sort());
});
