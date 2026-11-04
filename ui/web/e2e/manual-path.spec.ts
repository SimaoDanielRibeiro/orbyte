/**
 * O caminho à mão (etapa 10.4): do início aos modelos da tarefa, passando pelos dados, pelo
 * perfil e pela preparação.
 */
import { expect, test } from "@playwright/test";

import { loadSample, prepareWithSuggestions } from "./helpers";

test("do início aos modelos da tarefa, pelos dados, o perfil e a preparação", async ({ page }) => {
  test.setTimeout(120_000);
  await page.goto("/");
  await page.getByRole("button", { name: "Start by hand" }).click();
  await expect(page).toHaveURL(/\/data$/);

  await page.getByRole("button", { name: "Load Titanic" }).click();
  await expect(page.getByRole("heading", { level: 2, name: "Titanic" })).toBeVisible();
  await page.getByRole("tab", { name: "Columns and types" }).click();
  // O tipo de cada coluna, com a razão no "?" ao lado.
  await expect(page.getByRole("button", { name: "Why PassengerId is identifier" })).toBeVisible();
  await page.getByRole("link", { name: "Explore this dataset" }).first().click();

  await expect(page).toHaveURL(/\/explore$/);
  await expect(page.getByRole("heading", { level: 2, name: "Warnings" })).toBeVisible();
  // O perfil lê-se por partes, todas à vista; o índice no topo salta para cada uma.
  const parts = page.getByRole("navigation", { name: "The parts of the profile" });
  await parts.getByRole("link", { name: /Correlations/ }).click();
  await expect(page.getByRole("img", { name: /Pearson correlation/ })).toBeInViewport();
  await parts.getByRole("link", { name: /Target/ }).click();
  await expect(page.getByRole("heading", { name: "The classes of Survived" })).toBeInViewport();
  await expect(parts.getByRole("link", { name: /Target/ })).toHaveAttribute(
    "aria-current",
    "location",
  );
  await page.getByRole("link", { name: "Prepare the data" }).first().click();

  await expect(page).toHaveURL(/\/prepare$/);
  await expect(page.getByRole("heading", { level: 2, name: "The result" })).toBeVisible();
  await page.getByRole("tab", { name: "Lineage" }).click();
  await expect(
    page.getByText("Dropped the columns PassengerId, Name, Ticket, Cabin."),
  ).toBeVisible();
  await page.getByRole("link", { name: "Choose the model" }).first().click();

  // Os modelos chegam filtrados pela tarefa preparada: os de regressão ficam de fora.
  await expect(page).toHaveURL(/\/models/);
  await expect(page.getByRole("combobox", { name: "Task" })).toHaveText(/Binary classification/);
  await expect(page.getByText(/the ones for the task you prepared/)).toBeVisible();
  await expect(page.getByRole("link", { name: /Logistic regression/ }).first()).toBeVisible();
  await expect(page.getByRole("link", { name: /^Linear regression/ })).toHaveCount(0);
});

test("trocar de alvo no Preparar volta a preparar com as sugestões do alvo novo", async ({
  page,
}) => {
  test.setTimeout(120_000);
  await loadSample(page, "Titanic");
  await prepareWithSuggestions(page);
  // Com classes (o Survived), a proporção das classes em cada parte tem o seu separador.
  await expect(page.getByRole("tab", { name: "Classes", exact: true })).toBeVisible();

  await page.getByRole("combobox", { name: /Target column/ }).click();
  await page.getByRole("option", { name: "Fare" }).click();

  // Um número contínuo pede regressão: a tarefa e a preparação seguem a sugestão nova (sem
  // classes, deixa de haver a proporção das classes em cada parte).
  await expect(page.getByRole("combobox", { name: "Task type" })).toHaveText("Regression");
  await expect(page.getByRole("tab", { name: "Classes", exact: true })).toHaveCount(0, {
    timeout: 30_000,
  });
  await expect(page.getByRole("link", { name: "Choose the model" }).first()).toBeVisible();
});
