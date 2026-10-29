/**
 * O caminho à mão (etapa 10.4): do início aos modelos da tarefa, passando pelos dados, pelo
 * perfil e pela preparação.
 */
import { expect, test } from "@playwright/test";

import { loadSample, prepareWithSuggestions } from "./helpers";

test("do início aos modelos da tarefa, pelos dados, o perfil e a preparação", async ({ page }) => {
  test.setTimeout(120_000);
  await page.goto("/");
  await page.getByRole("button", { name: "Começar à mão" }).click();
  await expect(page).toHaveURL(/\/dados$/);

  await page.getByRole("button", { name: "Carregar Titanic" }).click();
  await expect(page.getByRole("heading", { level: 2, name: "Titanic" })).toBeVisible();
  await page.getByRole("tab", { name: "Colunas e tipos" }).click();
  // O tipo de cada coluna, com a razão no "?" ao lado.
  await expect(
    page.getByRole("button", { name: "Porque é que PassengerId é identificador" }),
  ).toBeVisible();
  await page.getByRole("link", { name: "Explorar este dataset" }).first().click();

  await expect(page).toHaveURL(/\/explorar$/);
  await expect(page.getByRole("heading", { level: 2, name: "Avisos" })).toBeVisible();
  // O perfil lê-se por partes, todas à vista; o índice no topo salta para cada uma.
  const parts = page.getByRole("navigation", { name: "As partes do perfil" });
  await parts.getByRole("link", { name: /Correlações/ }).click();
  await expect(page.getByRole("img", { name: /correlação de Pearson/ })).toBeInViewport();
  await parts.getByRole("link", { name: /Alvo/ }).click();
  await expect(page.getByRole("heading", { name: "As classes de Survived" })).toBeInViewport();
  await expect(parts.getByRole("link", { name: /Alvo/ })).toHaveAttribute(
    "aria-current",
    "location",
  );
  await page.getByRole("link", { name: "Preparar os dados" }).first().click();

  await expect(page).toHaveURL(/\/preparar$/);
  await expect(page.getByRole("heading", { level: 2, name: "O resultado" })).toBeVisible();
  await page.getByRole("tab", { name: "Linhagem" }).click();
  await expect(
    page.getByText("Removidas as colunas PassengerId, Name, Ticket, Cabin."),
  ).toBeVisible();
  await page.getByRole("link", { name: "Escolher o modelo" }).first().click();

  // Os modelos chegam filtrados pela tarefa preparada: os de regressão ficam de fora.
  await expect(page).toHaveURL(/\/modelos/);
  await expect(page.getByRole("combobox", { name: "Tarefa" })).toHaveText(/Classificação binária/);
  await expect(page.getByText(/os da tarefa que preparaste/)).toBeVisible();
  await expect(page.getByRole("link", { name: /Regressão logística/ }).first()).toBeVisible();
  await expect(page.getByRole("link", { name: /^Regressão linear/ })).toHaveCount(0);
});

test("trocar de alvo no Preparar volta a preparar com as sugestões do alvo novo", async ({
  page,
}) => {
  test.setTimeout(120_000);
  await loadSample(page, "Titanic");
  await prepareWithSuggestions(page);

  await page.getByRole("combobox", { name: /Coluna-alvo/ }).click();
  await page.getByRole("option", { name: "Fare" }).click();

  // Um número contínuo pede regressão: a tarefa e a preparação seguem a sugestão nova (sem
  // classes, deixa de haver a proporção das classes em cada parte).
  await expect(page.getByRole("combobox", { name: "Tipo de tarefa" })).toHaveText("Regressão");
  await expect(page.getByRole("tab", { name: "Classes em cada parte" })).toHaveCount(0, {
    timeout: 30_000,
  });
  await expect(page.getByRole("link", { name: "Escolher o modelo" }).first()).toBeVisible();
});
