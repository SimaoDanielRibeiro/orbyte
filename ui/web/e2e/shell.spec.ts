/**
 * A moldura (etapas 10.2 e 10.9): a entrada, as duas interfaces com as suas barras, o teclado, o
 * telemóvel e a acessibilidade. O critério: navega-se por tudo só com o teclado e, com a barra
 * aberta ou recolhida, o conteúdo reorganiza-se sem cortes a 1440, 1024 e 390 px.
 */
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Locator, type Page } from "@playwright/test";

const WIDTHS = [
  { width: 1440, height: 900 },
  { width: 1024, height: 768 },
  { width: 390, height: 844 },
] as const;

async function mainWidth(page: Page): Promise<number> {
  const box = await page.locator('[data-slot="sidebar-inset"]').boundingBox();
  if (box === null) {
    throw new Error("A área do conteúdo não está à vista.");
  }
  return box.width;
}

async function noHorizontalScroll(page: Page): Promise<boolean> {
  return page.evaluate(
    () => document.documentElement.scrollWidth <= document.documentElement.clientWidth,
  );
}

/** Carrega no Tab até o foco chegar a `target` (ou desiste ao fim de `limit` vezes). */
async function tabUntil(page: Page, target: Locator, limit = 30): Promise<void> {
  for (let presses = 0; presses < limit; presses++) {
    await page.keyboard.press("Tab");
    if (await target.evaluate((element) => element === document.activeElement)) {
      break;
    }
  }
  await expect(target).toBeFocused();
}

test("a entrada mostra o nome, a API ligada e as duas interfaces, sem barra", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { level: 1, name: "ML Lab" })).toBeVisible();
  await expect(page.getByRole("heading", { level: 2, name: "Manual" })).toBeVisible();
  await expect(page.getByRole("heading", { level: 2, name: "Agentes" })).toBeVisible();
  await expect(page.getByRole("status").filter({ hasText: "API ligada" })).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Ecrãs" })).toHaveCount(0);
});

test("cada interface tem as suas secções na barra, e troca-se por baixo dos primeiros passos", async ({
  page,
}) => {
  await page.goto("/");
  const nav = page.getByRole("navigation", { name: "Ecrãs" });

  await page.getByRole("button", { name: "Começar com os agentes" }).click();

  // Os primeiros passos (os Dados e o Explorar) são das duas interfaces; o resto é de cada uma.
  await expect(page).toHaveURL(/\/dados$/);
  await expect(nav.getByRole("radio", { name: "Agentes" })).toBeChecked();
  await expect(nav.getByRole("link", { name: "Explorar" })).toBeVisible();
  await expect(nav.getByRole("link", { name: "Agentes" })).toBeVisible();
  await expect(nav.getByRole("link", { name: "Preparar" })).toHaveCount(0);
  await expect(nav.getByRole("status")).toContainText("API ligada");

  // Os Dados existem nas duas: a troca fica no mesmo ecrã.
  await nav.getByRole("radio", { name: "Manual" }).click();

  await expect(page).toHaveURL(/\/dados$/);
  await expect(nav.getByRole("link", { name: "Explorar" })).toBeVisible();
  await expect(nav.getByRole("link", { name: "Preparar" })).toBeVisible();
  await expect(nav.getByRole("link", { name: "Agentes" })).toHaveCount(0);
});

test("um ecrã de uma só interface, aberto pelo endereço, traz a barra dela", async ({ page }) => {
  const nav = page.getByRole("navigation", { name: "Ecrãs" });

  await page.goto("/agentes");
  await expect(nav.getByRole("radio", { name: "Agentes" })).toBeChecked();

  await page.goto("/laboratorio");
  await expect(nav.getByRole("radio", { name: "Manual" })).toBeChecked();
  await expect(nav.getByRole("link", { name: "Laboratório" })).toHaveAttribute(
    "aria-current",
    "page",
  );
});

test("a pesquisa (Ctrl K) salta para um ecrã", async ({ page }) => {
  await page.goto("/dados");
  // O atalho só existe depois de a moldura montar: espera-se pelo título.
  await expect(page.getByRole("heading", { level: 1, name: "Dados" })).toBeVisible();

  await page.keyboard.press("Control+k");
  const search = page.getByRole("dialog");
  await expect(search).toBeVisible();
  await page.keyboard.type("Métricas");
  await page.keyboard.press("Enter");

  await expect(page).toHaveURL(/\/metricas/);
  await expect(page.getByRole("heading", { level: 1, name: "Métricas" })).toBeVisible();
});

test("navega-se só com o teclado", async ({ page }) => {
  await page.goto("/");
  await tabUntil(page, page.getByRole("button", { name: "Começar à mão" }));
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/dados$/);

  const explore = page
    .getByRole("navigation", { name: "Ecrãs" })
    .getByRole("link", { name: "Explorar" });
  await tabUntil(page, explore);
  await page.keyboard.press("Enter");

  await expect(page).toHaveURL(/\/explorar$/);
  await expect(explore).toHaveAttribute("aria-current", "page");
  await expect(page.getByRole("heading", { level: 1, name: "Explorar" })).toBeVisible();
});

test("Ctrl+B recolhe a barra e o conteúdo ganha o espaço, sem cortes", async ({
  page,
}, testInfo) => {
  for (const size of WIDTHS.slice(0, 2)) {
    await page.setViewportSize(size);
    await page.goto("/dados");
    const open = await mainWidth(page);
    await testInfo.attach(`dados-${String(size.width)}-aberta`, {
      body: await page.screenshot(),
      contentType: "image/png",
    });

    await page.keyboard.press("Control+b");
    await expect(page.locator('[data-slot="sidebar"]')).toHaveAttribute("data-state", "collapsed");
    await expect.poll(() => mainWidth(page)).toBeGreaterThan(open + 150);
    expect(await noHorizontalScroll(page)).toBe(true);
    await testInfo.attach(`dados-${String(size.width)}-recolhida`, {
      body: await page.screenshot(),
      contentType: "image/png",
    });

    // O estado fica guardado: depois de recarregar, a barra continua recolhida.
    await page.reload();
    await expect(page.locator('[data-slot="sidebar"]')).toHaveAttribute("data-state", "collapsed");
    await page.keyboard.press("Control+b");
    await expect(page.locator('[data-slot="sidebar"]')).toHaveAttribute("data-state", "expanded");
  }
});

test("no telemóvel, a barra é uma gaveta que se fecha ao escolher um ecrã", async ({
  page,
}, testInfo) => {
  const phone = WIDTHS[2];
  await page.setViewportSize(phone);
  await page.goto("/dados");
  expect(await noHorizontalScroll(page)).toBe(true);
  await testInfo.attach("dados-390", { body: await page.screenshot(), contentType: "image/png" });

  await page.getByRole("button", { name: "Abrir ou recolher a barra lateral" }).click();
  const drawer = page.getByRole("dialog");
  await expect(drawer).toBeVisible();
  await drawer.getByRole("link", { name: "Experiências" }).click();

  await expect(page).toHaveURL(/\/experiencias$/);
  await expect(drawer).toBeHidden();
  await expect(page.getByRole("heading", { level: 1, name: "Experiências" })).toBeVisible();
});

test("no telemóvel, nenhum ecrã desliza para o lado", async ({ page }) => {
  test.setTimeout(120_000);
  await page.setViewportSize(WIDTHS[2]);
  for (const path of [
    "/",
    "/dados",
    "/modelos",
    "/modelos/knn",
    "/metricas",
    "/laboratorio",
    "/rl",
    "/experiencias",
  ]) {
    await page.goto(path);
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
    // Os gráficos e as fórmulas chegam depois do título: espera-se que a página assente.
    await page.waitForLoadState("networkidle");
    expect(await noHorizontalScroll(page), path).toBe(true);
  }
});

test("um endereço desconhecido diz que o ecrã não existe", async ({ page }) => {
  await page.goto("/nada-disto");

  await expect(page.getByRole("heading", { name: "Este ecrã não existe" })).toBeVisible();
});

// Os ecrãs que abrem sem nada antes; os outros passam pelo axe nos seus próprios testes (o
// Explicar, as Experiências, o Laboratório, o RL e as Métricas, com os dados à vista).
for (const path of [
  "/",
  "/dados",
  "/explorar",
  "/preparar",
  "/modelos",
  "/modelos/knn",
  "/treinar",
  "/explicar",
  "/agentes",
  "/metricas",
  "/laboratorio",
  "/rl",
  "/experiencias",
]) {
  test(`sem problemas de acessibilidade em ${path}`, async ({ page }) => {
    await page.goto(path);
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();

    const results = await new AxeBuilder({ page }).analyze();

    expect(results.violations.map((violation) => `${violation.id}: ${violation.help}`)).toEqual([]);
  });
}
