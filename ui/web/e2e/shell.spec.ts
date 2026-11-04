/**
 * A moldura (etapas 10.2 e 10.9): a entrada, as duas interfaces com as suas barras, o teclado, o
 * telemóvel e a acessibilidade. O critério: navega-se por tudo só com o teclado e, com a barra
 * aberta ou recolhida, o conteúdo reorganiza-se sem cortes a 1440, 1024 e 390 px. Abaixo de
 * 1024 px (o tablet ao alto, 768 px) a barra é uma gaveta, como no telemóvel.
 */
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Locator, type Page } from "@playwright/test";

import { animationsSettled, loadSample } from "./helpers";

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

test("a entrada mostra o nome e as duas interfaces, sem barra nem estado da API", async ({
  page,
}) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { level: 1, name: "Orbyte" })).toBeVisible();
  await expect(page.getByRole("heading", { level: 2, name: "Manual" })).toBeVisible();
  await expect(page.getByRole("heading", { level: 2, name: "Agents" })).toBeVisible();
  // A API ligada é o normal: só a falha se escreve.
  await expect(page.getByText(/^API (online|offline)/)).toHaveCount(0);
  await expect(page.getByRole("navigation", { name: "Screens" })).toHaveCount(0);
});

test("cada interface tem as suas secções na barra, e troca-se por baixo dos primeiros passos", async ({
  page,
}) => {
  await page.goto("/");
  const nav = page.getByRole("navigation", { name: "Screens" });

  await page.getByRole("button", { name: "Start with the agents" }).click();

  // Os primeiros passos (os Dados e o Explorar) são das duas interfaces; o resto é de cada uma.
  await expect(page).toHaveURL(/\/data$/);
  await expect(nav.getByRole("radio", { name: "Agents" })).toBeChecked();
  await expect(nav.getByRole("link", { name: "Explore" })).toBeVisible();
  await expect(nav.getByRole("link", { name: "Agents" })).toBeVisible();
  await expect(nav.getByRole("link", { name: "Prepare" })).toHaveCount(0);
  // No fundo, o botão de recolher a barra e, por baixo, o Histórico.
  await expect(nav.getByRole("button", { name: "Collapse the sidebar" })).toBeVisible();
  await expect(nav.getByRole("link", { name: "History" })).toBeVisible();

  // Os Dados existem nas duas: a troca fica no mesmo ecrã.
  await nav.getByRole("radio", { name: "Manual" }).click();

  await expect(page).toHaveURL(/\/data$/);
  await expect(nav.getByRole("link", { name: "Explore" })).toBeVisible();
  await expect(nav.getByRole("link", { name: "Prepare" })).toBeVisible();
  await expect(nav.getByRole("link", { name: "Agents" })).toHaveCount(0);
});

test("com os dados abertos, trocar de interface num ecrã só de uma leva ao primeiro passo da outra", async ({
  page,
}) => {
  const nav = page.getByRole("navigation", { name: "Screens" });
  await loadSample(page, "Iris");

  // O Preparar só existe no Manual: nos Agentes, o passo equivalente é o dos Agentes, e não os
  // Dados outra vez (ADR 0050).
  await page.goto("/prepare");
  await nav.getByRole("radio", { name: "Agents" }).click();
  await expect(page).toHaveURL(/\/agents$/);
  await expect(page.getByRole("heading", { level: 1, name: "Agents" })).toBeVisible();

  await nav.getByRole("radio", { name: "Manual" }).click();
  await expect(page).toHaveURL(/\/prepare$/);
});

test("um ecrã de uma só interface, aberto pelo endereço, traz a barra dela", async ({ page }) => {
  const nav = page.getByRole("navigation", { name: "Screens" });

  await page.goto("/agents");
  await expect(nav.getByRole("radio", { name: "Agents" })).toBeChecked();

  await page.goto("/lab");
  await expect(nav.getByRole("radio", { name: "Manual" })).toBeChecked();
  await expect(nav.getByRole("link", { name: "Lab" })).toHaveAttribute("aria-current", "page");
});

test("a pesquisa (Ctrl K) salta para um ecrã", async ({ page }) => {
  await page.goto("/data");
  // O atalho só existe depois de a moldura montar: espera-se pelo título.
  await expect(page.getByRole("heading", { level: 1, name: "Data" })).toBeVisible();

  await page.keyboard.press("Control+k");
  const search = page.getByRole("dialog");
  await expect(search).toBeVisible();
  await page.keyboard.type("Metrics");
  await page.keyboard.press("Enter");

  await expect(page).toHaveURL(/\/metrics/);
  await expect(page.getByRole("heading", { level: 1, name: "Metrics" })).toBeVisible();
});

test("navega-se só com o teclado", async ({ page }) => {
  await page.goto("/");
  await tabUntil(page, page.getByRole("button", { name: "Start by hand" }));
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/data$/);

  const explore = page
    .getByRole("navigation", { name: "Screens" })
    .getByRole("link", { name: "Explore" });
  await tabUntil(page, explore);
  await page.keyboard.press("Enter");

  await expect(page).toHaveURL(/\/explore$/);
  await expect(explore).toHaveAttribute("aria-current", "page");
  await expect(page.getByRole("heading", { level: 1, name: "Explore" })).toBeVisible();
});

test("Ctrl+B recolhe a barra e o conteúdo ganha o espaço, sem cortes", async ({
  page,
}, testInfo) => {
  for (const size of WIDTHS.slice(0, 2)) {
    await page.setViewportSize(size);
    await page.goto("/data");
    const open = await mainWidth(page);
    await testInfo.attach(`data-${String(size.width)}-open`, {
      body: await page.screenshot(),
      contentType: "image/png",
    });

    await page.keyboard.press("Control+b");
    await expect(page.locator('[data-slot="sidebar"]')).toHaveAttribute("data-state", "collapsed");
    await expect.poll(() => mainWidth(page)).toBeGreaterThan(open + 150);
    expect(await noHorizontalScroll(page)).toBe(true);
    await testInfo.attach(`data-${String(size.width)}-collapsed`, {
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
  await page.goto("/data");
  expect(await noHorizontalScroll(page)).toBe(true);
  await testInfo.attach("data-390", { body: await page.screenshot(), contentType: "image/png" });

  await page.getByRole("button", { name: "Toggle the sidebar" }).first().click();
  const drawer = page.getByRole("dialog");
  await expect(drawer).toBeVisible();
  await drawer.getByRole("link", { name: "History" }).click();

  await expect(page).toHaveURL(/\/history$/);
  await expect(drawer).toBeHidden();
  await expect(page.getByRole("heading", { level: 1, name: "History" })).toBeVisible();
});

test("num tablet ao alto, a barra é uma gaveta e o conteúdo fica com a largura toda", async ({
  page,
}) => {
  // Com a barra fixa (18rem) e as margens do computador, sobravam 243 px a 768: menos do que no
  // telemóvel.
  await page.setViewportSize({ width: 768, height: 1024 });
  await page.goto("/data");

  await expect(page.getByRole("button", { name: "Toggle the sidebar" }).first()).toBeVisible();
  expect(await mainWidth(page)).toBeGreaterThan(740);
  expect(await noHorizontalScroll(page)).toBe(true);
});

test("no telemóvel, nenhum ecrã desliza para o lado", async ({ page }) => {
  test.setTimeout(120_000);
  await page.setViewportSize(WIDTHS[2]);
  for (const path of [
    "/",
    "/data",
    "/models",
    "/models/knn",
    "/metrics",
    "/lab",
    "/rl",
    "/history",
    "/save",
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

  await expect(page.getByRole("heading", { name: "This screen doesn't exist" })).toBeVisible();
});

// Os ecrãs que abrem sem nada antes; os outros passam pelo axe nos seus próprios testes (o
// Explicar, o Histórico, o Laboratório, o RL e as Métricas, com os dados à vista).
for (const path of [
  "/",
  "/data",
  "/explore",
  "/prepare",
  "/models",
  "/models/knn",
  "/train",
  "/explain",
  "/agents",
  "/metrics",
  "/lab",
  "/rl",
  "/history",
  "/save",
]) {
  test(`sem problemas de acessibilidade em ${path}`, async ({ page }) => {
    await page.goto(path);
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
    await animationsSettled(page);

    const results = await new AxeBuilder({ page }).analyze();

    expect(results.violations.map((violation) => `${violation.id}: ${violation.help}`)).toEqual([]);
  });
}
