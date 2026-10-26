/**
 * A moldura (etapa 10.2): a barra lateral, os dois caminhos, o teclado, o telemóvel e a
 * acessibilidade. O critério da etapa: navega-se por tudo só com o teclado e, com a barra
 * aberta ou recolhida, o conteúdo reorganiza-se sem cortes a 1440, 1024 e 390 px.
 */
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

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

test("o início abre com a API ligada e os dois caminhos", async ({ page }) => {
  await page.goto("/");

  await expect(
    page.getByRole("heading", { level: 1, name: "Vê um modelo aprender." }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { level: 2, name: "Como queres trabalhar?" }),
  ).toBeVisible();
  await expect(page.getByRole("heading", { name: "Testar à mão" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Procurar com agentes" })).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Ecrãs" }).getByRole("status")).toContainText(
    "API ligada",
  );
});

test("escolher um caminho mostra os passos dele na barra", async ({ page }) => {
  await page.goto("/");
  const nav = page.getByRole("navigation", { name: "Ecrãs" });

  await page.getByRole("button", { name: "Começar com os agentes" }).click();

  await expect(page).toHaveURL(/\/dados$/);
  await expect(nav.getByRole("link", { name: "Agentes" })).toBeVisible();
  await expect(nav.getByRole("link", { name: "Explorar" })).toHaveCount(0);

  await page.getByRole("button", { name: "Mudar para «Testar à mão»" }).click();

  await expect(nav.getByRole("link", { name: "Explorar" })).toBeVisible();
  await expect(nav.getByRole("link", { name: "Agentes" })).toHaveCount(0);
});

test("navega-se só com o teclado", async ({ page }) => {
  await page.goto("/");
  const dados = page
    .getByRole("navigation", { name: "Ecrãs" })
    .getByRole("link", { name: "Dados" });

  for (let presses = 0; presses < 10; presses++) {
    await page.keyboard.press("Tab");
    if (await dados.evaluate((element) => element === document.activeElement)) {
      break;
    }
  }
  await expect(dados).toBeFocused();
  await page.keyboard.press("Enter");

  await expect(page).toHaveURL(/\/dados$/);
  await expect(dados).toHaveAttribute("aria-current", "page");
  await expect(page.getByRole("heading", { level: 1, name: "Dados" })).toBeVisible();
});

test("Ctrl+B recolhe a barra e o conteúdo ganha o espaço, sem cortes", async ({
  page,
}, testInfo) => {
  for (const size of WIDTHS.slice(0, 2)) {
    await page.setViewportSize(size);
    await page.goto("/");
    const open = await mainWidth(page);
    await testInfo.attach(`inicio-${String(size.width)}-aberta`, {
      body: await page.screenshot(),
      contentType: "image/png",
    });

    await page.keyboard.press("Control+b");
    await expect(page.locator('[data-slot="sidebar"]')).toHaveAttribute("data-state", "collapsed");
    await expect.poll(() => mainWidth(page)).toBeGreaterThan(open + 150);
    expect(await noHorizontalScroll(page)).toBe(true);
    await testInfo.attach(`inicio-${String(size.width)}-recolhida`, {
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
  await page.goto("/");
  expect(await noHorizontalScroll(page)).toBe(true);
  await testInfo.attach("inicio-390", { body: await page.screenshot(), contentType: "image/png" });

  await page.getByRole("button", { name: "Abrir ou recolher a barra lateral" }).click();
  const drawer = page.getByRole("dialog");
  await expect(drawer).toBeVisible();
  await drawer.getByRole("link", { name: "Experiências" }).click();

  await expect(page).toHaveURL(/\/experiencias$/);
  await expect(drawer).toBeHidden();
  await expect(page.getByRole("heading", { level: 1, name: "Experiências" })).toBeVisible();
});

test("um endereço desconhecido diz que o ecrã não existe", async ({ page }) => {
  await page.goto("/nada-disto");

  await expect(page.getByRole("heading", { name: "Este ecrã não existe" })).toBeVisible();
});

for (const path of ["/", "/dados", "/modelos"]) {
  test(`sem problemas de acessibilidade em ${path}`, async ({ page }) => {
    await page.goto(path);
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();

    const results = await new AxeBuilder({ page }).analyze();

    expect(results.violations.map((violation) => `${violation.id}: ${violation.help}`)).toEqual([]);
  });
}
