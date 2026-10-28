/**
 * Os agentes (etapa 10.7): uma procura pequena passa por todas as fases no fluxograma, e pausar,
 * retomar e parar funcionam.
 */
import { expect, test, type Page } from "@playwright/test";

import { loadSample } from "./helpers";

/**
 * As fases que o fluxograma mostrou, pela ordem (guardadas na página, a cada mudança).
 *
 * O fluxograma só aparece quando chega o plano da procura, e esse pedido pode chegar depois
 * das primeiras mensagens do WebSocket: o fluxograma nasce então já numa fase. Por isso conta
 * também a fase com que ele aparece (`childList`), não só as mudanças do atributo.
 */
async function recordPhases(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const seen: string[] = [];
    Object.assign(window, { phases: seen });
    new MutationObserver(() => {
      const phase = document.querySelector("figure[data-phase]")?.getAttribute("data-phase");
      if (phase !== null && phase !== undefined && seen.at(-1) !== phase) {
        seen.push(phase);
      }
    }).observe(document, {
      attributes: true,
      childList: true,
      subtree: true,
      attributeFilter: ["data-phase"],
    });
  });
}

async function phases(page: Page): Promise<string[]> {
  return page.evaluate(() => (window as unknown as { phases: string[] }).phases);
}

/** Põe um controlo deslizante num valor, pelo teclado (do mínimo para cima). */
async function setSlider(page: Page, name: string, steps: number): Promise<void> {
  const slider = page.getByRole("slider", { name });
  await slider.focus();
  await page.keyboard.press("Home");
  for (let step = 0; step < steps; step++) {
    await page.keyboard.press("ArrowRight");
  }
}

/**
 * Começa uma procura pequena no Titanic com um agente rápido (os modelos lineares) e um lento
 * (as redes neuronais): dá tempo de a ver a trabalhar, e passa por todas as fases.
 */
async function startSlowSearch(page: Page): Promise<void> {
  await loadSample(page, "Titanic");
  await page.goto("/agentes");
  // Um grupo de botões de alternar é uma barra de ferramentas para os leitores de ecrã.
  const agents = page.getByRole("toolbar", { name: "Os agentes chamados" });
  await expect(agents).toBeVisible({ timeout: 60_000 });
  for (const name of await agents.getByRole("button").allTextContents()) {
    if (!["Modelos lineares", "Redes neuronais"].includes(name.trim())) {
      await agents.getByRole("button", { name: name.trim(), exact: true }).click();
    }
  }
  await expect(agents.getByRole("button", { pressed: true })).toHaveCount(2);
  await setSlider(page, "Orçamento (candidatos)", 5); // 6 candidatos
  await setSlider(page, "Folds da validação cruzada", 1); // 3 folds
  await expect(page.getByRole("slider", { name: "Orçamento (candidatos)" })).toHaveAttribute(
    "aria-valuenow",
    "6",
  );
  await page.getByRole("button", { name: "Pedir aos agentes" }).click();
  await expect(page).toHaveURL(/\/agentes\/search-/);
}

test("uma procura passa por todas as fases no fluxograma, com pausa e retoma", async ({ page }) => {
  test.setTimeout(300_000);
  await recordPhases(page);
  await startSlowSearch(page);

  await page.getByRole("button", { name: "Pausar" }).click();
  await expect(page.getByRole("status").filter({ hasText: "Em pausa" })).toBeVisible({
    timeout: 60_000,
  });
  await page.getByRole("button", { name: "Retomar" }).click();
  await expect(page.locator("figure[data-phase='done']")).toBeVisible({ timeout: 240_000 });

  // O fluxograma passou pela procura e acabou; as fases que duram muito pouco (a do agente de
  // dados, o treino do vencedor) podem acabar entre duas mensagens, mas a conversa tem todas.
  const shown = await phases(page);
  expect(shown).toContain("screening");
  expect(shown.at(-1)).toBe("done");
  const conversation = page.getByRole("heading", { name: "A conversa dos agentes" }).locator("..");
  for (const label of [
    "Preparar os dados",
    "Ronda 1 · triagem",
    "Ronda 2 · afinação",
    "Treinar e testar o vencedor",
  ]) {
    await expect(conversation.getByText(label).first()).toBeVisible();
  }
  await expect(page.getByRole("heading", { level: 2, name: "O vencedor" })).toBeVisible();
  await expect(
    page.getByRole("heading", { level: 2, name: "O quadro dos candidatos" }),
  ).toBeVisible();

  // O histórico das trocas repete a procura: no fim, o coordenador passa o melhor ao vencedor,
  // o vencedor vai ao teste, e a leitura do teste fecha a conversa.
  await page.getByRole("radio", { name: "Histórico das trocas" }).click();
  const moment = page.getByRole("slider", { name: "O momento da procura" });
  await moment.focus();
  await page.keyboard.press("End");
  await expect(moment).toHaveAttribute("aria-valuetext", /Coordenador, teste$/);
  await page.keyboard.press("ArrowLeft");
  await page.keyboard.press("ArrowLeft");
  await expect(moment).toHaveAttribute("aria-valuetext", /Coordenador, decisão$/);
  await expect(conversation.getByText("para Vencedor")).toBeVisible();
});

test("parar uma procura acaba-a com o melhor até aí", async ({ page }) => {
  test.setTimeout(300_000);
  await startSlowSearch(page);

  await page.getByRole("button", { name: "Parar" }).click();

  await expect(page.locator("figure[data-phase='done']")).toBeVisible({ timeout: 240_000 });
  await expect(page.getByText(/A procura parou a pedido/)).toBeVisible();
});
