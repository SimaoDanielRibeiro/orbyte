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
  await page.goto("/agents");
  // A escolha dos modelos: todos à partida; tira-se tudo e marcam-se duas famílias inteiras.
  const models = page.getByRole("region", { name: "The models" });
  await expect(models).toBeVisible({ timeout: 60_000 });
  await models.getByRole("button", { name: "None" }).click();
  await models.getByRole("checkbox", { name: "Linear models", exact: true }).click();
  await models.getByRole("checkbox", { name: "Neural networks", exact: true }).click();
  // Os três lineares (com os didáticos, que agora também entram) e a rede.
  await expect(models.getByRole("status")).toHaveText(/^4 of \d+ models$/);
  await setSlider(page, "Budget (candidates)", 5); // 6 candidatos
  await setSlider(page, "Cross-validation folds", 1); // 3 folds
  await expect(page.getByRole("slider", { name: "Budget (candidates)" })).toHaveAttribute(
    "aria-valuenow",
    "6",
  );
  await page.getByRole("button", { name: "Ask the agents" }).click();
  await expect(page).toHaveURL(/\/agents\/search-/);
}

test("uma procura passa por todas as fases no fluxograma, com pausa e retoma", async ({ page }) => {
  test.setTimeout(300_000);
  await recordPhases(page);
  await startSlowSearch(page);

  await page.getByRole("button", { name: "Pause" }).click();
  await expect(page.getByRole("status").filter({ hasText: "Paused" })).toBeVisible({
    timeout: 60_000,
  });
  await page.getByRole("button", { name: "Resume" }).click();
  await expect(page.locator("figure[data-phase='done']")).toBeVisible({ timeout: 240_000 });

  // O fluxograma passou pela procura e acabou; as fases que duram muito pouco (a do agente de
  // dados, o treino do vencedor) podem acabar entre duas mensagens, mas a conversa tem todas.
  const shown = await phases(page);
  expect(shown).toContain("screening");
  expect(shown.at(-1)).toBe("done");
  const conversation = page
    .getByRole("heading", { name: "The agents' conversation" })
    .locator("../..");
  for (const label of [
    "Prepare the data",
    "Round 1 · screening",
    "Round 2 · tuning",
    "Train and test the winner",
  ]) {
    await expect(conversation.getByText(label).first()).toBeVisible();
  }
  await expect(page.getByRole("heading", { level: 2, name: "The winner" })).toBeVisible();
  await expect(page.getByRole("heading", { level: 2, name: "The candidate board" })).toBeVisible();

  // O histórico das trocas repete a procura: no fim, o coordenador passa o melhor ao vencedor,
  // o vencedor vai ao teste, e a leitura do teste fecha a conversa.
  await page.getByRole("radio", { name: "Exchange history" }).click();
  const moment = page.getByRole("slider", { name: "The moment of the search" }).first();
  await moment.focus();
  await page.keyboard.press("End");
  await expect(moment).toHaveAttribute("aria-valuetext", /Coordinator, test$/);
  await page.keyboard.press("ArrowLeft");
  await page.keyboard.press("ArrowLeft");
  await expect(moment).toHaveAttribute("aria-valuetext", /Coordinator, decision$/);
  await expect(conversation.getByText("to Winner")).toBeVisible();
});

test("parar uma procura para tudo e volta ao pedido", async ({ page, request }) => {
  test.setTimeout(300_000);
  await startSlowSearch(page);
  const searchId = /\/agents\/(search-[^/?#]+)/.exec(page.url())?.[1];
  expect(searchId).toBeDefined();

  // Parar é parar tudo (ADR 0051): o ecrã volta logo ao pedido, para uma procura nova.
  await page.getByRole("button", { name: "Stop", exact: true }).click();
  await expect(page).toHaveURL(/\/agents$/);
  await expect(page.getByRole("button", { name: "Ask the agents" })).toBeVisible();

  // E a procura acaba sem treinar um vencedor: nada vai para o histórico.
  await expect
    .poll(
      async () => {
        const result = await request.get(`/api/automl/${String(searchId)}/result`);
        return result.ok()
          ? ((await result.json()) as { run_id: string | null }).run_id
          : "running";
      },
      { timeout: 240_000 },
    )
    .toBeNull();

  // Sai-se para o Treinar e volta-se aos Agentes: o pedido está lá, e não a procura parada.
  const nav = page.getByRole("navigation", { name: "Screens" });
  await nav.getByRole("link", { name: "Train & evaluate" }).click();
  await expect(page).toHaveURL(/\/train/);
  await nav.getByRole("link", { name: "Agents" }).click();
  await expect(page).toHaveURL(/\/agents$/);
  await expect(page.getByRole("button", { name: "Ask the agents" })).toBeVisible();
});
