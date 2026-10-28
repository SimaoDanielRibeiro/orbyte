/**
 * Reinforcement learning (etapa 10.8): treinar um agente, ver o que aprendeu episódio a
 * episódio, guardá-lo, e pausar e retomar um treino.
 */
import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test("treina um agente e mostra o que ele aprendeu", async ({ page }) => {
  test.setTimeout(120_000);
  await page.goto("/rl");
  await page.getByRole("radio", { name: "Penhasco" }).click();
  await page.getByRole("button", { name: "Treinar o agente" }).click();

  await expect(page).toHaveURL(/\/rl\/run-/);
  await expect(
    page.getByRole("heading", { level: 2, name: "O que o agente aprendeu" }),
  ).toBeVisible({ timeout: 60_000 });
  await expect(page.getByRole("heading", { level: 1, name: "Q-learning" })).toBeVisible();
  const episode = page.getByRole("slider", { name: "Episódio" });
  const last = await episode.getAttribute("aria-valuetext");
  await episode.focus();
  await page.keyboard.press("Home");
  await expect(episode).not.toHaveAttribute("aria-valuetext", last ?? "");
  await expect(page.getByRole("heading", { name: "A tabela Q, ação a ação" })).toBeVisible();

  await page.getByRole("button", { name: "Guardar experiência" }).click();
  await expect(page.getByRole("button", { name: "Experiência guardada" })).toBeDisabled();

  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations.map((violation) => `${violation.id}: ${violation.help}`)).toEqual([]);
});

test("pausar e retomar o treino de um agente", async ({ page }) => {
  test.setTimeout(180_000);
  await page.goto("/rl");
  // O DQN (uma rede no lugar da tabela) é o mais lento: dá tempo de o pausar.
  await page.getByRole("radio", { name: "DQN (Deep Q-Network)" }).click();
  await page.getByRole("button", { name: "Treinar o agente" }).click();
  await expect(page).toHaveURL(/\/rl\/run-/);

  await page.getByRole("button", { name: "Pausar" }).click();
  await expect(page.getByRole("status").filter({ hasText: "Em pausa" })).toBeVisible({
    timeout: 30_000,
  });
  await page.getByRole("button", { name: "Retomar" }).click();

  await expect(
    page.getByRole("heading", { level: 2, name: "O que o agente aprendeu" }),
  ).toBeVisible({ timeout: 120_000 });
});
