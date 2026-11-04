/**
 * Reinforcement learning (etapa 10.8): treinar um agente, ver o que aprendeu episódio a
 * episódio, guardá-lo, e pausar e retomar um treino.
 */
import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

import { animationsSettled } from "./helpers";

test("treina um agente e mostra o que ele aprendeu", async ({ page }) => {
  test.setTimeout(120_000);
  await page.goto("/rl");
  await page.getByRole("radio", { name: "Cliff" }).click();
  await page.getByRole("button", { name: "Train the agent" }).click();

  await expect(page).toHaveURL(/\/rl\/run-/);
  await expect(page.getByRole("heading", { level: 2, name: "What the agent learned" })).toBeVisible(
    { timeout: 60_000 },
  );
  await expect(page.getByRole("heading", { level: 1, name: "Q-learning" })).toBeVisible();
  const episode = page.getByRole("slider", { name: "Episode" });
  const last = await episode.getAttribute("aria-valuetext");
  await episode.focus();
  await page.keyboard.press("Home");
  await expect(episode).not.toHaveAttribute("aria-valuetext", last ?? "");
  await expect(page.getByRole("heading", { name: "The Q table, action by action" })).toBeVisible();

  await page.getByRole("button", { name: "Save the run" }).click();
  await expect(page.getByRole("button", { name: "Run saved" })).toBeDisabled();

  await animationsSettled(page);
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations.map((violation) => `${violation.id}: ${violation.help}`)).toEqual([]);
});

test("a lição mostra cada agente a aprender: o Q-learning e o DQN", async ({ page }) => {
  // A lição do DQN treina uma rede (uns 6 s sozinha); com os outros testes a treinar modelos na
  // mesma API ao mesmo tempo, o CPU reparte-se e chega a passar do minuto.
  test.setTimeout(240_000);
  await page.goto("/rl");
  const lesson = page.getByRole("region", { name: "How an agent learns" });
  await expect(lesson.getByRole("radio", { name: /Episode 160/ })).toBeVisible({
    timeout: 30_000,
  });

  // A do DQN treina-se a primeira vez que se pede (uns segundos), com outros episódios.
  await lesson
    .getByRole("radiogroup", { name: "The agent in the lesson" })
    .getByRole("radio", { name: "DQN (Deep Q-Network)" })
    .click();
  await expect(lesson.getByRole("radio", { name: /Episode 300/ })).toBeVisible({
    timeout: 180_000,
  });
  await expect(lesson.getByText(/the network takes a training step/)).toBeVisible();
});

test("pausar e retomar o treino de um agente", async ({ page }) => {
  test.setTimeout(180_000);
  await page.goto("/rl");
  // O DQN (uma rede no lugar da tabela) é o mais lento: dá tempo de o pausar. O do treino (a
  // lição tem outro botão com o mesmo nome).
  await page
    .getByRole("radiogroup", { name: "The agent", exact: true })
    .getByRole("radio", { name: "DQN (Deep Q-Network)" })
    .click();
  await page.getByRole("button", { name: "Train the agent" }).click();
  await expect(page).toHaveURL(/\/rl\/run-/);

  await page.getByRole("button", { name: "Pause" }).click();
  await expect(page.getByRole("status").filter({ hasText: "Paused" })).toBeVisible({
    timeout: 30_000,
  });
  await page.getByRole("button", { name: "Resume" }).click();

  await expect(page.getByRole("heading", { level: 2, name: "What the agent learned" })).toBeVisible(
    { timeout: 120_000 },
  );
});
