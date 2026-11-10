/**
 * As imagens do README e das publicações (`npm run screenshots`, com o
 * `playwright.screenshots.config.ts`).
 *
 * Todos os ecrãs saem em `docs/images/hd/`, com o dobro dos píxeis (para publicações; fica fora
 * do Git). Os que o README mostra (`README`) saem também em `docs/images/`, com o tamanho do ecrã
 * (1920 × 1080).
 *
 * Não são testes: tudo o que aqui se faz já os testes de ponta a ponta verificam. Cada passo só
 * espera que o ecrã esteja pronto (os dados chegaram, as animações de entrada acabaram) antes de o
 * fotografar.
 */
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { expect, test, type Locator, type Page } from "@playwright/test";

import { animationsSettled, loadSample, trainThroughApi, type ExperimentRequest } from "./helpers";

const IMAGES = fileURLToPath(new URL("../../../docs/images/", import.meta.url));

/** As imagens que o README mostra: as outras ficam só para publicações. */
const README: ReadonlySet<string> = new Set([
  "home",
  "agents-at-work",
  "agents-conversation",
  "analyst",
  "training",
  "metric",
  "explain-predictions",
  "lab",
  "reinforcement",
  "models",
  "history-compare",
]);

/** Fotografa o ecrã, com o dobro dos píxeis e, se o README a mostra, também com o tamanho dele. */
async function shoot(page: Page, name: string): Promise<void> {
  await page.mouse.move(0, 0); // nenhum botão fica com o ar de "rato por cima"
  if (README.has(name)) {
    await page.screenshot({ path: join(IMAGES, `${name}.png`), scale: "css" });
  }
  await page.screenshot({ path: join(IMAGES, "hd", `${name}.png`), scale: "device" });
}

/**
 * Espera que as animações de entrada acabem, sem esperar pela rede: nos ecrãs dos agentes, o
 * WebSocket e os pedidos ao modelo de linguagem não param (o `animationsSettled` espera por ela).
 */
async function entrancesDone(page: Page): Promise<void> {
  await page.waitForFunction(() =>
    document
      .getAnimations()
      .every(
        (animation) =>
          animation.playState !== "running" ||
          animation.effect?.getTiming().iterations === Infinity,
      ),
  );
}

/** Desliza a página até o elemento ficar no topo do ecrã, com uma margem por cima. */
async function scrollToTop(locator: Locator, margin = 32): Promise<void> {
  await locator.evaluate((element, gap) => {
    window.scrollTo({
      top: element.getBoundingClientRect().top + window.scrollY - gap,
      behavior: "instant",
    });
  }, margin);
}

const TITANIC: ExperimentRequest = {
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
};

function iris(c: number): ExperimentRequest {
  return {
    dataset: { kind: "sample", params: { name: "iris" } },
    preparation: { target: "species" },
    model: { name: "logistic_regression", hyperparameters: { C: c } },
    seed: 3,
  };
}

const QUESTION = "Why did the winner win?";

/** A segunda pergunta ao analista, no ecrã inteiro (uma das sugestões). */
const FOLLOW_UP = "Does the winner memorise the training data?";

test("a entrada", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await animationsSettled(page);
  // Os dois desenhos da entrada andam sem parar: dá-se-lhes tempo para a fronteira já estar a
  // meio do ajuste.
  await page.waitForTimeout(2_500);
  await shoot(page, "home");
});

test("os agentes: a procura ao vivo, a conversa, o vencedor e o analista", async ({ page }) => {
  test.setTimeout(45 * 60_000);
  await loadSample(page, "Titanic");
  // O aviso "Titanic loaded" sai sozinho ao fim de uns segundos.
  await expect(page.getByText(/Titanic loaded/)).toBeHidden({ timeout: 30_000 });
  await animationsSettled(page);
  await shoot(page, "data");

  await page.goto("/agents");
  await expect(page.getByRole("region", { name: "The models" })).toBeVisible({ timeout: 60_000 });
  // O agente de IA entra na afinação: o interruptor só liga com o Ollama pronto.
  const ai = page.getByRole("switch", { name: /Call the AI agent/ });
  await expect(ai).toBeEnabled({ timeout: 30_000 });
  await ai.click();
  await expect(ai).toBeChecked();
  await page.getByRole("button", { name: "Ask the agents" }).click();
  // Com o agente de IA, o pedido só volta depois de o Ollama carregar o modelo na memória.
  await expect(page).toHaveURL(/\/agents\/search-/, { timeout: 3 * 60_000 });

  // Ao vivo: com o agente de IA a pensar, e o raciocínio já a crescer na conversa (da primeira
  // vez, o Ollama ainda carrega o modelo, e o raciocínio fica uns segundos em "…").
  await expect(page.getByRole("button", { name: /Thinking about the next candidate/ })).toBeVisible(
    { timeout: 20 * 60_000 },
  );
  const reasoning = page
    .locator("li[aria-busy='true']", { hasText: "is thinking about candidate" })
    .first()
    .locator("p")
    .nth(1);
  await expect
    .poll(async () => (await reasoning.textContent())?.length ?? 0, { timeout: 5 * 60_000 })
    .toBeGreaterThan(300);
  await page.evaluate(() => {
    window.scrollTo({ top: 0, behavior: "instant" });
  });
  await shoot(page, "agents-at-work");

  // A conversa em grande, no "Expand": ao vivo, desliza sozinha até à última mensagem.
  await page.getByRole("button", { name: "Expand", exact: true }).click();
  const room = page.getByRole("dialog", { name: "Agent conversation" });
  await expect(room).toBeVisible();
  const messages = room.getByRole("list", { name: "Messages" }).locator("..");
  await expect
    .poll(() =>
      messages.evaluate(
        (element) => element.scrollHeight - element.scrollTop - element.clientHeight,
      ),
    )
    .toBeLessThan(80);
  await entrancesDone(page);
  await shoot(page, "agents-conversation");
  await page.keyboard.press("Escape");
  await expect(room).toBeHidden();

  // O fim: o fluxograma acabado, o vencedor e o quadro dos candidatos.
  await expect(page.locator("figure[data-phase='done']")).toBeVisible({ timeout: 30 * 60_000 });
  const winner = page.getByRole("heading", { level: 2, name: "The winner" });
  await expect(winner).toBeVisible({ timeout: 60_000 });
  await entrancesDone(page);
  await page.evaluate(() => {
    window.scrollTo({ top: 0, behavior: "instant" });
  });
  await shoot(page, "agents-done");
  await scrollToTop(winner);
  await entrancesDone(page);
  await shoot(page, "agents-winner");

  // O analista responde, com o modelo de linguagem a sério, porque ganhou o vencedor.
  const panel = page.getByRole("region", { name: "Ask the analyst" });
  const ask = panel.getByRole("button", { name: QUESTION });
  await expect(ask).toBeEnabled({ timeout: 60_000 });
  await ask.click();
  const turn = panel.getByRole("article", { name: `Question: ${QUESTION}` });
  const work = turn.getByText("How it got there");
  await expect(work).toBeVisible({ timeout: 15 * 60_000 });
  await scrollToTop(panel, 48);
  await entrancesDone(page);
  await shoot(page, "analyst");
  // E o que ele fez para lá chegar: cada ferramenta, com o que devolveu.
  await work.click();
  await expect(turn.getByRole("list", { name: "What the analyst did" })).toBeVisible();
  await entrancesDone(page);
  await shoot(page, "analyst-steps");

  // Em ecrã inteiro, como num chat (ADR 0057): mais uma pergunta, pela sugestão, e a conversa
  // toda; depois, os passos da resposta, com o pensamento que pediu a primeira ferramenta.
  await panel.getByRole("button", { name: "Expand" }).click();
  const chat = page.getByRole("dialog", { name: "Ask the analyst" });
  await expect(chat).toBeVisible();
  await chat.getByRole("button", { name: FOLLOW_UP }).click();
  const answer = chat.getByRole("article", { name: `Question: ${FOLLOW_UP}` });
  await expect(answer.getByText(/qwen3:8b · /)).toBeVisible({ timeout: 15 * 60_000 });
  await entrancesDone(page);
  await shoot(page, "analyst-expanded");
  await answer.getByText("How it got there").click();
  await answer
    .getByRole("button", { name: /^Thought · / })
    .first()
    .click();
  await answer.evaluate((element) => {
    element.scrollIntoView({ block: "start" });
  });
  await entrancesDone(page);
  await shoot(page, "analyst-expanded-steps");
  await page.keyboard.press("Escape");
  await expect(chat).toBeHidden();
});

test("treinar e avaliar: a rede neuronal nas luas, e a fórmula de uma métrica", async ({
  page,
  request,
}) => {
  test.setTimeout(5 * 60_000);
  const runId = await trainThroughApi(request, {
    dataset: { kind: "synthetic", params: { generator: "moons", seed: 7, parameters: {} } },
    preparation: { target: "target" },
    model: { name: "mlp" },
    seed: 7,
  });

  await page.goto(`/train/${runId}`);
  // A fronteira na última fotografia do treino, já à volta das duas luas.
  const snapshot = page.getByRole("slider", { name: "Training snapshot" });
  await expect(snapshot).toBeVisible({ timeout: 60_000 });
  await snapshot.focus();
  await page.keyboard.press("End");
  await snapshot.blur();
  await animationsSettled(page);
  await shoot(page, "training");

  const table = page.getByRole("table", { name: "The metrics" });
  await table.getByRole("button", { name: "F1", exact: true }).click();
  const drawer = page.getByRole("dialog", { name: "F1" });
  await expect(drawer.locator(".katex").first()).toBeVisible();
  await animationsSettled(page);
  await shoot(page, "metric");
});

test("explicar: a história do treino e o que pesa nas previsões", async ({ page, request }) => {
  test.setTimeout(5 * 60_000);
  const runId = await trainThroughApi(request, TITANIC);

  await page.goto(`/explain/${runId}`);
  const story = page.getByRole("region", { name: "What happened in the training" });
  await expect(story.getByText(/Logistic regression learned to predict Survived/)).toBeVisible({
    timeout: 60_000,
  });
  await animationsSettled(page);
  await shoot(page, "explain");

  await page.getByRole("button", { name: "Compute the importance" }).click();
  await expect(page.getByRole("img", { name: /when each feature is shuffled/ })).toBeVisible({
    timeout: 120_000,
  });
  await scrollToTop(page.getByRole("heading", { level: 2, name: "What drives the predictions" }));
  await animationsSettled(page);
  await shoot(page, "explain-predictions");
});

test("o laboratório: o overfitting de um polinómio", async ({ page }) => {
  await page.goto("/lab");
  await expect(page.getByRole("img", { name: /degree 3 polynomial/ })).toBeVisible({
    timeout: 60_000,
  });
  // Um grau alto: a curva passa por todos os pontos do treino e falha os do teste.
  const degree = page.getByRole("slider", { name: "Polynomial degree" });
  await degree.focus();
  for (let step = 0; step < 6; step++) {
    await page.keyboard.press("ArrowRight");
  }
  await expect(page.getByText(/With degree 9,/)).toBeVisible({ timeout: 60_000 });
  await degree.blur();
  await animationsSettled(page);
  await shoot(page, "lab");
});

test("os modelos: o catálogo", async ({ page }) => {
  await page.goto("/models");
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await animationsSettled(page);
  await shoot(page, "models");
});

test("reinforcement learning: a lição do Q-learning no penhasco", async ({ page }) => {
  test.setTimeout(3 * 60_000);
  await page.goto("/rl");
  const lesson = page.getByRole("region", { name: "How an agent learns" });
  const learned = lesson.getByRole("radio", { name: /Episode 160/ });
  await expect(learned).toBeVisible({ timeout: 60_000 });
  // O último episódio da lição: o valor já se espalhou a partir do objetivo, e o agente contorna
  // o penhasco. A lição anda sozinha: uns segundos, para ele já ir a meio do caminho.
  await learned.click();
  await scrollToTop(lesson);
  await animationsSettled(page);
  await page.waitForTimeout(5_000);
  await shoot(page, "reinforcement");
});

test("o histórico: dois runs lado a lado", async ({ page, request }) => {
  test.setTimeout(5 * 60_000);
  const runs = [
    { runId: await trainThroughApi(request, iris(10)), name: "Iris, C = 10" },
    { runId: await trainThroughApi(request, iris(0.01)), name: "Iris, C = 0.01" },
  ];
  for (const { runId, name } of runs) {
    await request.patch(`/api/history/runs/${runId}`, { data: { name } });
  }

  await page.goto("/history");
  for (const { name } of runs) {
    await page.getByRole("checkbox", { name: `Compare “${name}”` }).click();
  }
  const compared = page.getByRole("heading", { name: "What changes between the runs" });
  await expect(compared).toBeVisible();
  await animationsSettled(page);
  await shoot(page, "history");
  await scrollToTop(compared);
  await animationsSettled(page);
  await shoot(page, "history-compare");
});
