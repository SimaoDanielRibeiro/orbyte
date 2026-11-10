/**
 * O analista (etapa 13.3): depois de uma procura, pergunta-se porque ganhou o vencedor, e vê-se cada
 * ferramenta que usou e o que ela devolveu. A conversa também se abre em ecrã inteiro (ADR 0057).
 *
 * Estes testes não precisam do Ollama. A API dos testes procura-o numa porta fechada (ver
 * `playwright.config.ts`), por isso o painel mostra o estado de `GET /llm/status` e o que fazer.
 * Para perguntar, a página finge que o Ollama está pronto, e o pedido da pergunta leva as respostas
 * guardadas do modelo (`replies`): a API dá-as outra vez com o `ScriptedModel`, como faz para
 * repetir uma resposta. Tudo o resto é a sério: a procura, as ferramentas, o WebSocket e o ecrã.
 *
 * Os testes partilham uma procura, um depois do outro (os últimos continuam a conversa do segundo):
 * cada procura treina um vencedor, e a API dos testes, partilhada por todos, guarda poucos runs em
 * memória (um run de outro teste que saísse dela já não se podia guardar).
 */
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

import { animationsSettled } from "./helpers";

/** Uma procura pequena no Iris: três modelos rápidos, seis candidatos. */
const SEARCH = {
  dataset: { kind: "sample", params: { name: "iris" } },
  target: "species",
  models: ["logistic_regression", "knn", "decision_tree"],
  settings: { budget: 6, folds: 3, finalists: 2 },
  seed: 3,
};

/** O estado do Ollama quando está pronto, como `GET /llm/status` o devolve. */
const READY = {
  ready: true,
  running: true,
  url: "http://127.0.0.1:11434",
  model: "qwen3:8b",
  installed: true,
  version: "0.40.6",
  models: ["qwen3:8b"],
  thinks: true,
  notices: [],
};

const QUESTION = "Why did the winner win?";

/** O que um modelo de linguagem respondeu: o quadro, o vencedor contra o segundo e a resposta. */
function savedReplies(winner: number, second: number) {
  return [
    { thinking: "The board first.", tool_calls: [{ name: "list_candidates", arguments: {} }] },
    {
      thinking: "Now the winner against the second.",
      tool_calls: [{ name: "compare_candidates", arguments: { first: winner, second } }],
    },
    { thinking: "I know enough.", content: answerText(winner, second) },
  ];
}

function answerText(winner: number, second: number): string {
  return `#${String(winner)} is first on the board, and the comparison with #${String(second)} says why.`;
}

/** Uma procura acabada: o id, o vencedor e o segundo do quadro. */
interface Search {
  searchId: string;
  winner: number;
  second: number;
}

/** Começa a procura pela API e espera que acabe. */
async function finishedSearch(request: APIRequestContext): Promise<Search> {
  const started = await request.post("/api/automl", { data: SEARCH });
  expect(started.status()).toBe(202);
  const { run_id: searchId } = (await started.json()) as { run_id: string };
  await expect
    .poll(
      async () => {
        const job = await request.get(`/api/automl/${searchId}/job`);
        return ((await job.json()) as { status: string }).status;
      },
      { timeout: 120_000 },
    )
    .toBe("finished");
  const result = await request.get(`/api/automl/${searchId}/result`);
  const { ranking } = (await result.json()) as { ranking: number[] };
  const [winner, second] = ranking;
  if (winner === undefined || second === undefined) {
    throw new Error("The search should have at least two candidates on the board.");
  }
  return { searchId, winner, second };
}

/** A página finge o Ollama pronto, e cada pergunta leva as respostas guardadas do modelo. */
async function withSavedReplies(page: Page, { searchId, winner, second }: Search): Promise<void> {
  await page.route("**/api/llm/status", (route) => route.fulfill({ json: READY }));
  await page.route(`**/api/automl/${searchId}/questions`, async (route) => {
    if (route.request().method() !== "POST") {
      await route.continue();
      return;
    }
    const body = route.request().postDataJSON() as Record<string, unknown>;
    await route.continue({
      postData: JSON.stringify({ ...body, replies: savedReplies(winner, second) }),
    });
  });
}

test.describe.configure({ mode: "serial" });

let search: Search;

test.beforeAll(async ({ playwright }, testInfo) => {
  test.setTimeout(180_000);
  const request = await playwright.request.newContext({ baseURL: testInfo.project.use.baseURL });
  try {
    search = await finishedSearch(request);
  } finally {
    await request.dispose();
  }
});

test("sem o Ollama, o painel do analista diz como o ligar", async ({ page }) => {
  await page.goto(`/agents/${search.searchId}`);

  const panel = page.getByRole("region", { name: "Ask the analyst" });
  await expect(panel.getByRole("status", { name: "The analyst's language model" })).toContainText(
    "Ollama isn't running",
    { timeout: 15_000 },
  );
  await expect(panel.getByRole("button", { name: "Ask", exact: true })).toBeDisabled();
  await expect(panel.getByRole("button", { name: QUESTION })).toBeDisabled();
  // As ferramentas que ele usaria estão à vista na mesma, com o que cada uma faz.
  await expect(panel.getByRole("list", { name: "The analyst's tools" })).toContainText(
    "The candidate board",
  );
});

test("o analista explica porque ganhou o vencedor, e vê-se cada ferramenta que usou", async ({
  page,
}) => {
  const { searchId, winner, second } = search;
  await withSavedReplies(page, search);

  await page.goto(`/agents/${searchId}`);
  const panel = page.getByRole("region", { name: "Ask the analyst" });
  await panel.getByRole("button", { name: QUESTION }).click();

  const turn = panel.getByRole("article", { name: `Question: ${QUESTION}` });
  // A resposta fica em texto corrido; os passos fecham-se numa linha com as ferramentas que usou.
  await expect(turn.getByText(answerText(winner, second))).toBeVisible({ timeout: 60_000 });
  const work = turn.getByText("How it got there");
  await expect(work).toBeVisible();
  await work.click();
  const steps = turn.getByRole("list", { name: "What the analyst did" });
  // Os passos pelo que ele fez (ADR 0057): o quadro, a comparação e a resposta, cada um com o
  // pensamento que o levou lá fechado por baixo.
  await expect(steps.getByRole("listitem")).toHaveCount(3);
  await expect(steps.getByText("The candidate board")).toBeVisible();
  await expect(steps.getByText("list_candidates()")).toBeVisible();
  await expect(steps.getByText("Compare two candidates")).toBeVisible();
  // Exato: o quadro também sugere esta chamada, no JSON que devolveu.
  await expect(
    steps.getByText(`compare_candidates(first=${String(winner)}, second=${String(second)})`, {
      exact: true,
    }),
  ).toBeVisible();
  await expect(steps.getByText("Answer", { exact: true })).toBeVisible();
  // Fechado, o pensamento diz o tamanho, e não o princípio; aberto, tem o texto inteiro.
  await expect(steps.getByText("The board first.")).toBeHidden();
  const thought = steps.getByRole("button", { name: "Thought · 3 words" }).first();
  await thought.click();
  await expect(thought).toHaveAttribute("aria-expanded", "true");
  await expect(steps.getByText("The board first.")).toBeVisible();
  // A leitura do backend: os números da resposta estão no que as ferramentas devolveram, num selo
  // com a mensagem na dica.
  const checked = turn.getByText("Numbers checked");
  await expect(checked).toBeVisible();
  await checked.hover();
  await expect(page.getByRole("tooltip")).toHaveText(
    "Every number in the answer is in what the tools returned.",
  );
  await expect(turn.getByText(/saved answers · 2 tools/)).toBeVisible();

  // O que o quadro devolveu é o JSON que o modelo leu.
  await steps.getByRole("button", { name: "What it returned" }).first().click();
  await expect(steps.locator("pre").first()).toContainText(`"winner": {`);

  // "Copy" leva a resposta tal como o analista a escreveu.
  await page.context().grantPermissions(["clipboard-read", "clipboard-write"]);
  await turn.getByRole("button", { name: "Copy" }).click();
  await expect(turn.getByRole("button", { name: "Copied" })).toBeVisible();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(
    answerText(winner, second),
  );

  await animationsSettled(page);
  const results = await new AxeBuilder({ page })
    .include('section[aria-labelledby="analyst-title"]')
    .analyze();
  expect(results.violations.map((violation) => `${violation.id}: ${violation.help}`)).toEqual([]);

  // A resposta fica com a procura: depois de recarregar, continua lá.
  await page.reload();
  await expect(
    page
      .getByRole("region", { name: "Ask the analyst" })
      .getByRole("article", { name: `Question: ${QUESTION}` }),
  ).toBeVisible();
});

test("só a conversa desliza: a página fica onde está, e as sugestões que faltam ficam em baixo", async ({
  page,
}) => {
  const { searchId } = search;
  await withSavedReplies(page, search);
  await page.goto(`/agents/${searchId}`);
  const panel = page.getByRole("region", { name: "Ask the analyst" });
  const conversation = panel.getByRole("region", { name: "The conversation with the analyst" });
  await expect(panel.getByRole("article", { name: `Question: ${QUESTION}` })).toBeVisible();

  // Mais duas perguntas, escritas na caixa (Enter pergunta), para a conversa passar da altura do
  // painel.
  const box = panel.getByRole("textbox", { name: "Your question to the analyst" });
  for (const question of ["How close was the second best?", "What did the data agent do?"]) {
    await box.fill(question);
    await box.press("Enter");
    await expect(
      panel.getByRole("article", { name: `Question: ${question}` }).getByText(/saved answers/),
    ).toBeVisible({ timeout: 60_000 });
  }
  // Uma pergunta já feita sai das sugestões; as outras ficam por cima da caixa da pergunta.
  const suggested = panel.getByRole("list", { name: "Suggested questions" });
  await expect(suggested.getByRole("button", { name: QUESTION })).toHaveCount(0);
  await expect(
    suggested.getByRole("button", { name: "How close was the second best?" }),
  ).toHaveCount(0);

  // O painel tem a altura fixa: a conversa desliza por dentro, e a página não se mexe.
  const sizes = await conversation.evaluate((element) => ({
    scroll: element.scrollHeight,
    client: element.clientHeight,
  }));
  expect(sizes.scroll).toBeGreaterThan(sizes.client);
  await conversation.hover();
  const pageTop = await page.evaluate(() => window.scrollY);
  await page.mouse.wheel(0, -5000);
  await expect.poll(() => conversation.evaluate((element) => element.scrollTop)).toBe(0);
  expect(await page.evaluate(() => window.scrollY)).toBe(pageTop);
  // Quem sobe para ler deixa de seguir a conversa, e tem um botão para voltar ao fim.
  await expect(panel.getByRole("button", { name: "Latest" })).toBeVisible();
  await panel.getByRole("button", { name: "Latest" }).click();
  await expect(panel.getByRole("button", { name: "Latest" })).toBeHidden();
});

test("o Expand abre o analista em ecrã inteiro, e a conversa continua lá", async ({ page }) => {
  const { searchId } = search;
  await withSavedReplies(page, search);
  await page.goto(`/agents/${searchId}`);
  const panel = page.getByRole("region", { name: "Ask the analyst" });
  await expect(panel.getByRole("article", { name: `Question: ${QUESTION}` })).toBeVisible();
  const cardHeight = await panel
    .getByRole("region", { name: "The conversation with the analyst" })
    .evaluate((element) => element.clientHeight);
  // O que se começou a escrever no cartão segue para o ecrã inteiro.
  const draft = "Which model came last?";
  await panel.getByRole("textbox", { name: "Your question to the analyst" }).fill(draft);

  await panel.getByRole("button", { name: "Expand" }).click();
  const room = page.getByRole("dialog", { name: "Ask the analyst" });
  await expect(room).toBeVisible();
  // A mesma conversa, com mais altura, e a caixa da pergunta pronta a escrever.
  await expect(room.getByRole("article", { name: `Question: ${QUESTION}` })).toBeVisible();
  const roomHeight = await room
    .getByRole("region", { name: "The conversation with the analyst" })
    .evaluate((element) => element.clientHeight);
  expect(roomHeight).toBeGreaterThan(cardHeight);
  const box = room.getByRole("textbox", { name: "Your question to the analyst" });
  await expect(box).toBeFocused();
  await expect(box).toHaveValue(draft);
  await box.press("Enter");
  await expect(
    room.getByRole("article", { name: `Question: ${draft}` }).getByText(/saved answers/),
  ).toBeVisible({ timeout: 60_000 });

  await animationsSettled(page);
  const results = await new AxeBuilder({ page }).include('[role="dialog"]').analyze();
  expect(results.violations.map((violation) => `${violation.id}: ${violation.help}`)).toEqual([]);

  // O "Collapse" volta ao cartão, com a resposta nova, e o foco volta ao "Expand".
  await room.getByRole("button", { name: "Collapse" }).click();
  await expect(room).toBeHidden();
  await expect(panel.getByRole("article", { name: `Question: ${draft}` })).toBeVisible();
  const expand = panel.getByRole("button", { name: "Expand" });
  await expect(expand).toBeFocused();
  // O Esc também fecha.
  await expand.click();
  await expect(room).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(room).toBeHidden();
});
