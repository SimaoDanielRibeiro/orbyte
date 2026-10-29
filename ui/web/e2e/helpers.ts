/**
 * O que os testes de ponta a ponta fazem muitas vezes: abrir um dataset de exemplo, prepará-lo e
 * esperar por um treino lançado diretamente na API.
 */
import { expect, type APIRequestContext, type Page } from "@playwright/test";

/** Os campos de um pedido de treino que os testes usam (o resto fica por omissão). */
export interface ExperimentRequest {
  dataset: { kind: string; params: Record<string, unknown> };
  preparation: Record<string, unknown>;
  model: { name: string; hyperparameters?: Record<string, unknown> };
  seed?: number;
}

/** Carrega um dataset de exemplo no ecrã Dados e espera que fique aberto. */
export async function loadSample(page: Page, title: string): Promise<void> {
  await page.goto("/dados");
  await page.getByRole("button", { name: `Carregar ${title}` }).click();
  await expect(page.getByRole("heading", { level: 2, name: title })).toBeVisible();
}

/** Prepara o dataset aberto com as sugestões do backend e espera que a preparação sirva. */
export async function prepareWithSuggestions(page: Page): Promise<void> {
  await page.goto("/preparar");
  await expect(page.getByRole("link", { name: "Escolher o modelo" }).first()).toBeVisible({
    timeout: 30_000,
  });
}

/** Começa um treino pela API (pelo mesmo proxy da interface) e espera que acabe. */
export async function trainThroughApi(
  request: APIRequestContext,
  body: ExperimentRequest,
): Promise<string> {
  const started = await request.post("/api/experiments", { data: body });
  expect(started.status()).toBe(202);
  const { run_id: runId } = (await started.json()) as { run_id: string };
  await expect
    .poll(
      async () => {
        const job = await request.get(`/api/experiments/${runId}/job`);
        return ((await job.json()) as { status: string }).status;
      },
      { timeout: 60_000 },
    )
    .toBe("finished");
  return runId;
}

/**
 * O mesmo pacote `.mllab`, sem a assinatura (`manifest.sig`): continua intacto (os ficheiros
 * conferem com o manifesto), mas para o ML Lab passa a ser de outra instalação, e o modelo só
 * abre depois do aviso. As entradas do ZIP copiam-se tal como estão (comprimidas); só se refazem
 * o diretório central e o registo final, com as posições novas.
 */
export function withoutSignature(zip: Buffer): Buffer {
  const end = zip.lastIndexOf(Buffer.from([0x50, 0x4b, 0x05, 0x06]));
  const count = zip.readUInt16LE(end + 10);
  let entry = zip.readUInt32LE(end + 16);
  const kept: Buffer[] = [];
  const directory: Buffer[] = [];
  let offset = 0;
  for (let index = 0; index < count; index += 1) {
    const nameLength = zip.readUInt16LE(entry + 28);
    const length = 46 + nameLength + zip.readUInt16LE(entry + 30) + zip.readUInt16LE(entry + 32);
    const header = Buffer.from(zip.subarray(entry, entry + length));
    const name = header.toString("utf8", 46, 46 + nameLength);
    entry += length;
    if (name === "manifest.sig") {
      continue;
    }
    const local = header.readUInt32LE(42);
    const size =
      30 + zip.readUInt16LE(local + 26) + zip.readUInt16LE(local + 28) + header.readUInt32LE(20);
    kept.push(zip.subarray(local, local + size));
    header.writeUInt32LE(offset, 42);
    directory.push(header);
    offset += size;
  }
  const central = Buffer.concat(directory);
  const record = Buffer.alloc(22);
  record.writeUInt32LE(0x06054b50, 0);
  record.writeUInt16LE(directory.length, 8);
  record.writeUInt16LE(directory.length, 10);
  record.writeUInt32LE(central.length, 12);
  record.writeUInt32LE(offset, 16);
  return Buffer.concat([...kept, central, record]);
}
