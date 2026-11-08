/**
 * Apagar vários runs do histórico de uma vez (os escolhidos na tabela do History).
 *
 * É o mesmo `DELETE /history/runs/{run_id}` de apagar um run, uma vez por run, todos ao mesmo
 * tempo: a API não precisa de uma rota nova para isto. Se algum falhar, os outros apagam-se na
 * mesma, e o erro do primeiro que falhou chega a quem pediu (a tabela volta a pedir-se e mostra
 * o que ficou).
 */
import { historyDelete } from "./generated/sdk.gen";

export async function deleteRuns(runIds: readonly string[]): Promise<void> {
  const results = await Promise.allSettled(
    runIds.map((runId) => historyDelete({ path: { run_id: runId }, throwOnError: true })),
  );
  const failure = results.find(
    (result): result is PromiseRejectedResult => result.status === "rejected",
  );
  if (failure !== undefined) {
    throw failure.reason instanceof Error
      ? failure.reason
      : new Error("Some runs couldn't be deleted. Try again.");
  }
}
