/**
 * O histórico muda quando se guarda, renomeia ou apaga um run, e quando um treino acaba (fica
 * registado sozinho): as listas, as entradas e as comparações voltam a pedir-se.
 */
import type { QueryClient } from "@tanstack/react-query";

/** As consultas do histórico, pelo `_id` que o cliente gerado lhes dá. */
const HISTORY_QUERIES = ["historyRuns", "historyEntry", "history-comparison"] as const;

export function invalidateHistory(queryClient: QueryClient): void {
  for (const id of HISTORY_QUERIES) {
    void queryClient.invalidateQueries({
      predicate: (query) => {
        const [first] = query.queryKey;
        return (
          first === id ||
          (typeof first === "object" && first !== null && "_id" in first && first._id === id)
        );
      },
    });
  }
}
