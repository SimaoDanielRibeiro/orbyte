/**
 * A comparação de runs (`POST /history/comparison`): de 2 a 5 runs lado a lado.
 *
 * É um POST (a lista de runs vai no corpo), mas não muda nada: pede-se como uma consulta,
 * guardada pelos ids, que volta a pedir-se quando o histórico muda (um run renomeado ou apagado).
 */
import { queryOptions } from "@tanstack/react-query";

import { historyCompare } from "./generated/sdk.gen";

export function comparisonQuery(runIds: readonly string[]) {
  return queryOptions({
    queryKey: ["history-comparison", [...runIds]],
    queryFn: async ({ signal }) => {
      const { data } = await historyCompare({
        body: { run_ids: [...runIds] },
        signal,
        throwOnError: true,
      });
      return data;
    },
    enabled: runIds.length >= 2,
  });
}
