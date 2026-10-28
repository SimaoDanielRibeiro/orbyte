import { useQuery } from "@tanstack/react-query";

import { ApiRequestError } from "@/api/errors";
import {
  experimentsResultOptions,
  historyOpenOptions,
} from "@/api/generated/@tanstack/react-query.gen";

/**
 * O resultado de um run terminado, de onde estiver: da memória da API (os últimos runs) ou, se
 * já lá não estiver (`run_not_in_memory`), do histórico, que o reabre. Um run guardado volta do
 * histórico com o modelo, e com ele os gráficos e o ecrã Explicar.
 */
export function useRunResult(runId: string) {
  const session = useQuery({
    ...experimentsResultOptions({ path: { run_id: runId } }),
    staleTime: "static",
  });
  const fromHistory =
    session.error instanceof ApiRequestError && session.error.code === "run_not_in_memory";
  const history = useQuery({
    ...historyOpenOptions({ path: { run_id: runId } }),
    enabled: fromHistory,
    staleTime: "static",
  });
  return { result: fromHistory ? history : session, fromHistory };
}
