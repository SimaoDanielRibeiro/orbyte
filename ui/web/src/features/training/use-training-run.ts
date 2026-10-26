import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import type { JobState, StepRecord, TrainingAnimation, TrainingUpdate } from "@/api/generated";
import { experimentsResultOptions } from "@/api/generated/@tanstack/react-query.gen";
import { useJobStream } from "@/api/stream";

/** O treino a decorrer, tal como chega pelo WebSocket. */
export interface LiveTraining {
  state: JobState | null;
  /** Os passos até agora, sem repetições. */
  steps: StepRecord[];
  /** A fronteira no último snapshot. */
  preview: TrainingAnimation | null;
}

const EMPTY: LiveTraining = { state: null, steps: [], preview: null };

/**
 * Um run: o treino ao vivo (pelo WebSocket, até acabar) e depois o resultado.
 *
 * O WebSocket recomeça do princípio a cada ligação: os passos que já se têm ignoram-se. Um run
 * que já não está a treinar (aberto do histórico, ou de antes de a API recomeçar) não tem
 * transmissão: pede-se logo o resultado, e é ele que diz se o run ainda existe.
 */
export function useTrainingRun(runId: string) {
  const [live, setLive] = useState<LiveTraining>(EMPTY);
  const stream = useJobStream<TrainingUpdate>(`/experiments/${runId}/stream`, (update) => {
    setLive((previous) => {
      const last = previous.steps.at(-1)?.step ?? 0;
      const fresh = update.steps.filter((record) => record.step > last);
      return {
        state: update.state,
        steps: fresh.length > 0 ? [...previous.steps, ...fresh] : previous.steps,
        preview: update.preview ?? previous.preview,
      };
    });
  });
  const status = live.state?.status ?? null;
  const detached = stream.error !== null;
  const result = useQuery({
    ...experimentsResultOptions({ path: { run_id: runId } }),
    enabled: status === "finished" || detached,
    staleTime: "static",
  });
  return {
    live,
    status,
    /** Se o run não tem transmissão (já não está a treinar nesta sessão da API). */
    detached,
    result,
    connected: stream.connected,
  };
}
