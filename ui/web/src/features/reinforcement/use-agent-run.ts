import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { ApiRequestError } from "@/api/errors";
import type {
  GridLayout,
  JobState,
  PolicyFrame,
  ReinforcementUpdate,
  StepRecord,
} from "@/api/generated";
import {
  historyOpenReinforcementOptions,
  reinforcementResultOptions,
} from "@/api/generated/@tanstack/react-query.gen";
import { useJobStream } from "@/api/stream";

/** O treino de um agente a decorrer, tal como chega pelo WebSocket. */
export interface LiveAgent {
  state: JobState | null;
  /** Os episódios até agora, sem repetições. */
  episodes: StepRecord[];
  /** O que o agente sabia no último snapshot, e a grelha onde se desenha. */
  preview: PolicyFrame | null;
  layout: GridLayout | null;
}

const EMPTY: LiveAgent = { state: null, episodes: [], preview: null, layout: null };

/**
 * O treino de um agente de reinforcement learning: ao vivo pelo WebSocket, até acabar, e depois
 * o resultado. Como num modelo, um run que a API já não tem em memória reabre-se pelo
 * histórico (um run guardado volta com o que o agente sabia em cada snapshot).
 */
export function useAgentRun(runId: string) {
  const [live, setLive] = useState<LiveAgent>(EMPTY);
  const stream = useJobStream<ReinforcementUpdate>(`/reinforcement/${runId}/stream`, (update) => {
    setLive((previous) => {
      const last = previous.episodes.at(-1)?.step ?? 0;
      const fresh = update.steps.filter((record) => record.step > last);
      return {
        state: update.state,
        episodes: fresh.length > 0 ? [...previous.episodes, ...fresh] : previous.episodes,
        preview: update.preview ?? previous.preview,
        layout: update.layout ?? previous.layout,
      };
    });
  });
  const status = live.state?.status ?? null;
  const detached = stream.error !== null;
  const session = useQuery({
    ...reinforcementResultOptions({ path: { run_id: runId } }),
    enabled: status === "finished" || detached,
    staleTime: "static",
  });
  const fromHistory =
    session.error instanceof ApiRequestError && session.error.code === "run_not_in_memory";
  const history = useQuery({
    ...historyOpenReinforcementOptions({ path: { run_id: runId } }),
    enabled: fromHistory,
    staleTime: "static",
  });
  return {
    live,
    status,
    detached,
    result: fromHistory ? history : session,
    fromHistory,
  };
}
