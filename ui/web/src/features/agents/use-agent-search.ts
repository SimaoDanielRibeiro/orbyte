import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import type {
  AgentMessage,
  AutoMlUpdate,
  Candidate,
  JobState,
  ObjectiveOption,
  SearchPhase,
  Thought,
  WorkInProgress,
} from "@/api/generated";
import {
  automlResultOptions,
  automlSearchPlanOptions,
} from "@/api/generated/@tanstack/react-query.gen";
import { useJobStream } from "@/api/stream";

/** A procura a decorrer, tal como chega pelo WebSocket. */
export interface LiveSearch {
  state: JobState | null;
  phase: SearchPhase | null;
  objective: ObjectiveOption | null;
  /** Os candidatos medidos até agora, sem repetições, pela ordem. */
  candidates: Candidate[];
  /** A conversa dos agentes até agora, sem repetições, pela ordem. */
  messages: AgentMessage[];
  leader: Candidate | null;
  /** O candidato que um agente está a medir agora (`null` entre candidatos). */
  working: WorkInProgress | null;
  /** O que o agente de IA está a pensar agora, até aqui (`null` quando não pensa). */
  thought: Thought | null;
}

const EMPTY: LiveSearch = {
  state: null,
  phase: null,
  objective: null,
  candidates: [],
  messages: [],
  leader: null,
  working: null,
  thought: null,
};

/**
 * Junta o que chegou ao que já se tinha (o WebSocket recomeça do princípio a cada ligação). Serve
 * as mensagens e os candidatos da procura, e os passos do analista (`use-question.ts`).
 */
export function merge<T extends { number: number }>(known: T[], fresh: readonly T[]): T[] {
  const last = known.at(-1)?.number ?? 0;
  const added = fresh.filter((item) => item.number > last);
  return added.length > 0 ? [...known, ...added] : known;
}

/**
 * Uma procura dos agentes: o plano (para desenhar os agentes), a procura ao vivo e, no fim, o
 * resultado. Uma procura que já não está a decorrer nesta sessão da API não tem transmissão:
 * pede-se logo o resultado.
 */
export function useAgentSearch(searchId: string) {
  const [live, setLive] = useState<LiveSearch>(EMPTY);
  const stream = useJobStream<AutoMlUpdate>(`/automl/${searchId}/stream`, (update) => {
    setLive((previous) => ({
      state: update.state,
      phase: update.phase,
      objective: update.objective,
      candidates: merge(previous.candidates, update.candidates),
      messages: merge(previous.messages, update.messages),
      leader: update.leader ?? previous.leader,
      working: update.working,
      thought: update.thought,
    }));
  });
  const plan = useQuery({
    ...automlSearchPlanOptions({ path: { search_id: searchId } }),
    staleTime: "static",
  });
  const status = live.state?.status ?? null;
  const detached = stream.error !== null;
  const result = useQuery({
    ...automlResultOptions({ path: { search_id: searchId } }),
    enabled: status === "finished" || detached,
    staleTime: "static",
  });
  return { live, status, detached, plan, result };
}
