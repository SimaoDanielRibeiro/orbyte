/**
 * Uma pergunta ao analista, a decorrer: os passos que vão chegando pelo WebSocket e o que ele está
 * a pensar e a escrever agora. Quando acaba, pede-se a resposta inteira (com a leitura do backend).
 *
 * O componente que o usa tem a chave da pergunta (`key={questionId}`): uma pergunta nova começa
 * com o estado vazio.
 */
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import type { AnalystStep, AnalystUpdate, JobState, LiveTurn } from "@/api/generated";
import { automlQuestionResultOptions } from "@/api/generated/@tanstack/react-query.gen";
import { useJobStream } from "@/api/stream";

import { merge } from "./use-agent-search";

/** A pergunta tal como chega pelo WebSocket. */
export interface LiveQuestion {
  state: JobState | null;
  /** Os passos até agora, sem repetições, pela ordem. */
  trace: AnalystStep[];
  /** O que o modelo de linguagem está a pensar e a escrever agora (`null` entre vezes). */
  live: LiveTurn | null;
}

const EMPTY: LiveQuestion = { state: null, trace: [], live: null };

/** Acompanha a pergunta `questionId` até à resposta. */
export function useQuestion(questionId: string) {
  const [live, setLive] = useState<LiveQuestion>(EMPTY);
  const stream = useJobStream<AnalystUpdate>(`/automl/questions/${questionId}/stream`, (update) => {
    setLive((previous) => ({
      state: update.state,
      trace: merge(previous.trace, update.trace),
      live: update.live,
    }));
  });
  const status = live.state?.status ?? null;
  const detached = stream.error !== null;
  const result = useQuery({
    ...automlQuestionResultOptions({ path: { question_id: questionId } }),
    enabled: status === "finished" || detached,
    staleTime: "static",
  });
  return { live, status, detached, result };
}
