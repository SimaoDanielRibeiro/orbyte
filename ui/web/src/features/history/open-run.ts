import { useNavigate } from "@tanstack/react-router";
import { useCallback } from "react";

import type { TaskType } from "@/api/generated";
import { useVocabulary } from "@/api/vocabulary";
import { useSession } from "@/state/session";

/**
 * Abrir um run no ecrã que é o dele: o de um modelo em Treinar e avaliar, o de um agente de
 * reinforcement learning no ecrã do RL. Quem decide é a tarefa (o vocabulário diz se ela aprende
 * com um dataset ou num ambiente).
 */
export function useOpenRun(): (runId: string, taskType: TaskType, title?: string) => void {
  const navigate = useNavigate();
  const vocabulary = useVocabulary();
  const openRun = useSession((state) => state.openRun);
  const openAgentRun = useSession((state) => state.openAgentRun);
  return useCallback(
    (runId, taskType, title) => {
      const task = vocabulary.tasks.find((one) => one.task_type === taskType);
      if (task?.learns_from_data === false) {
        openAgentRun(runId);
        void navigate({ to: "/rl/$runId", params: { runId } });
        return;
      }
      openRun(runId, title);
      void navigate({ to: "/train/$runId", params: { runId } });
    },
    [navigate, vocabulary, openRun, openAgentRun],
  );
}
