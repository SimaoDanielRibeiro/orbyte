import { createFileRoute } from "@tanstack/react-router";

import { ModelsScreen } from "@/features/models/models-screen";

interface ModelsSearch {
  /** A tarefa do filtro (`todas` para ver todos); sem ela, a da preparação. */
  tarefa?: string;
}

export const Route = createFileRoute("/modelos/")({
  validateSearch: (search: Record<string, unknown>): ModelsSearch =>
    typeof search.tarefa === "string" ? { tarefa: search.tarefa } : {},
  component: ModelsScreen,
});
