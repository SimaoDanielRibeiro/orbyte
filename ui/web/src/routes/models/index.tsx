import { createFileRoute } from "@tanstack/react-router";

import { ModelsScreen } from "@/features/models/models-screen";

interface ModelsSearch {
  /** A tarefa do filtro (`all` para ver todos); sem ela, a da preparação. */
  task?: string;
}

export const Route = createFileRoute("/models/")({
  validateSearch: (search: Record<string, unknown>): ModelsSearch =>
    typeof search.task === "string" ? { task: search.task } : {},
  component: ModelsScreen,
});
