import { createFileRoute } from "@tanstack/react-router";

import { MetricsScreen } from "@/features/metrics/metrics-screen";

interface MetricsSearch {
  /** A tarefa do filtro; sem ela, todas as métricas. */
  task?: string;
}

export const Route = createFileRoute("/metrics")({
  validateSearch: (search: Record<string, unknown>): MetricsSearch =>
    typeof search.task === "string" ? { task: search.task } : {},
  component: MetricsScreen,
});
