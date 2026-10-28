import { createFileRoute } from "@tanstack/react-router";

import { MetricsScreen } from "@/features/metrics/metrics-screen";

interface MetricsSearch {
  /** A tarefa do filtro; sem ela, todas as métricas. */
  tarefa?: string;
}

export const Route = createFileRoute("/metricas")({
  validateSearch: (search: Record<string, unknown>): MetricsSearch =>
    typeof search.tarefa === "string" ? { tarefa: search.tarefa } : {},
  component: MetricsScreen,
});
