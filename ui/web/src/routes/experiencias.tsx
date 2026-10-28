import { createFileRoute } from "@tanstack/react-router";

import { HistoryScreen } from "@/features/history/history-screen";

interface HistorySearch {
  /** Os filtros do histórico: a tarefa, o dataset (ou ambiente) e o modelo (ou agente). */
  tarefa?: string;
  dataset?: string;
  modelo?: string;
}

export const Route = createFileRoute("/experiencias")({
  validateSearch: (search: Record<string, unknown>): HistorySearch => {
    const kept: HistorySearch = {};
    for (const key of ["tarefa", "dataset", "modelo"] as const) {
      const value = search[key];
      if (typeof value === "string" && value !== "") {
        kept[key] = value;
      }
    }
    return kept;
  },
  component: HistoryScreen,
});
