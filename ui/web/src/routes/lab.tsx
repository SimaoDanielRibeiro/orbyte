import { createFileRoute } from "@tanstack/react-router";

import { LaboratoryScreen } from "@/features/laboratory/laboratory-screen";

interface LaboratorySearch {
  /** A demonstração aberta (por omissão, a primeira). */
  demo?: string;
}

export const Route = createFileRoute("/lab")({
  validateSearch: (search: Record<string, unknown>): LaboratorySearch =>
    typeof search.demo === "string" ? { demo: search.demo } : {},
  component: LaboratoryScreen,
});
