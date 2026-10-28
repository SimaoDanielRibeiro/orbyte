import { createFileRoute } from "@tanstack/react-router";

import { ExplainScreen } from "@/features/explain/explain-screen";

export const Route = createFileRoute("/explicar/$runId")({
  component: ExplainScreen,
});
