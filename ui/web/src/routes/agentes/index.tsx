import { createFileRoute } from "@tanstack/react-router";

import { AgentsScreen } from "@/features/agents/agents-screen";

export const Route = createFileRoute("/agentes/")({
  component: AgentsScreen,
});
