import { createFileRoute } from "@tanstack/react-router";

import { SearchScreen } from "@/features/agents/search-screen";

export const Route = createFileRoute("/agentes/$searchId")({
  component: SearchScreen,
});
