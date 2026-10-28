import { createFileRoute } from "@tanstack/react-router";

import { AgentRunScreen } from "@/features/reinforcement/agent-run-screen";

export const Route = createFileRoute("/rl/$runId")({
  component: AgentRunScreen,
});
