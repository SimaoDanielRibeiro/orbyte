import { createFileRoute } from "@tanstack/react-router";

import { SetupScreen } from "@/features/reinforcement/setup-screen";

export const Route = createFileRoute("/rl/")({
  component: SetupScreen,
});
