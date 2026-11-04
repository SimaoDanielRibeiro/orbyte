import { createFileRoute } from "@tanstack/react-router";

import { PrepareScreen } from "@/features/prepare/prepare-screen";

export const Route = createFileRoute("/prepare")({
  component: PrepareScreen,
});
