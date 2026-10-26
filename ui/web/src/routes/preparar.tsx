import { createFileRoute } from "@tanstack/react-router";

import { PrepareScreen } from "@/features/prepare/prepare-screen";

export const Route = createFileRoute("/preparar")({
  component: PrepareScreen,
});
