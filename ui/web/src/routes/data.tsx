import { createFileRoute } from "@tanstack/react-router";

import { DataScreen } from "@/features/data/data-screen";

export const Route = createFileRoute("/data")({
  component: DataScreen,
});
