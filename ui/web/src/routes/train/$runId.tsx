import { createFileRoute } from "@tanstack/react-router";

import { TrainScreen } from "@/features/training/train-screen";

export const Route = createFileRoute("/train/$runId")({
  component: TrainScreen,
});
