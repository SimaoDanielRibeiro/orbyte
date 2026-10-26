import { createFileRoute } from "@tanstack/react-router";

import { ExploreScreen } from "@/features/explore/explore-screen";

export const Route = createFileRoute("/explorar")({
  component: ExploreScreen,
});
