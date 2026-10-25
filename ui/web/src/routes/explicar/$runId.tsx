import { createFileRoute } from "@tanstack/react-router";

import { ScreenPlaceholder } from "@/components/shell/screen-placeholder";
import { SCREENS } from "@/components/shell/screens";

export const Route = createFileRoute("/explicar/$runId")({
  component: () => <ScreenPlaceholder screen={SCREENS.explain} milestone="10.8" />,
});
