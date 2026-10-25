import { createFileRoute } from "@tanstack/react-router";

import { ScreenPlaceholder } from "@/components/shell/screen-placeholder";
import { SCREENS } from "@/components/shell/screens";

export const Route = createFileRoute("/treinar/$runId")({
  component: () => <ScreenPlaceholder screen={SCREENS.train} milestone="10.6" />,
});
