import { createFileRoute } from "@tanstack/react-router";

import { ScreenPlaceholder } from "@/components/shell/screen-placeholder";
import { SCREENS } from "@/components/shell/screens";

export const Route = createFileRoute("/agentes/$searchId")({
  component: () => <ScreenPlaceholder screen={SCREENS.agents} milestone="10.7" />,
});
