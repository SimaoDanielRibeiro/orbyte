import { createFileRoute } from "@tanstack/react-router";

import { ScreenPlaceholder } from "@/components/shell/screen-placeholder";
import { SCREENS } from "@/components/shell/screens";

export const Route = createFileRoute("/rl")({
  component: () => <ScreenPlaceholder screen={SCREENS.reinforcement} milestone="10.8" />,
});
