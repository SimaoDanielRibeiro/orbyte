import { createFileRoute } from "@tanstack/react-router";

import { ScreenPlaceholder } from "@/components/shell/screen-placeholder";
import { SCREENS } from "@/components/shell/screens";

export const Route = createFileRoute("/explorar")({
  component: () => <ScreenPlaceholder screen={SCREENS.explore} milestone="10.4" />,
});
