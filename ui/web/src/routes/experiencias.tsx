import { createFileRoute } from "@tanstack/react-router";

import { ScreenPlaceholder } from "@/components/shell/screen-placeholder";
import { SCREENS } from "@/components/shell/screens";

export const Route = createFileRoute("/experiencias")({
  component: () => <ScreenPlaceholder screen={SCREENS.history} milestone="10.8" />,
});
