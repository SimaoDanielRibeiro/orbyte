import { createFileRoute } from "@tanstack/react-router";

import { ScreenPlaceholder } from "@/components/shell/screen-placeholder";
import { SCREENS } from "@/components/shell/screens";

export const Route = createFileRoute("/modelos/")({
  component: () => <ScreenPlaceholder screen={SCREENS.models} milestone="10.5" />,
});
