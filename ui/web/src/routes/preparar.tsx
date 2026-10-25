import { createFileRoute } from "@tanstack/react-router";

import { ScreenPlaceholder } from "@/components/shell/screen-placeholder";
import { SCREENS } from "@/components/shell/screens";

export const Route = createFileRoute("/preparar")({
  component: () => <ScreenPlaceholder screen={SCREENS.prepare} milestone="10.4" />,
});
