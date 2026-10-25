import { createFileRoute } from "@tanstack/react-router";

import { ScreenPlaceholder } from "@/components/shell/screen-placeholder";
import { SCREENS } from "@/components/shell/screens";

export const Route = createFileRoute("/dados")({
  component: () => <ScreenPlaceholder screen={SCREENS.data} milestone="10.4" />,
});
