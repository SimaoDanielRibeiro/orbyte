import { createFileRoute } from "@tanstack/react-router";

import { ScreenPlaceholder } from "@/components/shell/screen-placeholder";
import { SCREENS } from "@/components/shell/screens";

export const Route = createFileRoute("/laboratorio")({
  component: () => <ScreenPlaceholder screen={SCREENS.laboratory} milestone="10.8" />,
});
