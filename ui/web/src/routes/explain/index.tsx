import { createFileRoute, redirect } from "@tanstack/react-router";

import { NeedsStep } from "@/components/shell/blocks";
import { Page, PageHeader } from "@/components/shell/page";
import { SCREENS } from "@/components/shell/screens";
import { useSession } from "@/state/session";

export const Route = createFileRoute("/explain/")({
  // Com um run aberto, explica-se o dele; sem nenhum, diz onde se treina um.
  beforeLoad: () => {
    const { runId } = useSession.getState();
    if (runId !== null) {
      // O redirect do router lança-se, como um erro (é assim que ele o apanha).
      // eslint-disable-next-line @typescript-eslint/only-throw-error
      throw redirect({ to: "/explain/$runId", params: { runId } });
    }
  },
  component: NoRun,
});

function NoRun() {
  return (
    <Page>
      <PageHeader title={SCREENS.explain.title} description={SCREENS.explain.description} />
      <NeedsStep
        message="To explain a model's predictions, you need to train it first. Choose a model and press “Train with these values”."
        to="/models"
        action="Choose the model"
      />
    </Page>
  );
}
