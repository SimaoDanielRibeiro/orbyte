import { createFileRoute, redirect } from "@tanstack/react-router";

import { NeedsStep } from "@/components/shell/blocks";
import { Page, PageHeader } from "@/components/shell/page";
import { SCREENS } from "@/components/shell/screens";
import { useSession } from "@/state/session";

export const Route = createFileRoute("/train/")({
  // Com um run aberto, o ecrã é o dele; sem nenhum, diz onde se começa um.
  beforeLoad: () => {
    const { runId } = useSession.getState();
    if (runId !== null) {
      // O redirect do router lança-se, como um erro (é assim que ele o apanha).
      // eslint-disable-next-line @typescript-eslint/only-throw-error
      throw redirect({ to: "/train/$runId", params: { runId } });
    }
  },
  component: NoRun,
});

function NoRun() {
  return (
    <Page>
      <PageHeader title={SCREENS.train.title} description={SCREENS.train.description} />
      <NeedsStep
        message="You haven't trained a model yet. Choose a model, tune its hyperparameters and press “Train with these values”."
        to="/models"
        action="Choose the model"
      />
    </Page>
  );
}
