import { createFileRoute, redirect } from "@tanstack/react-router";

import { SaveScreen } from "@/features/save/save-screen";
import { useSession } from "@/state/session";

export const Route = createFileRoute("/guardar/")({
  // Com um run aberto, guarda-se o dele; sem nenhum, o ecrã abre na mesma (pode carregar-se um
  // modelo já treinado).
  beforeLoad: () => {
    const { runId } = useSession.getState();
    if (runId !== null) {
      // O redirect do router lança-se, como um erro (é assim que ele o apanha).
      // eslint-disable-next-line @typescript-eslint/only-throw-error
      throw redirect({ to: "/guardar/$runId", params: { runId } });
    }
  },
  component: () => <SaveScreen runId={null} />,
});
