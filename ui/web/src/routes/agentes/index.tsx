import { createFileRoute, redirect } from "@tanstack/react-router";

import { AgentsScreen } from "@/features/agents/agents-screen";
import { useSession } from "@/state/session";

export const Route = createFileRoute("/agentes/")({
  // Com uma procura aberta, os Agentes são essa procura: sai-se para o Treinar e volta-se a ela
  // tal como estava. O pedido só aparece depois de "Nova procura" (que a esquece).
  beforeLoad: () => {
    const { searchId } = useSession.getState();
    if (searchId !== null) {
      // O redirect do router lança-se, como um erro (é assim que ele o apanha).
      // eslint-disable-next-line @typescript-eslint/only-throw-error
      throw redirect({ to: "/agentes/$searchId", params: { searchId }, replace: true });
    }
  },
  component: AgentsScreen,
});
