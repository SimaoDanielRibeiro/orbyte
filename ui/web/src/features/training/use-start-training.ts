import { useMutation, useQuery } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";

import {
  datasetsSummaryOptions,
  experimentsStartMutation,
} from "@/api/generated/@tanstack/react-query.gen";
import type { Values } from "@/components/forms/schema";
import { useSession } from "@/state/session";

/**
 * Começar um treino com a preparação da sessão: os mesmos dados, a mesma preparação e a mesma
 * seed (a da divisão), com o modelo e os hiperparâmetros escolhidos. O treino corre em segundo
 * plano na API; daqui segue-se para o ecrã Treinar e avaliar, que o acompanha.
 */
export function useStartTraining() {
  const preparation = useSession((state) => state.preparation);
  const openRun = useSession((state) => state.openRun);
  const navigate = useNavigate();
  const dataset = useQuery({
    ...datasetsSummaryOptions({ path: { dataset_id: preparation?.datasetId ?? "" } }),
    enabled: preparation !== null,
  });
  const mutation = useMutation(experimentsStartMutation());

  /** Começa o treino de `model` (com o nome a mostrar `title`) e segue para o acompanhar. */
  function start(model: string, hyperparameters: Values, title: string) {
    if (preparation === null || dataset.data === undefined) {
      return;
    }
    mutation.mutate(
      {
        body: {
          dataset: dataset.data.source,
          preparation: preparation.config,
          model: { name: model, hyperparameters },
          seed: preparation.seed,
        },
      },
      {
        onSuccess: (job) => {
          openRun(job.run_id, title);
          void navigate({ to: "/treinar/$runId", params: { runId: job.run_id } });
        },
      },
    );
  }

  return {
    preparation,
    datasetName: dataset.data?.name ?? null,
    /** Se há uma preparação (e o dataset dela ainda está na API). */
    ready: preparation !== null && dataset.data !== undefined,
    start,
    pending: mutation.isPending,
    error: mutation.error ?? dataset.error,
  };
}
