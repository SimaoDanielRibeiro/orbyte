import { useQueryClient } from "@tanstack/react-query";
import { useCallback } from "react";
import { toast } from "sonner";

import type { DatasetSummary } from "@/api/generated";
import {
  datasetsLoadedQueryKey,
  datasetsSummaryQueryKey,
} from "@/api/generated/@tanstack/react-query.gen";
import { formatQuantity } from "@/lib/format";
import { useSession } from "@/state/session";

/**
 * O que acontece quando um dataset chega (de um exemplo, de um ficheiro ou de um gerador): fica
 * na cache com o resumo que a API devolveu, passa a ser o aberto e a lista dos carregados
 * atualiza-se.
 */
export function useDatasetLoaded(): (dataset: DatasetSummary) => void {
  const queryClient = useQueryClient();
  const openDataset = useSession((state) => state.openDataset);
  return useCallback(
    (dataset: DatasetSummary) => {
      queryClient.setQueryData(
        datasetsSummaryQueryKey({ path: { dataset_id: dataset.id } }),
        dataset,
      );
      void queryClient.invalidateQueries({ queryKey: datasetsLoadedQueryKey() });
      openDataset(dataset.id);
      toast.success(
        `${dataset.name} aberto: ${formatQuantity(dataset.n_rows, "linha", "linhas")} e ${formatQuantity(dataset.n_columns, "coluna", "colunas")}.`,
      );
    },
    [queryClient, openDataset],
  );
}
