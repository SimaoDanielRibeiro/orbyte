import { useQuery } from "@tanstack/react-query";

import { ApiRequestError } from "@/api/errors";
import { datasetsSummaryOptions } from "@/api/generated/@tanstack/react-query.gen";
import { useSession } from "@/state/session";

/**
 * O dataset aberto na sessão, se houver.
 *
 * `lost` diz que a API já não o tem: os datasets carregados vivem na memória dela e perdem-se
 * quando ela recomeça, mas o id fica guardado no navegador.
 */
export function useOpenDataset() {
  const datasetId = useSession((state) => state.datasetId);
  const query = useQuery({
    ...datasetsSummaryOptions({ path: { dataset_id: datasetId ?? "" } }),
    enabled: datasetId !== null,
  });
  const lost = query.error instanceof ApiRequestError && query.error.status === 404;
  return { datasetId, query, lost };
}
