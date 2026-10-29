/**
 * A biblioteca dos pacotes muda quando se guarda, carrega, abre ou apaga um pacote: a lista, os
 * runs que se podem guardar e o histórico (guardar e carregar põem lá o run) voltam a pedir-se.
 */
import type { QueryClient } from "@tanstack/react-query";

import { API_BASE } from "@/api/client-config";
import { invalidateHistory } from "@/features/history/cache";

/** As consultas que mudam quando a biblioteca muda, pelo `_id` que o cliente gerado lhes dá. */
const PACKAGE_QUERIES = new Set(["packagesSaved", "packagesCandidates", "packagesContents"]);

/** Volta a pedir a biblioteca, os runs que se podem guardar e o histórico. */
export function invalidatePackages(queryClient: QueryClient): void {
  void queryClient.invalidateQueries({
    predicate: (query) => {
      const [first] = query.queryKey;
      return (
        typeof first === "object" &&
        first !== null &&
        "_id" in first &&
        PACKAGE_QUERIES.has(String(first._id))
      );
    },
  });
  invalidateHistory(queryClient);
}

/** O endereço do ficheiro de um pacote (descarrega-se com uma ligação normal). */
export function packageFileUrl(packageId: string): string {
  return `${API_BASE}/packages/${packageId}/file`;
}
