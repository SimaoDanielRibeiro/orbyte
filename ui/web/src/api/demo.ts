/**
 * A demonstração de um modelo (`POST /catalog/models/{name}/demo`): o modelo a treinar num
 * conjunto de exemplo pequeno, com o campo de decisão passo a passo.
 *
 * É um POST, mas não muda nada (não é um run, não vai para o histórico) e a mesma pergunta dá
 * sempre a mesma resposta: por isso pede-se como uma consulta, guardada pelos hiperparâmetros e
 * pelo exemplo. A API também guarda as últimas, e voltar a pedir uma não volta a treinar.
 */
import { queryOptions } from "@tanstack/react-query";

import { catalogDemo } from "./generated/sdk.gen";

export function demoQuery(
  name: string,
  hyperparameters: Readonly<Record<string, unknown>> = {},
  example: string | null = null,
) {
  return queryOptions({
    queryKey: ["catalog-demo", name, hyperparameters, example],
    queryFn: async ({ signal }) => {
      const { data } = await catalogDemo({
        path: { name },
        body: { hyperparameters: { ...hyperparameters }, example },
        signal,
        throwOnError: true,
      });
      return data;
    },
    staleTime: "static",
  });
}
