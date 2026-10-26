/**
 * Os temas didáticos: os do painel "Porquê?" e o "como ler" de cada gráfico, escritos uma só
 * vez no backend (regra 5 do `CONTRIBUTING.md`).
 *
 * Pedem-se todos de uma vez, ao arrancar (o carregamento da rota raiz espera por eles, como pelo
 * vocabulário), e ficam na cache para sempre: só mudam com o backend. Os nomes que a interface
 * pede estão guardados por um teste do lado do Python (`tests/contract/test_topics_in_use.py`),
 * que falha se um deles deixar de existir.
 */
import { queryOptions, useSuspenseQuery } from "@tanstack/react-query";

import type { Explanation } from "./generated";
import { explanationsTopicsOptions } from "./generated/@tanstack/react-query.gen";

export const topicsQuery = queryOptions({
  ...explanationsTopicsOptions(),
  staleTime: "static",
  gcTime: Infinity,
});

/** Todos os temas, pelo nome. */
export function useTopics(): Readonly<Record<string, Explanation>> {
  return useSuspenseQuery(topicsQuery).data;
}

/** Um tema (`"split"`, `"correlation_matrix"`…). */
export function useTopic(name: string): Explanation | undefined {
  return useTopics()[name];
}
