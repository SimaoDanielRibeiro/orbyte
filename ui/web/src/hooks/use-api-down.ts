import { useQuery } from "@tanstack/react-query";

import { metaHealthOptions } from "@/api/generated/@tanstack/react-query.gen";

/** De quanto em quanto tempo se pergunta à API se está a correr. */
const HEALTH_INTERVAL_MS = 15_000;

/**
 * Se a API caiu: a mesma pergunta para a barra lateral e para a entrada.
 *
 * Só a falha conta. Ligada é o normal, e escrevê-lo em todas as páginas era texto a mais; quando
 * cai, nenhum ecrã funciona, e por isso diz-se logo, com o que fazer (ADR 0048).
 */
export function useApiDown(): boolean {
  const health = useQuery({
    ...metaHealthOptions(),
    refetchInterval: HEALTH_INTERVAL_MS,
    retry: false,
  });
  return health.isError;
}
