import { QueryClient } from "@tanstack/react-query";

import { ApiRequestError } from "./errors";

/**
 * Repete um pedido falhado só quando pode valer a pena: um erro do servidor ou uma falha de
 * rede. Um 404 ou um 422 voltaria a falhar da mesma maneira.
 */
function shouldRetry(failures: number, error: Error): boolean {
  if (error instanceof ApiRequestError && error.status > 0 && error.status < 500) {
    return false;
  }
  return failures < 2;
}

/** A cache de tudo o que vem da API. A API corre no mesmo computador: as respostas são rápidas. */
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: shouldRetry,
      refetchOnWindowFocus: false,
    },
  },
});
