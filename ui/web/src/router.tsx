import { createRouter } from "@tanstack/react-router";

import { queryClient } from "@/api/query";

import { routeTree } from "./routeTree.gen";

/**
 * O router da interface. As rotas são os ficheiros de `src/routes/` (o plugin do Vite gera a
 * árvore em `routeTree.gen.ts`), e cada uma recebe a cache da API no contexto, para pedir os
 * dados antes de se mostrar.
 */
export const router = createRouter({
  routeTree,
  context: { queryClient },
  defaultPreload: "intent",
  // Quem decide se os dados estão frescos é o TanStack Query, não o router.
  defaultPreloadStaleTime: 0,
  scrollRestoration: true,
});

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
