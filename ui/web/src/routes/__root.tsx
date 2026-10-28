import type { QueryClient } from "@tanstack/react-query";
import { createRootRouteWithContext, Outlet, useRouterState } from "@tanstack/react-router";

import { topicsQuery } from "@/api/topics";
import { vocabularyQuery } from "@/api/vocabulary";
import { AppFrame, Providers } from "@/components/shell/app-shell";
import { NotFound, RootError } from "@/components/shell/root-error";

export interface RouterContext {
  queryClient: QueryClient;
}

export const Route = createRootRouteWithContext<RouterContext>()({
  // O vocabulário e os temas chegam antes de qualquer ecrã: assim nenhum enum aparece pelo
  // valor cru e nenhum gráfico fica sem a explicação de como se lê.
  loader: ({ context }) =>
    Promise.all([
      context.queryClient.query(vocabularyQuery),
      context.queryClient.query(topicsQuery),
    ]),
  component: RootLayout,
  errorComponent: RootError,
  notFoundComponent: NotFound,
});

/** A landing (o início) ocupa o ecrã todo; as outras páginas têm a moldura com a barra lateral. */
function RootLayout() {
  const landing = useRouterState({ select: (state) => state.location.pathname === "/" });
  return (
    <Providers>
      {landing ? (
        <Outlet />
      ) : (
        <AppFrame>
          <Outlet />
        </AppFrame>
      )}
    </Providers>
  );
}
