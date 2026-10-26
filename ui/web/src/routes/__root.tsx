import type { QueryClient } from "@tanstack/react-query";
import { createRootRouteWithContext, Outlet } from "@tanstack/react-router";

import { topicsQuery } from "@/api/topics";
import { vocabularyQuery } from "@/api/vocabulary";
import { AppShell } from "@/components/shell/app-shell";
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

function RootLayout() {
  return (
    <AppShell>
      <Outlet />
    </AppShell>
  );
}
