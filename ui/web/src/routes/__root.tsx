import type { QueryClient } from "@tanstack/react-query";
import { createRootRouteWithContext, Outlet } from "@tanstack/react-router";

import { vocabularyQuery } from "@/api/vocabulary";
import { AppShell } from "@/components/shell/app-shell";
import { NotFound, RootError } from "@/components/shell/root-error";

export interface RouterContext {
  queryClient: QueryClient;
}

export const Route = createRootRouteWithContext<RouterContext>()({
  // O vocabulário chega antes de qualquer ecrã: assim nenhum enum aparece pelo valor cru.
  loader: ({ context }) => context.queryClient.query(vocabularyQuery),
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
