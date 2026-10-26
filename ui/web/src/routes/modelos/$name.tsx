import { createFileRoute } from "@tanstack/react-router";

import { catalogModelOptions } from "@/api/generated/@tanstack/react-query.gen";
import { ModelScreen } from "@/features/models/model-screen";

export const Route = createFileRoute("/modelos/$name")({
  // O cartão chega antes de a página se mostrar: assim o nome que vem da lista tem logo onde
  // aterrar (a transição do título precisa dos dois no mesmo instante).
  loader: ({ context, params }) =>
    context.queryClient.query({
      ...catalogModelOptions({ path: { name: params.name } }),
      staleTime: "static",
    }),
  component: ModelScreen,
});
