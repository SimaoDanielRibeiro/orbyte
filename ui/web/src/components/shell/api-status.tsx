import { useQuery } from "@tanstack/react-query";

import { metaHealthOptions } from "@/api/generated/@tanstack/react-query.gen";
import { SidebarMenuButton } from "@/components/ui/sidebar";

/** De quanto em quanto tempo se pergunta à API se está a correr. */
const HEALTH_INTERVAL_MS = 15_000;

/**
 * Se a API está ligada, no rodapé da barra lateral.
 *
 * Com a barra recolhida, fica só o ponto (a dica diz o resto). Quando a API cai, o texto diz
 * como a voltar a arrancar.
 */
export function ApiStatus() {
  const health = useQuery({
    ...metaHealthOptions(),
    refetchInterval: HEALTH_INTERVAL_MS,
    retry: false,
  });

  const down = health.isError;
  const label = health.isPending
    ? "A ligar à API…"
    : down
      ? "API desligada"
      : `API ligada (versão ${health.data.version})`;
  const hint = down ? "Arranca-a com uv run python -m ml_lab.api" : label;

  return (
    <SidebarMenuButton
      asChild
      size="sm"
      tooltip={hint}
      className="cursor-default text-muted-foreground hover:bg-transparent hover:text-muted-foreground"
    >
      <div role="status" aria-live="polite">
        <span
          aria-hidden="true"
          className={
            down
              ? "ml-1.5 size-2 shrink-0 rounded-full bg-destructive"
              : "ml-1.5 size-2 shrink-0 rounded-full border border-muted-foreground"
          }
        />
        <span className={down ? "text-destructive" : undefined}>{label}</span>
      </div>
    </SidebarMenuButton>
  );
}
