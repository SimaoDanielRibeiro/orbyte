import { CircleAlert } from "lucide-react";

import { SidebarMenu, SidebarMenuButton, SidebarMenuItem } from "@/components/ui/sidebar";
import { useApiDown } from "@/hooks/use-api-down";

/**
 * O aviso de que a API caiu, no fundo da barra lateral (recolhida, fica o ícone e a dica). Com a
 * API ligada não aparece nada: é o normal, e não precisa de texto.
 */
export function ApiStatus() {
  const down = useApiDown();
  if (!down) {
    return null;
  }
  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <SidebarMenuButton
          asChild
          size="sm"
          tooltip="API offline: start it with uv run python -m ml_lab.api"
          className="cursor-default text-negative hover:bg-transparent hover:text-negative"
        >
          <div role="alert">
            <CircleAlert aria-hidden="true" />
            <span>API offline</span>
          </div>
        </SidebarMenuButton>
      </SidebarMenuItem>
    </SidebarMenu>
  );
}
