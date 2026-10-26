import { MotionConfig } from "motion/react";
import type { ReactNode } from "react";

import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useSession } from "@/state/session";

import { AppSidebar } from "./app-sidebar";
import { Logo } from "./logo";

/**
 * A moldura de todos os ecrãs: a barra lateral e a área do conteúdo.
 *
 * O conteúdo é um *container* (`@container/main`): as páginas reorganizam-se pela largura que
 * têm, e por isso adaptam-se sozinhas quando a barra se recolhe ou abre.
 *
 * As animações do Motion seguem a preferência "reduzir movimento" do sistema: com ela ligada,
 * as mudanças de posição acontecem sem animação.
 */
export function AppShell({ children }: { children: ReactNode }) {
  const open = useSession((state) => state.sidebarOpen);
  const setOpen = useSession((state) => state.setSidebarOpen);

  return (
    <MotionConfig reducedMotion="user">
      <TooltipProvider delayDuration={300}>
        <SidebarProvider open={open} onOpenChange={setOpen}>
          <AppSidebar />
          <SidebarInset>
            <MobileHeader />
            <div className="@container/main flex-1">{children}</div>
          </SidebarInset>
        </SidebarProvider>
        <Toaster position="bottom-right" />
      </TooltipProvider>
    </MotionConfig>
  );
}

/** No telemóvel não há barra à vista: este cabeçalho abre a gaveta. */
function MobileHeader() {
  return (
    <header className="sticky top-0 z-20 flex h-12 items-center gap-2 border-b bg-background px-3 md:hidden">
      <SidebarTrigger />
      <Logo className="size-6" />
      <span className="font-serif text-base">ML Lab</span>
    </header>
  );
}
