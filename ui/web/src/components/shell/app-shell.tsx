import { useRouterState } from "@tanstack/react-router";
import { Search } from "lucide-react";
import { MotionConfig } from "motion/react";
import { useEffect, useState, type ReactNode } from "react";

import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useSession } from "@/state/session";

import { AppSidebar } from "./app-sidebar";
import { CommandMenu } from "./command-menu";
import { Logo } from "./logo";
import { interfaceOf } from "./screens";

/**
 * O que todas as páginas têm, com ou sem barra lateral: as animações do Motion seguem a
 * preferência "reduzir movimento" do sistema, as dicas e os avisos (toasts).
 */
export function Providers({ children }: { children: ReactNode }) {
  return (
    <MotionConfig reducedMotion="user">
      <TooltipProvider delayDuration={300}>
        {children}
        <Toaster position="bottom-right" />
      </TooltipProvider>
    </MotionConfig>
  );
}

/**
 * A moldura das duas interfaces: a barra lateral da interface aberta, a área do conteúdo e a
 * pesquisa.
 *
 * Abrir um ecrã que só existe numa das interfaces (o Preparar, os Agentes) passa para ela: a
 * barra mostra sempre as secções de onde se está, mesmo vindo de um endereço guardado.
 *
 * O conteúdo é um *container* (`@container/main`): as páginas reorganizam-se pela largura que
 * têm, e por isso adaptam-se sozinhas quando a barra se recolhe ou abre.
 */
export function AppFrame({ children }: { children: ReactNode }) {
  const open = useSession((state) => state.sidebarOpen);
  const setOpen = useSession((state) => state.setSidebarOpen);
  const path = useSession((state) => state.path);
  const choosePath = useSession((state) => state.choosePath);
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const [searching, setSearching] = useState(false);

  const only = interfaceOf(pathname);
  useEffect(() => {
    if (only !== null && only !== path) {
      choosePath(only);
    }
  }, [only, path, choosePath]);

  return (
    <SidebarProvider open={open} onOpenChange={setOpen}>
      {/* Para quem usa o teclado: o primeiro Tab salta a barra toda. */}
      <a
        href="#conteudo"
        className="sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-50 focus:rounded-[10px] focus:bg-surface-high focus:px-4 focus:py-2 focus:text-sm focus:text-strong focus:ring-2 focus:ring-ring focus:outline-none"
      >
        Saltar para o conteúdo
      </a>
      <AppSidebar
        path={only ?? path}
        onChoosePath={choosePath}
        onSearch={() => {
          setSearching(true);
        }}
      />
      <SidebarInset>
        <MobileHeader
          onSearch={() => {
            setSearching(true);
          }}
        />
        <div id="conteudo" tabIndex={-1} className="@container/main flex-1 outline-none">
          {children}
        </div>
      </SidebarInset>
      <CommandMenu open={searching} onOpenChange={setSearching} />
    </SidebarProvider>
  );
}

/** No telemóvel não há barra à vista: este cabeçalho abre a gaveta e a pesquisa. */
function MobileHeader({ onSearch }: { onSearch: () => void }) {
  return (
    <header className="sticky top-0 z-20 flex h-14 items-center gap-2 border-b bg-background/95 px-3 backdrop-blur md:hidden">
      <SidebarTrigger />
      <Logo className="h-4 w-5" />
      <span translate="no" className="text-base font-semibold text-strong">
        ML Lab
      </span>
      <button
        type="button"
        onClick={onSearch}
        className="ml-auto flex size-9 items-center justify-center rounded-[10px] text-soft outline-none hover:bg-surface hover:text-strong focus-visible:ring-2 focus-visible:ring-ring"
      >
        <Search aria-hidden="true" className="size-[1.125rem]" />
        <span className="sr-only">Procurar</span>
      </button>
    </header>
  );
}
