import { useRouterState } from "@tanstack/react-router";
import { Search } from "lucide-react";
import { MotionConfig, useReducedMotion, type Transition } from "motion/react";
import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";

import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useSession } from "@/state/session";

import { AppSidebar } from "./app-sidebar";
import { CommandMenu } from "./command-menu";
import { Logo } from "./logo";
import { interfaceOf } from "./screens";

/**
 * A transição por omissão do Motion (ADR 0048): a mesma curva das entradas do CSS
 * (`--ease-out-expo`), rápida a partir e longa a assentar. Serve o que não pede outra: o nome de
 * um modelo a deslizar da lista para o título, os blocos que aparecem e desaparecem.
 */
const SETTLE: Transition = { duration: 0.55, ease: [0.16, 1, 0.3, 1] };

/** Quanto demora a barra a abrir ou a fechar: o mesmo do `duration-500` dela (`sidebar.tsx`). */
const SIDEBAR_MS = 500;

/** A curva da barra (`--ease-out-expo`), para o conteúdo deslizar ao mesmo ritmo. */
const EASE_OUT_EXPO = "cubic-bezier(0.16, 1, 0.3, 1)";

/**
 * O que todas as páginas têm, com ou sem barra lateral: as animações do Motion seguem a
 * preferência "reduzir movimento" do sistema, as dicas e os avisos (toasts).
 */
export function Providers({ children }: { children: ReactNode }) {
  return (
    <MotionConfig reducedMotion="user" transition={SETTLE}>
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
 *
 * **Recolher ou abrir a barra** (ADR 0050): o lugar dela muda de uma vez, e o conteúdo
 * reorganiza-se só uma vez, logo à largura final; depois desliza de onde estava até lá, ao
 * ritmo da barra (a técnica FLIP: medir antes, medir depois, animar a diferença com um
 * `transform`, que não volta a reorganizar nada). Com a largura animada, o conteúdo e cada
 * gráfico redesenhavam-se a cada fotograma, aos solavancos.
 */
export function AppFrame({ children }: { children: ReactNode }) {
  const open = useSession((state) => state.sidebarOpen);
  const setOpen = useSession((state) => state.setSidebarOpen);
  const path = useSession((state) => state.path);
  const choosePath = useSession((state) => state.choosePath);
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const [searching, setSearching] = useState(false);
  const reduced = useReducedMotion() ?? false;
  const inset = useRef<HTMLElement>(null);
  // Onde começava o conteúdo antes de a barra mudar (`null` fora de uma mudança).
  const before = useRef<number | null>(null);

  const only = interfaceOf(pathname);
  useEffect(() => {
    if (only !== null && only !== path) {
      choosePath(only);
    }
  }, [only, path, choosePath]);

  const changeOpen = useCallback(
    (next: boolean) => {
      before.current = inset.current?.getBoundingClientRect().left ?? null;
      setOpen(next);
    },
    [setOpen],
  );

  // Depois da mudança (já com o lugar novo da barra): o conteúdo volta, num instante, para onde
  // estava, e desliza até ao sítio novo.
  useLayoutEffect(() => {
    const element = inset.current;
    const start = before.current;
    before.current = null;
    if (element === null || start === null || reduced) {
      return;
    }
    const shift = start - element.getBoundingClientRect().left;
    if (Math.abs(shift) < 1) {
      return;
    }
    element.animate(
      [{ transform: `translateX(${String(shift)}px)` }, { transform: "translateX(0)" }],
      { duration: SIDEBAR_MS, easing: EASE_OUT_EXPO },
    );
  }, [open, reduced]);

  return (
    <SidebarProvider open={open} onOpenChange={changeOpen}>
      {/* Para quem usa o teclado: o primeiro Tab salta a barra toda. */}
      <a
        href="#content"
        className="sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-50 focus:rounded-[10px] focus:bg-surface-high focus:px-4 focus:py-2 focus:text-sm focus:text-strong focus:ring-2 focus:ring-ring focus:outline-none"
      >
        Skip to the content
      </a>
      <AppSidebar path={only ?? path} onChoosePath={choosePath} />
      <SidebarInset ref={inset}>
        <MobileHeader
          onSearch={() => {
            setSearching(true);
          }}
        />
        <div id="content" tabIndex={-1} className="@container/main flex-1 outline-none">
          {children}
        </div>
      </SidebarInset>
      <CommandMenu open={searching} onOpenChange={setSearching} />
    </SidebarProvider>
  );
}

/**
 * No telemóvel e no tablet ao alto (abaixo de 1024 px) não há barra à vista: este cabeçalho abre a
 * gaveta e a pesquisa.
 */
function MobileHeader({ onSearch }: { onSearch: () => void }) {
  return (
    <header className="sticky top-0 z-20 flex h-14 items-center gap-2 border-b bg-background/95 px-3 backdrop-blur lg:hidden">
      <SidebarTrigger />
      <Logo className="h-4 w-5" />
      <span
        translate="no"
        className="font-display text-base font-medium tracking-[-0.02em] text-strong"
      >
        Orbyte
      </span>
      <button
        type="button"
        onClick={onSearch}
        className="ml-auto flex size-9 items-center justify-center rounded-[10px] text-soft outline-none hover:bg-surface hover:text-strong focus-visible:ring-2 focus-visible:ring-ring"
      >
        <Search aria-hidden="true" className="size-[1.125rem]" />
        <span className="sr-only">Search</span>
      </button>
    </header>
  );
}
