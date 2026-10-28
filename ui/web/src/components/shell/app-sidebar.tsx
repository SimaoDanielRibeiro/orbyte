import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate, useRouterState } from "@tanstack/react-router";
import { cn } from "cn";
import { ArrowLeftRight, PanelLeft, Search } from "lucide-react";
import { ToggleGroup } from "radix-ui";

import {
  catalogMetricsOptions,
  catalogModelsOptions,
  historyRunsOptions,
} from "@/api/generated/@tanstack/react-query.gen";
import { Kbd } from "@/components/ui/kbd";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
  SidebarSeparator,
  useSidebar,
} from "@/components/ui/sidebar";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { formatCount } from "@/lib/format";
import { shortcut } from "@/lib/keys";
import type { WorkPath } from "@/state/session";

import { ApiStatus } from "./api-status";
import { Logo } from "./logo";
import {
  belongsTo,
  INTERFACE_NAMES,
  isCurrent,
  NAVIGATION,
  SCREENS,
  type Screen,
  type ScreenPath,
} from "./screens";

interface AppSidebarProps {
  /** A interface aberta: a barra mostra as secções dela. */
  path: WorkPath;
  onChoosePath: (path: WorkPath) => void;
  onSearch: () => void;
}

/**
 * A barra lateral, à maneira da referência: a marca e o botão de a recolher, a pesquisa, a troca
 * entre as duas interfaces e as secções da interface aberta, cada uma com o seu rótulo pequeno.
 * Recolhida, fica só com os ícones, e o nome de cada ecrã aparece numa dica.
 */
export function AppSidebar({ path, onChoosePath, onSearch }: AppSidebarProps) {
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const counts = useCounts();

  return (
    <Sidebar collapsible="icon">
      {/* Toda a barra é a navegação (com o estado da API e o botão de a recolher). */}
      <nav aria-label="Ecrãs" className="flex size-full min-h-0 flex-col">
        <SidebarHeader className="gap-4 px-3 pt-4 pb-3">
          <div className="flex items-center justify-between gap-2 group-data-[collapsible=icon]:flex-col">
            <Link
              to="/"
              className="flex min-w-0 items-center gap-2.5 rounded-lg px-2 py-1 outline-none group-data-[collapsible=icon]:px-1 focus-visible:ring-2 focus-visible:ring-sidebar-ring"
              aria-label="ML Lab, voltar ao início"
            >
              <Logo className="h-5 w-6 shrink-0" />
              <span
                translate="no"
                className="truncate text-[1.0625rem] font-semibold text-strong group-data-[collapsible=icon]:hidden"
              >
                ML Lab
              </span>
            </Link>
            <CollapseButton />
          </div>
          <SearchButton onSearch={onSearch} />
          <InterfaceSwitch path={path} pathname={pathname} onChoosePath={onChoosePath} />
        </SidebarHeader>

        <SidebarSeparator className="mx-0" />

        <SidebarContent className="gap-1 px-1 py-3">
          {NAVIGATION[path].map((group) => (
            <SidebarGroup key={group.label} className="py-1">
              <SidebarGroupLabel>{group.label}</SidebarGroupLabel>
              <SidebarMenu className="gap-0.5">
                {group.screens.map((screen) => (
                  <NavItem
                    key={screen.to}
                    screen={screen}
                    pathname={pathname}
                    count={counts[screen.to]}
                  />
                ))}
              </SidebarMenu>
            </SidebarGroup>
          ))}
        </SidebarContent>

        <SidebarSeparator className="mx-0" />
        <SidebarFooter className="px-3 py-3">
          <SidebarMenu>
            <SidebarMenuItem>
              <ApiStatus />
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarFooter>
        <SidebarRail />
      </nav>
    </Sidebar>
  );
}

/** As contagens à direita de alguns itens, como na referência: modelos, métricas e runs. */
function useCounts(): Partial<Record<ScreenPath, number>> {
  const models = useQuery({ ...catalogModelsOptions(), staleTime: "static" });
  const metrics = useQuery({ ...catalogMetricsOptions(), staleTime: "static" });
  const runs = useQuery({ ...historyRunsOptions(), staleTime: 30_000 });
  return {
    "/modelos": models.data?.length,
    "/metricas": metrics.data?.length,
    "/experiencias": runs.data?.length,
  };
}

interface NavItemProps {
  screen: Screen;
  pathname: string;
  count?: number;
}

function NavItem({ screen, pathname, count }: NavItemProps) {
  const { isMobile, setOpenMobile } = useSidebar();
  const Icon = screen.icon;
  return (
    <SidebarMenuItem>
      <SidebarMenuButton asChild isActive={isCurrent(screen.to, pathname)} tooltip={screen.title}>
        <Link
          to={screen.to}
          onClick={() => {
            // No telemóvel, a gaveta fecha-se para mostrar o ecrã escolhido.
            if (isMobile) {
              setOpenMobile(false);
            }
          }}
        >
          <Icon aria-hidden="true" />
          <span>{screen.title}</span>
        </Link>
      </SidebarMenuButton>
      {count !== undefined && (
        <SidebarMenuBadge className="font-mono text-xs font-normal text-dim tabular-nums">
          {formatCount(count)}
        </SidebarMenuBadge>
      )}
    </SidebarMenuItem>
  );
}

/** A pesquisa: parece um campo, como na referência, mas abre a janela da pesquisa. */
function SearchButton({ onSearch }: { onSearch: () => void }) {
  const { state, isMobile } = useSidebar();
  const keys = shortcut("K");
  if (state === "collapsed" && !isMobile) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            onClick={onSearch}
            className="mx-auto flex size-9 items-center justify-center rounded-[10px] text-soft outline-none hover:bg-surface hover:text-strong focus-visible:ring-2 focus-visible:ring-sidebar-ring"
          >
            <Search aria-hidden="true" className="size-[1.125rem]" />
            <span className="sr-only">Procurar</span>
          </button>
        </TooltipTrigger>
        <TooltipContent side="right">Procurar ({keys})</TooltipContent>
      </Tooltip>
    );
  }
  return (
    <button
      type="button"
      onClick={onSearch}
      aria-keyshortcuts={keys.replace(" ", "+").replace("⌘", "Meta").replace("Ctrl", "Control")}
      className="flex h-10 w-full items-center gap-2.5 rounded-[10px] border border-[#2a2a2a] bg-surface px-3 text-sm text-soft outline-none hover:border-[#333] hover:text-foreground focus-visible:ring-2 focus-visible:ring-sidebar-ring"
    >
      <Search aria-hidden="true" className="size-4 text-dim" />
      <span className="flex-1 text-left">Procurar</span>
      <Kbd>{keys}</Kbd>
    </button>
  );
}

interface InterfaceSwitchProps {
  path: WorkPath;
  pathname: string;
  onChoosePath: (path: WorkPath) => void;
}

/**
 * A troca entre as duas interfaces. Fica-se no mesmo ecrã se ele também existir na outra; se
 * não, vai-se para os Dados, onde as duas começam.
 */
function InterfaceSwitch({ path, pathname, onChoosePath }: InterfaceSwitchProps) {
  const navigate = useNavigate();
  const { state, isMobile } = useSidebar();
  const other: WorkPath = path === "manual" ? "agents" : "manual";

  function choose(target: WorkPath) {
    if (target === path) {
      return;
    }
    onChoosePath(target);
    const current = Object.values(SCREENS).find(
      (screen) => screen.to !== "/" && isCurrent(screen.to, pathname),
    );
    if (current === undefined || !belongsTo(current.to, target)) {
      void navigate({ to: "/dados" });
    }
  }

  if (state === "collapsed" && !isMobile) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            onClick={() => {
              choose(other);
            }}
            className="mx-auto flex size-9 items-center justify-center rounded-[10px] text-soft outline-none hover:bg-surface hover:text-strong focus-visible:ring-2 focus-visible:ring-sidebar-ring"
          >
            <ArrowLeftRight aria-hidden="true" className="size-[1.125rem]" />
            <span className="sr-only">Mudar para a interface {INTERFACE_NAMES[other]}</span>
          </button>
        </TooltipTrigger>
        <TooltipContent side="right">Mudar para «{INTERFACE_NAMES[other]}»</TooltipContent>
      </Tooltip>
    );
  }

  // Um grupo de alternância: um só ponto de paragem do Tab, e as setas trocam (como em Radix).
  return (
    <ToggleGroup.Root
      type="single"
      value={path}
      onValueChange={(value) => {
        if (value === "manual" || value === "agents") {
          choose(value);
        }
      }}
      aria-label="Interface"
      className="grid grid-cols-2 gap-1 rounded-[10px] border border-[#2a2a2a] bg-surface p-1"
    >
      {(["manual", "agents"] as const).map((option) => (
        <ToggleGroup.Item
          key={option}
          value={option}
          className={cn(
            "h-8 rounded-[8px] border border-transparent text-sm text-soft transition-colors outline-none hover:text-strong focus-visible:ring-2 focus-visible:ring-sidebar-ring",
            "data-[state=on]:border-[#333] data-[state=on]:bg-surface-high data-[state=on]:font-medium data-[state=on]:text-strong data-[state=on]:shadow-[inset_0_1px_0_rgb(255_255_255/0.05)]",
          )}
        >
          {INTERFACE_NAMES[option]}
        </ToggleGroup.Item>
      ))}
    </ToggleGroup.Root>
  );
}

/** O botão que recolhe e abre a barra (também com `Ctrl+B`), no canto, como na referência. */
function CollapseButton() {
  const { open, isMobile, toggleSidebar } = useSidebar();
  if (isMobile) {
    return null;
  }
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={toggleSidebar}
          aria-expanded={open}
          className="flex size-8 shrink-0 items-center justify-center rounded-lg text-dim outline-none hover:bg-surface hover:text-strong focus-visible:ring-2 focus-visible:ring-sidebar-ring"
        >
          <PanelLeft aria-hidden="true" className="size-[1.125rem]" />
          <span className="sr-only">{open ? "Recolher a barra" : "Abrir a barra"}</span>
        </button>
      </TooltipTrigger>
      <TooltipContent side="right">
        <span className="flex items-center gap-2">
          {open ? "Recolher a barra" : "Abrir a barra"} <Kbd>{shortcut("B")}</Kbd>
        </span>
      </TooltipContent>
    </Tooltip>
  );
}
