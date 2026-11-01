import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate, useRouterState } from "@tanstack/react-router";
import { cn } from "cn";
import { ArrowLeftRight, PanelLeft } from "lucide-react";
import { motion, type Transition } from "motion/react";
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
  FIRST_STEPS,
  INTERFACE_NAMES,
  isCurrent,
  NAVIGATION,
  SCREENS,
  stepsOf,
  type Screen,
  type ScreenPath,
} from "./screens";

interface AppSidebarProps {
  /** A interface aberta: a barra mostra as secções dela. */
  path: WorkPath;
  onChoosePath: (path: WorkPath) => void;
}

/**
 * O deslizar do destaque de uma secção para a outra: uma mola curta, que acelera, chega e
 * assenta com um ressalto pequeno, como se o cartão fosse arrastado até lá.
 */
const SLIDE: Transition = { type: "spring", bounce: 0.22, visualDuration: 0.42 };

/**
 * A barra lateral, pela ordem em que se trabalha (ADR 0036):
 *
 * 1. os primeiros passos, iguais nas duas interfaces (os Dados e o Explorar);
 * 2. a troca entre o Manual e os Agentes, porque é aí que os caminhos se separam;
 * 3. os passos da interface aberta, que continuam os primeiros;
 * 4. o que explica (o grupo "Aprender": os modelos, as métricas, o laboratório) e, por fim, o
 *    Histórico.
 *
 * Só o "Aprender" tem o rótulo à vista: os passos leem-se pelo trilho e o Histórico pelo nome
 * (os rótulos dos outros grupos ficam só para os leitores de ecrã).
 *
 * Os passos estão ligados por um trilho, com uma luz que o percorre sem parar e um ponto por
 * passo: aceso no passo aberto, cheio nos de trás, vazio nos que faltam. Não há filetes a separar
 * os grupos: o espaço faz isso. Recolhida, a barra fica só com os ícones.
 *
 * A secção aberta tem um cartão claro por trás, e ao abrir outra o cartão desliza até ela (e o
 * ponto aceso corre pelo trilho): vê-se de onde se veio e para onde se foi (ADR 0037). A pesquisa
 * não tem botão na barra: abre-se com `Ctrl K` (e, no telemóvel, no cabeçalho).
 */
export function AppSidebar({ path, onChoosePath }: AppSidebarProps) {
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const counts = useCounts();
  const [flow, ...rest] = NAVIGATION[path];
  const steps = stepsOf(path);
  const current = steps.findIndex((screen) => isCurrent(screen.to, pathname));

  return (
    <Sidebar collapsible="icon">
      {/* Toda a barra é a navegação (com o estado da API e o botão de a recolher). */}
      <nav aria-label="Ecrãs" className="flex size-full min-h-0 flex-col">
        <SidebarHeader className="gap-4 px-3 pt-4 pb-2">
          <div className="flex items-center justify-between gap-2 group-data-[collapsible=icon]:flex-col">
            <Link
              to="/"
              className="flex min-w-0 items-center gap-2.5 rounded-lg px-2 py-1 outline-none group-data-[collapsible=icon]:px-1 focus-visible:ring-2 focus-visible:ring-sidebar-ring"
              aria-label="ML Lab, voltar ao início"
            >
              <Logo className="h-5 w-6 shrink-0" />
              <span
                translate="no"
                className="truncate text-lg font-semibold tracking-[-0.01em] text-strong group-data-[collapsible=icon]:hidden"
              >
                ML Lab
              </span>
            </Link>
            <CollapseButton />
          </div>
        </SidebarHeader>

        <SidebarContent className="gap-0 px-2 pt-4 pb-6">
          <SidebarGroup className="py-0">
            {/* Sem rótulo à vista: os passos leem-se pela ordem (o trilho liga-os). */}
            <SidebarGroupLabel className="sr-only">Primeiros passos</SidebarGroupLabel>
            <StepList screens={FIRST_STEPS} first={0} current={current} pathname={pathname} />
          </SidebarGroup>

          {/* Por cima dos itens: o cartão da secção aberta passa por baixo da troca. */}
          <div className="relative z-30 px-2 pt-4 pb-3 group-data-[collapsible=icon]:px-0">
            <InterfaceSwitch path={path} pathname={pathname} onChoosePath={onChoosePath} />
          </div>

          {flow !== undefined && (
            <SidebarGroup className="py-0">
              <SidebarGroupLabel className="sr-only">
                {`${flow.label}: ${INTERFACE_NAMES[path]}`}
              </SidebarGroupLabel>
              <StepList
                screens={flow.screens}
                first={FIRST_STEPS.length}
                current={current}
                pathname={pathname}
                late
              />
            </SidebarGroup>
          )}

          {rest.map((group) => (
            <SidebarGroup key={group.label} className="pt-7 pb-0">
              <SidebarGroupLabel className={cn(group.hidden === true && "sr-only")}>
                {group.label}
              </SidebarGroupLabel>
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

        <SidebarFooter className="px-3 pt-2 pb-4">
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

/** Onde um passo está em relação ao passo aberto (sem passo aberto: nenhum se destaca). */
type StepState = "done" | "now" | "next" | "idle";

interface StepListProps {
  screens: readonly Screen[];
  /** A posição do primeiro destes passos entre todos os passos da interface. */
  first: number;
  /** A posição do passo aberto (-1 se a página aberta não for um passo). */
  current: number;
  pathname: string;
  /** O segundo troço do trilho: a luz parte mais tarde, como se continuasse o primeiro. */
  late?: boolean;
}

/**
 * Passos ligados por um trilho: uma linha fina que passa pelos pontos de cada passo, e uma luz
 * que desce por ela sem parar (a classe `rail-light`, que pára com "reduzir movimento").
 */
function StepList({ screens, first, current, pathname, late = false }: StepListProps) {
  return (
    <SidebarMenu className="relative gap-1">
      <span
        aria-hidden="true"
        className="pointer-events-none absolute top-5 bottom-5 left-[13px] w-px overflow-hidden bg-[#2c2c2c] group-data-[collapsible=icon]:hidden"
      >
        <span className={cn("rail-light", late && "rail-light-late")} />
      </span>
      {screens.map((screen, index) => {
        const position = first + index;
        const state: StepState =
          current < 0
            ? "idle"
            : position < current
              ? "done"
              : position === current
                ? "now"
                : "next";
        return <NavItem key={screen.to} screen={screen} pathname={pathname} step={state} />;
      })}
    </SidebarMenu>
  );
}

interface NavItemProps {
  screen: Screen;
  pathname: string;
  count?: number;
  /** Num passo: o ponto no trilho, à esquerda. */
  step?: StepState;
}

function NavItem({ screen, pathname, count, step }: NavItemProps) {
  const { isMobile, setOpenMobile } = useSidebar();
  const Icon = screen.icon;
  const active = isCurrent(screen.to, pathname);
  return (
    <SidebarMenuItem
      className={cn(step !== undefined && "pl-6 group-data-[collapsible=icon]:pl-0")}
    >
      {step !== undefined && (
        <span
          aria-hidden="true"
          className={cn(
            "absolute top-1/2 left-[9px] size-[9px] -translate-y-1/2 rounded-full border transition-colors group-data-[collapsible=icon]:hidden",
            step === "done" && "border-[#8a8a8a] bg-[#8a8a8a]",
            // O ponto aceso é o que corre pelo trilho (em baixo): aqui fica o vazio, por baixo.
            step !== "done" && "border-[#4a4a4a] bg-ink",
          )}
        />
      )}
      {step === "now" && (
        <motion.span
          aria-hidden="true"
          layoutId="trilho-agora"
          layoutDependency={pathname}
          transition={SLIDE}
          className="pointer-events-none absolute top-1/2 left-[9px] z-20 -mt-[4.5px] size-[9px] rounded-full border border-white bg-white shadow-[0_0_6px_rgb(255_255_255/0.45)] group-data-[collapsible=icon]:hidden"
        />
      )}
      {active && (
        // O cartão da secção aberta: um só, com o mesmo `layoutId` em todos os itens, e por isso
        // desliza do item de onde se saiu para o novo. Fica fora do botão, que corta o que sai
        // dele, para se ver o caminho todo.
        <motion.span
          aria-hidden="true"
          layoutId="barra-aberta"
          layoutDependency={pathname}
          transition={SLIDE}
          // O raio no estilo (e não numa classe): o Motion corrige-o enquanto o cartão estica.
          style={{ borderRadius: 10 }}
          className={cn(
            "pointer-events-none absolute inset-y-0 right-0 left-0 z-0 border border-[#3a3a3a] bg-[#262626] shadow-[inset_0_1px_0_rgb(255_255_255/0.06)]",
            "group-data-[collapsible=icon]:right-auto group-data-[collapsible=icon]:w-9",
            step !== undefined && "left-6 group-data-[collapsible=icon]:left-0",
          )}
        >
          {/* Nos passos, quem marca é o ponto aceso do trilho; nos outros, um traço de luz. */}
          {step === undefined && (
            <span className="absolute top-1/2 left-0 h-4 w-[2px] -translate-y-1/2 rounded-r-full bg-white group-data-[collapsible=icon]:hidden" />
          )}
        </motion.span>
      )}
      <SidebarMenuButton
        asChild
        isActive={active}
        tooltip={screen.title}
        // O fundo da secção aberta é o cartão que desliza: o botão fica transparente por cima.
        className="z-10 data-active:border-transparent data-active:bg-transparent data-active:text-strong data-active:shadow-none data-active:hover:bg-transparent"
      >
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

interface InterfaceSwitchProps {
  path: WorkPath;
  pathname: string;
  onChoosePath: (path: WorkPath) => void;
}

/**
 * A troca entre as duas interfaces, por baixo dos primeiros passos (que são das duas). Fica-se
 * no mesmo ecrã se ele também existir na outra; se não, vai-se para os Dados, onde as duas
 * começam.
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
