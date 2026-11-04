import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate, useRouterState } from "@tanstack/react-router";
import { cn } from "cn";
import { ArrowLeftRight, PanelLeftClose, PanelLeftOpen } from "lucide-react";
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
import { useSession, type WorkPath } from "@/state/session";

import { ApiStatus } from "./api-status";
import { Logo } from "./logo";
import {
  belongsTo,
  FIRST_STEPS,
  INTERFACE_NAMES,
  isCurrent,
  landingOf,
  NAVIGATION,
  PINNED,
  SCREENS,
  type Screen,
  type ScreenPath,
} from "./screens";

interface AppSidebarProps {
  /** A interface aberta: a barra mostra as secções dela. */
  path: WorkPath;
  onChoosePath: (path: WorkPath) => void;
}

/**
 * O deslizar do destaque de uma secção para a outra: uma mola que acelera, chega e assenta com
 * um ressalto quase nenhum, como se o cartão fosse arrastado até lá e pousado (ADR 0048: o
 * ressalto maior lia-se como um salto, não como um deslizar).
 */
const SLIDE: Transition = { type: "spring", bounce: 0.08, visualDuration: 0.5 };

/**
 * A barra lateral, pela ordem em que se trabalha (ADR 0036):
 *
 * 1. os primeiros passos, iguais nas duas interfaces (os Dados e o Explorar);
 * 2. a troca entre o Manual e os Agentes, porque é aí que os caminhos se separam;
 * 3. os passos da interface aberta, que continuam os primeiros;
 * 4. o que explica (o grupo "Learn": os modelos, as métricas, o laboratório);
 * 5. no fundo, o botão de recolher a barra e, por baixo dele, o Histórico: é para onde se volta,
 *    e fica sempre no mesmo sítio, nas duas interfaces. O estado da API só aparece quando ela
 *    falha (ligada, é o normal e não precisa de texto).
 *
 * Só o "Learn" tem o rótulo à vista: os passos leem-se pela ordem e o Histórico pelo nome (os
 * rótulos dos outros grupos ficam só para os leitores de ecrã). Não há filetes a separar os
 * grupos: o espaço faz isso.
 *
 * **Simples** (ADR 0050): só os itens, com o ícone e o nome. Não há trilho a ligar os passos (nem
 * a luz que o percorria): no browser, aberto ou recolhido, lia-se como um defeito.
 *
 * **Os ícones não se mexem** (ADR 0049): aberta ou recolhida, cada ícone fica com o centro a
 * 35 px da borda, à mesma altura. Recolher a barra só a estreita e apaga os nomes; por isso nada
 * muda de altura (o rótulo do "Learn" apaga-se mas guarda o lugar, e a troca de interface tem a
 * mesma altura nas duas formas).
 *
 * A secção aberta tem um cartão claro por trás, e ao abrir outra o cartão desliza até ela: vê-se
 * de onde se veio e para onde se foi (ADR 0037). A pesquisa não tem botão na barra: abre-se com
 * `Ctrl K` (e, no telemóvel, no cabeçalho).
 */
export function AppSidebar({ path, onChoosePath }: AppSidebarProps) {
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const counts = useCounts();
  const [flow, ...rest] = NAVIGATION[path];

  return (
    <Sidebar collapsible="icon">
      {/* Toda a barra é a navegação (com o estado da API e o botão de a recolher). */}
      <nav aria-label="Screens" className="flex size-full min-h-0 flex-col">
        {/*
          Tudo alinha pela mesma vertical, aberta ou recolhida: 12 px de margem dos dois lados, e
          os ícones (22 px) a 11 px do contorno de cada botão (1 px), com o centro a 35 px da borda
          da barra. Recolhida, a barra tem 68 px: a margem, o botão de 44 px e a outra margem.
        */}
        <SidebarHeader className="px-3 pt-5 pb-3">
          <Link
            to="/"
            // O logótipo (32 px) com o centro na vertical dos ícones: 12 + 7 + 16 = 35 px.
            className="flex min-w-0 items-center gap-3 overflow-hidden rounded-lg py-1 pr-3 pl-[7px] outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring"
            aria-label="Orbyte, back to home"
          >
            <Logo className="h-7 w-8 shrink-0" />
            <span
              translate="no"
              className="truncate font-display text-[1.375rem] font-medium tracking-[-0.02em] text-strong transition-opacity delay-100 duration-300 group-data-[collapsible=icon]:opacity-0 group-data-[collapsible=icon]:delay-0 group-data-[collapsible=icon]:duration-150"
            >
              Orbyte
            </span>
          </Link>
        </SidebarHeader>

        {/* `overflow-x-hidden`: enquanto a barra estreita, o que é largo (a troca de interface)
            fica cortado, em vez de fazer aparecer uma barra de deslocamento. */}
        <SidebarContent className="gap-0 overflow-x-hidden px-3 pt-5 pb-6">
          <SidebarGroup className="p-0">
            {/* Sem rótulo à vista: os passos leem-se pela ordem. */}
            <SidebarGroupLabel className="sr-only">First steps</SidebarGroupLabel>
            <NavList screens={FIRST_STEPS} pathname={pathname} />
          </SidebarGroup>

          {/* Por cima dos itens: o cartão da secção aberta passa por baixo da troca. */}
          <div className="relative z-30 pt-6 pb-4">
            <InterfaceSwitch path={path} pathname={pathname} onChoosePath={onChoosePath} />
          </div>

          {flow !== undefined && (
            <SidebarGroup className="p-0">
              <SidebarGroupLabel className="sr-only">
                {`${flow.label}: ${INTERFACE_NAMES[path]}`}
              </SidebarGroupLabel>
              <NavList screens={flow.screens} pathname={pathname} />
            </SidebarGroup>
          )}

          {rest.map((group) => (
            <SidebarGroup key={group.label} className="px-0 pt-9 pb-0">
              {/* O nome do grupo em letra normal, discreto: é um título, não um dado. Recolhida a
                  barra, apaga-se mas guarda o lugar, e os ícones de baixo não sobem. */}
              <SidebarGroupLabel className="px-[11px] font-sans text-sm font-medium tracking-normal text-dim">
                {group.label}
              </SidebarGroupLabel>
              <NavList screens={group.screens} pathname={pathname} counts={counts} />
            </SidebarGroup>
          ))}
        </SidebarContent>

        <SidebarFooter className="gap-1 border-t border-[#1f1f1f] px-3 pt-3 pb-5">
          {/* A API só se mostra quando falha: ligada, é o normal e não precisa de texto. */}
          <ApiStatus />
          <SidebarMenu className="gap-1">
            <SidebarMenuItem>
              <CollapseButton />
            </SidebarMenuItem>
            {/* O Histórico, no fundo: o sítio para onde se volta, igual nas duas interfaces. */}
            {PINNED.map((screen) => (
              <NavItem
                key={screen.to}
                screen={screen}
                pathname={pathname}
                count={counts[screen.to]}
              />
            ))}
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
    "/models": models.data?.length,
    "/metrics": metrics.data?.length,
    "/history": runs.data?.length,
  };
}

interface NavListProps {
  screens: readonly Screen[];
  pathname: string;
  /** As contagens à direita de alguns itens (os modelos, as métricas). */
  counts?: Partial<Record<ScreenPath, number>>;
}

/** Uma lista de ecrãs da barra, um por linha. */
function NavList({ screens, pathname, counts }: NavListProps) {
  return (
    <SidebarMenu className="gap-1">
      {screens.map((screen) => (
        <NavItem key={screen.to} screen={screen} pathname={pathname} count={counts?.[screen.to]} />
      ))}
    </SidebarMenu>
  );
}

interface NavItemProps {
  screen: Screen;
  pathname: string;
  count?: number;
}

function NavItem({ screen, pathname, count }: NavItemProps) {
  const { isMobile, setOpenMobile } = useSidebar();
  const Icon = screen.icon;
  const active = isCurrent(screen.to, pathname);
  return (
    <SidebarMenuItem>
      {active && (
        // O cartão da secção aberta: um só, com o mesmo `layoutId` em todos os itens, e por isso
        // desliza do item de onde se saiu para o novo. Fica fora do botão, que corta o que sai
        // dele, para se ver o caminho todo. Com a barra recolhida, estreita com ela.
        <motion.span
          aria-hidden="true"
          layoutId="barra-aberta"
          layoutDependency={pathname}
          transition={SLIDE}
          // O raio no estilo (e não numa classe): o Motion corrige-o enquanto o cartão estica.
          style={{ borderRadius: 10 }}
          className="pointer-events-none absolute inset-y-0 left-0 z-0 w-full border border-[#3a3a3a] bg-[#262626] shadow-[inset_0_1px_0_rgb(255_255_255/0.06)] transition-[width] duration-500 ease-out-expo group-data-[collapsible=icon]:w-11"
        />
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
 * no mesmo ecrã se ele também existir na outra. Se não existir, vai-se para o primeiro passo da
 * outra (o Preparar, ou os Agentes), quando os dados já estão abertos; sem dados, para os Dados,
 * onde as duas começam (ADR 0050). Antes ia sempre para os Dados, e quem já os tinha aberto
 * voltava atrás.
 *
 * A troca aberta (os dois botões) e a recolhida (um ícone) estão sempre lá, uma por cima da
 * outra, e passa-se de uma à outra a desvanecer, ao ritmo da barra: trocá-las de uma vez fazia a
 * aberta encolher e cortar os nomes enquanto a barra estreitava.
 */
function InterfaceSwitch({ path, pathname, onChoosePath }: InterfaceSwitchProps) {
  const navigate = useNavigate();
  const datasetOpen = useSession((session) => session.datasetId !== null);
  const { state, isMobile } = useSidebar();
  const other: WorkPath = path === "manual" ? "agents" : "manual";
  const collapsed = state === "collapsed" && !isMobile;

  function choose(target: WorkPath) {
    if (target === path) {
      return;
    }
    onChoosePath(target);
    const current = Object.values(SCREENS).find(
      (screen) => screen.to !== "/" && isCurrent(screen.to, pathname),
    );
    if (current === undefined || !belongsTo(current.to, target)) {
      void navigate({ to: landingOf(target, datasetOpen) });
    }
  }

  return (
    <div className="relative h-12">
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            inert={!collapsed}
            onClick={() => {
              choose(other);
            }}
            // A altura da troca aberta (48 px): o que está por baixo não sobe ao recolher.
            className={cn(
              "absolute top-0 left-0 flex h-12 w-11 items-center justify-center rounded-[10px] border border-transparent text-soft transition-[opacity,background-color,color] outline-none hover:bg-surface hover:text-strong focus-visible:ring-2 focus-visible:ring-sidebar-ring",
              collapsed ? "opacity-100 delay-100 duration-300" : "opacity-0 duration-150",
            )}
          >
            <ArrowLeftRight aria-hidden="true" className="size-[22px]" />
            <span className="sr-only">Switch to the {INTERFACE_NAMES[other]} interface</span>
          </button>
        </TooltipTrigger>
        <TooltipContent side="right">Switch to {INTERFACE_NAMES[other]}</TooltipContent>
      </Tooltip>
      {/* Um grupo de alternância: um só ponto de paragem do Tab, e as setas trocam (como em
          Radix). Com uma largura mínima: enquanto a barra estreita, fica cortado, não encolhido. */}
      <ToggleGroup.Root
        type="single"
        value={path}
        inert={collapsed}
        onValueChange={(value) => {
          if (value === "manual" || value === "agents") {
            choose(value);
          }
        }}
        aria-label="Interface"
        className={cn(
          "grid h-12 w-full min-w-[18rem] grid-cols-2 gap-1 rounded-[10px] border border-[#2a2a2a] bg-surface p-1 transition-opacity",
          collapsed ? "opacity-0 duration-150" : "opacity-100 delay-100 duration-300",
        )}
      >
        {(["manual", "agents"] as const).map((option) => (
          <ToggleGroup.Item
            key={option}
            value={option}
            className={cn(
              "h-full rounded-[8px] border border-transparent text-base text-soft transition-colors outline-none hover:text-strong focus-visible:ring-2 focus-visible:ring-sidebar-ring",
              "data-[state=on]:border-[#333] data-[state=on]:bg-surface-high data-[state=on]:font-medium data-[state=on]:text-strong data-[state=on]:shadow-[inset_0_1px_0_rgb(255_255_255/0.05)]",
            )}
          >
            {INTERFACE_NAMES[option]}
          </ToggleGroup.Item>
        ))}
      </ToggleGroup.Root>
    </div>
  );
}

/**
 * O botão que recolhe e abre a barra (também com `Ctrl+B`), no fundo, por cima do Histórico: é
 * uma linha como as outras, alinhada pelos mesmos ícones, e recolhida fica no mesmo sítio.
 */
function CollapseButton() {
  const { open, isMobile, toggleSidebar } = useSidebar();
  if (isMobile) {
    return null;
  }
  const Icon = open ? PanelLeftClose : PanelLeftOpen;
  const name = open ? "Collapse the sidebar" : "Expand the sidebar";
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <SidebarMenuButton
          onClick={toggleSidebar}
          aria-expanded={open}
          aria-label={name}
          className="text-dim hover:text-strong"
        >
          <Icon aria-hidden="true" />
          <span>Collapse</span>
        </SidebarMenuButton>
      </TooltipTrigger>
      <TooltipContent side="right">
        <span className="flex items-center gap-2">
          {name} <Kbd>{shortcut("B")}</Kbd>
        </span>
      </TooltipContent>
    </Tooltip>
  );
}
