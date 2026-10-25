import { Link, useRouterState } from "@tanstack/react-router";
import { ArrowLeftRight, PanelLeftClose, PanelLeftOpen } from "lucide-react";

import { Kbd } from "@/components/ui/kbd";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupAction,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
  useSidebar,
} from "@/components/ui/sidebar";
import { useSession, type WorkPath } from "@/state/session";

import { ApiStatus } from "./api-status";
import { Logo } from "./logo";
import {
  isCurrent,
  LEARN_SCREENS,
  OTHER_SCREENS,
  PATH_NAMES,
  PATH_STEPS,
  SCREENS,
  type Screen,
} from "./screens";

/**
 * A barra lateral: o início, os passos do caminho escolhido (numerados, porque são uma
 * sequência), os ecrãs para aprender e os restantes. Recolhida, fica só com os ícones e os
 * números, e o nome de cada ecrã aparece numa dica.
 */
export function AppSidebar() {
  const path = useSession((state) => state.path);
  const choosePath = useSession((state) => state.choosePath);
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const other: WorkPath = path === "manual" ? "agents" : "manual";

  return (
    <Sidebar collapsible="icon">
      {/* Toda a barra é a navegação (com o estado da API e o botão de a recolher). */}
      <nav aria-label="Ecrãs" className="flex size-full min-h-0 flex-col">
        <SidebarHeader>
          <SidebarMenu>
            <SidebarMenuItem>
              <SidebarMenuButton asChild size="lg" tooltip="ML Lab">
                <Link to="/">
                  <Logo className="size-8! shrink-0" />
                  <span className="font-serif text-lg">ML Lab</span>
                </Link>
              </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarHeader>

        <SidebarContent>
          <SidebarGroup>
            <SidebarMenu>
              <NavItem screen={SCREENS.home} pathname={pathname} />
            </SidebarMenu>
          </SidebarGroup>

          <SidebarGroup>
            <SidebarGroupLabel>{PATH_NAMES[path]}</SidebarGroupLabel>
            <SidebarGroupAction
              title={`Mudar para «${PATH_NAMES[other]}»`}
              onClick={() => {
                choosePath(other);
              }}
            >
              <ArrowLeftRight />
              <span className="sr-only">Mudar para «{PATH_NAMES[other]}»</span>
            </SidebarGroupAction>
            <SidebarMenu>
              {PATH_STEPS[path].map((screen, index) => (
                <NavItem key={screen.to} screen={screen} step={index + 1} pathname={pathname} />
              ))}
            </SidebarMenu>
          </SidebarGroup>

          <SidebarGroup>
            <SidebarGroupLabel>Aprender</SidebarGroupLabel>
            <SidebarMenu>
              {LEARN_SCREENS[path].map((screen) => (
                <NavItem key={screen.to} screen={screen} pathname={pathname} />
              ))}
            </SidebarMenu>
          </SidebarGroup>

          <SidebarGroup>
            <SidebarMenu>
              {OTHER_SCREENS.map((screen) => (
                <NavItem key={screen.to} screen={screen} pathname={pathname} />
              ))}
            </SidebarMenu>
          </SidebarGroup>
        </SidebarContent>

        <SidebarFooter>
          <SidebarMenu>
            <SidebarMenuItem>
              <ApiStatus />
            </SidebarMenuItem>
            <SidebarMenuItem>
              <CollapseButton />
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarFooter>
        <SidebarRail />
      </nav>
    </Sidebar>
  );
}

interface NavItemProps {
  screen: Screen;
  pathname: string;
  /** O número do passo, nos ecrãs de um caminho. */
  step?: number;
}

function NavItem({ screen, pathname, step }: NavItemProps) {
  const { isMobile, setOpenMobile } = useSidebar();
  const Icon = screen.icon;
  return (
    <SidebarMenuItem>
      <SidebarMenuButton
        asChild
        isActive={isCurrent(screen.to, pathname)}
        tooltip={step === undefined ? screen.title : `${step}. ${screen.title}`}
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
          {step === undefined ? <Icon aria-hidden="true" /> : <StepNumber value={step} />}
          <span>{screen.title}</span>
        </Link>
      </SidebarMenuButton>
    </SidebarMenuItem>
  );
}

/** O número de um passo, num círculo do tamanho de um ícone (para caber na barra recolhida). */
function StepNumber({ value }: { value: number }) {
  return (
    <span className="flex size-4 shrink-0 items-center justify-center rounded-full border border-current text-[0.6875rem] leading-none tabular-nums">
      {value}
    </span>
  );
}

/** O botão que recolhe e abre a barra (também com `Ctrl+B`). No telemóvel, a gaveta fecha-se sozinha. */
function CollapseButton() {
  const { open, isMobile, toggleSidebar } = useSidebar();
  if (isMobile) {
    return null;
  }
  const Icon = open ? PanelLeftClose : PanelLeftOpen;
  return (
    <SidebarMenuButton
      onClick={toggleSidebar}
      aria-expanded={open}
      className="text-muted-foreground"
      tooltip={{
        children: (
          <span className="flex items-center gap-2">
            Abrir a barra <Kbd>Ctrl B</Kbd>
          </span>
        ),
      }}
    >
      <Icon aria-hidden="true" />
      <span>{open ? "Recolher a barra" : "Abrir a barra"}</span>
      <Kbd className="ml-auto">Ctrl B</Kbd>
    </SidebarMenuButton>
  );
}
