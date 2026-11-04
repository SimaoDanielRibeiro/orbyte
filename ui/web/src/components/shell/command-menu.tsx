/**
 * A pesquisa (`Ctrl K`, ou `⌘ K` no Mac): salta para um ecrã, um modelo ou uma métrica sem
 * passar pela barra lateral. Os ecrãs das duas interfaces estão todos aqui; abrir um que só
 * existe na outra muda de interface (a moldura trata disso).
 */
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { Command } from "cmdk";
import { Search } from "lucide-react";
import { Dialog } from "radix-ui";
import { useEffect } from "react";

import {
  catalogMetricsOptions,
  catalogModelsOptions,
} from "@/api/generated/@tanstack/react-query.gen";
import { useLabel } from "@/api/vocabulary";
import { Kbd } from "@/components/ui/kbd";
import { metricAnchor } from "@/features/metrics/names";

import { INTERFACE_NAMES, interfaceOf, SCREENS, type Screen } from "./screens";

const ALL_SCREENS: readonly Screen[] = Object.values(SCREENS);

const GROUP =
  "[&_[cmdk-group-heading]]:small-label [&_[cmdk-group-heading]]:px-3 [&_[cmdk-group-heading]]:pt-3 [&_[cmdk-group-heading]]:pb-1.5";
const ITEM =
  "flex h-10 cursor-pointer items-center gap-3 rounded-[10px] px-3 text-sm text-soft select-none data-[selected=true]:bg-surface-high data-[selected=true]:text-strong [&_svg]:size-4 [&_svg]:shrink-0";

interface CommandMenuProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function CommandMenu({ open, onOpenChange }: CommandMenuProps) {
  const navigate = useNavigate();
  const label = useLabel();
  // Os modelos e as métricas só se pedem quando a pesquisa abre, e ficam na cache.
  const models = useQuery({ ...catalogModelsOptions(), staleTime: "static", enabled: open });
  const metrics = useQuery({ ...catalogMetricsOptions(), staleTime: "static", enabled: open });

  // Ctrl+K (ou Cmd+K) abre e fecha a pesquisa em qualquer ecrã.
  useEffect(() => {
    function toggle(event: KeyboardEvent) {
      if (event.key.toLowerCase() === "k" && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        onOpenChange(!open);
      }
    }
    window.addEventListener("keydown", toggle);
    return () => {
      window.removeEventListener("keydown", toggle);
    };
  }, [open, onOpenChange]);

  function go(action: () => Promise<void>) {
    onOpenChange(false);
    void action();
  }

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/70 data-open:animate-in data-open:fade-in-0 data-closed:animate-out data-closed:fade-out-0" />
        <Dialog.Content
          aria-describedby={undefined}
          className="fixed top-[14vh] left-1/2 z-50 w-[min(38rem,calc(100%-2rem))] -translate-x-1/2 overflow-hidden panel shadow-2xl outline-none data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95"
        >
          <Dialog.Title className="sr-only">Search</Dialog.Title>
          <Command label="Search screens, models and metrics" loop>
            <div className="flex items-center gap-3 border-b px-4">
              <Search aria-hidden="true" className="size-4 text-dim" />
              <Command.Input
                placeholder="Search screens, models and metrics"
                className="h-13 flex-1 bg-transparent text-base text-strong outline-none placeholder:text-dim"
              />
              <Kbd>Esc</Kbd>
            </div>
            <Command.List className="max-h-[min(26rem,60vh)] scroll-py-2 overflow-y-auto overscroll-contain p-2">
              <Command.Empty className="px-3 py-8 text-center text-sm text-dim">
                Nothing by that name.
              </Command.Empty>

              <Command.Group heading="Screens" className={GROUP}>
                {ALL_SCREENS.map((screen) => {
                  const only = interfaceOf(screen.to);
                  const Icon = screen.icon;
                  return (
                    <Command.Item
                      key={screen.to}
                      value={`screen ${screen.title}`}
                      className={ITEM}
                      onSelect={() => {
                        go(() => navigate({ to: screen.to }));
                      }}
                    >
                      <Icon aria-hidden="true" />
                      <span className="flex-1 truncate">{screen.title}</span>
                      {only !== null && (
                        <span className="small-label">{INTERFACE_NAMES[only]}</span>
                      )}
                    </Command.Item>
                  );
                })}
              </Command.Group>

              {models.data !== undefined && (
                <Command.Group heading="Models" className={GROUP}>
                  {models.data.map((model) => (
                    <Command.Item
                      key={model.name}
                      value={`model ${model.display_name} ${model.name}`}
                      className={ITEM}
                      onSelect={() => {
                        go(() => navigate({ to: "/models/$name", params: { name: model.name } }));
                      }}
                    >
                      <span className="flex-1 truncate">{model.display_name}</span>
                      <span className="small-label">{label("ModelFamily", model.family)}</span>
                    </Command.Item>
                  ))}
                </Command.Group>
              )}

              {metrics.data !== undefined && (
                <Command.Group heading="Metrics" className={GROUP}>
                  {metrics.data.map((metric) => (
                    <Command.Item
                      key={metric.name}
                      value={`metric ${metric.display_name} ${metric.name}`}
                      className={ITEM}
                      onSelect={() => {
                        go(() => navigate({ to: "/metrics", hash: metricAnchor(metric.name) }));
                      }}
                    >
                      <span className="flex-1 truncate">{metric.display_name}</span>
                    </Command.Item>
                  ))}
                </Command.Group>
              )}
            </Command.List>
          </Command>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
