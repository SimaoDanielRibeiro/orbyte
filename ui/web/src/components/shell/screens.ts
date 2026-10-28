/**
 * Os ecrãs da interface: o caminho de cada um, o nome, o que se faz nele e o ícone, e as secções
 * de cada uma das duas interfaces (a manual e a dos agentes).
 *
 * A barra lateral, os cabeçalhos das páginas, a pesquisa e os estados vazios leem daqui, para que
 * um ecrã tenha o mesmo nome em todo o lado.
 */
import type { LucideIcon } from "lucide-react";
import {
  Activity,
  Boxes,
  Database,
  FlaskConical,
  Grid3x3,
  History,
  House,
  Lightbulb,
  ScanSearch,
  Sigma,
  SlidersHorizontal,
  Workflow,
} from "lucide-react";

import type { WorkPath } from "@/state/session";

export type ScreenPath =
  | "/"
  | "/dados"
  | "/explorar"
  | "/preparar"
  | "/modelos"
  | "/treinar"
  | "/explicar"
  | "/agentes"
  | "/metricas"
  | "/laboratorio"
  | "/rl"
  | "/experiencias";

export interface Screen {
  to: ScreenPath;
  /** O nome na barra lateral e o título da página. */
  title: string;
  /** O que se faz no ecrã, numa frase (aparece no "?" do cabeçalho, não à vista). */
  description: string;
  icon: LucideIcon;
}

export const SCREENS = {
  home: {
    to: "/",
    title: "Início",
    description: "Escolhe a interface: testar um modelo à mão ou deixar os agentes procurar.",
    icon: House,
  },
  data: {
    to: "/dados",
    title: "Dados",
    description: "Carrega um CSV ou um Excel, escolhe um dataset de exemplo ou gera um sintético.",
    icon: Database,
  },
  explore: {
    to: "/explorar",
    title: "Explorar",
    description: "Vê o perfil de cada coluna, as correlações, os nulos e os avisos sobre os dados.",
    icon: ScanSearch,
  },
  prepare: {
    to: "/preparar",
    title: "Preparar",
    description: "Escolhe o alvo e a tarefa, limpa os dados, transforma as colunas e divide-os.",
    icon: SlidersHorizontal,
  },
  models: {
    to: "/modelos",
    title: "Modelos",
    description: "Escolhe o modelo, vê como funciona e ajusta os hiperparâmetros.",
    icon: Boxes,
  },
  train: {
    to: "/treinar",
    title: "Treinar e avaliar",
    description: "Acompanha o treino passo a passo e lê a avaliação de cada métrica.",
    icon: Activity,
  },
  explain: {
    to: "/explicar",
    title: "Explicar",
    description: "Vê que features pesam nas previsões e porque é que o modelo decidiu cada uma.",
    icon: Lightbulb,
  },
  agents: {
    to: "/agentes",
    title: "Agentes",
    description: "Os agentes preparam os dados, experimentam os modelos e escolhem o melhor.",
    icon: Workflow,
  },
  metrics: {
    to: "/metricas",
    title: "Métricas",
    description: "O que cada métrica mede, a fórmula e como se lê.",
    icon: Sigma,
  },
  laboratory: {
    to: "/laboratorio",
    title: "Laboratório",
    description:
      "Mexe num controlo e vê o efeito: overfitting, taxa de aprendizagem, k e regularização.",
    icon: FlaskConical,
  },
  reinforcement: {
    to: "/rl",
    title: "Reinforcement learning",
    description: "Escolhe um ambiente e um agente, e vê-o aprender por tentativa e erro.",
    icon: Grid3x3,
  },
  history: {
    to: "/experiencias",
    title: "Experiências",
    description: "O histórico dos runs: abre, compara, guarda e reproduz.",
    icon: History,
  },
} as const satisfies Record<string, Screen>;

/** O nome de cada interface, como aparece na landing, na troca e na pesquisa. */
export const INTERFACE_NAMES: Readonly<Record<WorkPath, string>> = {
  manual: "Manual",
  agents: "Agentes",
};

export interface NavGroup {
  /** O rótulo pequeno do grupo (em mono, como na referência). */
  label: string;
  screens: readonly Screen[];
}

/**
 * As secções da barra lateral de cada interface. O fluxo vem primeiro, pela ordem em que se
 * trabalha; depois o que serve para aprender (ou consultar) e o histórico.
 */
export const NAVIGATION: Readonly<Record<WorkPath, readonly NavGroup[]>> = {
  manual: [
    {
      label: "Fluxo",
      screens: [
        SCREENS.data,
        SCREENS.explore,
        SCREENS.prepare,
        SCREENS.models,
        SCREENS.train,
        SCREENS.explain,
      ],
    },
    { label: "Aprender", screens: [SCREENS.laboratory, SCREENS.metrics, SCREENS.reinforcement] },
    { label: "Histórico", screens: [SCREENS.history] },
  ],
  agents: [
    { label: "Fluxo", screens: [SCREENS.data, SCREENS.agents, SCREENS.train, SCREENS.explain] },
    { label: "Consultar", screens: [SCREENS.models, SCREENS.metrics] },
    { label: "Histórico", screens: [SCREENS.history] },
  ],
};

/** Os ecrãs que só existem numa das interfaces: abri-los leva para ela. */
const ONLY_IN: readonly (readonly [ScreenPath, WorkPath])[] = [
  ["/explorar", "manual"],
  ["/preparar", "manual"],
  ["/laboratorio", "manual"],
  ["/rl", "manual"],
  ["/agentes", "agents"],
];

/** A interface a que a página aberta pertence, se só existir numa delas. */
export function interfaceOf(pathname: string): WorkPath | null {
  return ONLY_IN.find(([to]) => isCurrent(to, pathname))?.[1] ?? null;
}

/** Se o ecrã `to` existe na interface `path`. */
export function belongsTo(to: ScreenPath, path: WorkPath): boolean {
  return NAVIGATION[path].some((group) => group.screens.some((screen) => screen.to === to));
}

/** Se o ecrã `to` é o da página aberta (ou o de uma página dentro dele). */
export function isCurrent(to: ScreenPath, pathname: string): boolean {
  return to === "/" ? pathname === "/" : pathname === to || pathname.startsWith(`${to}/`);
}
