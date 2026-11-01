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
  Package,
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
  | "/guardar"
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
  save: {
    to: "/guardar",
    title: "Guardar",
    description:
      "Guarda o que preparaste e o modelo num ficheiro, vê-o antes de o gravar e carrega um modelo já treinado.",
    icon: Package,
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
    title: "Histórico",
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
  /**
   * Se o rótulo fica à vista. Um grupo de um só item (o Histórico) não precisa dele: o nome do
   * item já diz o que é, e o espaço por cima separa-o dos outros.
   */
  hidden?: boolean;
  screens: readonly Screen[];
}

/**
 * Os primeiros passos, iguais nas duas interfaces: abrir os dados e explorá-los. Na barra ficam
 * por cima da troca entre o Manual e os Agentes, porque é a partir daí que os caminhos se
 * separam (ADR 0036).
 */
export const FIRST_STEPS: readonly Screen[] = [SCREENS.data, SCREENS.explore];

/**
 * As secções da barra lateral de cada interface, por baixo da troca. O primeiro grupo continua
 * os passos, pela ordem em que se trabalha; depois o que explica (os modelos, as métricas) e,
 * por fim, o histórico.
 */
export const NAVIGATION: Readonly<Record<WorkPath, readonly NavGroup[]>> = {
  manual: [
    {
      label: "Os passos",
      screens: [SCREENS.prepare, SCREENS.models, SCREENS.train, SCREENS.explain, SCREENS.save],
    },
    { label: "Aprender", screens: [SCREENS.laboratory, SCREENS.metrics, SCREENS.reinforcement] },
    { label: "Histórico", hidden: true, screens: [SCREENS.history] },
  ],
  agents: [
    {
      label: "Os passos",
      screens: [SCREENS.agents, SCREENS.train, SCREENS.explain, SCREENS.save],
    },
    { label: "Aprender", screens: [SCREENS.models, SCREENS.metrics] },
    { label: "Histórico", hidden: true, screens: [SCREENS.history] },
  ],
};

/** Os passos de uma interface, do princípio ao fim: os primeiros e os dela. */
export function stepsOf(path: WorkPath): readonly Screen[] {
  return [...FIRST_STEPS, ...(NAVIGATION[path][0]?.screens ?? [])];
}

/** Os ecrãs que só existem numa das interfaces: abri-los leva para ela. */
const ONLY_IN: readonly (readonly [ScreenPath, WorkPath])[] = [
  ["/preparar", "manual"],
  ["/laboratorio", "manual"],
  ["/rl", "manual"],
  ["/agentes", "agents"],
];

/** A interface a que a página aberta pertence, se só existir numa delas. */
export function interfaceOf(pathname: string): WorkPath | null {
  return ONLY_IN.find(([to]) => isCurrent(to, pathname))?.[1] ?? null;
}

/** Se o ecrã `to` existe na interface `path` (os primeiros passos existem nas duas). */
export function belongsTo(to: ScreenPath, path: WorkPath): boolean {
  return (
    FIRST_STEPS.some((screen) => screen.to === to) ||
    NAVIGATION[path].some((group) => group.screens.some((screen) => screen.to === to))
  );
}

/** Se o ecrã `to` é o da página aberta (ou o de uma página dentro dele). */
export function isCurrent(to: ScreenPath, pathname: string): boolean {
  return to === "/" ? pathname === "/" : pathname === to || pathname.startsWith(`${to}/`);
}
