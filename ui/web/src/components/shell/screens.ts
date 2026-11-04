/**
 * Os ecrãs da interface: o caminho de cada um, o nome, o que se faz nele e o ícone, e as secções
 * de cada uma das duas interfaces (a manual e a dos agentes).
 *
 * A barra lateral, os cabeçalhos das páginas, a pesquisa e os estados vazios leem daqui, para que
 * um ecrã tenha o mesmo nome em todo o lado. Os textos estão em inglês, como todo o site (ADR
 * 0043); os caminhos também.
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
  | "/data"
  | "/explore"
  | "/prepare"
  | "/models"
  | "/train"
  | "/explain"
  | "/save"
  | "/agents"
  | "/metrics"
  | "/lab"
  | "/rl"
  | "/history";

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
    title: "Home",
    description: "Choose how to work: test a model by hand, or let the agents search for you.",
    icon: House,
  },
  data: {
    to: "/data",
    title: "Data",
    description:
      "Upload a CSV or an Excel file, pick a sample dataset or generate a synthetic one.",
    icon: Database,
  },
  explore: {
    to: "/explore",
    title: "Explore",
    description:
      "See each column's profile, the correlations, the missing values and the warnings.",
    icon: ScanSearch,
  },
  prepare: {
    to: "/prepare",
    title: "Prepare",
    description:
      "Choose the target and the task, clean the data, transform the columns and split them.",
    icon: SlidersHorizontal,
  },
  models: {
    to: "/models",
    title: "Models",
    description:
      "Choose a model, see how it works and tune its hyperparameters. The colour is the problem it solves: cool for learning from a target, warm for finding structure on its own.",
    icon: Boxes,
  },
  train: {
    to: "/train",
    title: "Train & evaluate",
    description: "Follow the training step by step and read the evaluation of each metric.",
    icon: Activity,
  },
  explain: {
    to: "/explain",
    title: "Explain",
    description:
      "See what happened in the training, what weighs on the predictions, and why the model decided each one.",
    icon: Lightbulb,
  },
  save: {
    to: "/save",
    title: "Save",
    description:
      "Save the preparation and the model in one file, preview it before writing it, and load a trained model.",
    icon: Package,
  },
  agents: {
    to: "/agents",
    title: "Agents",
    description: "The agents prepare the data, try the models and pick the best one.",
    icon: Workflow,
  },
  metrics: {
    to: "/metrics",
    title: "Metrics",
    description: "What each metric measures, its formula and how to read it.",
    icon: Sigma,
  },
  laboratory: {
    to: "/lab",
    title: "Lab",
    description:
      "Move a control and see the effect: overfitting, learning rate, k and regularization.",
    icon: FlaskConical,
  },
  reinforcement: {
    to: "/rl",
    title: "Reinforcement learning",
    description: "Choose an environment and an agent, and watch it learn by trial and error.",
    icon: Grid3x3,
  },
  history: {
    to: "/history",
    title: "History",
    description: "Every run you trained: open, compare, save and replay.",
    icon: History,
  },
} as const satisfies Record<string, Screen>;

/** O nome de cada interface, como aparece na landing, na troca e na pesquisa. */
export const INTERFACE_NAMES: Readonly<Record<WorkPath, string>> = {
  manual: "Manual",
  agents: "Agents",
};

export interface NavGroup {
  /** O rótulo pequeno do grupo (em mono, como na referência). */
  label: string;
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
 * os passos, pela ordem em que se trabalha; depois o que explica (os modelos, as métricas).
 */
export const NAVIGATION: Readonly<Record<WorkPath, readonly NavGroup[]>> = {
  manual: [
    {
      label: "Steps",
      screens: [SCREENS.prepare, SCREENS.models, SCREENS.train, SCREENS.explain, SCREENS.save],
    },
    { label: "Learn", screens: [SCREENS.laboratory, SCREENS.metrics, SCREENS.reinforcement] },
  ],
  agents: [
    {
      label: "Steps",
      screens: [SCREENS.agents, SCREENS.train, SCREENS.explain, SCREENS.save],
    },
    { label: "Learn", screens: [SCREENS.models, SCREENS.metrics] },
  ],
};

/**
 * O que fica no fundo da barra, nas duas interfaces, por cima do estado da API: o Histórico. Não
 * é um passo nem explica nada; é para onde se volta, e por isso fica sempre no mesmo sítio.
 */
export const PINNED: readonly Screen[] = [SCREENS.history];

/** Os passos de uma interface, do princípio ao fim: os primeiros e os dela. */
export function stepsOf(path: WorkPath): readonly Screen[] {
  return [...FIRST_STEPS, ...(NAVIGATION[path][0]?.screens ?? [])];
}

/** Os ecrãs que só existem numa das interfaces: abri-los leva para ela. */
const ONLY_IN: readonly (readonly [ScreenPath, WorkPath])[] = [
  ["/prepare", "manual"],
  ["/lab", "manual"],
  ["/rl", "manual"],
  ["/agents", "agents"],
];

/** A interface a que a página aberta pertence, se só existir numa delas. */
export function interfaceOf(pathname: string): WorkPath | null {
  return ONLY_IN.find(([to]) => isCurrent(to, pathname))?.[1] ?? null;
}

/** Se o ecrã `to` existe na interface `path` (os primeiros passos e o fundo existem nas duas). */
export function belongsTo(to: ScreenPath, path: WorkPath): boolean {
  return (
    FIRST_STEPS.some((screen) => screen.to === to) ||
    PINNED.some((screen) => screen.to === to) ||
    NAVIGATION[path].some((group) => group.screens.some((screen) => screen.to === to))
  );
}

/**
 * Para onde se vai ao trocar para a interface `path` a partir de um ecrã que ela não tem: o
 * primeiro passo dela (o Preparar, ou os Agentes), se os dados já estiverem abertos; sem dados,
 * os Dados, onde as duas começam (ADR 0050).
 */
export function landingOf(path: WorkPath, datasetOpen: boolean): ScreenPath {
  const first = NAVIGATION[path][0]?.screens[0];
  return datasetOpen && first !== undefined ? first.to : SCREENS.data.to;
}

/** Se o ecrã `to` é o da página aberta (ou o de uma página dentro dele). */
export function isCurrent(to: ScreenPath, pathname: string): boolean {
  return to === "/" ? pathname === "/" : pathname === to || pathname.startsWith(`${to}/`);
}
