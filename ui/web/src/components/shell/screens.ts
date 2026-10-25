/**
 * Os ecrãs da interface: o caminho de cada um, o nome, o que se faz nele e o ícone.
 *
 * A barra lateral, os cabeçalhos das páginas e os estados vazios leem daqui, para que um
 * ecrã tenha o mesmo nome em todo o lado.
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
  /** O que se faz no ecrã, numa frase. */
  description: string;
  icon: LucideIcon;
}

export const SCREENS = {
  home: {
    to: "/",
    title: "Início",
    description:
      "Escolhe como queres trabalhar: testar um modelo à mão ou deixar os agentes procurar.",
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
  model: {
    to: "/modelos",
    title: "Modelo",
    description: "Escolhe o modelo, vê como funciona e ajusta os hiperparâmetros.",
    icon: Boxes,
  },
  models: {
    to: "/modelos",
    title: "Modelos",
    description: "Todos os modelos do laboratório, com o que cada um faz e como aprende.",
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

/** O nome de cada caminho, como aparece no início e na barra lateral. */
export const PATH_NAMES: Readonly<Record<WorkPath, string>> = {
  manual: "Testar à mão",
  agents: "Procurar com agentes",
};

/** Os passos de cada caminho, pela ordem: a barra lateral numera-os. */
export const PATH_STEPS: Readonly<Record<WorkPath, readonly Screen[]>> = {
  manual: [
    SCREENS.data,
    SCREENS.explore,
    SCREENS.prepare,
    SCREENS.model,
    SCREENS.train,
    SCREENS.explain,
  ],
  agents: [SCREENS.data, SCREENS.agents, SCREENS.train, SCREENS.explain],
};

/**
 * Os ecrãs para aprender. No caminho manual, os modelos já são o passo 4, por isso não se
 * repetem aqui: um ecrã aparece uma só vez na barra.
 */
export const LEARN_SCREENS: Readonly<Record<WorkPath, readonly Screen[]>> = {
  manual: [SCREENS.metrics, SCREENS.laboratory],
  agents: [SCREENS.models, SCREENS.metrics, SCREENS.laboratory],
};

/** Os outros ecrãs, fora dos dois caminhos. */
export const OTHER_SCREENS: readonly Screen[] = [SCREENS.reinforcement, SCREENS.history];

/** Se o ecrã `to` é o da página aberta (ou o de uma página dentro dele). */
export function isCurrent(to: ScreenPath, pathname: string): boolean {
  return to === "/" ? pathname === "/" : pathname === to || pathname.startsWith(`${to}/`);
}
