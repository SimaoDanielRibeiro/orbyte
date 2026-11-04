/**
 * O fluxograma dos agentes, ao vivo: os dados passam ao coordenador, que distribui os
 * candidatos pelos agentes de cada família; o melhor é o líder (o vencedor, no fim), que se testa
 * uma só vez.
 *
 * As ligações são SVG, por baixo; os nós são HTML por cima, e são botões quando o ecrã sabe
 * mostrar o que um agente fez (`onSelect`): abre-se o painel dele ao lado. O movimento segue a
 * ideia de um mapa vivo (ADR 0034):
 *
 * - ao abrir, os nós entram coluna a coluna e as ligações desenham-se atrás deles;
 * - uma luz corre na ligação ativa, na direção do trabalho, com a cor do agente;
 * - nas ligações dos agentes que já trabalharam passa, de vez em quando, uma luz mais lenta.
 *
 * Com "reduzir movimento" tudo aparece já no fim, sem luzes a correr (o estado lê-se nas cores).
 *
 * O agente de IA (ADR 0042), quando entra, é a última linha da coluna dos agentes: contorno
 * tracejado, o seu lilás e um ícone, para não se confundir com os agentes de regras. Acende-se
 * enquanto pensa e enquanto a proposta dele se mede.
 */
import { cn } from "cn";
import { FlaskConical, Sparkles, Trophy, Workflow, Database, type LucideIcon } from "lucide-react";
import { motion, useReducedMotion } from "motion/react";
import { useLayoutEffect, useRef } from "react";

import type {
  Candidate,
  MetricResult,
  ModelAgentCard,
  ModelFamily,
  ObjectiveOption,
  SearchPhase,
  Thought,
  WorkInProgress,
} from "@/api/generated";
import { useLabel } from "@/api/vocabulary";
import { useSize } from "@/hooks/use-size";
import { withAlpha } from "@/lib/colors";
import { formatCount, formatMetric, formatQuantity } from "@/lib/format";

import { AI_AGENT_COLOR, familyColor } from "./agent-colors";
import { objectiveValue } from "./values";

/** Um nó do fluxograma: um dos quatro fixos, o agente de IA ou o agente de uma família. */
export type FlowNode = "data" | "coordinator" | "winner" | "test" | "ai" | ModelFamily;

/** O agente de IA, quando entra na procura. */
export interface FlowAi {
  /** O modelo de linguagem que pensa por ele ("qwen3:8b"). */
  model: string | null;
}

export interface FlowchartProps {
  agents: readonly ModelAgentCard[];
  phase: SearchPhase | null;
  working: WorkInProgress | null;
  candidates: readonly Candidate[];
  objective: ObjectiveOption | null;
  /** O melhor até agora (ou o vencedor, no fim). */
  leader: Candidate | null;
  budget: number | null;
  trainRows: number | null;
  /** No fim: o valor do vencedor no teste. */
  test: MetricResult | null;
  /** O nó com o painel aberto. */
  selected?: FlowNode | null;
  /** Sem esta função os nós não se clicam (não há painel para abrir). */
  onSelect?: (node: FlowNode) => void;
  /** Uma troca a desenhar por cima (o histórico das trocas). */
  exchange?: FlowExchange | null;
  /** O agente de IA, se entrar na afinação. */
  ai?: FlowAi | null;
  /** O que o agente de IA está a pensar agora (acende o nó dele). */
  thought?: Thought | null;
}

/** Uma troca do histórico: uma luz que vai de quem fala para quem ouve. */
export interface FlowExchange {
  /** Muda a cada troca: é o que faz a luz partir outra vez. */
  key: number;
  from: FlowNode;
  to: readonly FlowNode[];
  /**
   * O agente do vencedor: a decisão e o teste passam por ele (vê-se o vencedor a seguir pelo
   * fluxo, do coordenador ao agente que o propôs, ao vencedor e ao teste).
   */
  via: ModelFamily | "ai" | null;
  color: string;
  /** Quanto tempo leva a luz, em segundos. */
  duration: number;
}

type NodeState = "idle" | "active" | "done";

/** A largura das colunas de um só nó: fixa, para os nomes caberem sempre. */
const SIDES = { data: 148, coordinator: 156, winner: 156, test: 138 } as const;
const SIDES_WIDTH = SIDES.data + SIDES.coordinator + SIDES.winner + SIDES.test;
/** A coluna dos agentes cresce com o espaço, até um limite; o resto vai para as ligações. */
const AGENTS = { min: 188, max: 300 } as const;
/** O espaço mínimo entre colunas: as etiquetas das ligações que não cabem nele escondem-se. */
const MIN_GAP = 24;
/** A largura de uma letra das etiquetas e dos pormenores (Geist Mono a 12px). */
const MONO_CHAR = 7.2;
/** O que a etiqueta ocupa além das letras: as margens de dentro, o contorno e uma folga. */
const LABEL_EXTRA = 28;
const EDGE = 4;
/** Abaixo disto, os nomes não cabem: o fluxograma encolhe ou rola na horizontal. */
const MIN_WIDTH = SIDES_WIDTH + AGENTS.min + 4 * MIN_GAP + 2 * EDGE;
/**
 * Até onde o fluxograma encolhe, por igual, para caber sem rolar: num portátil falta-lhe pouco (a
 * 1280 px, com a barra aberta e as margens de 3 cm, tem 723 px), e cortar o teste do lado direito
 * escondia o fim do fluxo. Mais estreito do que isto (o tablet e o telemóvel), as letras ficavam
 * pequenas demais: rola.
 */
const MIN_SCALE = 0.8;
const NODE_HEIGHT = 60;
const ROW_GAP = 12;
const PADDING = 28;

type Columns = Record<
  "data" | "coordinator" | "agents" | "winner" | "test",
  { x: number; width: number }
>;

/** As colunas na largura que há: os nós não encolhem, as ligações esticam (e as etiquetas cabem). */
function columnsFor(width: number): Columns {
  const agents = Math.min(
    AGENTS.max,
    Math.max(AGENTS.min, width - SIDES_WIDTH - 4 * MIN_GAP - 2 * EDGE),
  );
  const gap = (width - SIDES_WIDTH - agents - 2 * EDGE) / 4;
  const widths = [SIDES.data, SIDES.coordinator, agents, SIDES.winner, SIDES.test];
  const keys = ["data", "coordinator", "agents", "winner", "test"] as const;
  let x = EDGE;
  const columns = {} as Columns;
  keys.forEach((key, index) => {
    const columnWidth = widths[index] ?? 0;
    columns[key] = { x, width: columnWidth };
    x += columnWidth + gap;
  });
  return columns;
}

interface PlacedNode {
  key: FlowNode;
  x: number;
  y: number;
  width: number;
  state: NodeState;
  title: string;
  /** Da versão mais longa à mais curta: fica a que cabe. */
  detail: readonly string[];
  icon?: LucideIcon;
  /** A cor do agente (as famílias). */
  color?: string;
  progress: number | null;
  /** A coluna, para a entrada em cascata. */
  column: number;
  /** O contorno tracejado (o agente de IA: outro tipo de agente). */
  dashed?: boolean;
}

interface PlacedEdge {
  key: string;
  from: [number, number];
  to: [number, number];
  state: NodeState;
  color?: string;
  /** Uma etiqueta a meio da ligação ("119 linhas", "fold 2 de 5"). */
  label?: string;
  column: number;
  /** A ordem das luzes lentas, para não passarem todas ao mesmo tempo. */
  order: number;
}

export function Flowchart({
  agents,
  phase,
  working,
  candidates,
  objective,
  leader,
  budget,
  trainRows,
  test,
  selected = null,
  onSelect,
  exchange = null,
  ai = null,
  thought = null,
}: FlowchartProps) {
  const label = useLabel();
  const { ref, width: available } = useSize<HTMLDivElement>();
  const shortfall = available / MIN_WIDTH;
  const scale = available > 0 && shortfall < 1 && shortfall >= MIN_SCALE ? shortfall : 1;
  const width = Math.max(MIN_WIDTH, available / scale);
  const columns = columnsFor(width);
  const rows = Math.max(1, agents.length + (ai === null ? 0 : 1));
  const stack = rows * NODE_HEIGHT + (rows - 1) * ROW_GAP;
  const height = Math.max(stack, NODE_HEIGHT * 3) + PADDING * 2;
  const middle = height / 2;
  const agentTop = (index: number) => middle - stack / 2 + index * (NODE_HEIGHT + ROW_GAP);

  const searching = phase === "screening" || phase === "tuning";
  const after = (stage: number) => order(phase) > stage;
  const states = {
    data: phase === "data" ? "active" : after(0) ? "done" : "idle",
    coordinator: searching && working === null ? "active" : after(2) || searching ? "done" : "idle",
    winner: phase === "final" ? "active" : phase === "done" ? "done" : "idle",
    test: phase === "done" ? "done" : "idle",
  } satisfies Record<string, NodeState>;
  // Os candidatos de cada agente: os de uma família são os sorteios dela; os do agente de IA, as
  // propostas dele (de qualquer família).
  const ownBy = (family: ModelFamily | "ai") =>
    candidates.filter((candidate) =>
      family === "ai" ? candidate.ai : candidate.family === family && !candidate.ai,
    );
  const agentState = (family: ModelFamily): NodeState => {
    if (searching && working?.family === family && !working.ai) {
      return "active";
    }
    return ownBy(family).length > 0 ? "done" : "idle";
  };
  const aiActive = searching && (thought !== null || working?.ai === true);
  const aiState: NodeState = aiActive ? "active" : ownBy("ai").length > 0 ? "done" : "idle";
  const best = (family: ModelFamily | "ai") => {
    const values = ownBy(family)
      .filter((candidate) => candidate.mean !== null)
      .map((candidate) => candidate.mean ?? 0);
    if (values.length === 0) {
      return null;
    }
    return objective?.higher_is_better === false ? Math.min(...values) : Math.max(...values);
  };
  const leaderColor =
    leader === null ? undefined : leader.ai ? AI_AGENT_COLOR : familyColor(leader.family);
  const measured = candidates.length;
  const now = describeNow(phase, working, thought, label);
  const center = (x: number) => [x, middle] as [number, number];
  const aiWorking = working?.ai === true ? working : null;

  /** O nó do agente de IA: a pensar, a ver a proposta medida, ou o que já propôs. */
  const aiNode = (): PlacedNode => {
    const own = ownBy("ai");
    const value = best("ai");
    return {
      key: "ai",
      x: columns.agents.x,
      y: agentTop(agents.length),
      width: columns.agents.width,
      state: aiState,
      title: label("AgentRole", "ai"),
      icon: Sparkles,
      color: AI_AGENT_COLOR,
      dashed: true,
      detail:
        searching && thought !== null
          ? ["Thinking about the next candidate", "Thinking"]
          : searching && aiWorking !== null
            ? [`Measuring ${aiWorking.model_display_name}`, aiWorking.model_display_name]
            : own.length > 0
              ? [
                  `${formatQuantity(own.length, "proposal", "proposals")}, best ${objectiveValue(value, objective)}`,
                  `Best ${objectiveValue(value, objective)} in ${formatCount(own.length)}`,
                ]
              : [`Waiting · ${ai?.model ?? "language model"}`, "Waiting"],
      progress:
        searching && aiWorking !== null ? aiWorking.fold / Math.max(1, aiWorking.folds) : null,
      column: 2,
    };
  };

  /** As ligações do agente de IA: do coordenador até ele, e dele ao líder (se liderar). */
  const aiEdges = (): PlacedEdge[] => {
    const top = agentTop(agents.length) + NODE_HEIGHT / 2;
    const leads = leader?.ai === true && order(phase) >= 3;
    return [
      {
        key: "para-ai",
        from: center(columns.coordinator.x + columns.coordinator.width),
        to: [columns.agents.x, top],
        state: aiState,
        color: AI_AGENT_COLOR,
        label:
          searching && thought !== null
            ? "thinking"
            : searching && aiWorking !== null
              ? `fold ${formatCount(aiWorking.fold)} of ${formatCount(aiWorking.folds)}`
              : undefined,
        column: 2,
        order: agents.length,
      },
      {
        key: "de-ai",
        from: [columns.agents.x + columns.agents.width, top],
        to: center(columns.winner.x),
        state: leads ? (states.winner === "active" ? "active" : "done") : "idle",
        color: AI_AGENT_COLOR,
        column: 3,
        order: agents.length,
      },
    ];
  };

  const nodes: PlacedNode[] = [
    {
      key: "data",
      x: columns.data.x,
      y: middle - NODE_HEIGHT / 2,
      width: columns.data.width,
      state: states.data,
      title: "Data",
      icon: Database,
      detail:
        phase === "data"
          ? ["Preparing"]
          : trainRows === null
            ? ["The data"]
            : [`${formatCount(trainRows)} training rows`, `${formatCount(trainRows)} rows`],
      progress: null,
      column: 0,
    },
    {
      key: "coordinator",
      x: columns.coordinator.x,
      y: middle - NODE_HEIGHT / 2,
      width: columns.coordinator.width,
      state: states.coordinator,
      title: label("AgentRole", "coordinator"),
      icon: Workflow,
      detail:
        budget === null
          ? [formatQuantity(measured, "candidate", "candidates")]
          : [
              `${formatCount(measured)} of ${formatCount(budget)} candidates`,
              `${formatCount(measured)} of ${formatCount(budget)} measured`,
              `${formatCount(measured)} of ${formatCount(budget)}`,
            ],
      progress: budget === null || measured === 0 ? null : measured / Math.max(1, budget),
      column: 1,
    },
    ...agents.map((agent, index): PlacedNode => {
      const state = agentState(agent.family);
      const own = ownBy(agent.family);
      const value = best(agent.family);
      const active = state === "active" && working !== null;
      return {
        key: agent.family,
        x: columns.agents.x,
        y: agentTop(index),
        width: columns.agents.width,
        state,
        title: label("ModelFamily", agent.family),
        color: familyColor(agent.family),
        detail: active
          ? [`Measuring ${working.model_display_name}`, working.model_display_name]
          : own.length === 0
            ? ["Waiting"]
            : [
                `${formatQuantity(own.length, "candidate", "candidates")}, best ${objectiveValue(value, objective)}`,
                `Best ${objectiveValue(value, objective)} in ${formatCount(own.length)}`,
              ],
        progress: active ? working.fold / Math.max(1, working.folds) : null,
        column: 2,
      };
    }),
    ...(ai === null ? [] : [aiNode()]),
    {
      key: "winner",
      x: columns.winner.x,
      y: middle - NODE_HEIGHT / 2,
      width: columns.winner.width,
      state: states.winner,
      title: phase === "done" || phase === "final" ? "Winner" : "Leader",
      icon: Trophy,
      color: leaderColor,
      detail:
        leader === null
          ? ["None yet"]
          : [
              `${objectiveValue(leader.mean, objective)}, ${leader.model_display_name}`,
              objectiveValue(leader.mean, objective),
            ],
      progress: null,
      column: 3,
    },
    {
      key: "test",
      x: columns.test.x,
      y: middle - NODE_HEIGHT / 2,
      width: columns.test.width,
      state: states.test,
      title: "Test",
      icon: FlaskConical,
      detail: test === null ? ["Only at the end", "At the end"] : [formatMetric(test)],
      progress: null,
      column: 4,
    },
  ];

  const edges: PlacedEdge[] = [
    {
      key: "dados",
      from: center(columns.data.x + columns.data.width),
      to: center(columns.coordinator.x),
      state: states.data,
      label: trainRows === null || phase === null ? undefined : `${formatCount(trainRows)} rows`,
      column: 1,
      order: 0,
    },
    ...agents.flatMap((agent, index): PlacedEdge[] => {
      const top = agentTop(index) + NODE_HEIGHT / 2;
      const state = agentState(agent.family);
      const color = familyColor(agent.family);
      const active = state === "active" && working !== null;
      const leads = leader?.family === agent.family && !leader.ai && order(phase) >= 3;
      return [
        {
          key: `para-${agent.family}`,
          from: center(columns.coordinator.x + columns.coordinator.width),
          to: [columns.agents.x, top],
          state,
          color,
          label: !active
            ? undefined
            : working.fold === 0
              ? "starting"
              : `fold ${formatCount(working.fold)} of ${formatCount(working.folds)}`,
          column: 2,
          order: index,
        },
        {
          key: `de-${agent.family}`,
          from: [columns.agents.x + columns.agents.width, top],
          to: center(columns.winner.x),
          state: leads ? (states.winner === "active" ? "active" : "done") : "idle",
          color,
          column: 3,
          order: index,
        },
      ];
    }),
    ...(ai === null ? [] : aiEdges()),
    {
      key: "teste",
      from: center(columns.winner.x + columns.winner.width),
      to: center(columns.test.x),
      state: states.test === "done" ? "done" : states.winner === "active" ? "active" : "idle",
      label: test === null ? undefined : formatMetric(test),
      column: 4,
      order: 0,
    },
  ];

  const routes =
    exchange === null
      ? []
      : exchange.to
          .map((target) => route(exchange.from, target, exchange.via, edges, nodes))
          .filter((path): path is string => path !== null);
  const lit = exchange === null ? null : new Set<FlowNode>([exchange.from, ...exchange.to]);

  return (
    <figure className="flex flex-col gap-3" data-phase={phase ?? "none"}>
      <div ref={ref} className="overflow-x-auto">
        {/* A caixa de fora tem o tamanho já encolhido; o desenho, o tamanho inteiro. */}
        <div style={{ width: width * scale, height: height * scale }}>
          <div
            className="relative origin-top-left"
            style={{ width, height, transform: scale < 1 ? `scale(${String(scale)})` : undefined }}
          >
            <svg
              width={width}
              height={height}
              aria-hidden="true"
              className="absolute inset-0 block overflow-visible"
            >
              {edges.map((edge) => (
                <Edge key={edge.key} edge={edge} />
              ))}
              {exchange !== null &&
                routes.map((path, index) => (
                  <Trail
                    key={`${String(exchange.key)}-${String(index)}`}
                    path={path}
                    color={exchange.color}
                    duration={exchange.duration}
                  />
                ))}
            </svg>
            {edges.map(
              (edge) =>
                edge.label !== undefined &&
                // Uma etiqueta que não cabe entre os dois nós ficava por cima deles: não se mostra (o
                // que diz também está no nó, ou no painel do agente).
                fits(edge) && (
                  <span
                    key={`etiqueta-${edge.key}`}
                    aria-hidden="true"
                    className="pointer-events-none absolute -translate-x-1/2 -translate-y-1/2 rounded-full border border-[#2e2e2e] bg-ground px-2 py-0.5 font-mono text-[12px] whitespace-nowrap text-soft"
                    style={{
                      left: (edge.from[0] + edge.to[0]) / 2,
                      // Numa ligação reta a etiqueta fica por cima da linha, a deixá-la à vista.
                      top: (edge.from[1] + edge.to[1]) / 2 - (edge.from[1] === edge.to[1] ? 16 : 0),
                    }}
                  >
                    {edge.label}
                  </span>
                ),
            )}
            {nodes.map((node) => (
              <NodeBox
                key={node.key}
                node={node}
                selected={selected === node.key}
                onSelect={onSelect}
                lit={lit?.has(node.key) === true ? exchange?.color : undefined}
              />
            ))}
            {/* A luz que viaja vai por cima dos nós (passa por dentro do agente do vencedor). */}
            {exchange !== null && (
              <svg
                width={width}
                height={height}
                aria-hidden="true"
                className="pointer-events-none absolute inset-0 block overflow-visible"
              >
                {routes.map((path, index) => (
                  <Pulse
                    key={`${String(exchange.key)}-${String(index)}`}
                    path={path}
                    color={exchange.color}
                    duration={exchange.duration}
                  />
                ))}
              </svg>
            )}
          </div>
        </div>
      </div>
      <figcaption className="sr-only" aria-live="polite">
        {now}
      </figcaption>
    </figure>
  );
}

/** Se a etiqueta de uma ligação cabe no espaço entre os dois nós que ela liga. */
function fits(edge: PlacedEdge): boolean {
  const room = edge.to[0] - edge.from[0];
  return (edge.label ?? "").length * MONO_CHAR + LABEL_EXTRA <= room;
}

/** A posição de uma fase na procura (para saber o que já passou). */
function order(phase: SearchPhase | null): number {
  switch (phase) {
    case "data":
      return 0;
    case "screening":
      return 1;
    case "tuning":
      return 2;
    case "final":
      return 3;
    case "done":
      return 4;
    default:
      return -1;
  }
}

/** O que está a acontecer agora, numa frase (para quem usa um leitor de ecrã). */
function describeNow(
  phase: SearchPhase | null,
  working: WorkInProgress | null,
  thought: Thought | null,
  label: ReturnType<typeof useLabel>,
): string {
  if (phase === null) {
    return "The search hasn't started yet.";
  }
  if (thought !== null) {
    return `Now: ${thought.author} thinking about candidate ${String(thought.candidate)}.`;
  }
  if (working !== null) {
    return `Now: ${working.author} measuring ${working.model_display_name}, fold ${String(working.fold)} of ${String(working.folds)}.`;
  }
  return `Now: ${label("SearchPhase", phase)}.`;
}

/** Um nó: um botão quando abre o painel do agente, uma caixa quando não há painel. */
function NodeBox({
  node,
  selected,
  onSelect,
  lit,
}: {
  node: PlacedNode;
  selected: boolean;
  onSelect?: (node: FlowNode) => void;
  /** A cor de quem fala, quando o nó entra na troca à vista (o histórico). */
  lit?: string;
}) {
  const reduced = useReducedMotion() ?? false;
  const { state, icon: Icon, color } = node;
  const active = state === "active";
  // Um agente de uma família não tem ícone: a cor dele é o contorno da caixa (ADR 0038).
  const family = Icon === undefined && color !== undefined;
  // A largura do texto: a caixa menos o ícone (se houver) e as margens.
  const letters = Math.floor((node.width - (family ? 30 : 66)) / MONO_CHAR);
  const detail = fit(node.detail, letters);
  const className = cn(
    "absolute flex items-center gap-2.5 overflow-hidden rounded-xl border px-3 text-left transition-[border-color,box-shadow,background-color] duration-300 outline-none",
    state === "idle" && "border-[#262626] bg-[#141414]",
    state === "done" && "border-[#363636] bg-[#171717]",
    active && "border-white/80 bg-surface-high",
    selected && "border-white bg-surface-high",
    lit !== undefined && "bg-surface-high",
    family && "px-3.5",
    node.dashed === true && "border-dashed",
    onSelect !== undefined &&
      "cursor-pointer hover:border-[#4d4d4d] focus-visible:ring-2 focus-visible:ring-ring",
  );
  const style = {
    left: node.x,
    top: node.y,
    width: node.width,
    height: NODE_HEIGHT,
    ...((family || node.dashed === true) && color !== undefined
      ? {
          // Fraco à espera, mais forte depois de trabalhar, inteiro a trabalhar ou aberto.
          // Um contorno tracejado tem metade da tinta: vai mais forte, para se ver como os outros.
          borderColor:
            active || selected
              ? color
              : withAlpha(color, (state === "done" ? 0.6 : 0.3) + (node.dashed === true ? 0.3 : 0)),
          boxShadow: selected
            ? `inset 0 0 0 1px ${color}`
            : active
              ? `0 0 20px -10px ${color}`
              : "none",
        }
      : {}),
    ...(lit === undefined ? {} : { borderColor: lit, boxShadow: `0 0 18px -4px ${lit}` }),
  };
  const content = (
    <>
      {Icon !== undefined && (
        <span
          aria-hidden="true"
          className="flex size-7 shrink-0 items-center justify-center rounded-lg border border-[#2e2e2e] bg-[#1f1f1f]"
        >
          <Icon
            className="size-3.5"
            style={{ color: color ?? (state === "idle" ? "var(--text-dim)" : "var(--text)") }}
          />
        </span>
      )}
      <span className="flex min-w-0 flex-col">
        <span
          className={cn(
            "truncate text-[14px] font-medium",
            state === "idle" ? "text-soft" : "text-strong",
          )}
        >
          {node.title}
        </span>
        <span className="truncate font-mono text-[12px] text-dim">{detail}</span>
      </span>
      {active && (
        <span
          aria-hidden="true"
          className="absolute top-2 right-2 size-1.5 animate-pulse rounded-full bg-white"
        />
      )}
      {node.progress !== null && (
        <span aria-hidden="true" className="absolute inset-x-3 bottom-1.5 h-0.5 rounded bg-[#333]">
          <motion.span
            className="block h-full rounded"
            style={{ background: color ?? "var(--now)" }}
            initial={false}
            animate={{ width: `${String(Math.round(Math.min(1, node.progress) * 100))}%` }}
            transition={{ duration: reduced ? 0 : 0.3 }}
          />
        </span>
      )}
    </>
  );
  const entrance = reduced
    ? {}
    : {
        initial: { opacity: 0, y: 8 },
        animate: { opacity: 1, y: 0 },
        transition: { duration: 0.4, delay: 0.12 * node.column, ease: "easeOut" as const },
      };
  if (onSelect === undefined) {
    return (
      <motion.div className={className} style={style} {...entrance}>
        {content}
      </motion.div>
    );
  }
  return (
    <motion.button
      type="button"
      className={className}
      style={style}
      aria-haspopup="dialog"
      data-flow-node=""
      aria-label={`${node.title}: ${node.detail[0] ?? ""}. See what it did`}
      onClick={() => {
        onSelect(node.key);
      }}
      {...entrance}
    >
      {content}
    </motion.button>
  );
}

/**
 * Uma ligação curva. Desenha-se ao abrir; a ativa tem uma luz que corre na direção do trabalho,
 * e as dos agentes que já trabalharam uma luz lenta, de vez em quando (as classes `comet` e
 * `comet-slow` de `globals.css`: a luz é um traço curto numa ligação de comprimento 1).
 */
function Edge({ edge }: { edge: PlacedEdge }) {
  const reduced = useReducedMotion() ?? false;
  const [x1, y1] = edge.from;
  const [x2, y2] = edge.to;
  const bend = (x2 - x1) / 2;
  const path = `M${String(x1)},${String(y1)} C${String(x1 + bend)},${String(y1)} ${String(x2 - bend)},${String(y2)} ${String(x2)},${String(y2)}`;
  const tone = edge.color ?? "#fafafa";
  const idle = edge.state === "idle";
  return (
    <g>
      <motion.path
        d={path}
        fill="none"
        stroke={
          idle ? "var(--line-strong)" : edge.state === "done" ? (edge.color ?? "#5c5c5c") : tone
        }
        strokeOpacity={idle ? 0.9 : edge.state === "done" ? 0.5 : 0.9}
        strokeWidth={idle ? 1 : 1.5}
        initial={reduced ? false : { pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 0.6, delay: 0.12 * edge.column, ease: "easeOut" }}
      />
      {!idle && !reduced && (
        <path
          d={path}
          pathLength={1}
          fill="none"
          stroke={tone}
          strokeWidth={edge.state === "active" ? 2.5 : 2}
          strokeLinecap="round"
          className={edge.state === "active" ? "comet" : "comet-slow"}
          style={{
            filter: `drop-shadow(0 0 5px ${tone})`,
            animationDelay: edge.state === "active" ? undefined : `${String(edge.order * 0.7)}s`,
          }}
        />
      )}
    </g>
  );
}

type Point = readonly [number, number];

/** Uma curva como as das ligações: sai na horizontal e chega na horizontal. */
function curve(from: Point, to: Point): string {
  const [x1, y1] = from;
  const [x2, y2] = to;
  const bend = (x2 - x1) / 2;
  return `C${String(x1 + bend)},${String(y1)} ${String(x2 - bend)},${String(y2)} ${String(x2)},${String(y2)}`;
}

/**
 * O caminho de uma troca, de um nó para outro. Vai pelas ligações quando as há (ao contrário,
 * numa resposta); a decisão e o teste vão pelo agente do vencedor (`via`), atravessando-o; o
 * resto faz um arco por cima dos dois nós.
 */
function route(
  from: FlowNode,
  to: FlowNode,
  via: ModelFamily | "ai" | null,
  edges: readonly PlacedEdge[],
  nodes: readonly PlacedNode[],
): string | null {
  const byKey = new Map(edges.map((edge) => [edge.key, edge]));
  const legs: [Point, Point][] = [];
  const follow = (key: string, backwards = false) => {
    const edge = byKey.get(key);
    if (edge !== undefined) {
      legs.push(backwards ? [edge.to, edge.from] : [edge.from, edge.to]);
    }
  };
  const fixed = ["data", "coordinator", "winner", "test"];
  const family = (node: FlowNode) => !fixed.includes(node);
  if (from === "data" && to === "coordinator") {
    follow("dados");
  } else if (from === "coordinator" && to === "data") {
    follow("dados", true);
  } else if (from === "coordinator" && family(to)) {
    follow(`para-${to}`);
  } else if (family(from) && to === "coordinator") {
    follow(`para-${from}`, true);
  } else if (family(from) && to === "winner") {
    follow(`de-${from}`);
  } else if (from === "winner" && to === "test") {
    follow("teste");
  } else if (from === "coordinator" && (to === "winner" || to === "test") && via !== null) {
    follow(`para-${via}`);
    follow(`de-${via}`);
    if (to === "test") {
      follow("teste");
    }
  }
  if (legs.length > 0) {
    return legs
      .map(
        ([start, end], index) =>
          // Entre duas ligações, a luz atravessa o nó em linha reta.
          `${index === 0 ? "M" : "L"}${String(start[0])},${String(start[1])} ${curve(start, end)}`,
      )
      .join(" ");
  }
  const source = nodes.find((node) => node.key === from);
  const target = nodes.find((node) => node.key === to);
  if (source === undefined || target === undefined) {
    return null;
  }
  const [x1, y1] = [source.x + source.width / 2, source.y];
  const [x2, y2] = [target.x + target.width / 2, target.y];
  const top = Math.min(y1, y2) - 70;
  return `M${String(x1)},${String(y1)} C${String(x1)},${String(top)} ${String(x2)},${String(top)} ${String(x2)},${String(y2)}`;
}

/** O rasto de uma troca: o caminho desenha-se à velocidade da luz que o percorre. */
function Trail({ path, color, duration }: { path: string; color: string; duration: number }) {
  const reduced = useReducedMotion() ?? false;
  return (
    <motion.path
      d={path}
      fill="none"
      stroke={color}
      strokeWidth={2.5}
      strokeLinecap="round"
      strokeOpacity={0.85}
      style={{ filter: `drop-shadow(0 0 6px ${color})` }}
      initial={reduced ? false : { pathLength: 0 }}
      animate={{ pathLength: 1 }}
      transition={{ duration, ease: [0.4, 0, 0.2, 1] }}
    />
  );
}

/**
 * A luz de uma troca: um ponto que percorre o caminho (`animateMotion` do SVG, que segue o
 * caminho tal como foi desenhado). Começa quando aparece; com "reduzir movimento" não há luz,
 * só o rasto.
 */
function Pulse({ path, color, duration }: { path: string; color: string; duration: number }) {
  const reduced = useReducedMotion() ?? false;
  const motionRef = useRef<SVGAnimateMotionElement>(null);
  useLayoutEffect(() => {
    // Um `animateMotion` acrescentado depois de a página abrir começa só quando se pede.
    motionRef.current?.beginElement();
  }, []);
  if (reduced) {
    return null;
  }
  return (
    <circle
      r={5}
      fill="#fafafa"
      stroke={color}
      strokeWidth={2}
      style={{ filter: `drop-shadow(0 0 6px ${color}) drop-shadow(0 0 14px ${color})` }}
    >
      <animateMotion
        ref={motionRef}
        path={path}
        dur={`${String(duration)}s`}
        begin="indefinite"
        fill="freeze"
        calcMode="spline"
        keyTimes="0;1"
        keySplines="0.4 0 0.2 1"
      />
    </circle>
  );
}

function clip(text: string, length: number): string {
  return text.length > length ? `${text.slice(0, Math.max(1, length - 1))}…` : text;
}

/** A versão mais longa que cabe em `length` letras; se nenhuma couber, a mais curta, cortada. */
function fit(versions: readonly string[], length: number): string {
  return (
    versions.find((version) => version.length <= length) ?? clip(versions.at(-1) ?? "", length)
  );
}
