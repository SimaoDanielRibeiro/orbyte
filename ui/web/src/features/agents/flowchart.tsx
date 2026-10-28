/**
 * O fluxograma dos agentes, ao vivo: os dados passam ao coordenador, que distribui os
 * candidatos pelos agentes de cada família; o melhor é o vencedor, que se testa uma só vez.
 *
 * Só o que está a acontecer agora fica a amarelo (o giz amarelo marca o agora, ADR 0029): o
 * agente de dados a decidir, o coordenador a escolher o próximo candidato, um agente a medir um
 * candidato (com o fold em que vai) ou o vencedor a treinar. O que já passou fica a giz; o que
 * falta, apagado. A ligação ativa corre; com "reduzir movimento", fica só pintada.
 */
import { motion, useReducedMotion } from "motion/react";

import type {
  Candidate,
  MetricResult,
  ModelAgentCard,
  ModelFamily,
  ObjectiveOption,
  SearchPhase,
  WorkInProgress,
} from "@/api/generated";
import { useLabel } from "@/api/vocabulary";
import { useSize } from "@/hooks/use-size";
import { formatCount, formatMetric, formatQuantity } from "@/lib/format";

import { objectiveValue } from "./values";

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
}

type NodeState = "idle" | "active" | "done";

/** A largura para que as colunas foram pensadas; o desenho estica até à largura que tem. */
const DESIGN_WIDTH = 960;
/** Abaixo disto, os nomes não cabem: o fluxograma rola na horizontal. */
const MIN_WIDTH = 680;
const NODE_HEIGHT = 56;
const ROW_GAP = 10;
const PADDING = 24;
const DESIGN_COLUMNS = {
  data: { x: 12, width: 140 },
  coordinator: { x: 184, width: 140 },
  agents: { x: 368, width: 250 },
  winner: { x: 662, width: 150 },
  test: { x: 846, width: 104 },
} as const;

type Columns = Record<keyof typeof DESIGN_COLUMNS, { x: number; width: number }>;

/** As colunas na largura que há (o texto fica sempre do mesmo tamanho). */
function columnsFor(width: number): Columns {
  const scale = width / DESIGN_WIDTH;
  return Object.fromEntries(
    Object.entries(DESIGN_COLUMNS).map(([key, column]) => [
      key,
      { x: column.x * scale, width: column.width * scale },
    ]),
  ) as Columns;
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
}: FlowchartProps) {
  const label = useLabel();
  const { ref, width: available } = useSize<HTMLDivElement>();
  const width = Math.max(MIN_WIDTH, available);
  const COLUMNS = columnsFor(width);
  const rows = Math.max(1, agents.length);
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
  const agentState = (family: ModelFamily): NodeState => {
    if (searching && working?.family === family) {
      return "active";
    }
    return candidates.some((candidate) => candidate.family === family) ? "done" : "idle";
  };
  const best = (family: ModelFamily) => {
    const own = candidates.filter(
      (candidate) => candidate.family === family && candidate.mean !== null,
    );
    const values = own.map((candidate) => candidate.mean ?? 0);
    if (values.length === 0) {
      return null;
    }
    return objective?.higher_is_better === false ? Math.min(...values) : Math.max(...values);
  };
  const measured = candidates.length;
  const now = describeNow(phase, working, label);

  return (
    <figure className="flex flex-col gap-3" data-phase={phase ?? "none"}>
      <div ref={ref} className="relative overflow-x-auto">
        <svg
          width={width}
          height={height}
          className="block"
          role="img"
          aria-label={`O fluxograma dos agentes. ${now}`}
        >
          {/* As ligações primeiro, para ficarem por baixo dos nós. */}
          <Edge
            from={[COLUMNS.data.x + COLUMNS.data.width, middle]}
            to={[COLUMNS.coordinator.x, middle]}
            state={states.data === "done" ? "done" : states.data === "active" ? "active" : "idle"}
          />
          {agents.map((agent, index) => {
            const top = agentTop(index) + NODE_HEIGHT / 2;
            const state = agentState(agent.family);
            return (
              <g key={`ligacoes-${agent.family}`}>
                <Edge
                  from={[COLUMNS.coordinator.x + COLUMNS.coordinator.width, middle]}
                  to={[COLUMNS.agents.x, top]}
                  state={state}
                />
                <Edge
                  from={[COLUMNS.agents.x + COLUMNS.agents.width, top]}
                  to={[COLUMNS.winner.x, middle]}
                  state={
                    leader?.family === agent.family && order(phase) >= 3
                      ? states.winner === "active"
                        ? "active"
                        : "done"
                      : "idle"
                  }
                />
              </g>
            );
          })}
          <Edge
            from={[COLUMNS.winner.x + COLUMNS.winner.width, middle]}
            to={[COLUMNS.test.x, middle]}
            state={states.test === "done" ? "done" : states.winner === "active" ? "active" : "idle"}
          />

          <Node
            x={COLUMNS.data.x}
            y={middle - NODE_HEIGHT / 2}
            width={COLUMNS.data.width}
            state={states.data}
            title="Dados"
            detail={
              phase === "data" || trainRows === null
                ? ["A preparar"]
                : [`${formatCount(trainRows)} linhas de treino`, `${formatCount(trainRows)} linhas`]
            }
          />
          <Node
            x={COLUMNS.coordinator.x}
            y={middle - NODE_HEIGHT / 2}
            width={COLUMNS.coordinator.width}
            state={states.coordinator}
            title={label("AgentRole", "coordinator")}
            detail={
              budget === null
                ? [formatQuantity(measured, "candidato", "candidatos")]
                : [
                    `${formatCount(measured)} de ${formatCount(budget)} candidatos`,
                    `${formatCount(measured)} de ${formatCount(budget)} medidos`,
                    `${formatCount(measured)} de ${formatCount(budget)}`,
                  ]
            }
          />
          {agents.map((agent, index) => {
            const state = agentState(agent.family);
            const own = candidates.filter((candidate) => candidate.family === agent.family);
            const value = best(agent.family);
            const active = state === "active" && working !== null;
            return (
              <Node
                key={agent.family}
                x={COLUMNS.agents.x}
                y={agentTop(index)}
                width={COLUMNS.agents.width}
                state={state}
                title={label("ModelFamily", agent.family)}
                detail={
                  active
                    ? [
                        `${working.model_display_name}, fold ${formatCount(working.fold)} de ${formatCount(working.folds)}`,
                        `Fold ${formatCount(working.fold)} de ${formatCount(working.folds)}, ${working.model_display_name}`,
                      ]
                    : own.length === 0
                      ? ["À espera"]
                      : [
                          `${formatQuantity(own.length, "candidato", "candidatos")}, melhor ${objectiveValue(value, objective)}`,
                          `Melhor ${objectiveValue(value, objective)} em ${formatCount(own.length)}`,
                        ]
                }
                progress={active ? working.fold / Math.max(1, working.folds) : null}
              />
            );
          })}
          <Node
            x={COLUMNS.winner.x}
            y={middle - NODE_HEIGHT / 2}
            width={COLUMNS.winner.width}
            state={states.winner}
            title={phase === "done" || phase === "final" ? "Vencedor" : "Líder"}
            detail={
              leader === null
                ? ["Ainda nenhum"]
                : [
                    `${objectiveValue(leader.mean, objective)}, ${leader.model_display_name}`,
                    objectiveValue(leader.mean, objective),
                  ]
            }
          />
          <Node
            x={COLUMNS.test.x}
            y={middle - NODE_HEIGHT / 2}
            width={COLUMNS.test.width}
            state={states.test}
            title="Teste"
            detail={[test === null ? "Só no fim" : formatMetric(test)]}
          />
        </svg>
      </div>
      <figcaption className="sr-only" aria-live="polite">
        {now}
      </figcaption>
    </figure>
  );
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
  label: ReturnType<typeof useLabel>,
): string {
  if (phase === null) {
    return "A procura ainda não começou.";
  }
  if (working !== null) {
    return `Agora: ${working.author} a medir ${working.model_display_name}, fold ${String(working.fold)} de ${String(working.folds)}.`;
  }
  return `Agora: ${label("SearchPhase", phase)}.`;
}

interface NodeProps {
  x: number;
  y: number;
  width: number;
  state: NodeState;
  title: string;
  /** O que o nó diz por baixo do nome, da versão mais longa à mais curta: fica a que cabe. */
  detail: readonly string[];
  /** A parte do trabalho feita (o fold), numa barra em baixo. */
  progress?: number | null;
}

function Node({ x, y, width, state, title, detail, progress = null }: NodeProps) {
  const reduced = useReducedMotion() ?? false;
  const active = state === "active";
  return (
    <g data-state={state}>
      <rect
        x={x}
        y={y}
        width={width}
        height={NODE_HEIGHT}
        rx={8}
        fill={active ? "var(--secondary)" : "var(--card)"}
        stroke={
          active ? "var(--now)" : state === "done" ? "var(--muted-foreground)" : "var(--input)"
        }
        strokeWidth={active ? 2 : 1}
        strokeOpacity={state === "idle" ? 0.6 : 1}
      />
      {active && (
        <motion.circle
          cx={x + width - 12}
          cy={y + 12}
          r={4}
          fill="var(--now)"
          animate={reduced ? undefined : { opacity: [1, 0.25, 1] }}
          transition={reduced ? undefined : { duration: 1.2, repeat: Infinity }}
        />
      )}
      <text
        x={x + 10}
        y={y + 23}
        className={state === "idle" ? "fill-muted-foreground" : "fill-foreground"}
        style={{ fontSize: 13, fontWeight: 500 }}
      >
        {/* O texto não se mede: estima-se a largura média de uma letra em cada tamanho. */}
        {clip(title, Math.floor((width - 20) / 6.9))}
      </text>
      <text x={x + 10} y={y + 41} className="fill-muted-foreground" style={{ fontSize: 11.5 }}>
        {fit(detail, Math.floor((width - 20) / 5.9))}
      </text>
      {progress !== null && (
        <g>
          <rect
            x={x + 1}
            y={y + NODE_HEIGHT - 4}
            width={width - 2}
            height={3}
            rx={1.5}
            fill="var(--muted)"
          />
          <motion.rect
            x={x + 1}
            y={y + NODE_HEIGHT - 4}
            height={3}
            rx={1.5}
            fill="var(--now)"
            initial={false}
            animate={{ width: Math.max(0, (width - 2) * Math.min(1, progress)) }}
            transition={{ duration: reduced ? 0 : 0.3 }}
          />
        </g>
      )}
    </g>
  );
}

interface EdgeProps {
  from: [number, number];
  to: [number, number];
  state: NodeState;
}

/** Uma ligação curva; a ativa corre na direção do trabalho. */
function Edge({ from, to, state }: EdgeProps) {
  const reduced = useReducedMotion() ?? false;
  const [x1, y1] = from;
  const [x2, y2] = to;
  const bend = (x2 - x1) / 2;
  const path = `M${String(x1)},${String(y1)} C${String(x1 + bend)},${String(y1)} ${String(x2 - bend)},${String(y2)} ${String(x2)},${String(y2)}`;
  if (state === "active") {
    return (
      <motion.path
        d={path}
        fill="none"
        stroke="var(--now)"
        strokeWidth={2}
        strokeDasharray="6 6"
        animate={reduced ? undefined : { strokeDashoffset: [12, 0] }}
        transition={reduced ? undefined : { duration: 0.6, repeat: Infinity, ease: "linear" }}
      />
    );
  }
  return (
    <path
      d={path}
      fill="none"
      stroke={state === "done" ? "var(--muted-foreground)" : "var(--input)"}
      strokeOpacity={state === "done" ? 0.8 : 0.45}
      strokeWidth={1}
    />
  );
}

function clip(text: string, length: number): string {
  return text.length > length ? `${text.slice(0, length - 1)}…` : text;
}

/** A versão mais longa que cabe em `length` letras; se nenhuma couber, a mais curta, cortada. */
function fit(versions: readonly string[], length: number): string {
  return (
    versions.find((version) => version.length <= length) ?? clip(versions.at(-1) ?? "", length)
  );
}
