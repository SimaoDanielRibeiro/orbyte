/**
 * "Como um agente aprende": a lição que abre o ecrã Reinforcement learning (ADR 0037).
 *
 * Um Q-learning a aprender a atravessar o penhasco, episódio a episódio e passo a passo, com os
 * dados do backend (`GET /reinforcement/lesson`, que treina o agente a sério): a interface só os
 * anima.
 *
 * - Em cima, os episódios escolhidos (o primeiro, uma queda, o primeiro êxito, um do meio e o
 *   último), e a frase do backend sobre o que se vê no episódio aberto.
 * - No meio, a grelha: o agente anda, a recompensa salta da célula onde ele chega, e a cor de
 *   cada célula muda quando o agente aprende com ela. Episódio a episódio, vê-se o valor do
 *   objetivo a espalhar-se para trás.
 * - Em baixo, o ciclo de cada passo em quatro cartões, que acendem à vez: o estado, a ação (ao
 *   acaso ou a melhor que conhece), a recompensa e a correção do Q. Os textos são temas do
 *   backend; por baixo de cada um, o que aconteceu neste passo.
 *
 * Começa sozinha quando aparece no ecrã e pára quando sai dele. Com "reduzir movimento" não
 * começa sozinha, e o agente salta de célula em célula, sem deslizar.
 */
import { useQuery } from "@tanstack/react-query";
import { cn } from "cn";
import {
  Compass,
  Dices,
  Gift,
  MapPin,
  Pause,
  Play,
  RefreshCw,
  RotateCcw,
  type LucideIcon,
} from "lucide-react";
import { AnimatePresence, motion, useInView, useReducedMotion } from "motion/react";
import { useRef, type ReactNode } from "react";

import type { GridLayout, LessonEpisode, LessonStep } from "@/api/generated";
import { reinforcementLessonOptions } from "@/api/generated/@tanstack/react-query.gen";
import { useTopic } from "@/api/topics";
import { cellFill } from "@/components/charts/grid-fill";
import { CellSymbol } from "@/components/charts/grid-world";
import { Explain } from "@/components/explain/explain";
import { Prose } from "@/components/markdown/prose";
import { ErrorNote } from "@/components/shell/blocks";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { useSize } from "@/hooks/use-size";
import { CATEGORICAL, diverging, LIGHT_INK, SURFACE } from "@/lib/colors";
import { formatCount, formatNumber } from "@/lib/format";

import { useLessonClock, type Speed } from "./use-lesson-clock";

/** As quatro fases de cada passo, pela ordem do ciclo. */
const STAGES = [
  { topic: "rl_state", icon: MapPin },
  { topic: "rl_action", icon: Compass },
  { topic: "rl_reward", icon: Gift },
  { topic: "rl_update", icon: RefreshCw },
] as const satisfies readonly { topic: string; icon: LucideIcon }[];

const SPEEDS: readonly Speed[] = [1, 2, 4];

/** O lado máximo de uma célula da grelha, em px. */
const MAX_CELL = 64;

export function RlLesson() {
  const lesson = useQuery({ ...reinforcementLessonOptions(), staleTime: "static" });
  return (
    <section aria-labelledby="licao-titulo" className="mt-10">
      <div className="mb-6 flex flex-wrap items-center gap-x-3 gap-y-1">
        <div className="flex items-center gap-1.5">
          <h2 id="licao-titulo" className="text-xl">
            Como um agente aprende
          </h2>
          <Explain topic="reinforcement_learning" />
        </div>
        {lesson.data !== undefined && (
          <span className="font-mono text-xs text-dim">
            {lesson.data.agent_display_name} em {lesson.data.environment_name}, ao longo de{" "}
            {formatCount(lesson.data.episodes)} episódios
          </span>
        )}
      </div>
      <ErrorNote error={lesson.error} />
      {lesson.data === undefined ? (
        lesson.error === null && <Skeleton className="h-96 w-full rounded-2xl" />
      ) : (
        <Player
          layout={lesson.data.layout}
          kept={lesson.data.kept}
          learningRate={lesson.data.learning_rate}
        />
      )}
    </section>
  );
}

interface PlayerProps {
  layout: GridLayout;
  kept: readonly LessonEpisode[];
  learningRate: number;
}

function Player({ layout, kept, learningRate }: PlayerProps) {
  const box = useRef<HTMLDivElement>(null);
  const inView = useInView(box, { amount: 0.35 });
  const reduced = useReducedMotion() ?? false;
  const clock = useLessonClock(kept, { visible: inView, autoPlay: !reduced });
  const episode = kept[clock.episode];
  if (episode === undefined) {
    return null;
  }
  const { steps } = episode;
  const step = steps[clock.step];

  return (
    <div ref={box} className="flex flex-col gap-6 panel p-5 md:p-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <EpisodePicker
          kept={kept}
          layout={layout}
          current={clock.episode}
          onPick={clock.goToEpisode}
        />
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="icon-sm"
            aria-label={
              clock.playing
                ? "Pausar a lição"
                : clock.atEnd
                  ? "Ver a lição outra vez"
                  : "Reproduzir a lição"
            }
            onClick={clock.toggle}
          >
            {clock.playing ? <Pause /> : clock.atEnd ? <RotateCcw /> : <Play />}
          </Button>
          <ToggleGroup
            type="single"
            variant="outline"
            size="sm"
            value={String(clock.speed)}
            onValueChange={(value) => {
              if (value !== "") {
                clock.setSpeed(Number(value) as Speed);
              }
            }}
            aria-label="Velocidade da lição"
          >
            {SPEEDS.map((speed) => (
              <ToggleGroupItem key={speed} value={String(speed)} className="px-2.5 font-mono">
                {speed}×
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </div>
      </div>

      <p className="max-w-[80ch] text-[0.9375rem] leading-relaxed text-soft" aria-live="polite">
        <span className="mr-2 font-mono text-xs text-dim">
          Episódio {formatCount(episode.episode)}
        </span>
        {episode.story}
      </p>

      <LessonGrid
        layout={layout}
        episode={episode}
        stepIndex={clock.step}
        phase={clock.phase}
        finished={clock.finished}
        episodeKey={clock.episode}
        reduced={reduced}
      />

      <div className="flex flex-col gap-3">
        <LessonLegend />
        <div className="flex items-center gap-3">
          <span className="h-1 flex-1 overflow-hidden rounded-full bg-[#262626]">
            <span
              className="block h-full rounded-full bg-strong transition-[width] duration-200"
              style={{
                width: `${String(((clock.finished ? steps.length : clock.step + 1) / Math.max(1, steps.length)) * 100)}%`,
              }}
            />
          </span>
          <span className="font-mono text-xs text-dim tabular-nums">
            passo {formatCount(Math.min(steps.length, clock.step + 1))} de{" "}
            {formatCount(steps.length)}
          </span>
        </div>
      </div>

      {step !== undefined && (
        <Cycle
          layout={layout}
          step={step}
          phase={clock.finished ? null : clock.phase}
          learningRate={learningRate}
        />
      )}
    </div>
  );
}

/** O que as cores e as setas querem dizer. */
function LessonLegend() {
  return (
    <ul className="flex flex-wrap items-center gap-x-6 gap-y-1.5 text-xs text-soft">
      <li className="flex items-center gap-1.5">
        <span className="font-mono text-dim tabular-nums">−1</span>
        <span
          aria-hidden="true"
          className="h-2 w-20 rounded-[2px]"
          style={{
            background: `linear-gradient(to right, ${diverging(-1)}, ${diverging(0)}, ${diverging(1)})`,
          }}
        />
        <span className="font-mono text-dim tabular-nums">1</span>
        <span>o valor de cada célula</span>
      </li>
      <li className="flex items-center gap-2">
        <svg aria-hidden="true" width={22} height={8} className="shrink-0">
          <line
            x1={1}
            y1={4}
            x2={21}
            y2={4}
            stroke={CATEGORICAL[1]}
            strokeWidth={2}
            strokeDasharray="4 3"
          />
        </svg>
        ao acaso (explorar)
      </li>
      <li className="flex items-center gap-2">
        <svg aria-hidden="true" width={22} height={8} className="shrink-0">
          <line x1={1} y1={4} x2={21} y2={4} stroke={LIGHT_INK} strokeWidth={2} />
        </svg>
        a melhor que conhece (aproveitar)
      </li>
    </ul>
  );
}

/** Os episódios guardados, como separadores: o número e o que acontece nele. */
function EpisodePicker({
  kept,
  layout,
  current,
  onPick,
}: {
  kept: readonly LessonEpisode[];
  layout: GridLayout;
  current: number;
  onPick: (index: number) => void;
}) {
  const firstSuccess = kept.findIndex((episode) => episode.reached);
  return (
    <ToggleGroup
      type="single"
      value={String(current)}
      onValueChange={(value) => {
        if (value !== "") {
          onPick(Number(value));
        }
      }}
      aria-label="Episódio da lição"
      className="flex flex-wrap justify-start gap-1.5"
    >
      {kept.map((episode, index) => {
        const last = episode.steps.at(-1);
        const fell = last !== undefined && layout.cells[last.next_state] === "trap";
        const tag =
          index === firstSuccess
            ? "1.º objetivo"
            : index === kept.length - 1 && episode.reached
              ? "aprendeu"
              : episode.reached
                ? "objetivo"
                : fell
                  ? "cai"
                  : "às voltas";
        return (
          <ToggleGroupItem
            key={episode.episode}
            value={String(index)}
            className="h-8 gap-2 rounded-full border border-[#2a2a2a] px-3 text-[0.8125rem] data-[state=on]:border-[#4a4a4a] data-[state=on]:bg-surface-high"
          >
            <span
              aria-hidden="true"
              className="size-1.5 rounded-full"
              style={{ background: episode.reached ? diverging(1) : diverging(-1) }}
            />
            Episódio {formatCount(episode.episode)}
            <span className="font-mono text-[0.6875rem] text-dim">{tag}</span>
          </ToggleGroupItem>
        );
      })}
    </ToggleGroup>
  );
}

interface LessonGridProps {
  layout: GridLayout;
  episode: LessonEpisode;
  stepIndex: number;
  phase: number;
  finished: boolean;
  /** Muda a cada episódio: o agente volta à partida sem deslizar. */
  episodeKey: number;
  reduced: boolean;
}

/**
 * A grelha do episódio: a cor de cada célula é o valor que o agente lhe dá (a mesma escala do
 * resto do ecrã), o agente é o ponto branco, e o rasto é o caminho deste episódio.
 */
function LessonGrid({
  layout,
  episode,
  stepIndex,
  phase,
  finished,
  episodeKey,
  reduced,
}: LessonGridProps) {
  const { ref, width } = useSize<HTMLDivElement>();
  const cell = Math.max(24, Math.min(MAX_CELL, Math.floor(width / layout.columns)));
  const gridWidth = cell * layout.columns;
  const gridHeight = cell * layout.rows;
  const { steps } = episode;
  const step = steps[stepIndex];
  const center = (state: number): [number, number] => [
    (state % layout.columns) * cell + cell / 2,
    Math.floor(state / layout.columns) * cell + cell / 2,
  ];

  // Os valores à vista: os do início do episódio, com o que o agente já aprendeu nele.
  const values = [...episode.before];
  const learned = finished ? steps.length : stepIndex + (phase >= 3 ? 1 : 0);
  for (const done of steps.slice(0, learned)) {
    values[done.state] = done.value;
  }
  const moved = finished ? steps.length : stepIndex + (phase >= 2 ? 1 : 0);
  const visited = [steps[0]?.state ?? 0, ...steps.slice(0, moved).map((one) => one.next_state)];
  const here = visited.at(-1) ?? 0;
  const [agentX, agentY] = center(here);
  const trail = visited.map((state) => center(state).join(",")).join(" L");
  const glide = { duration: reduced ? 0 : 0.22, ease: "easeInOut" as const };

  return (
    <div ref={ref} className="w-full">
      {width > 0 && (
        <svg
          width={gridWidth}
          height={gridHeight}
          role="img"
          aria-label={`A grelha do episódio ${String(episode.episode)}: o agente, o caminho que fez até agora e o valor que dá a cada célula.`}
          className="mx-auto block max-w-full"
        >
          {layout.cells.map((kind, state) => {
            const [cx, cy] = center(state);
            return (
              <g key={state}>
                <rect
                  x={cx - cell / 2 + 1}
                  y={cy - cell / 2 + 1}
                  width={cell - 2}
                  height={cell - 2}
                  rx={4}
                  fill={cellFill(kind, values[state] ?? null)}
                  style={{ transition: reduced ? undefined : "fill 300ms ease" }}
                />
                <CellSymbol kind={kind} cx={cx} cy={cy} size={cell} />
              </g>
            );
          })}
          {/* A célula que o agente acabou de corrigir acende por um instante. */}
          {step !== undefined && phase >= 3 && !finished && (
            <motion.rect
              key={`corrige-${String(episodeKey)}-${String(stepIndex)}`}
              x={center(step.state)[0] - cell / 2 + 1}
              y={center(step.state)[1] - cell / 2 + 1}
              width={cell - 2}
              height={cell - 2}
              rx={4}
              fill="none"
              stroke={LIGHT_INK}
              strokeWidth={2}
              initial={{ opacity: 1 }}
              animate={{ opacity: 0 }}
              transition={{ duration: reduced ? 0 : 0.6 }}
            />
          )}
          {visited.length > 1 && (
            <path
              d={`M${trail}`}
              fill="none"
              stroke={LIGHT_INK}
              strokeOpacity={0.4}
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
              pointerEvents="none"
            />
          )}
          {step !== undefined && phase === 1 && !finished && (
            <ActionArrow layout={layout} step={step} cell={cell} center={center(step.state)} />
          )}
          <motion.circle
            key={`agente-${String(episodeKey)}`}
            r={cell * 0.2}
            fill={LIGHT_INK}
            stroke={SURFACE}
            strokeWidth={3}
            initial={false}
            animate={{ cx: agentX, cy: agentY }}
            transition={glide}
            style={{ filter: "drop-shadow(0 0 6px rgb(255 255 255 / 0.55))" }}
          />
          <AnimatePresence>
            {step !== undefined && phase >= 2 && !finished && (
              <RewardPop
                key={`recompensa-${String(episodeKey)}-${String(stepIndex)}`}
                step={step}
                at={center(step.next_state)}
                cell={cell}
                reduced={reduced}
              />
            )}
          </AnimatePresence>
        </svg>
      )}
    </div>
  );
}

/** A seta da ação escolhida: amarela e tracejada quando é ao acaso, branca quando é a melhor. */
function ActionArrow({
  layout,
  step,
  cell,
  center: [cx, cy],
}: {
  layout: GridLayout;
  step: LessonStep;
  cell: number;
  center: [number, number];
}) {
  const [rows = 0, columns = 0] = layout.action_moves[step.action] ?? [];
  const length = cell * 0.62;
  const [tipX, tipY] = [cx + columns * length, cy + rows * length];
  const color = step.explored ? CATEGORICAL[1] : LIGHT_INK;
  const head = cell * 0.12;
  return (
    <g pointerEvents="none" stroke={color} fill={color} strokeLinecap="round">
      <line
        x1={cx}
        y1={cy}
        x2={tipX}
        y2={tipY}
        strokeWidth={2.5}
        strokeDasharray={step.explored ? "4 4" : undefined}
      />
      <path
        d={`M${String(tipX)},${String(tipY)}L${String(tipX - columns * head - rows * head)},${String(tipY - rows * head + columns * head)}L${String(tipX - columns * head + rows * head)},${String(tipY - rows * head - columns * head)}Z`}
      />
    </g>
  );
}

/** A recompensa do passo, a saltar da célula onde o agente chegou. */
function RewardPop({
  step,
  at: [x, y],
  cell,
  reduced,
}: {
  step: LessonStep;
  at: [number, number];
  cell: number;
  reduced: boolean;
}) {
  const big = Math.abs(step.reward) >= 0.5;
  return (
    <motion.text
      x={x}
      y={y - cell * 0.32}
      textAnchor="middle"
      fontSize={big ? 15 : 11}
      fontWeight={big ? 600 : 400}
      fill={big ? diverging(Math.sign(step.reward)) : "#bdbdbd"}
      stroke={SURFACE}
      strokeWidth={3}
      paintOrder="stroke"
      className="font-mono"
      pointerEvents="none"
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: reduced ? 0 : 0.25 }}
    >
      {signed(step.reward, big ? 0 : 2)}
    </motion.text>
  );
}

/**
 * O ciclo de um passo, em quatro cartões: o tema do backend em cada um e, por baixo, o que
 * aconteceu neste passo. O cartão da fase de agora acende.
 */
function Cycle({
  layout,
  step,
  phase,
  learningRate,
}: {
  layout: GridLayout;
  step: LessonStep;
  /** A fase de agora (de 0 a 3), ou `null` no fim do episódio. */
  phase: number | null;
  learningRate: number;
}) {
  const row = Math.floor(step.state / layout.columns) + 1;
  const column = (step.state % layout.columns) + 1;
  const arrived = layout.cells[step.next_state];
  const facts: readonly ReactNode[] = [
    <>
      linha {formatCount(row)}, coluna {formatCount(column)}
      {layout.cells[step.state] === "start" ? ", a partida" : ""}
    </>,
    <>
      {layout.action_labels[step.action] ?? ""},{" "}
      {step.explored ? "ao acaso (explorar)" : "a melhor que conhece (aproveitar)"}
    </>,
    <>
      {signed(step.reward, Math.abs(step.reward) >= 0.5 ? 0 : 2)}
      {arrived === "goal"
        ? ", chegou ao objetivo"
        : arrived === "trap"
          ? ", caiu numa armadilha"
          : ""}
    </>,
    <>
      surpresa {signed(step.surprise, 3)}; o Q anda α = {formatNumber(learningRate, 1)} dela, e a
      célula fica com {signed(step.value, 3)}
    </>,
  ];
  return (
    <ol className="grid gap-3 @3xl/main:grid-cols-2 @5xl/main:grid-cols-4">
      {STAGES.map((stage, index) => (
        <Stage
          key={stage.topic}
          index={index}
          topic={stage.topic}
          icon={index === 1 && step.explored ? Dices : stage.icon}
          now={phase === index}
          fact={facts[index]}
        />
      ))}
    </ol>
  );
}

function Stage({
  index,
  topic,
  icon: Icon,
  now,
  fact,
}: {
  index: number;
  topic: string;
  icon: LucideIcon;
  now: boolean;
  fact: ReactNode;
}) {
  const explanation = useTopic(topic);
  return (
    <li
      aria-current={now ? "step" : undefined}
      className={cn(
        "flex flex-col gap-2.5 rounded-xl border p-3.5 transition-[border-color,background-color,box-shadow] duration-200",
        now
          ? "border-white/45 bg-surface-high shadow-[0_0_24px_-10px_rgb(255_255_255/0.5)]"
          : "border-[#262626] bg-ground",
      )}
    >
      <div className="flex items-center gap-2.5">
        <span
          aria-hidden="true"
          className={cn(
            "flex size-7 items-center justify-center rounded-lg border transition-colors",
            now ? "border-white/40 text-strong" : "border-[#2e2e2e] text-dim",
          )}
        >
          <Icon className="size-3.5" />
        </span>
        <span className="font-mono text-[0.6875rem] text-dim">{index + 1}</span>
        <h3 className="text-[0.9375rem] font-medium text-strong">{explanation?.title}</h3>
      </div>
      {explanation !== undefined && <Prose text={explanation.text} size="small" />}
      <p className="mt-auto font-mono text-xs leading-relaxed text-soft">{fact}</p>
    </li>
  );
}

/** Um número com o sinal à frente (o menos tipográfico), com `decimals` casas. */
function signed(value: number, decimals: number): string {
  const text = formatNumber(Math.abs(value), decimals);
  if (text === formatNumber(0, decimals)) {
    return text;
  }
  return value > 0 ? `+${text}` : `−${text}`;
}
