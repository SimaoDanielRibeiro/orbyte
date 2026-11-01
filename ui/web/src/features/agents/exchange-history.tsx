/**
 * O histórico visual das trocas: a procura repetida mensagem a mensagem, no fluxograma.
 *
 * Cada mensagem da conversa diz quem fala e a quem (`kind` e `recipients`, do backend). Aqui,
 * cada uma é uma luz que parte de quem fala e acende quem ouve: o agente de dados entrega o
 * plano ao coordenador; o coordenador anuncia as rondas aos agentes, dá a vez a um de cada vez
 * e recebe a resposta; no fim passa o melhor ao vencedor, que vai ao teste.
 *
 * O fluxograma mostra a procura como estava nesse momento: os candidatos já medidos, quem
 * liderava, o agente a quem o coordenador acabou de dar a vez. Por baixo, o que se disse, e a
 * linha do tempo (uma marca por mensagem, com a cor de quem a disse), que se arrasta com o rato
 * ou com as setas. A reprodução começa sozinha; com "reduzir movimento", também, mas sem luzes.
 */
import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";

import type {
  AgentMessage,
  AutoMlPlan,
  Candidate,
  MessageKind,
  MetricResult,
  ObjectiveOption,
  Participant,
  WorkInProgress,
} from "@/api/generated";
import { useLabel } from "@/api/vocabulary";
import { Prose } from "@/components/markdown/prose";
import { Button } from "@/components/ui/button";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { formatCount } from "@/lib/format";

import {
  AI_AGENT_COLOR,
  COORDINATOR_COLOR,
  DATA_AGENT_COLOR,
  familyColor,
  messageColor,
} from "./agent-colors";
import { Flowchart, type FlowExchange, type FlowNode } from "./flowchart";
import { bestCandidate } from "./values";

/** Quanto tempo leva a luz de uma troca, e a pausa até à seguinte (à velocidade normal). */
const TRAVEL_SECONDS = 1.1;
const PAUSE_SECONDS = 0.5;
const SPEEDS = [1, 2, 4] as const;
type Speed = (typeof SPEEDS)[number];

/** A altura da marca de cada mensagem na linha do tempo: o que decide mais, mais alto. */
const TICK_HEIGHT: Readonly<Record<MessageKind, string>> = {
  report: "55%",
  announcement: "75%",
  request: "35%",
  reply: "55%",
  decision: "90%",
  test: "90%",
  thought: "45%",
};

interface ExchangeHistoryProps {
  plan: AutoMlPlan;
  messages: readonly AgentMessage[];
  candidates: readonly Candidate[];
  objective: ObjectiveOption | null;
  /** No fim: o valor do vencedor no teste. */
  test: MetricResult | null;
  /** Se a procura já acabou (a última mensagem é a leitura do teste). */
  finished: boolean;
  /** A mensagem à vista: a conversa ao lado acompanha-a. */
  onMoment?: (message: number | null) => void;
  selected?: FlowNode | null;
  onSelect?: (node: FlowNode) => void;
}

export function ExchangeHistory({
  plan,
  messages,
  candidates,
  objective,
  test,
  finished,
  onMoment,
  selected,
  onSelect,
}: ExchangeHistoryProps) {
  const label = useLabel();
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState<Speed>(1);
  const last = messages.length - 1;
  const position = Math.max(0, Math.min(index, last));
  const current = messages[position] ?? null;
  const running = playing && position < last;

  // Uma troca de cada vez: quando a luz chega e passa a pausa, segue para a próxima.
  useEffect(() => {
    if (!running) {
      return;
    }
    const timer = window.setTimeout(
      () => {
        setIndex((value) => value + 1);
      },
      ((TRAVEL_SECONDS + PAUSE_SECONDS) / speed) * 1000,
    );
    return () => {
      window.clearTimeout(timer);
    };
  }, [running, position, speed]);

  const moment = current?.number ?? null;
  useEffect(() => {
    onMoment?.(moment);
  }, [moment, onMoment]);

  if (current === null) {
    return <p className="text-sm text-soft">Os agentes ainda não trocaram nenhuma mensagem.</p>;
  }

  // A procura como estava neste momento.
  const said = messages.slice(0, position + 1);
  const measured = Math.max(
    0,
    ...said
      .filter((message) => message.kind === "reply" && message.candidate !== null)
      .map((message) => message.candidate ?? 0),
  );
  const seen = candidates.filter((candidate) => candidate.number <= measured);
  const asked =
    current.kind === "request" && current.candidate !== null
      ? (candidates.find((candidate) => candidate.number === current.candidate) ?? null)
      : null;
  const working: WorkInProgress | null =
    asked === null
      ? null
      : {
          candidate: asked.number,
          phase: asked.phase,
          family: asked.family,
          author: asked.ai
            ? label("AgentRole", "ai")
            : (plan.agents.find((agent) => agent.family === asked.family)?.name ?? ""),
          model: asked.model,
          model_display_name: asked.model_display_name,
          changes: asked.changes,
          fold: 0,
          folds: plan.settings.folds ?? asked.folds.length,
          ai: asked.ai,
        };
  const atEnd = finished && position === last;
  // O vencedor passa pelo agente que o propôs: o da família, ou o agente de IA.
  const proposed = candidates.find((candidate) => candidate.number === current.candidate);
  const winnerFamily =
    current.candidate === null || proposed === undefined
      ? null
      : proposed.ai
        ? ("ai" as const)
        : proposed.family;
  const color = messageColor(current);
  // A leitura do teste não tem destinatários (é para quem acompanha): acende-se o teste.
  const reading = current.kind === "test" && current.recipients.length === 0;
  const exchange: FlowExchange = {
    key: current.number,
    from: reading ? "test" : senderOf(current),
    to: current.recipients.map(nodeOf),
    via: current.kind === "decision" || current.kind === "test" ? winnerFamily : null,
    color,
    duration: TRAVEL_SECONDS / speed,
  };

  const go = (target: number) => {
    setPlaying(false);
    setIndex(Math.max(0, Math.min(target, last)));
  };

  return (
    <div className="flex flex-col gap-6">
      <Flowchart
        agents={plan.agents}
        phase={atEnd ? "done" : current.phase}
        working={working}
        candidates={seen}
        objective={objective}
        leader={bestCandidate(seen, objective)}
        budget={plan.settings.budget ?? null}
        trainRows={plan.train_rows}
        test={atEnd ? test : null}
        selected={selected}
        onSelect={onSelect}
        exchange={exchange}
        ai={plan.ai_agent ? { model: plan.ai_model } : null}
      />

      <div className="grid gap-x-12 gap-y-6 @5xl/main:grid-cols-[minmax(0,1fr)_auto]">
        <div className="flex min-w-0 flex-col gap-2">
          <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
            <Who color={color} name={current.author} />
            {current.recipients.length > 0 && (
              <>
                <span aria-hidden="true" className="text-dim">
                  →
                </span>
                <span className="sr-only">para</span>
                <Recipients recipients={current.recipients} plan={plan} label={label} />
              </>
            )}
            <span className="font-mono text-xs text-dim">
              {label("MessageKind", current.kind)},{" "}
              {label("SearchPhase", current.phase).toLowerCase()}
            </span>
          </p>
          <Prose text={current.text} size="small" className="max-w-[80ch] text-foreground/90" />
        </div>
        <div className="flex items-center gap-2 self-start">
          <Button
            variant="outline"
            size="icon-sm"
            aria-label="A mensagem anterior"
            disabled={position === 0}
            onClick={() => {
              go(position - 1);
            }}
          >
            <SkipBack aria-hidden="true" />
          </Button>
          <Button
            size="icon-sm"
            aria-label={running ? "Pausar o histórico" : "Reproduzir o histórico"}
            onClick={() => {
              if (running) {
                setPlaying(false);
                return;
              }
              if (position >= last) {
                setIndex(0);
              }
              setPlaying(true);
            }}
          >
            {running ? <Pause aria-hidden="true" /> : <Play aria-hidden="true" />}
          </Button>
          <Button
            variant="outline"
            size="icon-sm"
            aria-label="A mensagem seguinte"
            disabled={position >= last}
            onClick={() => {
              go(position + 1);
            }}
          >
            <SkipForward aria-hidden="true" />
          </Button>
          <ToggleGroup
            type="single"
            variant="outline"
            size="sm"
            aria-label="A velocidade"
            value={String(speed)}
            onValueChange={(value) => {
              const chosen = SPEEDS.find((one) => String(one) === value);
              if (chosen !== undefined) {
                setSpeed(chosen);
              }
            }}
            className="ml-2"
          >
            {SPEEDS.map((one) => (
              <ToggleGroupItem key={one} value={String(one)} className="px-2.5 font-mono text-xs">
                {one}×
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </div>
      </div>

      <Timeline messages={messages} position={position} onPosition={go} label={label} />
    </div>
  );
}

/** O nó de quem diz uma mensagem. */
function senderOf(message: AgentMessage): FlowNode {
  if (message.role === "data") {
    return "data";
  }
  if (message.role === "ai") {
    return "ai";
  }
  if (message.role === "model" && message.family !== null) {
    return message.family;
  }
  return "coordinator";
}

/** O nó de um destinatário. */
function nodeOf(participant: Participant): FlowNode {
  return participant.kind === "model" && participant.family !== null
    ? participant.family
    : participant.kind === "model"
      ? "coordinator"
      : participant.kind;
}

/** Um participante da troca: o nome, num contorno com a cor dele (a do seu nó). */
function Who({ color, name }: { color: string; name: string }) {
  return (
    <span
      className="inline-flex h-6 items-center rounded-full border px-2.5 text-strong"
      style={{ borderColor: color }}
    >
      {name}
    </span>
  );
}

/** A cor de um destinatário (a mesma do seu nó). */
function participantColor(participant: Participant): string {
  if (participant.kind === "model" && participant.family !== null) {
    return familyColor(participant.family);
  }
  if (participant.kind === "ai") {
    return AI_AGENT_COLOR;
  }
  return participant.kind === "data" ? DATA_AGENT_COLOR : COORDINATOR_COLOR;
}

/** A quem se dirige uma mensagem: os nomes, ou "os N agentes" quando são muitos. */
function Recipients({
  recipients,
  plan,
  label,
}: {
  recipients: readonly Participant[];
  plan: AutoMlPlan;
  label: ReturnType<typeof useLabel>;
}) {
  if (recipients.length > 3) {
    return (
      <span className="inline-flex h-6 items-center rounded-full border border-[#3a3a3a] px-2.5 text-strong">
        {`os ${formatCount(recipients.length)} agentes`}
      </span>
    );
  }
  return (
    <>
      {recipients.map((recipient, index) => (
        <Who
          key={index}
          color={participantColor(recipient)}
          name={
            (recipient.kind === "model"
              ? plan.agents.find((agent) => agent.family === recipient.family)?.name
              : undefined) ?? label("ParticipantKind", recipient.kind)
          }
        />
      ))}
    </>
  );
}

/**
 * A linha do tempo: uma marca por mensagem, com a cor de quem a disse (mais alta quando decide
 * mais), e as fases por baixo. É um cursor: arrasta-se com o rato ou o dedo, e com as setas,
 * Home e End no teclado.
 */
function Timeline({
  messages,
  position,
  onPosition,
  label,
}: {
  messages: readonly AgentMessage[];
  position: number;
  onPosition: (index: number) => void;
  label: ReturnType<typeof useLabel>;
}) {
  const strip = useRef<HTMLDivElement>(null);
  const count = messages.length;
  const current = messages[position];
  const pick = (clientX: number) => {
    const box = strip.current?.getBoundingClientRect();
    if (box === undefined || box.width === 0) {
      return;
    }
    onPosition(Math.floor(((clientX - box.left) / box.width) * count));
  };
  const keys: Readonly<Record<string, number>> = {
    ArrowLeft: position - 1,
    ArrowDown: position - 1,
    ArrowRight: position + 1,
    ArrowUp: position + 1,
    PageDown: position - 10,
    PageUp: position + 10,
    Home: 0,
    End: count - 1,
  };
  // As fases por ordem, cada uma com quantas mensagens tem (a largura do seu troço).
  const phases: { phase: AgentMessage["phase"]; count: number }[] = [];
  for (const message of messages) {
    const previous = phases.at(-1);
    if (previous?.phase === message.phase) {
      previous.count += 1;
    } else {
      phases.push({ phase: message.phase, count: 1 });
    }
  }
  return (
    <div className="flex flex-col gap-1.5">
      <div
        ref={strip}
        role="slider"
        tabIndex={0}
        aria-label="O momento da procura"
        aria-valuemin={1}
        aria-valuemax={count}
        aria-valuenow={position + 1}
        aria-valuetext={
          current === undefined
            ? undefined
            : `Mensagem ${String(position + 1)} de ${String(count)}: ${current.author}, ${label("MessageKind", current.kind).toLowerCase()}`
        }
        onKeyDown={(event: KeyboardEvent<HTMLDivElement>) => {
          const target = keys[event.key];
          if (target !== undefined) {
            event.preventDefault();
            onPosition(target);
          }
        }}
        onPointerDown={(event: PointerEvent<HTMLDivElement>) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          pick(event.clientX);
        }}
        onPointerMove={(event: PointerEvent<HTMLDivElement>) => {
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            pick(event.clientX);
          }
        }}
        className="flex h-12 cursor-pointer touch-none items-end gap-[2px] rounded-md outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-4 focus-visible:ring-offset-ground"
      >
        {messages.map((message, index) => {
          const color = messageColor(message);
          const now = index === position;
          return (
            <span
              key={message.number}
              aria-hidden="true"
              className="min-w-0 flex-1 rounded-[2px] transition-[height,opacity] duration-200"
              style={{
                height: now ? "100%" : TICK_HEIGHT[message.kind],
                background: color,
                opacity: index <= position ? 1 : 0.22,
                boxShadow: now ? `0 0 12px ${color}` : undefined,
              }}
            />
          );
        })}
      </div>
      <div aria-hidden="true" className="flex gap-[2px]">
        {phases.map(({ phase, count: size }, index) => (
          <span
            key={`${phase}-${String(index)}`}
            className="min-w-0 truncate border-t border-[#333] pt-1.5 font-mono text-[12px] text-dim"
            style={{ flex: size }}
          >
            {label("SearchPhase", phase)}
          </span>
        ))}
      </div>
      <p className="font-mono text-xs text-dim tabular-nums">
        Mensagem {formatCount(position + 1)} de {formatCount(count)}
      </p>
    </div>
  );
}
