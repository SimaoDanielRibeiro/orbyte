/**
 * As partes do ecrã de uma procura: as fases, os candidatos medidos (ao vivo e o quadro final)
 * e a conversa dos agentes.
 *
 * Ao vivo, os candidatos aparecem pela ordem em que são medidos (o mais recente em cima): o
 * lugar de cada um só se decide no fim, pelo backend (`ranking`), e é por essa ordem que o
 * quadro final se lê.
 */
import { cn } from "cn";
import { Check, Sparkles } from "lucide-react";
import { motion, useReducedMotion } from "motion/react";
import { useEffect, useRef, useState } from "react";

import type {
  AgentMessage,
  AutoMlResult,
  Candidate,
  ObjectiveOption,
  SearchPhase,
  Thought,
} from "@/api/generated";
import { useLabel } from "@/api/vocabulary";
import { Prose } from "@/components/markdown/prose";
import { formatCount, formatNumber, formatOrdinal } from "@/lib/format";

import { AI_AGENT_COLOR, messageColor, ROLE_ICONS } from "./agent-colors";
import { changesText, objectiveValue, recipientsText } from "./values";

/** A marca dos candidatos que o agente de IA propôs (e não o sorteio de um agente de família). */
export function AiMark() {
  return (
    <span
      className="ml-2 inline-flex items-center gap-1 rounded-full border px-1.5 py-px text-[12px] font-medium"
      style={{ borderColor: AI_AGENT_COLOR, color: AI_AGENT_COLOR }}
    >
      <Sparkles aria-hidden="true" className="size-3" />
      AI
    </span>
  );
}

const PHASES: readonly SearchPhase[] = ["data", "screening", "tuning", "final", "done"];

/**
 * As fases da procura, por ordem: a de agora a amarelo, as que passaram com um visto. Compactas
 * (círculos pequenos, traços curtos), para as cinco caberem numa linha num portátil: a última
 * sozinha na linha de baixo parecia outra coisa.
 */
export function PhaseSteps({ phase }: { phase: SearchPhase | null }) {
  const label = useLabel();
  const current = phase === null ? -1 : PHASES.indexOf(phase);
  return (
    <ol
      className="flex flex-wrap items-center gap-x-1.5 gap-y-2 text-sm"
      aria-label="The phases of the search"
    >
      {PHASES.map((one, index) => {
        const done = index < current || phase === "done";
        const now = index === current && phase !== "done";
        return (
          <li
            key={one}
            className="flex items-center gap-1.5"
            aria-current={now ? "step" : undefined}
          >
            <span
              aria-hidden="true"
              className={cn(
                "flex size-5 items-center justify-center rounded-full border text-[12px] tabular-nums",
                now && "border-now bg-now text-background",
                done && "border-muted-foreground text-foreground",
                !now && !done && "text-muted-foreground",
              )}
            >
              {done ? <Check className="size-3" /> : index + 1}
            </span>
            <span className={cn(now ? "text-foreground" : "text-muted-foreground")}>
              {label("SearchPhase", one)}
            </span>
            {index < PHASES.length - 1 && (
              <span aria-hidden="true" className="mx-0.5 h-px w-4 bg-input" />
            )}
          </li>
        );
      })}
    </ol>
  );
}

interface FeedProps {
  candidates: readonly Candidate[];
  objective: ObjectiveOption | null;
  leader: Candidate | null;
  /** Quantos se mostram (os mais recentes). */
  limit?: number;
}

/** Os candidatos medidos, o mais recente em cima; o melhor até agora fica marcado. */
export function CandidateFeed({ candidates, objective, leader, limit = 12 }: FeedProps) {
  const label = useLabel();
  const reduced = useReducedMotion() ?? false;
  const recent = [...candidates].reverse().slice(0, limit);
  if (recent.length === 0) {
    return <p className="text-sm text-dim">No candidates measured yet.</p>;
  }
  return (
    <div className="relative overflow-x-auto">
      <table className="w-full text-sm whitespace-nowrap">
        <thead>
          <tr className="border-b text-left">
            <th scope="col" className="px-3 py-2.5 text-xs font-medium text-dim">
              #
            </th>
            <th scope="col" className="px-3 py-2.5 text-xs font-medium text-dim">
              Model
            </th>
            <th scope="col" className="px-3 py-2.5 text-xs font-medium text-dim">
              Round
            </th>
            <th scope="col" className="px-3 py-2.5 text-xs font-medium text-dim">
              What changed
            </th>
            <th scope="col" className="px-3 py-2.5 text-right text-xs font-medium text-dim">
              {objective?.label ?? "Value"}
            </th>
          </tr>
        </thead>
        <tbody>
          {recent.map((candidate) => {
            const best = leader?.number === candidate.number;
            return (
              <motion.tr
                key={candidate.number}
                initial={reduced ? false : { opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: reduced ? 0 : 0.25 }}
                className={cn(
                  "border-b border-border/60 last:border-0",
                  best && "bg-surface-high text-strong",
                )}
              >
                <td className="px-3 py-2 text-muted-foreground tabular-nums">
                  {formatCount(candidate.number)}
                </td>
                <td className="px-3 py-2">
                  {candidate.model_display_name}
                  {candidate.ai && <AiMark />}
                  {best && <span className="ml-2 text-xs text-muted-foreground">best so far</span>}
                </td>
                <td className="px-3 py-2 text-muted-foreground">
                  {label("SearchPhase", candidate.phase)}
                </td>
                <td className="max-w-[20rem] truncate px-3 py-2 text-muted-foreground">
                  {changesText(candidate)}
                </td>
                <td className="px-3 py-2 text-right tabular-nums">
                  {candidate.error === null
                    ? `${objectiveValue(candidate.mean, objective)} ± ${objectiveValue(candidate.std, objective)}`
                    : "failed"}
                </td>
              </motion.tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** O quadro final: pela ordem do backend (`ranking`), e os que falharam no fim. */
export function Board({ result }: { result: AutoMlResult }) {
  const label = useLabel();
  const objective = result.plan.objective;
  const byNumber = new Map(result.candidates.map((candidate) => [candidate.number, candidate]));
  const ranked = result.ranking
    .map((number) => byNumber.get(number))
    .filter((candidate): candidate is Candidate => candidate !== undefined);
  const failed = result.candidates.filter(
    (candidate) => !result.ranking.includes(candidate.number),
  );
  return (
    <div className="relative overflow-x-auto">
      <table className="w-full text-sm whitespace-nowrap">
        <thead>
          <tr className="border-b text-left">
            <th scope="col" className="px-3 py-2.5 text-xs font-medium text-dim">
              Place
            </th>
            <th scope="col" className="px-3 py-2.5 text-xs font-medium text-dim">
              Model
            </th>
            <th scope="col" className="px-3 py-2.5 text-xs font-medium text-dim">
              Agent
            </th>
            <th scope="col" className="px-3 py-2.5 text-xs font-medium text-dim">
              Round
            </th>
            <th scope="col" className="px-3 py-2.5 text-xs font-medium text-dim">
              What changed
            </th>
            <th scope="col" className="px-3 py-2.5 text-right text-xs font-medium text-dim">
              {objective.label}
            </th>
            <th scope="col" className="px-3 py-2.5 text-right text-xs font-medium text-dim">
              On training
            </th>
            <th scope="col" className="px-3 py-2.5 text-right text-xs font-medium text-dim">
              Seconds
            </th>
          </tr>
        </thead>
        <tbody>
          {[...ranked, ...failed].map((candidate) => {
            const place = result.ranking.indexOf(candidate.number) + 1;
            return (
              <tr
                key={candidate.number}
                className={cn(
                  "border-b border-border/60 last:border-0",
                  candidate.number === result.winner && "bg-surface-high text-strong",
                )}
              >
                <td className="px-3 py-2 tabular-nums">{place > 0 ? formatOrdinal(place) : "—"}</td>
                <td className="px-3 py-2">
                  {candidate.model_display_name}
                  <span className="ml-2 text-xs text-muted-foreground tabular-nums">
                    #{formatCount(candidate.number)}
                  </span>
                  {candidate.ai && <AiMark />}
                </td>
                <td className="px-3 py-2 text-muted-foreground">
                  {candidate.ai ? label("AgentRole", "ai") : label("ModelFamily", candidate.family)}
                </td>
                <td className="px-3 py-2 text-muted-foreground">
                  {label("SearchPhase", candidate.phase)}
                </td>
                <td className="max-w-[18rem] truncate px-3 py-2 text-muted-foreground">
                  {candidate.error ?? changesText(candidate)}
                </td>
                <td className="px-3 py-2 text-right whitespace-nowrap tabular-nums">
                  {candidate.error === null
                    ? `${objectiveValue(candidate.mean, objective)} ± ${objectiveValue(candidate.std, objective)}`
                    : "failed"}
                </td>
                <td className="px-3 py-2 text-right text-muted-foreground tabular-nums">
                  {objectiveValue(candidate.train_mean, objective)}
                </td>
                <td className="px-3 py-2 text-right text-muted-foreground tabular-nums">
                  {formatNumber(candidate.seconds, 2)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/**
 * A conversa dos agentes, pela ordem; segue a mensagem mais recente enquanto chegam. Cada agente
 * tem a sua cor (a do nó dele no fluxograma), no contorno da caixa do ícone: lê-se quem fala sem
 * ler o nome. O texto fica nas cores do texto.
 *
 * O agente de IA pensa antes de responder: enquanto pensa, o raciocínio aparece no fim da
 * conversa, a crescer (`thought`); depois fica guardado numa mensagem "Thought", fechada, que
 * se abre para o ler todo. Para a ler como um chat, em ecrã inteiro, há a `ConversationRoom`.
 */
export function Conversation({
  messages,
  active = null,
  thought = null,
  className,
}: {
  messages: readonly AgentMessage[];
  /** A mensagem à vista no histórico das trocas (fica marcada, e a conversa vai até ela). */
  active?: number | null;
  /** O que o agente de IA está a pensar agora (aparece no fim, ao vivo). */
  thought?: Thought | null;
  className?: string;
}) {
  const label = useLabel();
  const reduced = useReducedMotion() ?? false;
  const list = useRef<HTMLOListElement>(null);
  const count = messages.length;
  // Só entram a deslizar as mensagens que chegam depois de a conversa aparecer: abrir uma
  // procura terminada não anima centenas de mensagens de uma vez. (Um estado que nunca muda, e
  // não uma ref: o desenho pode lê-lo.)
  const [settled] = useState(count);
  const thinking = thought?.text.length ?? -1;
  // Só a caixa da conversa rola (nunca a página): quem está a ver o fluxograma não salta. Ao
  // vivo segue a última mensagem; no histórico, a que está à vista.
  useEffect(() => {
    const element = list.current;
    if (element === null) {
      return;
    }
    const behavior = reduced ? "auto" : "smooth";
    if (active === null) {
      element.scrollTo({ top: element.scrollHeight, behavior });
      return;
    }
    const item = element.querySelector<HTMLElement>(`[data-message="${String(active)}"]`);
    if (item !== null) {
      element.scrollTo({ top: item.offsetTop - element.clientHeight / 3, behavior });
    }
  }, [count, active, reduced, thinking]);
  if (count === 0) {
    return <p className="text-sm text-dim">The agents haven't said anything yet.</p>;
  }
  return (
    <ol ref={list} className={cn("relative flex flex-col gap-4 overflow-y-auto pr-2", className)}>
      {messages.map((message, index) => {
        const Icon = ROLE_ICONS[message.role];
        const color = messageColor(message);
        const now = active === message.number;
        const enter = reduced || index < settled ? false : { opacity: 0, x: -6 };
        // Um pedido do coordenador (dar a vez a um agente) diz porquê (ADR 0049): lê-se inteiro,
        // em vez de se cortar numa linha, mas mais discreto do que as respostas.
        if (message.kind === "request") {
          return (
            <motion.li
              key={message.number}
              data-message={message.number}
              initial={enter}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: reduced ? 0 : 0.25 }}
              className={cn(
                "grid grid-cols-[1.75rem_minmax(0,1fr)] items-start gap-3 rounded-lg lazy-paint",
                now && "bg-surface-high ring-1 ring-[#3a3a3a]",
              )}
            >
              <span aria-hidden="true" className="flex justify-center py-0.5 text-xs text-dim">
                →
              </span>
              <p className="min-w-0 py-0.5 text-xs leading-snug text-soft">{message.text}</p>
            </motion.li>
          );
        }
        // O raciocínio do agente de IA: fechado, com o princípio à vista; abre-se para o ler todo.
        if (message.kind === "thought") {
          return (
            <motion.li
              key={message.number}
              data-message={message.number}
              initial={enter}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: reduced ? 0 : 0.25 }}
              className={cn(
                "grid grid-cols-[1.75rem_minmax(0,1fr)] gap-3 rounded-lg lazy-paint",
                now && "bg-surface-high py-2 ring-1 ring-[#3a3a3a]",
              )}
            >
              <span aria-hidden="true" className="flex justify-center pt-1 text-dim">
                <Sparkles className="size-3.5" style={{ color: AI_AGENT_COLOR }} />
              </span>
              <details className="group min-w-0">
                <summary className="cursor-pointer list-none truncate font-mono text-xs text-dim hover:text-soft">
                  <span className="text-soft">{label("MessageKind", "thought")}</span> ·{" "}
                  {message.text}
                </summary>
                <p
                  className="mt-2 max-h-72 overflow-y-auto border-l pl-3 font-mono text-xs leading-relaxed whitespace-pre-wrap text-soft"
                  style={{ borderColor: AI_AGENT_COLOR }}
                >
                  {message.text}
                </p>
              </details>
            </motion.li>
          );
        }
        const recipients = recipientsText(message.recipients, label);
        return (
          <motion.li
            key={message.number}
            data-message={message.number}
            initial={enter}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: reduced ? 0 : 0.25 }}
            className={cn(
              "grid grid-cols-[1.75rem_minmax(0,1fr)] gap-3 rounded-lg lazy-paint",
              now && "bg-surface-high py-2 ring-1 ring-[#3a3a3a]",
            )}
          >
            {/* Quem fala: o ícone numa caixa com o contorno da cor dele (a do nó no fluxograma). */}
            <span
              aria-hidden="true"
              className="mt-0.5 flex size-7 items-center justify-center rounded-lg border bg-[#1a1a1a]"
              style={{ borderColor: color }}
            >
              <Icon className="size-3.5" style={{ color }} />
            </span>
            <div className="min-w-0">
              <p className="text-sm">
                <span className="font-medium text-strong">{message.author}</span>
                {recipients !== null && <span className="ml-1.5 text-soft">to {recipients}</span>}
                <span className="ml-2 text-xs text-dim">{label("SearchPhase", message.phase)}</span>
              </p>
              <Prose text={message.text} size="small" className="mt-0.5 text-foreground/85" />
            </div>
          </motion.li>
        );
      })}
      {thought !== null && <LiveThought thought={thought} />}
    </ol>
  );
}

/** O agente de IA a pensar, agora: o raciocínio a crescer, com um cursor a piscar no fim. */
export function LiveThought({ thought }: { thought: Thought }) {
  return (
    <li
      className="grid grid-cols-[1.75rem_minmax(0,1fr)] gap-3 rounded-lg"
      aria-live="polite"
      aria-busy="true"
    >
      <span
        aria-hidden="true"
        className="mt-0.5 flex size-7 items-center justify-center rounded-lg border border-dashed bg-[#1a1a1a]"
        style={{ borderColor: AI_AGENT_COLOR }}
      >
        <Sparkles className="size-3.5 animate-pulse" style={{ color: AI_AGENT_COLOR }} />
      </span>
      <div className="min-w-0">
        <p className="text-sm">
          <span className="font-medium text-strong">{thought.author}</span>
          <span className="ml-1.5 text-soft">
            is thinking about candidate #{formatCount(thought.candidate)}
          </span>
        </p>
        <p
          className="mt-1 max-h-60 overflow-y-auto border-l pl-3 font-mono text-xs leading-relaxed whitespace-pre-wrap text-soft"
          style={{ borderColor: AI_AGENT_COLOR }}
        >
          {thought.text === "" ? "…" : thought.text}
          <span
            aria-hidden="true"
            className="ml-0.5 inline-block h-3 w-1.5 animate-pulse align-middle"
            style={{ background: AI_AGENT_COLOR }}
          />
        </p>
      </div>
    </li>
  );
}
