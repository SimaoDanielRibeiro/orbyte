/**
 * As partes do ecrã de uma procura: as fases, os candidatos medidos (ao vivo e o quadro final)
 * e a conversa dos agentes.
 *
 * Ao vivo, os candidatos aparecem pela ordem em que são medidos (o mais recente em cima): o
 * lugar de cada um só se decide no fim, pelo backend (`ranking`), e é por essa ordem que o
 * quadro final se lê.
 */
import { cn } from "cn";
import { Check, Database, Network, Workflow } from "lucide-react";
import { motion, useReducedMotion } from "motion/react";
import { useEffect, useRef } from "react";

import type {
  AgentMessage,
  AgentRole,
  AutoMlResult,
  Candidate,
  ObjectiveOption,
  SearchPhase,
} from "@/api/generated";
import { useLabel } from "@/api/vocabulary";
import { Prose } from "@/components/markdown/prose";
import { formatCount, formatNumber } from "@/lib/format";

import { messageColor } from "./agent-colors";
import { changesText, objectiveValue } from "./values";

const PHASES: readonly SearchPhase[] = ["data", "screening", "tuning", "final", "done"];

/** As fases da procura, por ordem: a de agora a amarelo, as que passaram com um visto. */
export function PhaseSteps({ phase }: { phase: SearchPhase | null }) {
  const label = useLabel();
  const current = phase === null ? -1 : PHASES.indexOf(phase);
  return (
    <ol
      className="flex flex-wrap items-center gap-x-2 gap-y-2 text-sm"
      aria-label="As fases da procura"
    >
      {PHASES.map((one, index) => {
        const done = index < current || phase === "done";
        const now = index === current && phase !== "done";
        return (
          <li key={one} className="flex items-center gap-2" aria-current={now ? "step" : undefined}>
            <span
              aria-hidden="true"
              className={cn(
                "flex size-6 items-center justify-center rounded-full border text-xs tabular-nums",
                now && "border-now bg-now text-background",
                done && "border-muted-foreground text-foreground",
                !now && !done && "text-muted-foreground",
              )}
            >
              {done ? <Check className="size-3.5" /> : index + 1}
            </span>
            <span className={cn(now ? "text-foreground" : "text-muted-foreground")}>
              {label("SearchPhase", one)}
            </span>
            {index < PHASES.length - 1 && (
              <span aria-hidden="true" className="mx-1 h-px w-5 bg-input" />
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
    return <p className="font-mono text-xs text-dim">Ainda não há candidatos medidos.</p>;
  }
  return (
    <div className="relative overflow-x-auto">
      <table className="w-full text-sm whitespace-nowrap">
        <thead>
          <tr className="border-b text-left">
            <th scope="col" className="px-3 py-2.5 font-mono text-xs font-normal text-dim">
              N.º
            </th>
            <th scope="col" className="px-3 py-2.5 font-mono text-xs font-normal text-dim">
              Modelo
            </th>
            <th scope="col" className="px-3 py-2.5 font-mono text-xs font-normal text-dim">
              Ronda
            </th>
            <th scope="col" className="px-3 py-2.5 font-mono text-xs font-normal text-dim">
              O que mudou
            </th>
            <th
              scope="col"
              className="px-3 py-2.5 text-right font-mono text-xs font-normal text-dim"
            >
              {objective?.label ?? "Valor"}
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
                  {best && (
                    <span className="ml-2 text-xs text-muted-foreground">o melhor até agora</span>
                  )}
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
                    : "falhou"}
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
            <th scope="col" className="px-3 py-2.5 font-mono text-xs font-normal text-dim">
              Lugar
            </th>
            <th scope="col" className="px-3 py-2.5 font-mono text-xs font-normal text-dim">
              Modelo
            </th>
            <th scope="col" className="px-3 py-2.5 font-mono text-xs font-normal text-dim">
              Agente
            </th>
            <th scope="col" className="px-3 py-2.5 font-mono text-xs font-normal text-dim">
              Ronda
            </th>
            <th scope="col" className="px-3 py-2.5 font-mono text-xs font-normal text-dim">
              O que mudou
            </th>
            <th
              scope="col"
              className="px-3 py-2.5 text-right font-mono text-xs font-normal text-dim"
            >
              {objective.label}
            </th>
            <th
              scope="col"
              className="px-3 py-2.5 text-right font-mono text-xs font-normal text-dim"
            >
              No treino
            </th>
            <th
              scope="col"
              className="px-3 py-2.5 text-right font-mono text-xs font-normal text-dim"
            >
              Segundos
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
                <td className="px-3 py-2 tabular-nums">
                  {place > 0 ? `${formatCount(place)}.º` : "—"}
                </td>
                <td className="px-3 py-2">
                  {candidate.model_display_name}
                  <span className="ml-2 text-xs text-muted-foreground tabular-nums">
                    n.º {formatCount(candidate.number)}
                  </span>
                </td>
                <td className="px-3 py-2 text-muted-foreground">
                  {label("ModelFamily", candidate.family)}
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
                    : "falhou"}
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

const ROLE_ICONS: Readonly<Record<AgentRole, typeof Database>> = {
  data: Database,
  coordinator: Workflow,
  model: Network,
};

/**
 * A conversa dos agentes, pela ordem; segue a mensagem mais recente enquanto chegam. Cada agente
 * tem a sua cor (a do nó dele no fluxograma), no contorno da caixa do ícone: lê-se quem fala sem
 * ler o nome. O texto fica nas cores do texto.
 */
export function Conversation({
  messages,
  active = null,
  className,
}: {
  messages: readonly AgentMessage[];
  /** A mensagem à vista no histórico das trocas (fica marcada, e a conversa vai até ela). */
  active?: number | null;
  className?: string;
}) {
  const label = useLabel();
  const reduced = useReducedMotion() ?? false;
  const list = useRef<HTMLOListElement>(null);
  const count = messages.length;
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
  }, [count, active, reduced]);
  if (count === 0) {
    return <p className="font-mono text-xs text-dim">Os agentes ainda não disseram nada.</p>;
  }
  const to = (message: AgentMessage) => {
    const { recipients } = message;
    if (recipients.length === 0) {
      return null;
    }
    if (recipients.length > 3) {
      return `para os ${formatCount(recipients.length)} agentes`;
    }
    return `para ${recipients
      .map((recipient) =>
        recipient.kind === "model" && recipient.family !== null
          ? label("ModelFamily", recipient.family)
          : label("ParticipantKind", recipient.kind),
      )
      .join(", ")}`;
  };
  return (
    <ol ref={list} className={cn("relative flex flex-col gap-4 overflow-y-auto pr-2", className)}>
      {messages.map((message) => {
        const Icon = ROLE_ICONS[message.role];
        const color = messageColor(message);
        const now = active === message.number;
        // Um pedido do coordenador (dar a vez a um agente) é curto: fica numa linha só.
        if (message.kind === "request") {
          return (
            <motion.li
              key={message.number}
              data-message={message.number}
              initial={reduced ? false : { opacity: 0, x: -6 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: reduced ? 0 : 0.25 }}
              className={cn(
                "grid grid-cols-[1.75rem_minmax(0,1fr)] items-center gap-3 rounded-lg",
                now && "bg-surface-high ring-1 ring-[#3a3a3a]",
              )}
            >
              <span aria-hidden="true" className="flex justify-center font-mono text-xs text-dim">
                →
              </span>
              <p className="min-w-0 truncate py-0.5 font-mono text-xs text-dim">{message.text}</p>
            </motion.li>
          );
        }
        const recipients = to(message);
        return (
          <motion.li
            key={message.number}
            data-message={message.number}
            initial={reduced ? false : { opacity: 0, x: -6 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: reduced ? 0 : 0.25 }}
            className={cn(
              "grid grid-cols-[1.75rem_minmax(0,1fr)] gap-3 rounded-lg",
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
                {recipients !== null && <span className="ml-1.5 text-soft">{recipients}</span>}
                <span className="ml-2 font-mono text-xs text-dim">
                  {label("SearchPhase", message.phase)}
                </span>
              </p>
              <Prose text={message.text} size="small" className="mt-0.5 text-foreground/85" />
            </div>
          </motion.li>
        );
      })}
    </ol>
  );
}
