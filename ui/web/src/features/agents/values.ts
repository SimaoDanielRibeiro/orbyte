import type { Candidate, ObjectiveOption, Participant } from "@/api/generated";
import type { useLabel } from "@/api/vocabulary";
import { formatCount, formatNumber, formatPercent, formatValue } from "@/lib/format";

/** Um valor da métrica que os agentes otimizam, escrito como ela se escreve. */
export function objectiveValue(value: number | null, objective: ObjectiveOption | null): string {
  if (value === null) {
    return "—";
  }
  return objective?.as_percent === true ? formatPercent(value) : formatNumber(value, 3);
}

/**
 * O melhor de uns candidatos medidos, pela mesma regra do coordenador (`rank`, no backend): a
 * melhor média no sentido da métrica, depois o menor desvio, depois o que veio primeiro. Serve
 * para o histórico mostrar quem liderava num momento passado (ao vivo, o líder vem da API).
 */
export function bestCandidate(
  candidates: readonly Candidate[],
  objective: ObjectiveOption | null,
): Candidate | null {
  const higher = objective?.higher_is_better !== false;
  let best: Candidate | null = null;
  for (const candidate of candidates) {
    if (candidate.mean === null || candidate.error !== null) {
      continue;
    }
    if (best?.mean == null) {
      best = candidate;
      continue;
    }
    const better = higher ? candidate.mean > best.mean : candidate.mean < best.mean;
    const tied = candidate.mean === best.mean;
    const steadier = (candidate.std ?? Infinity) < (best.std ?? Infinity);
    if (better || (tied && steadier)) {
      best = candidate;
    }
  }
  return best;
}

/**
 * Os candidatos que, quando foram medidos, passaram a ser o melhor até ali (pela mesma regra de
 * `bestCandidate`): na conversa, a resposta deles leva a marca "new best".
 */
export function leaderNumbers(
  candidates: readonly Candidate[],
  objective: ObjectiveOption | null,
): ReadonlySet<number> {
  const leaders = new Set<number>();
  const sorted = [...candidates].sort((a, b) => a.number - b.number);
  let best: Candidate | null = null;
  for (const candidate of sorted) {
    const next = bestCandidate(best === null ? [candidate] : [best, candidate], objective);
    if (next !== null && next.number === candidate.number) {
      leaders.add(candidate.number);
    }
    best = next ?? best;
  }
  return leaders;
}

/** O que muda num candidato em relação aos valores por omissão, numa linha. */
export function changesText(candidate: Pick<Candidate, "changes">): string {
  const entries = Object.entries(candidate.changes);
  if (entries.length === 0) {
    return "default values";
  }
  return entries.map(([name, value]) => `${name} ${formatValue(value)}`).join(", ");
}

/**
 * A quem se dirige uma mensagem, numa linha: os nomes, ou "all N agents" quando são mais de
 * três; `null` quando não tem destinatários (a leitura do teste é para quem acompanha).
 */
export function recipientsText(
  recipients: readonly Participant[],
  label: ReturnType<typeof useLabel>,
): string | null {
  if (recipients.length === 0) {
    return null;
  }
  if (recipients.length > 3) {
    return `all ${formatCount(recipients.length)} agents`;
  }
  return recipients
    .map((recipient) =>
      recipient.kind === "model" && recipient.family !== null
        ? label("ModelFamily", recipient.family)
        : label("ParticipantKind", recipient.kind),
    )
    .join(", ");
}
