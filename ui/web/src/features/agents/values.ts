import type { Candidate, ObjectiveOption } from "@/api/generated";
import { formatNumber, formatPercent, formatValue } from "@/lib/format";

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

/** O que muda num candidato em relação aos valores por omissão, numa linha. */
export function changesText(candidate: Pick<Candidate, "changes">): string {
  const entries = Object.entries(candidate.changes);
  if (entries.length === 0) {
    return "valores por omissão";
  }
  return entries.map(([name, value]) => `${name} ${formatValue(value)}`).join(", ");
}
