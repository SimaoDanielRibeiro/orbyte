import type { Candidate, ObjectiveOption } from "@/api/generated";
import { formatNumber, formatPercent, formatValue } from "@/lib/format";

/** Um valor da métrica que os agentes otimizam, escrito como ela se escreve. */
export function objectiveValue(value: number | null, objective: ObjectiveOption | null): string {
  if (value === null) {
    return "—";
  }
  return objective?.as_percent === true ? formatPercent(value) : formatNumber(value, 3);
}

/** O que muda num candidato em relação aos valores por omissão, numa linha. */
export function changesText(candidate: Pick<Candidate, "changes">): string {
  const entries = Object.entries(candidate.changes);
  if (entries.length === 0) {
    return "valores por omissão";
  }
  return entries.map(([name, value]) => `${name} ${formatValue(value)}`).join(", ");
}
