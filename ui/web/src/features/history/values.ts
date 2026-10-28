import type { RunEntry } from "@/api/generated";
import { formatNumber, formatPercent } from "@/lib/format";

/** Quantos runs se comparam, no máximo (o backend recusa mais). */
export const MAX_COMPARED = 5;

/** O valor da métrica principal de um run, escrito (`—` se não tiver). */
export function primaryValue(entry: RunEntry): string {
  const value = entry.primary_metric === null ? null : entry.metrics[entry.primary_metric];
  if (value === null || value === undefined) {
    return "—";
  }
  return entry.primary_as_percent ? formatPercent(value) : formatNumber(value, 3);
}

/** Um valor de uma comparação de runs: percentagem com uma casa, os outros com três. */
export function comparedValue(value: number | null, asPercent: boolean): string {
  if (value === null) {
    return "—";
  }
  return asPercent ? formatPercent(value) : formatNumber(value, 3);
}
