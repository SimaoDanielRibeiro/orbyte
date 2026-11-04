/** Para que lado uma métrica é melhor, numa frase (a gaveta de cada métrica e o glossário). */
export function directionText(higherIsBetter: boolean | null): string {
  if (higherIsBetter === null) {
    return "It describes the result: there is no better or worse.";
  }
  return higherIsBetter ? "Higher is better." : "Lower is better.";
}
