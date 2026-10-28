/** Para que lado uma métrica é melhor, numa frase (a gaveta de cada métrica e o glossário). */
export function directionText(higherIsBetter: boolean | null): string {
  if (higherIsBetter === null) {
    return "Descreve o resultado: não há melhor nem pior.";
  }
  return higherIsBetter ? "Quanto maior, melhor." : "Quanto menor, melhor.";
}
