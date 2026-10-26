/** As casas decimais que um passo entre marcas pede (0,2 dá uma; 0,05 dá duas). */
export function stepDecimals(ticks: readonly number[]): number {
  const [first, second] = ticks;
  if (first === undefined || second === undefined) {
    return 0;
  }
  const step = Math.abs(second - first);
  return step > 0 ? Math.max(0, -Math.floor(Math.log10(step) + 1e-9)) : 0;
}
