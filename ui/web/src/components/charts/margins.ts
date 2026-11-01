/** A largura de uma letra dos eixos (Geist Mono a 12px: cada letra tem 0,6 da altura). */
const AXIS_CHAR = 7.2;

/**
 * A margem da esquerda que os números do eixo pedem: a do número mais comprido, mais o espaço
 * até à grelha e, se houver, o nome do eixo ao alto. Uma margem fixa cortava os números grandes
 * (um MSE de "12 000 000 000") e o nome do eixo caía por cima deles.
 */
export function leftMargin(labels: readonly string[], named: boolean, minimum = 48): number {
  const longest = Math.max(0, ...labels.map((text) => text.length));
  return Math.max(minimum, Math.ceil(longest * AXIS_CHAR) + 14 + (named ? 22 : 0));
}
