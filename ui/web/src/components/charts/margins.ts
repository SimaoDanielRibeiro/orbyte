/** A largura de uma letra dos eixos (Geist Mono a 13px: cada letra tem 0,6 da altura). */
const AXIS_CHAR = 7.8;

/**
 * A margem da esquerda que os números do eixo pedem: a do número mais comprido, mais o espaço
 * até à grelha e, se houver, o nome do eixo ao alto. Uma margem fixa cortava os números grandes
 * (um MSE de "12 000 000 000") e o nome do eixo caía por cima deles.
 */
export function leftMargin(labels: readonly string[], named: boolean, minimum = 48): number {
  const longest = Math.max(0, ...labels.map((text) => text.length));
  return Math.max(minimum, Math.ceil(longest * AXIS_CHAR) + 14 + (named ? 22 : 0));
}

/**
 * Quanto um rótulo inclinado avança para a direita do ponto onde começa (o nome de uma coluna
 * rodado `degrees` graus, a subir). O último rótulo de uma fila precisa desta margem à direita:
 * com uma fixa, no telemóvel passava da borda do ecrã.
 */
export function slantedOverhang(label: string, degrees: number, minimum = 8): number {
  const radians = (degrees * Math.PI) / 180;
  return Math.max(minimum, Math.ceil(label.length * AXIS_CHAR * Math.cos(radians)));
}
