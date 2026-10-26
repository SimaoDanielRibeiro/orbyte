/**
 * As formas dos pontos. Só três cores se distinguem com segurança numa dispersão (todas as
 * pares da paleta): com mais categorias, a forma também muda.
 */
import {
  symbol,
  symbolCircle,
  symbolCross,
  symbolDiamond,
  symbolSquare,
  symbolTriangle,
  symbolWye,
} from "d3";

export type PointShape = "circle" | "square" | "triangle" | "diamond" | "cross" | "wye";

/** As formas pela ordem em que se atribuem. */
export const SHAPE_ORDER: readonly PointShape[] = [
  "circle",
  "square",
  "triangle",
  "diamond",
  "cross",
  "wye",
];

const SYMBOLS = {
  circle: symbolCircle,
  square: symbolSquare,
  triangle: symbolTriangle,
  diamond: symbolDiamond,
  cross: symbolCross,
  wye: symbolWye,
} as const;

/** O desenho de cada forma, centrado em (0, 0), com a área de um círculo de raio 4 px. */
export const SHAPE_PATHS: Readonly<Record<PointShape, string>> = Object.fromEntries(
  SHAPE_ORDER.map((shape) => [shape, symbol(SYMBOLS[shape], 48)() ?? ""]),
) as Record<PointShape, string>;

/** A forma da categoria número `index`. */
export function shapeOf(index: number): PointShape {
  return SHAPE_ORDER[index % SHAPE_ORDER.length] ?? "circle";
}
