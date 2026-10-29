/**
 * As cores das células de uma grelha de reinforcement learning, as mesmas na grelha do treino
 * (`grid-world.tsx`) e na lição do ecrã (`features/reinforcement/lesson.tsx`).
 */
import type { CellKind } from "@/api/generated";
import { CATEGORICAL, diverging } from "@/lib/colors";

export const WALL = "#3a3a3a";
export const EMPTY_KIND = "#1e1e1e";
export const TRAP = CATEGORICAL[7];

/** A cor de uma célula: o valor, ou o desenho próprio das células sem escolha. */
export function cellFill(kind: CellKind, value: number | null): string {
  if (kind === "wall") {
    return WALL;
  }
  if (value === null) {
    return EMPTY_KIND;
  }
  return diverging(value);
}
