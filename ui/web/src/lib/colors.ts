/**
 * As cores dos dados (ADR 0029: "a cor pertence aos dados").
 *
 * A paleta categórica é a do skill de visualização de dados, nos passos para o tema escuro,
 * validada contra a ardósia (#141A21): todas as verificações passam nos pares vizinhos (barras,
 * linhas) e, nas três primeiras cores, em todos os pares (dispersões e campos). Com mais de três
 * classes num campo ou numa dispersão, os pontos ganham também uma forma diferente.
 *
 * As cores atribuem-se pela ordem e nunca se repetem: uma nona categoria fica cinzenta.
 */
import { color as parseColor, interpolateLab, piecewise } from "d3";

export const CATEGORICAL = [
  "#3987e5", // azul
  "#d95926", // laranja
  "#199e70", // água
  "#c98500", // amarelo
  "#d55181", // magenta
  "#008300", // verde
  "#9085e9", // violeta
  "#e66767", // vermelho
] as const;

/** Para o que fica de fora da paleta (uma nona categoria) e para o ruído do clustering. */
export const OTHER = "#5b6876";

/** O meio neutro das escalas divergentes: nem um lado nem o outro. */
export const NEUTRAL = "#383835";

/** A cor da categoria número `index` (pela ordem em que aparecem). */
export function categoryColor(index: number): string {
  return CATEGORICAL[index] ?? OTHER;
}

/** Quantas categorias se distinguem só pela cor numa dispersão (as outras precisam de forma). */
export const COLOR_ONLY_LIMIT = 3;

/**
 * A rampa azul, do passo mais escuro ao mais claro. No tema escuro, o "quase zero" fica perto
 * do fundo e os valores altos acendem.
 */
const BLUE_RAMP = [
  "#0d366b",
  "#104281",
  "#184f95",
  "#1c5cab",
  "#256abf",
  "#2a78d6",
  "#3987e5",
  "#5598e7",
  "#6da7ec",
  "#86b6ef",
  "#9ec5f4",
  "#b7d3f6",
  "#cde2fb",
];

const ORANGE_RAMP = ["#4a1d0c", "#7a2e12", "#a83f19", "#d95926", "#ec835a", "#f4ab8c", "#fbd3c1"];

const blue = piecewise(interpolateLab, BLUE_RAMP);
const orange = piecewise(interpolateLab, ORANGE_RAMP);

/** Uma magnitude de 0 a 1, na rampa azul. */
export function sequential(t: number): string {
  return String(blue(clamp(t)));
}

/** Uma segunda magnitude no mesmo ecrã (a pontuação de anomalia), na rampa laranja. */
export function sequentialAlt(t: number): string {
  return String(orange(clamp(t)));
}

const toBlue = interpolateLab(NEUTRAL, CATEGORICAL[0]);
const toRed = interpolateLab(NEUTRAL, CATEGORICAL[7]);
const toOrange = interpolateLab(NEUTRAL, CATEGORICAL[1]);

/** Uma polaridade de -1 a 1 (uma correlação): azul de um lado, vermelho do outro. */
export function diverging(t: number): string {
  const value = Math.max(-1, Math.min(1, t));
  return value < 0 ? toRed(-value) : toBlue(value);
}

/**
 * Uma probabilidade de 0 a 1 entre as duas classes de uma classificação binária: a cor de
 * cada classe (as mesmas dos pontos) e o cinzento neutro na fronteira, a 50%.
 */
export function betweenClasses(probability: number): string {
  const p = clamp(probability);
  return p < 0.5 ? toBlue((0.5 - p) * 2) : toOrange((p - 0.5) * 2);
}

function clamp(t: number): number {
  return Number.isFinite(t) ? Math.max(0, Math.min(1, t)) : 0;
}

/** O texto escrito em cima de `fill`: o giz nas cores escuras, a ardósia nas claras. */
export function inkOn(fill: string): string {
  const parsed = parseColor(fill)?.rgb();
  if (parsed === undefined) {
    return "var(--foreground)";
  }
  const luminance = (0.299 * parsed.r + 0.587 * parsed.g + 0.114 * parsed.b) / 255;
  return luminance > 0.55 ? "#141a21" : "#ece7dd";
}

/**
 * Os três canais (de 0 a 255) de uma cor em qualquer formato do CSS: `#rrggbb` (a paleta) ou
 * `rgb(…)` (o que as escalas do d3 devolvem). Para pintar o campo pixel a pixel.
 */
export function channels(value: string): [number, number, number] {
  const parsed = parseColor(value)?.rgb();
  return parsed === undefined
    ? [0, 0, 0]
    : [Math.round(parsed.r), Math.round(parsed.g), Math.round(parsed.b)];
}

/** Uma cor com uma transparência, para os véus por baixo dos pontos. */
export function withAlpha(value: string, alpha: number): string {
  const [r, g, b] = channels(value);
  return `rgba(${String(r)}, ${String(g)}, ${String(b)}, ${String(alpha)})`;
}
