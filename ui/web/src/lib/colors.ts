/**
 * As cores dos dados (ADR 0032: a cor fica para os dados, que brilham sobre o escuro).
 *
 * As cinco primeiras são as da referência, medidas na imagem (o azul, o amarelo, o rosa, o
 * verde e o roxo); as três seguintes juntam-se no mesmo tom para chegar às oito. A ordem é a
 * que passa no validador do skill de visualização de dados, sobre o cartão (#191919): as três
 * primeiras distinguem-se todas entre si, também com daltonismo (as dispersões e os campos
 * usam-nas assim), e as seguintes distinguem-se das vizinhas. O verde subiu um nada de tom
 * (#64c097 para #62c68c) para não se confundir com o rosa num daltonismo verde-vermelho.
 *
 * O validador queria-as um pouco mais escuras (a faixa de luminosidade do tema escuro), mas a
 * referência tem-nas assim, a brilhar: a escolha é do pedido, e as verificações que dizem se as
 * categorias se distinguem passam todas. Com mais de três classes numa dispersão, os pontos
 * ganham também uma forma diferente.
 *
 * As cores atribuem-se pela ordem e nunca se repetem: uma nona categoria fica cinzenta.
 */
import { color as parseColor, interpolateLab, piecewise } from "d3";

export const CATEGORICAL = [
  "#6996d4", // azul
  "#e2b94e", // amarelo
  "#d26da1", // rosa
  "#62c68c", // verde
  "#8e7dd8", // roxo
  "#e0884f", // laranja
  "#3db6d1", // ciano
  "#e36b6b", // vermelho
] as const;

/** Para o que fica de fora da paleta (uma nona categoria) e para o ruído do clustering. */
export const OTHER = "#707070";

/** O meio neutro das escalas divergentes: nem um lado nem o outro. */
export const NEUTRAL = "#2c2c2c";

/** O texto claro (e as linhas que se desenham a "branco"), sobre o escuro. */
export const LIGHT_INK = "#ededed";

/** O texto escuro, sobre as cores claras; também o fundo das páginas. */
export const DARK_INK = "#121212";

/** O fundo dos cartões, onde os gráficos se desenham: o anel à volta dos pontos tem esta cor. */
export const SURFACE = "#191919";

/** A cor da categoria número `index` (pela ordem em que aparecem). */
export function categoryColor(index: number): string {
  return CATEGORICAL[index] ?? OTHER;
}

/** Quantas categorias se distinguem só pela cor numa dispersão (as outras precisam de forma). */
export const COLOR_ONLY_LIMIT = 3;

/**
 * A rampa azul, do passo mais escuro ao mais claro (validada como escala ordenada: cada passo
 * distingue-se do vizinho, e o mais escuro ainda se vê sobre o cartão). No tema escuro, o "quase
 * zero" fica perto do fundo e os valores altos acendem.
 */
const BLUE_RAMP = ["#33507f", "#44669a", "#567eb7", "#6996d4", "#91b0e0", "#b7cbec", "#dbe6f8"];

/** A rampa amarela, para uma segunda magnitude no mesmo ecrã (validada como a azul). */
const AMBER_RAMP = ["#6a521e", "#a48435", "#e2b94e", "#efd28c", "#f9ecc8"];

const blue = piecewise(interpolateLab, BLUE_RAMP);
const amber = piecewise(interpolateLab, AMBER_RAMP);

/** Uma magnitude de 0 a 1, na rampa azul. */
export function sequential(t: number): string {
  return String(blue(clamp(t)));
}

/** Uma segunda magnitude no mesmo ecrã (a pontuação de anomalia), na rampa amarela. */
export function sequentialAlt(t: number): string {
  return String(amber(clamp(t)));
}

const toBlue = interpolateLab(NEUTRAL, CATEGORICAL[0]);
const toAmber = interpolateLab(NEUTRAL, CATEGORICAL[1]);
const toOrange = interpolateLab(NEUTRAL, CATEGORICAL[5]);

/**
 * Uma polaridade de -1 a 1 (uma correlação): azul de um lado, laranja do outro, e o cinzento
 * neutro no zero. Azul e laranja são o par que melhor se distingue com daltonismo.
 */
export function diverging(t: number): string {
  const value = Math.max(-1, Math.min(1, t));
  return value < 0 ? toOrange(-value) : toBlue(value);
}

/**
 * Uma probabilidade de 0 a 1 entre as duas classes de uma classificação binária: a cor de
 * cada classe (as mesmas dos pontos) e o cinzento neutro na fronteira, a 50%.
 */
export function betweenClasses(probability: number): string {
  const p = clamp(probability);
  return p < 0.5 ? toBlue((0.5 - p) * 2) : toAmber((p - 0.5) * 2);
}

function clamp(t: number): number {
  return Number.isFinite(t) ? Math.max(0, Math.min(1, t)) : 0;
}

/** O texto escrito em cima de `fill`: claro nas cores escuras, escuro nas claras. */
export function inkOn(fill: string): string {
  const parsed = parseColor(fill)?.rgb();
  if (parsed === undefined) {
    return "var(--foreground)";
  }
  const luminance = (0.299 * parsed.r + 0.587 * parsed.g + 0.114 * parsed.b) / 255;
  return luminance > 0.55 ? DARK_INK : LIGHT_INK;
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

/** Uma cor com uma transparência, para os véus por baixo dos pontos e o brilho das linhas. */
export function withAlpha(value: string, alpha: number): string {
  const [r, g, b] = channels(value);
  return `rgba(${String(r)}, ${String(g)}, ${String(b)}, ${String(alpha)})`;
}
