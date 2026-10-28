/**
 * Números escritos à portuguesa, com as mesmas regras de `ml_lab/formatting.py`: vírgula
 * decimal e espaço a separar os milhares, só a partir de cinco algarismos (1234, mas 12 345).
 *
 * As regras são as mesmas nos dois lados para que um número escrito pelo backend num aviso e o
 * mesmo número numa tabela da interface apareçam iguais. Por isso o arredondamento também é o
 * do Python: metade para o par (`halfEven`), sobre o valor exato do número. O `toFixed` do
 * JavaScript arredonda metade para cima, e 0,8125 daria 81,3% em vez de 81,2%.
 */
import type { MetricResult } from "@/api/generated";

const GROUPING_THRESHOLD = 10_000;
const SCIENTIFIC_FROM = 6;
const SUPERSCRIPTS: Readonly<Record<string, string>> = {
  "-": "⁻",
  "0": "⁰",
  "1": "¹",
  "2": "²",
  "3": "³",
  "4": "⁴",
  "5": "⁵",
  "6": "⁶",
  "7": "⁷",
  "8": "⁸",
  "9": "⁹",
};

const formats = new Map<string, Intl.NumberFormat>();

/** Os formatos do `Intl` custam a criar: guarda-se um por combinação de opções. */
function numberFormat(options: Intl.NumberFormatOptions): Intl.NumberFormat {
  const key = JSON.stringify(options);
  let format = formats.get(key);
  if (format === undefined) {
    // O formato americano tem o ponto decimal e a vírgula dos milhares, que depois se trocam,
    // como no Python: assim os dois lados escrevem exatamente os mesmos caracteres.
    format = new Intl.NumberFormat("en-US", { roundingMode: "halfEven", ...options });
    formats.set(key, format);
  }
  return format;
}

/** Troca a pontuação americana pela portuguesa, em dois passos para não se confundirem. */
function toPortuguese(text: string): string {
  return text.replaceAll(",", " ").replaceAll(".", ",");
}

function superscript(exponent: number): string {
  return String(exponent).replace(/[-0-9]/g, (character) => SUPERSCRIPTS[character] ?? character);
}

/** Escreve `value` com `decimals` casas decimais. Os não finitos dão `—` (em falta) ou `∞`. */
export function formatNumber(value: number, decimals = 2): string {
  if (Number.isNaN(value)) {
    return "—";
  }
  if (!Number.isFinite(value)) {
    return value > 0 ? "∞" : "-∞";
  }
  const format = numberFormat({
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
    useGrouping: Math.abs(value) >= GROUPING_THRESHOLD,
  });
  return toPortuguese(format.format(value));
}

/** Escreve um número inteiro (`20 640`). */
export function formatCount(value: number): string {
  return formatNumber(value, 0);
}

/** Escreve uma proporção como percentagem: `0.1987` dá `19,9%`. */
export function formatPercent(ratio: number, decimals = 1): string {
  return `${formatNumber(ratio * 100, decimals)}%`;
}

/** A mantissa e o expoente de `value` em notação científica, já arredondados. */
function scientificParts(value: number, fractionDigits: number): [string, number] {
  const format = numberFormat({
    notation: "scientific",
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  });
  const [mantissa = "", exponent = "0"] = format.format(value).split("E");
  return [mantissa, Number(exponent)];
}

/** Escreve `value` como potência de 10 (`2,5 × 10⁻⁵`), para números muito pequenos ou grandes. */
export function formatScientific(value: number, digits = 3): string {
  if (value === 0 || !Number.isFinite(value)) {
    return formatNumber(value, 0);
  }
  const [mantissa, exponent] = scientificParts(value, digits - 1);
  const trimmed = mantissa.includes(".")
    ? mantissa.replace(/0+$/, "").replace(/\.$/, "")
    : mantissa;
  return `${trimmed.replace(".", ",")} × 10${superscript(exponent)}`;
}

/**
 * Escreve `value` com `digits` algarismos significativos (uma loss, por exemplo). Os números
 * muito grandes (a partir de um milhão) ou muito pequenos ficam em potência de 10.
 */
export function formatSignificant(value: number, digits = 4): string {
  if (value === 0 || !Number.isFinite(value)) {
    return formatNumber(value, 0);
  }
  const magnitude = Math.floor(Math.log10(Math.abs(value)));
  if (magnitude >= SCIENTIFIC_FROM || magnitude < -3) {
    return formatScientific(value, digits);
  }
  return formatNumber(value, Math.max(0, digits - 1 - magnitude));
}

/**
 * Escreve uma estatística (uma média, um quartil) só com os algarismos que dizem alguma coisa:
 * até quatro significativos e sem zeros à direita (`0,42`, `29,7`, `80`). Os números a partir de
 * mil escrevem-se por inteiro (`206 856`), e os muito pequenos em potência de 10.
 */
export function formatStatistic(value: number): string {
  if (value === 0 || !Number.isFinite(value)) {
    return formatNumber(value, 0);
  }
  if (Math.abs(value) >= 1000) {
    return formatNumber(value, 0);
  }
  const magnitude = Math.floor(Math.log10(Math.abs(value)));
  if (magnitude < -3) {
    return formatScientific(value, 3);
  }
  const text = formatNumber(value, Math.max(0, 3 - magnitude));
  return text.includes(",") ? text.replace(/0+$/, "").replace(/,$/, "") : text;
}

/** Escreve uma quantidade com o nome no singular ou no plural (`12 linhas duplicadas`). */
export function formatQuantity(count: number, singular: string, plural: string): string {
  return `${formatCount(count)} ${count === 1 ? singular : plural}`;
}

/**
 * Escreve um valor qualquer (o de um hiperparâmetro, por exemplo) à portuguesa.
 *
 * Os decimais ficam com `digits` algarismos significativos, sem zeros à direita, e os muito
 * pequenos em potência de 10, como o formato `g` do Python. Uma diferença inevitável: o JSON
 * não distingue `3` de `3.0`, por isso um número sem parte decimal escreve-se como inteiro.
 */
export function formatValue(value: unknown, digits = 3): string {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "boolean") {
    return value ? "sim" : "não";
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      return formatNumber(value);
    }
    if (Number.isInteger(value)) {
      return formatCount(value);
    }
    const [, exponent] = scientificParts(value, digits - 1);
    if (exponent < -4 || exponent >= digits) {
      return formatScientific(value, digits);
    }
    const text = numberFormat({ maximumSignificantDigits: digits, useGrouping: false }).format(
      value,
    );
    return text.replace(".", ",");
  }
  if (typeof value === "string") {
    return value;
  }
  if (Array.isArray(value)) {
    // As camadas de uma rede, por exemplo: [64, 32].
    return `[${value.map((item) => formatValue(item, digits)).join(", ")}]`;
  }
  return JSON.stringify(value);
}

/**
 * Escreve um momento (em ISO 8601, como a API os manda) na hora local, por extenso curto:
 * `24/10/2026 15:30`.
 */
export function formatMoment(iso: string): string {
  const moment = new Date(iso);
  if (Number.isNaN(moment.getTime())) {
    return "—";
  }
  const two = (value: number) => String(value).padStart(2, "0");
  return `${two(moment.getDate())}/${two(moment.getMonth() + 1)}/${String(moment.getFullYear())} ${two(moment.getHours())}:${two(moment.getMinutes())}`;
}

/** Escreve uma lista como se diz em português, com "e" antes do último. */
export function formatList(items: readonly string[]): string {
  if (items.length <= 1) {
    return items.join("");
  }
  return `${items.slice(0, -1).join(", ")} e ${items.at(-1) ?? ""}`;
}

/**
 * O valor de uma métrica, como `MetricResult.formatted` no Python: percentagem com uma casa,
 * contagens sem casas, as outras com três; `—` quando não existe.
 */
export function formatMetric(
  metric: Pick<MetricResult, "value" | "as_percent" | "higher_is_better">,
): string {
  const { value } = metric;
  if (value === null || Number.isNaN(value)) {
    return "—";
  }
  if (metric.as_percent) {
    return formatPercent(value);
  }
  if (Number.isInteger(value) && metric.higher_is_better === null) {
    return formatNumber(value, 0);
  }
  return formatNumber(value, 3);
}
