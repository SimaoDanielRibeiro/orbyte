/**
 * Ler um JSON Schema do backend: as propriedades (com as referências resolvidas), os valores
 * por omissão e os passos dos controlos numéricos.
 */

export type Values = Record<string, unknown>;

export interface Property {
  type?: string;
  title?: string;
  description?: string;
  default?: unknown;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  enum?: unknown[];
  "x-scale"?: string;
  "x-labels"?: Record<string, string>;
}

/** As propriedades do esquema, com as referências (`$ref`) resolvidas. */
export function schemaProperties(schema: Record<string, unknown>): [string, Property][] {
  const definitions = (schema.$defs ?? {}) as Record<string, Property>;
  const properties = (schema.properties ?? {}) as Record<string, Property & { $ref?: string }>;
  return Object.entries(properties).map(([name, raw]) => {
    const { $ref: reference, ...rest } = raw;
    const resolved =
      reference === undefined
        ? rest
        : { ...definitions[reference.split("/").pop() ?? ""], ...rest };
    return [name, resolved];
  });
}

/** Os valores por omissão do esquema. */
export function schemaDefaults(schema: Record<string, unknown>): Values {
  return Object.fromEntries(
    schemaProperties(schema).map(([name, property]) => [name, property.default]),
  );
}

/** Os valores 1, 2 e 5 de cada potência de 10 entre os limites, e o valor atual. */
export function stepOptions(
  minimum: number,
  maximum: number,
  current: number,
  integer: boolean,
): number[] {
  const options = new Set<number>([current, minimum, maximum]);
  const low = Math.floor(Math.log10(Math.max(minimum, integer ? 1 : 1e-12)));
  const high = Math.ceil(Math.log10(maximum));
  for (let exponent = low; exponent <= high; exponent++) {
    for (const multiple of [1, 2, 5]) {
      const option = Number((multiple * 10 ** exponent).toPrecision(12));
      if (option >= minimum && option <= maximum && (!integer || Number.isInteger(option))) {
        options.add(option);
      }
    }
  }
  return [...options].sort((a, b) => a - b);
}

/** Um passo "redondo" (0,01, 0,05, 0,1…) que dá cerca de cem posições no controlo. */
export function niceStep(minimum: number, maximum: number): number {
  const raw = (maximum - minimum) / 100;
  if (raw <= 0) {
    return 0.01;
  }
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  for (const multiple of [1, 2, 5, 10]) {
    if (raw <= multiple * magnitude) {
      return Number((multiple * magnitude).toPrecision(12));
    }
  }
  return raw;
}
