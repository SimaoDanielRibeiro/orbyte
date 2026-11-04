/**
 * O mapa dos valores em falta: uma faixa por coluna e uma linha fina por linha do dataset (uma
 * seleção espalhada por ele todo). Os valores em falta são marcas de giz na ardósia; os presentes
 * ficam no tom da superfície.
 *
 * As linhas seguidas em falta juntam-se num só retângulo, para o desenho ficar leve.
 */
import { scaleBand } from "d3";
import { useState, type ReactNode } from "react";

import type { MissingMap } from "@/api/generated";
import { formatCount, formatList } from "@/lib/format";

import { Chart, ChartTooltip, TooltipRow } from "./chart";
import { slantedOverhang } from "./margins";

interface MissingMapViewProps {
  map: MissingMap;
  description: string;
  caption?: ReactNode;
}

const TOP = 64;
/** A inclinação dos nomes das colunas, por cima das faixas. */
const SLANT = 40;

/** O nome de uma coluna como fica por cima da faixa: os compridos encurtam. */
function shortName(name: string): string {
  return name.length > 14 ? `${name.slice(0, 13)}…` : name;
}

export function MissingMapView({ map, description, caption }: MissingMapViewProps) {
  const [hover, setHover] = useState<[number, number] | null>(null);
  const rows = map.rows.length;
  const height = Math.min(380, Math.max(160, rows * 1.6)) + TOP + 24;
  // O nome da última coluna sobe para a direita: a margem tem de lhe dar lugar.
  const right = slantedOverhang(shortName(map.columns.at(-1) ?? ""), SLANT);

  return (
    <Chart
      height={height}
      description={description}
      caption={caption}
      margin={{ top: TOP, left: 56, right, bottom: 24 }}
      overlay={(plot) => {
        if (hover === null) {
          return null;
        }
        const [row, column] = hover;
        const missing = map.missing[row] ?? [];
        const absent = map.columns.filter((_, index) => missing[index] === true);
        const x = scaleBand<number>()
          .domain(map.columns.map((_, index) => index))
          .range([0, plot.width]);
        return (
          <ChartTooltip
            x={(x(column) ?? 0) + x.bandwidth()}
            y={((row + 0.5) * plot.height) / rows}
            plot={plot}
          >
            <TooltipRow
              value={missing[column] === true ? "missing" : "present"}
              label={map.columns[column] ?? ""}
            />
            <div className="mt-1 text-muted-foreground">
              Row {formatCount((map.rows[row] ?? 0) + 1)}:{" "}
              {absent.length === 0 ? "nothing missing" : `missing ${formatList(absent)}`}
            </div>
          </ChartTooltip>
        );
      }}
    >
      {(plot) => {
        const x = scaleBand<number>()
          .domain(map.columns.map((_, index) => index))
          .range([0, plot.width])
          .paddingInner(0.08);
        const rowHeight = plot.height / Math.max(1, rows);
        return (
          <g>
            {map.columns.map((name, index) => (
              <text
                key={name}
                transform={`translate(${String((x(index) ?? 0) + x.bandwidth() / 2)},-8) rotate(${String(-SLANT)})`}
                className="fill-muted-foreground text-[13px]"
                aria-hidden="true"
              >
                {shortName(name)}
              </text>
            ))}
            {map.columns.map((name, column) => (
              <g key={name}>
                <rect
                  x={x(column) ?? 0}
                  y={0}
                  width={x.bandwidth()}
                  height={plot.height}
                  fill="var(--muted)"
                  opacity={hover?.[1] === column ? 0.9 : 0.55}
                />
                {runs(map.missing, column).map(([start, length]) => (
                  <rect
                    key={start}
                    x={x(column) ?? 0}
                    y={start * rowHeight}
                    width={x.bandwidth()}
                    height={Math.max(1, length * rowHeight)}
                    fill="var(--foreground)"
                    opacity={0.85}
                  />
                ))}
              </g>
            ))}
            {hover !== null && (
              <rect
                x={0}
                y={hover[0] * rowHeight}
                width={plot.width}
                height={Math.max(1, rowHeight)}
                fill="none"
                stroke="var(--foreground)"
                strokeWidth={1}
              />
            )}
            {/* A área da dica: a linha e a coluna saem da posição do rato. */}
            <rect
              width={plot.width}
              height={plot.height}
              fill="transparent"
              onPointerMove={(event) => {
                const box = event.currentTarget.getBoundingClientRect();
                const column = Math.floor(
                  ((event.clientX - box.left) / box.width) * map.columns.length,
                );
                const row = Math.floor(((event.clientY - box.top) / box.height) * rows);
                if (column >= 0 && column < map.columns.length && row >= 0 && row < rows) {
                  setHover([row, column]);
                }
              }}
              onPointerLeave={() => {
                setHover(null);
              }}
            />
            <text
              x={-8}
              y={0}
              dy="0.7em"
              textAnchor="end"
              className="fill-muted-foreground text-[13px] tabular-nums"
              aria-hidden="true"
            >
              {formatCount((map.rows[0] ?? 0) + 1)}
            </text>
            <text
              x={-8}
              y={plot.height}
              textAnchor="end"
              className="fill-muted-foreground text-[13px] tabular-nums"
              aria-hidden="true"
            >
              {formatCount((map.rows.at(-1) ?? 0) + 1)}
            </text>
          </g>
        );
      }}
    </Chart>
  );
}

/** Os troços de linhas seguidas em falta numa coluna: `[primeira, quantas]`. */
function runs(missing: readonly (readonly boolean[])[], column: number): [number, number][] {
  const found: [number, number][] = [];
  let start = -1;
  missing.forEach((row, index) => {
    if (row[column] === true) {
      if (start < 0) {
        start = index;
      }
    } else if (start >= 0) {
      found.push([start, index - start]);
      start = -1;
    }
  });
  if (start >= 0) {
    found.push([start, missing.length - start]);
  }
  return found;
}
