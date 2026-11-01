/**
 * Uma rede neuronal densa, da entrada (à esquerda) à saída (à direita): cada círculo é um
 * neurónio e cada camada liga-se toda à seguinte.
 *
 * Nas camadas escondidas, a cor é a ativação média do neurónio num passo do treino (na rampa
 * azul: quase apagado perto do fundo, aceso nos valores altos); um neurónio parado fica
 * cinzento. Com o passo a mudar, vê-se as camadas a organizarem-se.
 */
import { useState, type ReactNode } from "react";

import type { NetworkTrace } from "@/api/generated";
import { OTHER, sequential } from "@/lib/colors";
import { formatCount, formatNumber } from "@/lib/format";

import { Chart, ChartTooltip, TooltipRow } from "./chart";

interface NetworkViewProps {
  network: NetworkTrace;
  /** O passo do treino a mostrar (por omissão, o último). */
  step?: number;
  description: string;
  caption?: ReactNode;
  height?: number;
}

/** Com mais neurónios do que isto numa camada, desenham-se só os primeiros. */
const MAX_UNITS = 24;

interface Column {
  name: string;
  detail: string;
  units: number;
  hidden: boolean;
  activations: readonly number[];
  stuck: readonly boolean[];
}

export function NetworkView({
  network,
  step,
  description,
  caption,
  height = 300,
}: NetworkViewProps) {
  const [hover, setHover] = useState<[number, number] | null>(null);
  const frame =
    (step === undefined ? undefined : network.frames.find((one) => one.step === step)) ??
    network.frames.at(-1);
  const columns: Column[] = [
    {
      name: "Entrada",
      detail: `${formatCount(network.inputs)} features`,
      units: network.inputs,
      hidden: false,
      activations: [],
      stuck: [],
    },
    ...network.layers.map((layer, index) => ({
      name: layer.name,
      detail: `${formatCount(layer.units)}, ${layer.activation}`,
      units: layer.units,
      hidden: layer.hidden,
      activations: frame?.layers[index]?.activations ?? [],
      stuck: frame?.layers[index]?.stuck_units ?? [],
    })),
  ];
  const strongest = Math.max(1e-9, ...columns.flatMap((column) => column.activations));
  const tallest = Math.max(...columns.map((column) => Math.min(column.units, MAX_UNITS)));

  const layout = (width: number, plotHeight: number) => {
    const gapX = columns.length > 1 ? width / (columns.length - 1) : 0;
    const spacing = Math.min(22, (plotHeight - 8) / Math.max(1, tallest));
    const radius = Math.max(3, Math.min(8, spacing * 0.36));
    const positions = columns.map((column, c) => {
      const shown = Math.min(column.units, MAX_UNITS);
      const top = plotHeight / 2 - ((shown - 1) * spacing) / 2;
      return Array.from({ length: shown }, (_, u) => ({ x: c * gapX, y: top + u * spacing }));
    });
    return { positions, radius };
  };

  return (
    <Chart
      height={height}
      description={description}
      caption={caption}
      margin={{ top: 8, bottom: 40, left: 28, right: 28 }}
      legend={
        <ul className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
          <li className="flex items-center gap-1.5">
            <span
              aria-hidden="true"
              className="inline-block h-2 w-16 rounded-sm"
              style={{
                background: `linear-gradient(to right, ${sequential(0)}, ${sequential(0.5)}, ${sequential(1)})`,
              }}
            />
            Ativação média, de baixa a alta
          </li>
          <li className="flex items-center gap-1.5">
            <span
              aria-hidden="true"
              className="size-2 rounded-full"
              style={{ background: OTHER }}
            />
            Parado
          </li>
        </ul>
      }
      overlay={(plot) => {
        if (hover === null) {
          return null;
        }
        const [c, u] = hover;
        const column = columns[c];
        const point = layout(plot.width, plot.height).positions[c]?.[u];
        if (column === undefined || point === undefined) {
          return null;
        }
        const activation = column.activations[u];
        return (
          <ChartTooltip x={point.x} y={point.y} plot={plot}>
            <TooltipRow
              value={
                column.stuck[u] === true
                  ? "parado"
                  : activation === undefined
                    ? column.name
                    : formatNumber(activation, 2)
              }
              label={activation === undefined ? "" : "ativação média"}
            />
            <div className="text-muted-foreground">
              {column.name}, neurónio {formatCount(u + 1)}
            </div>
          </ChartTooltip>
        );
      }}
    >
      {(plot) => {
        const { positions, radius } = layout(plot.width, plot.height);
        return (
          <>
            {positions.slice(1).map((targets, index) => {
              const sources = positions[index] ?? [];
              return (
                <g key={`ligacoes-${String(index)}`} aria-hidden="true">
                  {sources.flatMap((source, s) =>
                    targets.map((target, t) => (
                      <line
                        key={`${String(s)}-${String(t)}`}
                        x1={source.x}
                        y1={source.y}
                        x2={target.x}
                        y2={target.y}
                        stroke="var(--input)"
                        strokeOpacity={0.22}
                      />
                    )),
                  )}
                </g>
              );
            })}
            {columns.map((column, c) => (
              <g key={column.name}>
                {(positions[c] ?? []).map((point, u) => {
                  const activation = column.activations[u];
                  const fill =
                    column.stuck[u] === true
                      ? OTHER
                      : column.hidden && activation !== undefined
                        ? sequential(activation / strongest)
                        : "var(--card)";
                  return (
                    <circle
                      key={u}
                      cx={point.x}
                      cy={point.y}
                      r={radius}
                      fill={fill}
                      stroke={
                        hover?.[0] === c && hover[1] === u ? "var(--foreground)" : "var(--input)"
                      }
                      strokeWidth={hover?.[0] === c && hover[1] === u ? 2 : 1}
                      onPointerEnter={() => {
                        setHover([c, u]);
                      }}
                      onPointerLeave={() => {
                        setHover(null);
                      }}
                    />
                  );
                })}
                {column.units > MAX_UNITS && (
                  <text
                    x={positions[c]?.[0]?.x ?? 0}
                    y={plot.height + 4}
                    textAnchor="middle"
                    className="fill-muted-foreground text-[11px]"
                  >
                    e mais {formatCount(column.units - MAX_UNITS)}
                  </text>
                )}
                <text
                  x={positions[c]?.[0]?.x ?? 0}
                  y={plot.height + 20}
                  textAnchor="middle"
                  className="fill-foreground text-[12px]"
                >
                  {column.name}
                </text>
                <text
                  x={positions[c]?.[0]?.x ?? 0}
                  y={plot.height + 34}
                  textAnchor="middle"
                  className="fill-muted-foreground text-[11px]"
                >
                  {column.detail}
                </text>
              </g>
            ))}
          </>
        );
      }}
    </Chart>
  );
}
