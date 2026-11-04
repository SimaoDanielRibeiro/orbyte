/**
 * A procura num gráfico: cada candidato pela ordem em que foi medido, com a média na validação
 * cruzada e a barra do desvio entre folds. A cor é o agente (a família), a forma é a ronda (a
 * triagem ou a afinação), e o vencedor tem um anel.
 */
import { extent, scaleLinear } from "d3";
import { useState, type ReactNode } from "react";

import type { Candidate, ModelFamily, ObjectiveOption } from "@/api/generated";
import { useLabel } from "@/api/vocabulary";
import {
  AxisBottom,
  AxisLeft,
  Chart,
  ChartTooltip,
  Legend,
  TooltipRow,
} from "@/components/charts/chart";
import { SHAPE_PATHS } from "@/components/charts/shapes";
import { stepDecimals } from "@/components/charts/ticks";
import { familyColor } from "@/lib/families";
import { formatCount, formatNumber } from "@/lib/format";

import { changesText, objectiveValue } from "./values";

interface SearchChartProps {
  candidates: readonly Candidate[];
  objective: ObjectiveOption;
  /** As famílias pela ordem do plano (a cor de cada agente). */
  families: readonly ModelFamily[];
  winner: number | null;
  description: string;
  caption?: ReactNode;
}

export function SearchChart({
  candidates,
  objective,
  families,
  winner,
  description,
  caption,
}: SearchChartProps) {
  const label = useLabel();
  const [hover, setHover] = useState<number | null>(null);
  const measured = candidates.filter(
    (candidate): candidate is Candidate & { mean: number } => candidate.mean !== null,
  );
  const values = measured.flatMap((candidate) => [
    candidate.mean - (candidate.std ?? 0),
    candidate.mean + (candidate.std ?? 0),
  ]);
  const [low = 0, high = 1] = extent(values);
  const last = Math.max(1, ...measured.map((candidate) => candidate.number));
  const phases = [...new Set(measured.map((candidate) => candidate.phase))];
  const format = (value: number) =>
    objective.as_percent ? `${formatNumber(value * 100, 0)}%` : formatNumber(value, 2);

  const scales = (width: number, height: number) => ({
    x: scaleLinear()
      .domain([0.5, last + 0.5])
      .range([0, width]),
    y: scaleLinear().domain([low, high]).nice().range([height, 0]),
  });

  return (
    <Chart
      height={300}
      description={description}
      caption={caption}
      margin={{ left: 56, bottom: 40 }}
      legend={
        <Legend
          items={[
            ...families
              .filter((family) => measured.some((candidate) => candidate.family === family))
              .map((family) => ({
                label: label("ModelFamily", family),
                color: familyColor(family),
                kind: "dot" as const,
              })),
            ...phases.map((phase) => ({
              label: label("SearchPhase", phase),
              color: "var(--muted-foreground)",
              shape: phase === "tuning" ? ("diamond" as const) : ("circle" as const),
            })),
          ]}
        />
      }
      overlay={(plot) => {
        const candidate = measured.find((one) => one.number === hover);
        if (candidate === undefined) {
          return null;
        }
        const { x, y } = scales(plot.width, plot.height);
        return (
          <ChartTooltip x={x(candidate.number)} y={y(candidate.mean)} plot={plot}>
            <TooltipRow
              value={`${objectiveValue(candidate.mean, objective)} ± ${objectiveValue(candidate.std, objective)}`}
              label={objective.label}
              color={familyColor(candidate.family)}
              kind="dot"
            />
            <div className="mt-1">{candidate.model_display_name}</div>
            <div className="text-muted-foreground">
              N.º {formatCount(candidate.number)}, {changesText(candidate)}
            </div>
          </ChartTooltip>
        );
      }}
    >
      {(plot) => {
        const { x, y } = scales(plot.width, plot.height);
        const yTicks = y.ticks(5);
        const xTicks = x
          .ticks(Math.min(last, Math.max(2, Math.floor(plot.width / 60))))
          .filter(Number.isInteger);
        return (
          <>
            <AxisLeft
              ticks={yTicks}
              position={y}
              format={
                objective.as_percent ? format : (value) => formatNumber(value, stepDecimals(yTicks))
              }
              plot={plot}
              label={objective.label}
            />
            <AxisBottom
              ticks={xTicks}
              position={x}
              format={(value) => formatCount(value)}
              plot={plot}
              label="Candidate (in the order it was measured)"
            />
            {measured.map((candidate) => {
              const cx = x(candidate.number);
              const spread = candidate.std ?? 0;
              return (
                <g
                  key={candidate.number}
                  onPointerEnter={() => {
                    setHover(candidate.number);
                  }}
                  onPointerLeave={() => {
                    setHover(null);
                  }}
                >
                  <rect
                    x={cx - 12}
                    y={0}
                    width={24}
                    height={plot.height}
                    fill={hover === candidate.number ? "var(--accent)" : "transparent"}
                    opacity={0.5}
                  />
                  {spread > 0 && (
                    <line
                      x1={cx}
                      x2={cx}
                      y1={y(candidate.mean - spread)}
                      y2={y(candidate.mean + spread)}
                      stroke={familyColor(candidate.family)}
                      strokeOpacity={0.6}
                      strokeWidth={2}
                    />
                  )}
                  {candidate.number === winner && (
                    <circle
                      cx={cx}
                      cy={y(candidate.mean)}
                      r={9}
                      fill="none"
                      stroke="var(--foreground)"
                      strokeWidth={1.5}
                    />
                  )}
                  <path
                    d={SHAPE_PATHS[candidate.phase === "tuning" ? "diamond" : "circle"]}
                    transform={`translate(${String(cx)},${String(y(candidate.mean))})`}
                    fill={familyColor(candidate.family)}
                    stroke="var(--card)"
                    strokeWidth={1.5}
                  />
                </g>
              );
            })}
          </>
        );
      }}
    </Chart>
  );
}
