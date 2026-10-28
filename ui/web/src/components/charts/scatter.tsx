/**
 * Pontos num plano: a projeção 2D dos dados, o mapa da redução, os resíduos.
 *
 * Cada ponto tem um anel da cor do fundo, para se ver quando se sobrepõem. A dica segue o ponto
 * mais próximo do rato (até 24 px), para não ser preciso acertar num ponto pequeno. Com mais de
 * três categorias, a forma também muda, porque só três cores se distinguem com segurança.
 */
import { Delaunay, extent, scaleLinear } from "d3";
import { useState, type ReactNode } from "react";

import { formatNumber } from "@/lib/format";

import { AxisBottom, AxisLeft, Chart, ChartTooltip, Legend, type LegendItem } from "./chart";
import { SHAPE_PATHS, type PointShape } from "./shapes";
import { stepDecimals } from "./ticks";

export interface ScatterPoint {
  x: number;
  y: number;
  color: string;
  shape?: PointShape;
  /** O que a dica diz sobre o ponto. */
  label?: string;
}

interface ScatterProps {
  points: readonly ScatterPoint[];
  description: string;
  caption?: ReactNode;
  legend?: readonly LegendItem[];
  height?: number;
  xLabel?: string;
  yLabel?: string;
  /** A diagonal y = x (no gráfico do previsto contra o real). */
  identity?: boolean;
  /** Uma linha horizontal (o zero dos resíduos). */
  zeroLine?: boolean;
}

export function Scatter({
  points,
  description,
  caption,
  legend,
  height = 320,
  xLabel,
  yLabel,
  identity = false,
  zeroLine = false,
}: ScatterProps) {
  const [hover, setHover] = useState<number | null>(null);
  const [xMin = 0, xMax = 1] = extent(points, (point) => point.x);
  const [yMin = 0, yMax = 1] = extent(points, (point) => point.y);
  const low = identity ? Math.min(xMin, yMin) : 0;
  const high = identity ? Math.max(xMax, yMax) : 0;

  const scales = (width: number, plotHeight: number) => {
    const x = scaleLinear()
      .domain(identity ? [low, high] : [xMin, xMax])
      .nice()
      .range([0, width]);
    const y = scaleLinear()
      .domain(identity ? [low, high] : [yMin, yMax])
      .nice()
      .range([plotHeight, 0]);
    return { x, y };
  };

  return (
    <Chart
      height={height}
      description={description}
      caption={caption}
      margin={{ bottom: xLabel === undefined ? 24 : 40, left: yLabel === undefined ? 48 : 60 }}
      legend={legend === undefined ? undefined : <Legend items={legend} />}
      overlay={(plot) => {
        const point = hover === null ? undefined : points[hover];
        if (point === undefined) {
          return null;
        }
        const { x, y } = scales(plot.width, plot.height);
        return (
          <ChartTooltip x={x(point.x)} y={y(point.y)} plot={plot}>
            {point.label !== undefined && <div className="font-medium">{point.label}</div>}
            <div className="text-muted-foreground tabular-nums">
              {xLabel ?? "x"}: {formatNumber(point.x, 3)}
            </div>
            <div className="text-muted-foreground tabular-nums">
              {yLabel ?? "y"}: {formatNumber(point.y, 3)}
            </div>
          </ChartTooltip>
        );
      }}
    >
      {(plot) => {
        const { x, y } = scales(plot.width, plot.height);
        const xTicks = x.ticks(Math.max(2, Math.floor(plot.width / 90)));
        const yTicks = y.ticks(5);
        const delaunay = Delaunay.from(
          points,
          (point) => x(point.x),
          (point) => y(point.y),
        );
        const hovered = hover === null ? undefined : points[hover];
        return (
          <>
            <AxisLeft
              ticks={yTicks}
              position={y}
              format={(value) => formatNumber(value, stepDecimals(yTicks))}
              plot={plot}
              label={yLabel}
            />
            <AxisBottom
              ticks={xTicks}
              position={x}
              format={(value) => formatNumber(value, stepDecimals(xTicks))}
              plot={plot}
              label={xLabel}
            />
            {identity && (
              <line
                x1={x(low)}
                y1={y(low)}
                x2={x(high)}
                y2={y(high)}
                stroke="var(--input)"
                strokeDasharray="4 4"
              />
            )}
            {zeroLine && <line x1={0} x2={plot.width} y1={y(0)} y2={y(0)} stroke="var(--input)" />}
            {points.map((point, index) => (
              <path
                key={index}
                d={SHAPE_PATHS[point.shape ?? "circle"]}
                transform={`translate(${String(x(point.x))},${String(y(point.y))})`}
                fill={point.color}
                stroke="var(--card)"
                strokeWidth={1.5}
              />
            ))}
            {hovered !== undefined && (
              <circle
                cx={x(hovered.x)}
                cy={y(hovered.y)}
                r={7}
                fill="none"
                stroke="var(--foreground)"
                strokeWidth={1.5}
              />
            )}
            <rect
              width={plot.width}
              height={plot.height}
              fill="transparent"
              onPointerMove={(event) => {
                const box = event.currentTarget.getBoundingClientRect();
                const px = event.clientX - box.left;
                const py = event.clientY - box.top;
                const index = delaunay.find(px, py);
                const point = points[index];
                const near =
                  point !== undefined && Math.hypot(x(point.x) - px, y(point.y) - py) <= 24;
                setHover(near ? index : null);
              }}
              onPointerLeave={() => {
                setHover(null);
              }}
            />
          </>
        );
      }}
    </Chart>
  );
}
