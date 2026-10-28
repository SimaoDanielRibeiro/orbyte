/**
 * Uma superfície vista de cima, em curvas de nível: o erro de um modelo para cada par de pesos
 * (a "taça" do gradiente descendente), com o caminho que os pesos fizeram por cima.
 *
 * As curvas de nível calculam-se com os contornos do D3, em escala logarítmica: o fundo da taça
 * e as paredes ficam com o mesmo número de curvas. A cor é a magnitude (a rampa azul, o erro
 * baixo junto ao escuro do fundo); o caminho e o fundo da taça ficam a branco, por cima.
 */
import { contours, extent, line, scaleLinear } from "d3";
import { useId, useMemo, useState, type ReactNode } from "react";

import { LIGHT_INK, SURFACE, sequential } from "@/lib/colors";
import { formatNumber, formatStatistic } from "@/lib/format";

import { AxisBottom, AxisLeft, Chart, ChartTooltip, Legend, TooltipRow } from "./chart";
import { stepDecimals } from "./ticks";

const LEVELS = 18;

interface ContourChartProps {
  /** Os valores de x da grelha (as colunas). */
  xs: readonly number[];
  /** Os valores de y da grelha (as linhas). */
  ys: readonly number[];
  /** `values[i][j]`: o valor em `ys[i]`, `xs[j]` (positivo: a escala é logarítmica). */
  values: readonly (readonly number[])[];
  /** O nome do valor (na dica). */
  valueLabel: string;
  /** Um caminho por cima da superfície (os pesos, época a época). */
  path?: { x: readonly number[]; y: readonly number[]; label: string };
  /** Um ponto marcado com uma cruz (o fundo da taça). */
  target?: { x: number; y: number; label: string };
  description: string;
  caption?: ReactNode;
  height?: number;
  xLabel?: string;
  yLabel?: string;
}

export function ContourChart({
  xs,
  ys,
  values,
  valueLabel,
  path,
  target,
  description,
  caption,
  height = 380,
  xLabel,
  yLabel,
}: ContourChartProps) {
  const [hover, setHover] = useState<{ x: number; y: number; i: number; j: number } | null>(null);
  const clipId = `curvas-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const [xMin = 0, xMax = 1] = [xs[0], xs[xs.length - 1]];
  const [yMin = 0, yMax = 1] = [ys[0], ys[ys.length - 1]];
  // As bandas calculam-se uma vez por superfície, no espaço da grelha (de 0 ao número de células).
  const bands = useMemo(() => {
    const flat = values.flat().map((value) => Math.log(Math.max(value, 1e-12)));
    const [low = 0, high = 1] = extent(flat);
    const thresholds = Array.from(
      { length: LEVELS },
      (_, level) => low + ((high - low) * level) / LEVELS,
    );
    const columns = values[0]?.length ?? 0;
    return contours()
      .size([columns, values.length])
      .thresholds(thresholds)(flat)
      .map((shape, level) => ({ level, rings: shape.coordinates.flat() as [number, number][][] }));
  }, [values]);

  return (
    <Chart
      height={height}
      description={description}
      caption={caption}
      margin={{ left: 56, bottom: 40, right: 12 }}
      legend={
        <Legend
          items={[
            ...(path === undefined
              ? []
              : [{ label: path.label, color: LIGHT_INK, kind: "line" as const }]),
            ...(target === undefined
              ? []
              : [{ label: target.label, color: LIGHT_INK, kind: "dot" as const }]),
          ]}
        />
      }
      overlay={(plot) => {
        const value = hover === null ? undefined : values[hover.i]?.[hover.j];
        if (hover === null || value === undefined) {
          return null;
        }
        return (
          <ChartTooltip x={hover.x} y={hover.y} plot={plot}>
            <TooltipRow value={formatStatistic(value)} label={valueLabel} />
            <div className="text-muted-foreground tabular-nums">
              {xLabel ?? "x"} {formatNumber(xs[hover.j] ?? 0, 2)}, {yLabel ?? "y"}{" "}
              {formatNumber(ys[hover.i] ?? 0, 2)}
            </div>
          </ChartTooltip>
        );
      }}
    >
      {(plot) => {
        const x = scaleLinear().domain([xMin, xMax]).range([0, plot.width]);
        const y = scaleLinear().domain([yMin, yMax]).range([plot.height, 0]);
        const xTicks = x.ticks(Math.max(2, Math.floor(plot.width / 90)));
        const yTicks = y.ticks(5);
        // Do espaço da grelha (os cantos das células) para o desenho.
        const toPlot = (gx: number, gy: number): [number, number] => [
          x(xMin + (gx - 0.5) * ((xMax - xMin) / Math.max(1, xs.length - 1))),
          y(yMin + (gy - 0.5) * ((yMax - yMin) / Math.max(1, ys.length - 1))),
        ];
        const ring = (points: [number, number][]) =>
          `M${points.map(([gx, gy]) => toPlot(gx, gy).join(",")).join("L")}Z`;
        const walk =
          path === undefined
            ? ""
            : (line<number>()
                .x((_, i) => x(path.x[i] ?? 0))
                .y((_, i) => y(path.y[i] ?? 0))(path.x.map((_, i) => i)) ?? "");
        return (
          <>
            <defs>
              <clipPath id={clipId}>
                <rect width={plot.width} height={plot.height} />
              </clipPath>
            </defs>
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
            <g clipPath={`url(#${clipId})`} pointerEvents="none">
              {bands.map(({ level, rings }) => (
                <path
                  key={level}
                  d={rings.map(ring).join("")}
                  // Cada nível cobre tudo o que está acima dele: pintados por ordem e opacos, ficam
                  // as bandas, do fundo da taça às paredes.
                  fill={sequential(level / (LEVELS - 1))}
                  stroke={SURFACE}
                  strokeOpacity={0.45}
                  strokeWidth={0.75}
                />
              ))}
              {path !== undefined && (
                <g>
                  <path
                    d={walk}
                    fill="none"
                    stroke={SURFACE}
                    strokeWidth={4.5}
                    strokeLinejoin="round"
                  />
                  <path
                    d={walk}
                    fill="none"
                    stroke={LIGHT_INK}
                    strokeWidth={2}
                    strokeLinejoin="round"
                  />
                  {path.x.map((px, i) => (
                    <circle
                      key={i}
                      cx={x(px)}
                      cy={y(path.y[i] ?? 0)}
                      r={i === 0 ? 4.5 : 2.5}
                      fill={i === 0 ? SURFACE : LIGHT_INK}
                      stroke={i === 0 ? LIGHT_INK : SURFACE}
                      strokeWidth={1.5}
                    />
                  ))}
                </g>
              )}
              {target !== undefined && (
                <g transform={`translate(${String(x(target.x))},${String(y(target.y))})`}>
                  <path
                    d="M-6,-6L6,6M-6,6L6,-6"
                    stroke={SURFACE}
                    strokeWidth={5}
                    strokeLinecap="round"
                  />
                  <path
                    d="M-6,-6L6,6M-6,6L6,-6"
                    stroke={LIGHT_INK}
                    strokeWidth={2}
                    strokeLinecap="round"
                  />
                </g>
              )}
            </g>
            <rect
              width={plot.width}
              height={plot.height}
              fill="transparent"
              onPointerMove={(event) => {
                const box = event.currentTarget.getBoundingClientRect();
                const px = event.clientX - box.left;
                const py = event.clientY - box.top;
                const j = Math.round(
                  ((x.invert(px) - xMin) / (xMax - xMin || 1)) * (xs.length - 1),
                );
                const i = Math.round(
                  ((y.invert(py) - yMin) / (yMax - yMin || 1)) * (ys.length - 1),
                );
                setHover({
                  x: px,
                  y: py,
                  i: Math.max(0, Math.min(ys.length - 1, i)),
                  j: Math.max(0, Math.min(xs.length - 1, j)),
                });
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
