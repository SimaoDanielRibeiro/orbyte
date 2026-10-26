/**
 * O campo do modelo: o que ele prevê em cada ponto do plano, numa fotografia do treino.
 *
 * É o desenho mais característico do laboratório. O campo pinta-se num canvas, com um véu da
 * cor de cada classe (mais forte onde o modelo tem mais certeza); a fronteira de decisão é uma
 * linha a giz, calculada com os contornos do D3 sobre a mesma grelha; os pontos de treino ficam
 * por cima, com um anel da cor do fundo. Com uma só feature, o campo é uma curva.
 *
 * Os números vêm todos do backend (`TrainingAnimation`): aqui só se desenha.
 */
import { contours, extent, line, scaleLinear } from "d3";
import { useEffect, useId, useMemo, useRef, useState, type ReactNode } from "react";

import type { TrainingAnimation } from "@/api/generated";
import {
  betweenClasses,
  categoryColor,
  channels,
  COLOR_ONLY_LIMIT,
  OTHER,
  sequential,
  sequentialAlt,
} from "@/lib/colors";
import { formatNumber, formatPercent, formatSignificant } from "@/lib/format";

import {
  AxisBottom,
  AxisLeft,
  Chart,
  ChartTooltip,
  Legend,
  TooltipRow,
  type LegendItem,
} from "./chart";
import { stepDecimals } from "./ticks";

const CHALK = "#ece7dd";
const SLATE = "#141a21";

interface FieldViewProps {
  animation: TrainingAnimation;
  /** A fotografia a mostrar (por omissão, a última). */
  frameIndex?: number;
  description: string;
  caption?: ReactNode;
  height?: number;
}

export function FieldView({
  animation,
  frameIndex,
  description,
  caption,
  height = 380,
}: FieldViewProps) {
  const index = frameIndex ?? animation.frames.length - 1;
  return animation.dimensions === 1 ? (
    <CurveView
      animation={animation}
      index={index}
      description={description}
      caption={caption}
      height={height}
    />
  ) : (
    <PlaneView
      animation={animation}
      index={index}
      description={description}
      caption={caption}
      height={height}
    />
  );
}

/** As cores e as formas das categorias, decididas uma vez por animação. */
function useEncoding(animation: TrainingAnimation) {
  return useMemo(() => {
    const kind = animation.value_kind;
    const classification =
      animation.field_labels.length > 0 &&
      animation.field_labels.length === animation.class_labels.length &&
      animation.field_labels.every((label, i) => label === animation.class_labels[i]);
    const allValues = animation.frames.flatMap((frame) => frame.values.flat());
    const [low = 0, high = 1] = extent(allValues);
    const span = high - low || 1;
    /** A cor de uma célula do campo, já com a transparência do véu. */
    const cell = (value: number): [string, number] => {
      if (kind === "probability") {
        return [betweenClasses(value), 0.18 + Math.abs(value - 0.5) * 0.7];
      }
      if (kind === "class") {
        const category = Math.round(value);
        const noise = animation.noise && category === animation.field_labels.length - 1;
        return [noise ? OTHER : categoryColor(category), 0.32];
      }
      const t = (value - low) / span;
      return [kind === "score" ? sequentialAlt(t) : sequential(t), 0.55];
    };
    /** A cor de um ponto: a da sua classe, ou a do valor (regressão), ou giz (referência). */
    const point = (i: number): string => {
      const label = animation.points.labels[i];
      if ((kind === "probability" || classification) && label !== undefined) {
        return categoryColor(animation.class_labels.indexOf(label));
      }
      const value = animation.points.values[i];
      if (value !== undefined) {
        return sequential((value - low) / span);
      }
      return CHALK;
    };
    /** Na referência do clustering e das anomalias, a classe vai na forma (a cor é do campo). */
    const shapeOf = (i: number): number => {
      const label = animation.points.labels[i];
      if (label === undefined) {
        return 0;
      }
      const position = animation.class_labels.indexOf(label);
      if (kind === "probability" || classification) {
        return animation.class_labels.length > COLOR_ONLY_LIMIT ? position : 0;
      }
      return position;
    };
    return { kind, classification, low, high, cell, point, shapeOf };
  }, [animation]);
}

function PlaneView({
  animation,
  index,
  description,
  caption,
  height,
}: {
  animation: TrainingAnimation;
  index: number;
  description: string;
  caption?: ReactNode;
  height: number;
}) {
  const encoding = useEncoding(animation);
  const [hover, setHover] = useState<{ x: number; y: number; i: number; j: number } | null>(null);
  const frame = animation.frames[Math.min(index, animation.frames.length - 1)];
  const xs = animation.xs;
  const ys = animation.ys;
  const [xMin = 0, xMax = 1] = [xs[0], xs[xs.length - 1]];
  const [yMin = 0, yMax = 1] = [ys[0], ys[ys.length - 1]];
  const clipId = `campo-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  // As fronteiras calculam-se uma vez por fotografia (não a cada movimento do rato).
  const lines = useMemo(
    () => (frame === undefined ? [] : boundaries(animation, frame.values, encoding.kind)),
    [animation, frame, encoding.kind],
  );

  const legend = fieldLegend(animation, encoding.kind, encoding.low, encoding.high);

  return (
    <Chart
      height={height}
      description={description}
      caption={caption}
      margin={{ left: 52, bottom: 40, right: 12 }}
      legend={<Legend items={legend} />}
      underlay={(plot) =>
        frame === undefined ? null : (
          <FieldCanvas
            animation={animation}
            frameIndex={index}
            encoding={encoding}
            left={plot.margin.left}
            top={plot.margin.top}
            width={plot.width}
            height={plot.height}
          />
        )
      }
      overlay={(plot) => {
        if (hover === null || frame === undefined) {
          return null;
        }
        const value = frame.values[hover.i]?.[hover.j];
        if (value === undefined) {
          return null;
        }
        return (
          <ChartTooltip x={hover.x} y={hover.y} plot={plot}>
            <TooltipRow value={writeValue(animation, value)} label={animation.value_label} />
          </ChartTooltip>
        );
      }}
    >
      {(plot) => {
        const x = scaleLinear().domain([xMin, xMax]).range([0, plot.width]);
        const y = scaleLinear().domain([yMin, yMax]).range([plot.height, 0]);
        const xTicks = x.ticks(Math.max(2, Math.floor(plot.width / 90)));
        const yTicks = y.ticks(5);
        const toPlot = (gx: number, gy: number): [number, number] => [
          x(xMin + (gx - 0.5) * ((xMax - xMin) / Math.max(1, xs.length - 1))),
          y(yMin + (gy - 0.5) * ((yMax - yMin) / Math.max(1, ys.length - 1))),
        ];
        return (
          <>
            <AxisLeft
              ticks={yTicks}
              position={y}
              format={(v) => formatNumber(v, stepDecimals(yTicks))}
              plot={plot}
              label={animation.axis_labels[1]}
            />
            <AxisBottom
              ticks={xTicks}
              position={x}
              format={(v) => formatNumber(v, stepDecimals(xTicks))}
              plot={plot}
              label={animation.axis_labels[0]}
            />
            <defs>
              <clipPath id={clipId}>
                <rect width={plot.width} height={plot.height} />
              </clipPath>
            </defs>
            <g clipPath={`url(#${clipId})`}>
              {lines.map((rings, k) => (
                <path
                  key={k}
                  d={rings
                    .map(
                      (ring) => `M${ring.map(([gx, gy]) => toPlot(gx, gy).join(",")).join("L")}Z`,
                    )
                    .join("")}
                  fill="none"
                  stroke={CHALK}
                  strokeWidth={encoding.kind === "probability" ? 2 : 1.25}
                  strokeOpacity={encoding.kind === "probability" ? 0.95 : 0.7}
                  strokeLinejoin="round"
                />
              ))}
              {frame?.outlines.map((outline, k) => (
                <path
                  key={`outline-${String(k)}`}
                  d={
                    line<number>()
                      .x((_, i) => x(outline.x[i] ?? 0))
                      .y((_, i) => y(outline.y[i] ?? 0))(outline.x.map((_, i) => i)) ?? ""
                  }
                  fill="none"
                  stroke={CHALK}
                  strokeWidth={1.25}
                  strokeDasharray="2 4"
                  strokeLinecap="round"
                />
              ))}
              {frame?.centers.map(([cx, cy], k) => (
                <g
                  key={`center-${String(k)}`}
                  transform={`translate(${String(x(cx))},${String(y(cy))})`}
                >
                  <path
                    d="M-6,-6L6,6M-6,6L6,-6"
                    stroke={SLATE}
                    strokeWidth={5}
                    strokeLinecap="round"
                  />
                  <path
                    d="M-6,-6L6,6M-6,6L6,-6"
                    stroke={CHALK}
                    strokeWidth={2}
                    strokeLinecap="round"
                  />
                </g>
              ))}
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
                  i: clampIndex(i, ys.length),
                  j: clampIndex(j, xs.length),
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

function clampIndex(value: number, length: number): number {
  return Math.max(0, Math.min(length - 1, value));
}

/** O canvas do campo: o véu de cada célula e os pontos de treino. */
function FieldCanvas({
  animation,
  frameIndex,
  encoding,
  left,
  top,
  width,
  height,
}: {
  animation: TrainingAnimation;
  frameIndex: number;
  encoding: ReturnType<typeof useEncoding>;
  left: number;
  top: number;
  width: number;
  height: number;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const element = canvas.current;
    const frame = animation.frames[Math.min(frameIndex, animation.frames.length - 1)];
    const context = element?.getContext("2d");
    if (
      element === null ||
      context === null ||
      context === undefined ||
      frame === undefined ||
      width <= 0 ||
      height <= 0
    ) {
      return;
    }
    const ratio = window.devicePixelRatio || 1;
    element.width = Math.round(width * ratio);
    element.height = Math.round(height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);

    // O campo, à resolução da grelha, esticado com suavização.
    const rows = frame.values.length;
    const columns = frame.values[0]?.length ?? 0;
    const grid = document.createElement("canvas");
    grid.width = columns;
    grid.height = rows;
    const gridContext = grid.getContext("2d");
    if (gridContext === null || columns === 0) {
      return;
    }
    const image = gridContext.createImageData(columns, rows);
    frame.values.forEach((row, i) => {
      row.forEach((value, j) => {
        const [color, alpha] = encoding.cell(value);
        const [r, g, b] = channels(color);
        const offset = ((rows - 1 - i) * columns + j) * 4; // a linha 0 da imagem é a de cima
        image.data[offset] = r;
        image.data[offset + 1] = g;
        image.data[offset + 2] = b;
        image.data[offset + 3] = Math.round(alpha * 255);
      });
    });
    gridContext.putImageData(image, 0, 0);
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(grid, 0, 0, width, height);

    // Os pontos de treino por cima, com o anel da cor do fundo.
    const xs = animation.xs;
    const ys = animation.ys;
    const [xMin = 0, xMax = 1] = [xs[0], xs[xs.length - 1]];
    const [yMin = 0, yMax = 1] = [ys[0], ys[ys.length - 1]];
    const toX = (value: number) => ((value - xMin) / (xMax - xMin || 1)) * width;
    const toY = (value: number) => height - ((value - yMin) / (yMax - yMin || 1)) * height;
    animation.points.x.forEach((px, i) => {
      const py = animation.points.y[i] ?? 0;
      drawMarker(context, toX(px), toY(py), encoding.shapeOf(i), encoding.point(i));
    });
  }, [animation, frameIndex, encoding, width, height]);

  return (
    <canvas
      ref={canvas}
      aria-hidden="true"
      className="absolute"
      style={{ left, top, width, height }}
    />
  );
}

/** Um ponto de treino: círculo, quadrado, triângulo, losango ou cruz, com o anel da ardósia. */
function drawMarker(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  shape: number,
  fill: string,
) {
  context.beginPath();
  const r = 4;
  switch (shape % 5) {
    case 1:
      context.rect(x - r, y - r, r * 2, r * 2);
      break;
    case 2:
      context.moveTo(x, y - r * 1.2);
      context.lineTo(x + r * 1.1, y + r * 0.8);
      context.lineTo(x - r * 1.1, y + r * 0.8);
      context.closePath();
      break;
    case 3:
      context.moveTo(x, y - r * 1.3);
      context.lineTo(x + r * 1.1, y);
      context.lineTo(x, y + r * 1.3);
      context.lineTo(x - r * 1.1, y);
      context.closePath();
      break;
    case 4:
      context.moveTo(x - r, y - r);
      context.lineTo(x + r, y + r);
      context.moveTo(x + r, y - r);
      context.lineTo(x - r, y + r);
      context.lineWidth = 4;
      context.strokeStyle = SLATE;
      context.stroke();
      context.lineWidth = 2;
      context.strokeStyle = fill;
      context.stroke();
      return;
    default:
      context.arc(x, y, r, 0, Math.PI * 2);
  }
  context.lineWidth = 1.5;
  context.strokeStyle = SLATE;
  context.fillStyle = fill;
  context.fill();
  context.stroke();
}

/** As fronteiras: a linha dos 50% (binária) ou as mudanças de categoria (multiclasse, grupos). */
function boundaries(
  animation: TrainingAnimation,
  values: readonly (readonly number[])[],
  kind: TrainingAnimation["value_kind"],
): [number, number][][][] {
  const rows = values.length;
  const columns = values[0]?.length ?? 0;
  if (rows < 2 || columns < 2 || (kind !== "probability" && kind !== "class")) {
    return [];
  }
  const flat = values.flat();
  const generator = contours().size([columns, rows]).smooth(true);
  if (kind === "probability") {
    return generator
      .thresholds([0.5])(flat)
      .map((shape) => shape.coordinates.flat() as [number, number][][]);
  }
  const categories = animation.field_labels.length;
  const result: [number, number][][][] = [];
  for (let category = 0; category < categories; category++) {
    const indicator = flat.map((value) => (Math.round(value) === category ? 1 : 0));
    if (!indicator.includes(1) || !indicator.includes(0)) {
      continue;
    }
    for (const shape of generator.thresholds([0.5])(indicator)) {
      result.push(shape.coordinates.flat() as [number, number][][]);
    }
  }
  return result;
}

/** A legenda do campo: as classes (ou os grupos), ou a escala do valor. */
function fieldLegend(
  animation: TrainingAnimation,
  kind: TrainingAnimation["value_kind"],
  low: number,
  high: number,
): LegendItem[] {
  if (kind === "probability") {
    return [
      ...animation.class_labels
        .slice(0, 2)
        .map((label, i) => ({ label, color: categoryColor(i), kind: "dot" as const })),
      { label: "Fronteira (50%)", color: CHALK, kind: "line" as const },
    ];
  }
  if (kind === "class") {
    return animation.field_labels.map((label, i) => ({
      label,
      color: animation.noise && i === animation.field_labels.length - 1 ? OTHER : categoryColor(i),
      kind: "rect" as const,
    }));
  }
  const color = kind === "score" ? sequentialAlt : sequential;
  return [
    {
      label: `${animation.value_label}: ${formatSignificant(low, 3)}`,
      color: color(0.15),
      kind: "rect",
    },
    { label: formatSignificant(high, 3), color: color(0.95), kind: "rect" },
  ];
}

function writeValue(animation: TrainingAnimation, value: number): string {
  if (animation.value_kind === "probability") {
    return formatPercent(value);
  }
  if (animation.value_kind === "class") {
    return animation.field_labels[Math.round(value)] ?? formatNumber(value, 0);
  }
  return formatSignificant(value, 4);
}

/** Com uma só feature: a previsão como curva, e os pontos de treino. */
function CurveView({
  animation,
  index,
  description,
  caption,
  height,
}: {
  animation: TrainingAnimation;
  index: number;
  description: string;
  caption?: ReactNode;
  height: number;
}) {
  const encoding = useEncoding(animation);
  const frame = animation.frames[Math.min(index, animation.frames.length - 1)];
  const curve = frame?.values[0] ?? [];
  const probability = animation.value_kind === "probability";
  const [xMin = 0, xMax = 1] = extent([...animation.xs, ...animation.points.x]);
  const [yLow = 0, yHigh = 1] = probability
    ? [0, 1]
    : extent([...animation.points.y, ...animation.frames.flatMap((f) => f.values[0] ?? [])]);

  return (
    <Chart
      height={height}
      description={description}
      caption={caption}
      margin={{ left: 56, bottom: 40 }}
      legend={
        <Legend
          items={[
            { label: animation.value_label, color: CHALK, kind: "line" },
            ...(probability
              ? animation.class_labels
                  .slice(0, 2)
                  .map((label, i) => ({ label, color: categoryColor(i), kind: "dot" as const }))
              : [{ label: "Pontos de treino", color: sequential(0.6), kind: "dot" as const }]),
          ]}
        />
      }
    >
      {(plot) => {
        const x = scaleLinear().domain([xMin, xMax]).nice().range([0, plot.width]);
        const y = scaleLinear().domain([yLow, yHigh]).nice().range([plot.height, 0]);
        const xTicks = x.ticks(Math.max(2, Math.floor(plot.width / 90)));
        const yTicks = y.ticks(5);
        const path = line<number>()
          .x((_, i) => x(animation.xs[i] ?? 0))
          .y((value) => y(value))([...curve]);
        return (
          <>
            <AxisLeft
              ticks={yTicks}
              position={y}
              format={(v) =>
                probability ? formatPercent(v, 0) : formatNumber(v, stepDecimals(yTicks))
              }
              plot={plot}
              label={probability ? animation.value_label : animation.axis_labels[1]}
            />
            <AxisBottom
              ticks={xTicks}
              position={x}
              format={(v) => formatNumber(v, stepDecimals(xTicks))}
              plot={plot}
              label={animation.axis_labels[0]}
            />
            {probability && (
              <line
                x1={0}
                x2={plot.width}
                y1={y(0.5)}
                y2={y(0.5)}
                stroke="var(--input)"
                strokeDasharray="4 4"
              />
            )}
            {animation.points.x.map((px, i) => (
              <circle
                key={i}
                cx={x(px)}
                cy={y(animation.points.y[i] ?? 0)}
                r={3.5}
                fill={probability ? encoding.point(i) : sequential(0.6)}
                fillOpacity={0.85}
                stroke="var(--background)"
                strokeWidth={1.5}
              />
            ))}
            <path
              d={path ?? ""}
              fill="none"
              stroke={CHALK}
              strokeWidth={2.5}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          </>
        );
      }}
    </Chart>
  );
}
