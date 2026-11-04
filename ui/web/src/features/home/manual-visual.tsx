/**
 * O desenho do Manual na entrada: tu conduzes cada passo.
 *
 * Em ciclo: as etapas acendem-se por ordem (os dados, a preparação, o modelo, o treino e a
 * avaliação); no treino, a fronteira de decisão ajusta-se aos dois grupos de pontos, passo a
 * passo, enquanto a loss desce e a exatidão sobe; no fim, aparece a nota. É geral (não é a
 * demonstração de nenhum modelo em particular), por isso não depende da API.
 *
 * As animações são do próprio SVG (SMIL): todas partilham o mesmo relógio e a mesma duração, e
 * por isso andam sempre juntas, sem renders do React. Com "reduzir movimento", mostra-se o fim,
 * parado.
 */
import { useReducedMotion } from "motion/react";
import { useId } from "react";

import { GlowFilter } from "@/components/charts/chart";

/** A duração de um ciclo, em segundos. */
const CYCLE = 9;
/** Quando o treino começa e acaba, em segundos desde o início do ciclo. */
const TRAIN_FROM = 2.1;
const TRAIN_TO = 6.6;
/** Quando tudo volta ao princípio (um pouco antes do fim, para o salto não se ver). */
const RESET = 8.85;

const STAGES = ["Data", "Prepare", "Model", "Train", "Evaluate"] as const;
/** Quando cada etapa se acende. */
const STAGE_AT = [0.3, 0.9, 1.5, TRAIN_FROM, TRAIN_TO + 0.15] as const;

/** A moldura do gráfico e o painel do treino, à direita. */
const PLOT = { x: 21, y: 46, width: 300, height: 200 } as const;
const PANEL_X = 341;
const PANEL_WIDTH = 158;

/** Os pontos das duas classes (dois deles do lado errado: a nota final não é 100%). */
const CLASS_A = [
  [48, 92],
  [72, 128],
  [44, 160],
  [96, 76],
  [120, 112],
  [88, 150],
  [146, 84],
  [170, 120],
  [196, 72],
  [222, 98],
  [252, 64],
  [130, 140],
  [58, 178],
  [280, 80],
  [300, 62],
  [178, 152],
] as const;
const CLASS_B = [
  [70, 228],
  [110, 214],
  [150, 200],
  [190, 186],
  [230, 160],
  [260, 140],
  [290, 124],
  [306, 150],
  [200, 222],
  [250, 200],
  [160, 234],
  [280, 180],
  [230, 232],
  [300, 214],
  [110, 168],
] as const;

/**
 * A fronteira em cada passo do treino: uma curva cúbica, do lado esquerdo ao direito. Todas têm
 * os mesmos comandos, para o SVG poder passar de uma para a outra.
 */
type Curve = readonly [
  start: number,
  c1x: number,
  c1y: number,
  c2x: number,
  c2y: number,
  end: number,
];
const BOUNDARIES: readonly Curve[] = [
  [120, 120, 125, 220, 135, 150],
  [150, 120, 150, 200, 135, 128],
  [175, 120, 175, 185, 128, 110],
  [195, 120, 192, 175, 122, 99],
  [205, 120, 200, 170, 120, 95],
];

const LEFT = PLOT.x;
const RIGHT = PLOT.x + PLOT.width;
const TOP = PLOT.y;
const BOTTOM = PLOT.y + PLOT.height;

function boundary([start, c1x, c1y, c2x, c2y, end]: Curve): string {
  return `M${String(LEFT)} ${String(start)} C${String(c1x)} ${String(c1y)} ${String(c2x)} ${String(c2y)} ${String(RIGHT)} ${String(end)}`;
}

/** A região de um dos lados da fronteira: até ao topo (`edge = TOP`) ou até ao fundo. */
function region([start, c1x, c1y, c2x, c2y, end]: Curve, edge: number): string {
  return `M${String(LEFT)} ${String(edge)} H${String(RIGHT)} V${String(end)} C${String(c2x)} ${String(c2y)} ${String(c1x)} ${String(c1y)} ${String(LEFT)} ${String(start)} Z`;
}

/** Os instantes de uma animação, como frações do ciclo (o que o SMIL pede em `keyTimes`). */
function times(...seconds: readonly number[]): string {
  return seconds.map((second) => String(Math.round((second / CYCLE) * 10000) / 10000)).join(";");
}

/** A fronteira ao longo do ciclo: parada até ao treino, um passo de cada vez, e de volta. */
function morph(shape: (curve: Curve) => string): { values: string; keyTimes: string } {
  const steps = BOUNDARIES.length - 1;
  const during = BOUNDARIES.map(
    (_, index) => TRAIN_FROM + ((TRAIN_TO - TRAIN_FROM) * index) / steps,
  );
  const first: Curve = BOUNDARIES[0] ?? [0, 0, 0, 0, 0, 0];
  return {
    values: [first, ...BOUNDARIES, BOUNDARIES.at(-1) ?? first, first].map(shape).join(";"),
    keyTimes: times(0, ...during, RESET, CYCLE),
  };
}

/** A curva da loss, no painel: desce depressa e depois devagar. */
const LOSS = `M${String(PANEL_X + 10)} 78 C${String(PANEL_X + 40)} 112 ${String(PANEL_X + 70)} 122 ${String(PANEL_X + PANEL_WIDTH - 10)} 128`;

export function ManualVisual() {
  const reduced = useReducedMotion() ?? false;
  const id = useId();
  const glow = `${id}-glow`;
  const clip = `${id}-plot`;
  const last = BOUNDARIES.at(-1) ?? BOUNDARIES[0];
  if (last === undefined) {
    return null;
  }
  const line = morph(boundary);
  const above = morph((curve) => region(curve, TOP));
  const below = morph((curve) => region(curve, BOTTOM));
  const training = times(0, TRAIN_FROM, TRAIN_TO, RESET, CYCLE);
  const result = times(0, TRAIN_TO + 0.15, TRAIN_TO + 0.4, RESET, CYCLE);
  return (
    <svg
      viewBox="0 0 520 260"
      role="img"
      aria-label="By hand: you pick the data, the preparation and the model, and watch the decision boundary fit the points step by step while the loss falls."
      className="mx-auto block h-auto max-h-80 w-full font-sans tabular-nums"
    >
      <defs>
        <GlowFilter id={glow} blur={3} />
        <clipPath id={clip}>
          <rect x={PLOT.x} y={PLOT.y} width={PLOT.width} height={PLOT.height} rx={10} />
        </clipPath>
      </defs>

      {/* As etapas: acendem-se por ordem, e ficam acesas até o ciclo recomeçar. */}
      {STAGES.map((stage, index) => {
        const x = 21 + index * 98;
        const at = STAGE_AT[index] ?? 0;
        return (
          <g key={stage}>
            {index > 0 && <path d={`M${String(x - 12)} 21 H${String(x)}`} stroke="#2e2e2e" />}
            <rect
              x={x + 0.5}
              y={10.5}
              width={86}
              height={22}
              rx={11}
              fill="#171717"
              stroke="#2a2a2a"
            />
            <text
              x={x + 43}
              y={21.5}
              dy="0.34em"
              textAnchor="middle"
              fontSize={10.5}
              fill="#6b6b6b"
            >
              {stage}
            </text>
            <g opacity={reduced ? 1 : 0}>
              {!reduced && (
                <animate
                  attributeName="opacity"
                  values="0;0;1;1;0"
                  keyTimes={times(0, at, at + 0.25, RESET, CYCLE)}
                  dur={`${String(CYCLE)}s`}
                  repeatCount="indefinite"
                />
              )}
              <rect
                x={x + 0.5}
                y={10.5}
                width={86}
                height={22}
                rx={11}
                fill="#202020"
                stroke="#5a5a5a"
              />
              <text
                x={x + 43}
                y={21.5}
                dy="0.34em"
                textAnchor="middle"
                fontSize={10.5}
                fill="#f5f5f5"
              >
                {stage}
              </text>
            </g>
          </g>
        );
      })}

      {/* O gráfico: as duas regiões, os pontos e a fronteira. */}
      <rect
        x={PLOT.x + 0.5}
        y={PLOT.y + 0.5}
        width={PLOT.width - 1}
        height={PLOT.height - 1}
        rx={10}
        fill="#131313"
        stroke="#262626"
      />
      <g clipPath={`url(#${clip})`}>
        <path d={region(last, TOP)} fill="var(--data-1)" fillOpacity={0.08}>
          {!reduced && (
            <animate
              attributeName="d"
              values={above.values}
              keyTimes={above.keyTimes}
              dur={`${String(CYCLE)}s`}
              repeatCount="indefinite"
            />
          )}
        </path>
        <path d={region(last, BOTTOM)} fill="var(--data-2)" fillOpacity={0.08}>
          {!reduced && (
            <animate
              attributeName="d"
              values={below.values}
              keyTimes={below.keyTimes}
              dur={`${String(CYCLE)}s`}
              repeatCount="indefinite"
            />
          )}
        </path>
        {CLASS_A.map(([x, y]) => (
          <circle
            key={`a-${String(x)}-${String(y)}`}
            cx={x}
            cy={y}
            r={4.2}
            fill="var(--data-1)"
            stroke="#131313"
          />
        ))}
        {CLASS_B.map(([x, y]) => (
          <circle
            key={`b-${String(x)}-${String(y)}`}
            cx={x}
            cy={y}
            r={4.2}
            fill="var(--data-2)"
            stroke="#131313"
          />
        ))}
        <path
          d={boundary(last)}
          fill="none"
          stroke="#f5f5f5"
          strokeWidth={2}
          strokeLinecap="round"
          filter={`url(#${glow})`}
        >
          {!reduced && (
            <animate
              attributeName="d"
              values={line.values}
              keyTimes={line.keyTimes}
              dur={`${String(CYCLE)}s`}
              repeatCount="indefinite"
            />
          )}
        </path>
      </g>

      {/* O painel do treino: a loss a descer e a exatidão a subir. */}
      <text x={PANEL_X} y={58} fontSize={10.5} fill="#7a7a7a">
        loss
      </text>
      <rect
        x={PANEL_X + 0.5}
        y={66.5}
        width={PANEL_WIDTH - 1}
        height={72}
        rx={8}
        fill="#131313"
        stroke="#262626"
      />
      <path
        d={LOSS}
        fill="none"
        stroke="var(--data-1)"
        strokeWidth={1.75}
        strokeLinecap="round"
        pathLength={1}
        strokeDasharray="1"
        strokeDashoffset={0}
      >
        {!reduced && (
          <animate
            attributeName="stroke-dashoffset"
            values="1;1;0;0;1"
            keyTimes={training}
            dur={`${String(CYCLE)}s`}
            repeatCount="indefinite"
          />
        )}
      </path>
      <circle
        r={3}
        fill="#f5f5f5"
        cx={reduced ? PANEL_X + PANEL_WIDTH - 10 : 0}
        cy={reduced ? 128 : 0}
      >
        {!reduced && (
          <animateMotion
            path={LOSS}
            keyPoints="0;0;1;1;0"
            keyTimes={training}
            calcMode="linear"
            dur={`${String(CYCLE)}s`}
            repeatCount="indefinite"
          />
        )}
      </circle>

      <text x={PANEL_X} y={164} fontSize={10.5} fill="#7a7a7a">
        accuracy
      </text>
      <rect x={PANEL_X} y={172} width={PANEL_WIDTH} height={8} rx={4} fill="#222" />
      <rect x={PANEL_X} y={172} width={reduced ? 147 : 16} height={8} rx={4} fill="#e5e5e5">
        {!reduced && (
          <animate
            attributeName="width"
            values="16;16;147;147;16"
            keyTimes={training}
            dur={`${String(CYCLE)}s`}
            repeatCount="indefinite"
          />
        )}
      </rect>

      {/* A avaliação: a nota aparece quando o treino acaba. */}
      <g opacity={reduced ? 1 : 0}>
        {!reduced && (
          <animate
            attributeName="opacity"
            values="0;0;1;1;0"
            keyTimes={result}
            dur={`${String(CYCLE)}s`}
            repeatCount="indefinite"
          />
        )}
        <text
          x={PANEL_X}
          y={222}
          fontSize={28}
          fontWeight={600}
          letterSpacing="-0.02em"
          fill="#fafafa"
          className="font-sans"
        >
          93%
        </text>
        <text x={PANEL_X} y={242} fontSize={10.5} fill="#7a7a7a">
          test accuracy
        </text>
      </g>
    </svg>
  );
}
