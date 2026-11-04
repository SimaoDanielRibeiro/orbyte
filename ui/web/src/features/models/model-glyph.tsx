/**
 * O desenho de um modelo, na lista do ecrã Modelos: uma ideia do que o modelo faz, em pequeno, a
 * mexer devagar (ADR 0037). A reta da regressão ajusta-se aos pontos, os vizinhos do KNN ligam-se
 * ao ponto novo, a árvore cresce, os centróides andam até aos grupos, a luz passa pela rede…
 *
 * O desenho escolhe-se pelo que o backend já diz de cada modelo (a família e as tarefas), não pelo
 * nome: um modelo novo de uma família conhecida tem desenho sem mudar nada aqui. A cor é a do
 * problema que o modelo resolve (`lib/problems.ts`, ADR 0048).
 *
 * O movimento são as classes `glyph-*` de `globals.css`: ciclos lentos, desencontrados entre
 * modelos (`delay`). Parado (com "reduzir movimento"), cada desenho fica no seu estado de base,
 * que é o desenho completo.
 */
import type { CSSProperties, ReactNode } from "react";

import type { GlyphKind } from "./glyphs";

/** Os pontos que não são a estrela do desenho (os dados, a outra classe). */
const MUTED = "#6b6b6b";
const SOFT = "#9a9a9a";
const INK = "#ededed";
const CARD = "#141414";

interface ModelGlyphProps {
  kind: GlyphKind;
  color: string;
  /** Segundos de avanço no ciclo, para os desenhos da lista não mexerem todos ao mesmo tempo. */
  delay?: number;
  className?: string;
}

export function ModelGlyph({ kind, color, delay = 0, className }: ModelGlyphProps) {
  const at = (extra = 0): CSSProperties => ({ animationDelay: `${String(-(delay + extra))}s` });
  return (
    <svg viewBox="0 0 56 36" aria-hidden="true" className={className ?? "glyph h-9 w-14"}>
      {DRAW[kind]({ color, at })}
    </svg>
  );
}

interface DrawProps {
  color: string;
  /** O atraso de uma peça (um negativo: o ciclo já vai a meio, e não há espera ao abrir). */
  at: (extra?: number) => CSSProperties;
}

/** Variáveis do CSS para as animações (o ponto de partida de um centróide, por exemplo). */
function vars(values: Record<`--${string}`, string>): CSSProperties {
  return values;
}

function Dot({ x, y, r = 1.8, fill = MUTED }: { x: number; y: number; r?: number; fill?: string }) {
  return <circle cx={x} cy={y} r={r} fill={fill} />;
}

/** Um anel que se expande e some à volta de um ponto (o ponto novo, a anomalia). */
function Ping({
  x,
  y,
  color,
  style,
}: {
  x: number;
  y: number;
  color: string;
  style: CSSProperties;
}) {
  return (
    <circle
      cx={x}
      cy={y}
      r={4.5}
      fill="none"
      stroke={color}
      strokeWidth={1}
      className="glyph-ping"
      style={style}
    />
  );
}

/** Um traço que se desenha, fica e se apaga (`pathLength` 1: o traço é o caminho inteiro). */
function Drawn({
  d,
  color,
  style,
  width = 1.4,
}: {
  d: string;
  color: string;
  style: CSSProperties;
  width?: number;
}) {
  return (
    <path
      d={d}
      pathLength={1}
      fill="none"
      stroke={color}
      strokeWidth={width}
      strokeLinecap="round"
      strokeLinejoin="round"
      className="glyph-draw"
      style={style}
    />
  );
}

const DRAW: Record<GlyphKind, (props: DrawProps) => ReactNode> = {
  /** A regressão: a reta roda até passar pelo meio dos pontos. */
  "line-fit": ({ color, at }) => (
    <>
      {[
        [8, 28],
        [14, 25],
        [20, 24],
        [27, 19],
        [33, 17],
        [40, 13],
        [47, 10],
      ].map(([x = 0, y = 0]) => (
        <Dot key={x} x={x} y={y} />
      ))}
      <line
        x1={7}
        y1={29}
        x2={49}
        y2={9}
        stroke={color}
        strokeWidth={1.8}
        strokeLinecap="round"
        className="glyph-fit"
        style={at()}
      />
    </>
  ),

  /** A classificação linear: a reta roda até separar as duas classes. */
  separator: ({ color, at }) => (
    <>
      {[
        [10, 9],
        [17, 6],
        [13, 15],
        [21, 11],
        [8, 21],
      ].map(([x = 0, y = 0]) => (
        <Dot key={`a${String(x)}`} x={x} y={y} fill={color} />
      ))}
      {[
        [36, 25],
        [44, 20],
        [41, 30],
        [49, 27],
        [32, 31],
      ].map(([x = 0, y = 0]) => (
        <Dot key={`b${String(x)}`} x={x} y={y} fill={SOFT} />
      ))}
      <line
        x1={9}
        y1={31}
        x2={47}
        y2={5}
        stroke={INK}
        strokeWidth={1.4}
        strokeLinecap="round"
        className="glyph-fit"
        style={at()}
      />
    </>
  ),

  /** O KNN: o ponto novo liga-se aos três vizinhos mais próximos, um a um. */
  neighbors: ({ color, at }) => {
    const near = [
      [20, 12],
      [36, 23],
      [31, 8],
    ] as const;
    return (
      <>
        {[
          [7, 7],
          [9, 28],
          [48, 7],
          [50, 28],
          [44, 31],
          [13, 19],
        ].map(([x = 0, y = 0]) => (
          <Dot key={x} x={x} y={y} />
        ))}
        {near.map(([x, y], index) => (
          <Drawn
            key={x}
            d={`M28,18 L${String(x)},${String(y)}`}
            color={color}
            style={at(-0.35 * index)}
            width={1.2}
          />
        ))}
        {near.map(([x, y]) => (
          <Dot key={`v${String(x)}`} x={x} y={y} fill={color} r={2} />
        ))}
        <Ping x={28} y={18} color={INK} style={at()} />
        <Dot x={28} y={18} r={2.6} fill={INK} />
      </>
    );
  },

  /** As anomalias locais: um ponto longe do grupo, assinalado. */
  outlier: ({ color, at }) => (
    <>
      {[
        [12, 14],
        [16, 21],
        [20, 12],
        [23, 22],
        [13, 26],
        [26, 16],
        [18, 17],
      ].map(([x = 0, y = 0]) => (
        <Dot key={`${String(x)}-${String(y)}`} x={x} y={y} fill={color} />
      ))}
      <Ping x={45} y={10} color={color} style={at()} />
      <Dot x={45} y={10} r={2.4} fill={INK} />
    </>
  ),

  /** O Naive Bayes: uma curva de sino por classe, a respirar. */
  bells: ({ color, at }) => (
    <>
      <line x1={3} y1={31} x2={53} y2={31} stroke="#333" strokeWidth={1} />
      <path
        d="M3,31 C 13,31 14,9 21,9 C 28,9 29,31 39,31 Z"
        fill={color}
        fillOpacity={0.14}
        stroke={color}
        strokeWidth={1.5}
        className="glyph-rise"
        style={{ ...at(), transformOrigin: "50% 100%" }}
      />
      <path
        d="M19,31 C 29,31 30,13 36,13 C 42,13 43,31 53,31 Z"
        fill={SOFT}
        fillOpacity={0.08}
        stroke={SOFT}
        strokeWidth={1.3}
        className="glyph-rise"
        style={{ ...at(-1.6), transformOrigin: "50% 100%" }}
      />
    </>
  ),

  /** A mistura de gaussianas: uma elipse à volta de cada grupo. */
  ellipses: ({ color, at }) => (
    <>
      {[
        [12, 12],
        [15, 18],
        [20, 13],
        [19, 19],
        [14, 15],
        [34, 21],
        [38, 26],
        [43, 19],
        [45, 24],
        [40, 22],
      ].map(([x = 0, y = 0]) => (
        <Dot key={`${String(x)}-${String(y)}`} x={x} y={y} />
      ))}
      <ellipse
        cx={16.5}
        cy={15.5}
        rx={9}
        ry={6.5}
        fill={color}
        fillOpacity={0.1}
        stroke={color}
        strokeWidth={1.3}
        className="glyph-breathe"
        style={at()}
      />
      <ellipse
        cx={39.5}
        cy={22.5}
        rx={9.5}
        ry={7}
        fill={color}
        fillOpacity={0.1}
        stroke={color}
        strokeWidth={1.3}
        className="glyph-breathe"
        style={at(-2)}
      />
    </>
  ),

  /** A árvore: as perguntas crescem de cima para baixo, nível a nível. */
  tree: ({ color, at }) => (
    <>
      <Drawn d="M28,7 L17,18" color={color} style={at()} />
      <Drawn d="M28,7 L39,18" color={color} style={at()} />
      <Drawn d="M17,18 L11,29 M17,18 L23,29" color={color} style={at(-0.5)} />
      <Drawn d="M39,18 L34,29 M39,18 L45,29" color={color} style={at(-0.5)} />
      <circle cx={28} cy={7} r={2.6} fill={INK} className="glyph-pop" style={at(0.1)} />
      {[
        [17, 18],
        [39, 18],
      ].map(([x = 0, y = 0]) => (
        <circle key={x} cx={x} cy={y} r={2.3} fill={color} className="glyph-pop" style={at(-0.3)} />
      ))}
      {[11, 23, 34, 45].map((x, index) => (
        <circle
          key={x}
          cx={x}
          cy={29}
          r={2}
          fill={index % 2 === 0 ? color : SOFT}
          className="glyph-pop"
          style={at(-0.8)}
        />
      ))}
    </>
  ),

  /** Os ensembles: várias árvores pequenas, que acendem à vez (a votar, ou em sequência). */
  forest: ({ color, at }) => (
    <>
      {[12, 28, 44].map((x, index) => (
        <g key={x} className="glyph-blink" style={at(-1.2 * index)}>
          <path
            d={`M${String(x)},9 L${String(x - 6)},19 M${String(x)},9 L${String(x + 6)},19 M${String(x - 6)},19 L${String(x - 9)},28 M${String(x - 6)},19 L${String(x - 3)},28 M${String(x + 6)},19 L${String(x + 9)},28`}
            fill="none"
            stroke={color}
            strokeWidth={1.3}
            strokeLinecap="round"
          />
          <circle cx={x} cy={9} r={2.2} fill={INK} />
          <circle cx={x - 6} cy={19} r={1.8} fill={color} />
          <circle cx={x + 6} cy={19} r={1.8} fill={color} />
        </g>
      ))}
    </>
  ),

  /** O Isolation Forest: cortes ao acaso isolam a anomalia em poucos passos. */
  isolation: ({ color, at }) => (
    <>
      {[
        [8, 20],
        [13, 14],
        [16, 24],
        [20, 18],
        [24, 27],
        [27, 15],
        [11, 29],
        [30, 23],
      ].map(([x = 0, y = 0]) => (
        <Dot key={`${String(x)}-${String(y)}`} x={x} y={y} />
      ))}
      <Drawn d="M37,3 L37,33" color={color} style={at()} width={1.2} />
      <Drawn d="M37,17 L53,17" color={color} style={at(-0.45)} width={1.2} />
      <Drawn d="M42,3 L42,17" color={color} style={at(-0.9)} width={1.2} />
      <Dot x={47} y={9} r={2.4} fill={INK} />
    </>
  ),

  /** O SVM: a fronteira com a maior margem entre as classes. */
  margin: ({ color, at }) => (
    <>
      {[
        [8, 10],
        [14, 6],
        [11, 17],
        [19, 10],
      ].map(([x = 0, y = 0]) => (
        <Dot key={`a${String(x)}`} x={x} y={y} fill={color} />
      ))}
      {[
        [37, 27],
        [45, 24],
        [41, 32],
        [49, 31],
      ].map(([x = 0, y = 0]) => (
        <Dot key={`b${String(x)}`} x={x} y={y} fill={SOFT} />
      ))}
      {/* As duas margens, a 3 px da fronteira, aproximam-se e afastam-se dela. */}
      <path
        d="M1,26 C 15,23 25,10 49,3"
        fill="none"
        stroke={color}
        strokeWidth={1}
        strokeDasharray="2 2"
        className="glyph-squeeze"
        style={{ ...at(), ...vars({ "--dx": "2px", "--dy": "2px" }) }}
      />
      <path
        d="M7,32 C 21,29 31,16 55,9"
        fill="none"
        stroke={color}
        strokeWidth={1}
        strokeDasharray="2 2"
        className="glyph-squeeze"
        style={{ ...at(), ...vars({ "--dx": "-2px", "--dy": "-2px" }) }}
      />
      <path d="M4,29 C 18,26 28,13 52,6" fill="none" stroke={INK} strokeWidth={1.4} />
      <circle cx={19} cy={10} r={3.6} fill="none" stroke={INK} strokeWidth={0.8} />
      <circle cx={37} cy={27} r={3.6} fill="none" stroke={INK} strokeWidth={0.8} />
    </>
  ),

  /** O SVR: uma curva com um tubo de tolerância à volta. */
  tube: ({ color, at }) => (
    <>
      <path
        d="M4,24 C 14,24 18,10 28,12 S 44,27 52,14"
        fill="none"
        stroke={color}
        strokeOpacity={0.16}
        strokeWidth={10}
        strokeLinecap="round"
        className="glyph-swell"
        style={at()}
      />
      {[
        [8, 22],
        [15, 18],
        [21, 13],
        [29, 14],
        [35, 18],
        [41, 22],
        [48, 17],
      ].map(([x = 0, y = 0]) => (
        <Dot key={x} x={x} y={y} />
      ))}
      <path
        d="M4,24 C 14,24 18,10 28,12 S 44,27 52,14"
        fill="none"
        stroke={color}
        strokeWidth={1.6}
        strokeLinecap="round"
      />
    </>
  ),

  /** O One-Class SVM: a fronteira justa à volta dos dados; o que fica fora é anómalo. */
  enclosure: ({ color, at }) => (
    <>
      <path
        d="M9,18 C 9,8 33,5 35,15 C 37,26 11,30 9,18 Z"
        fill={color}
        fillOpacity={0.08}
        stroke={color}
        strokeWidth={1.4}
        className="glyph-breathe"
        style={at()}
      />
      {[
        [14, 15],
        [18, 20],
        [22, 13],
        [26, 19],
        [17, 24],
        [29, 15],
      ].map(([x = 0, y = 0]) => (
        <Dot key={`${String(x)}-${String(y)}`} x={x} y={y} fill={SOFT} />
      ))}
      <Ping x={47} y={9} color={color} style={at(-0.8)} />
      <Dot x={47} y={9} r={2.3} fill={INK} />
    </>
  ),

  /** O K-means: os centróides partem ao acaso e andam até ao meio de cada grupo. */
  centroids: ({ color, at }) => {
    const groups = [
      { center: [12.5, 10.5], from: ["14px", "12px"] },
      { center: [42, 9.5], from: ["-16px", "13px"] },
      { center: [28.5, 26.5], from: ["8px", "-15px"] },
    ] as const;
    return (
      <>
        {[
          [9, 9],
          [14, 7],
          [16, 13],
          [11, 14],
          [38, 8],
          [44, 6],
          [46, 12],
          [40, 13],
          [24, 26],
          [30, 30],
          [33, 25],
          [27, 24],
        ].map(([x = 0, y = 0]) => (
          <Dot key={`${String(x)}-${String(y)}`} x={x} y={y} />
        ))}
        {groups.map(({ center: [x, y], from: [dx, dy] }) => (
          <path
            key={x}
            d={`M${String(x)},${String(y - 3.6)} L${String(x + 3.6)},${String(y)} L${String(x)},${String(y + 3.6)} L${String(x - 3.6)},${String(y)} Z`}
            fill={color}
            stroke={CARD}
            strokeWidth={1}
            className="glyph-home"
            style={{ ...at(), ...vars({ "--dx": dx, "--dy": dy }) }}
          />
        ))}
      </>
    );
  },

  /** O DBSCAN: a densidade espalha-se de vizinho em vizinho; o ruído fica de fora. */
  density: ({ color, at }) => {
    const arc = [
      [6, 22],
      [9, 15],
      [14, 10],
      [20, 8],
      [26, 10],
      [30, 15],
      [24, 22],
      [28, 26],
      [33, 29],
      [39, 30],
      [45, 27],
      [49, 21],
    ];
    return (
      <>
        {arc.map(([x = 0, y = 0], index) => (
          <circle
            key={`${String(x)}-${String(y)}`}
            cx={x}
            cy={y}
            r={2}
            fill={color}
            className="glyph-light"
            style={at(-0.18 * index)}
          />
        ))}
        <Dot x={46} y={7} fill={MUTED} />
        <Dot x={13} y={30} fill={MUTED} />
      </>
    );
  },

  /** O clustering hierárquico: os pontos juntam-se, dos mais próximos para os mais afastados. */
  dendrogram: ({ color, at }) => (
    <>
      <Drawn d="M8,31 V24 H17 V31" color={color} style={at()} width={1.3} />
      <Drawn d="M38,31 V21 H48 V31" color={color} style={at(-0.4)} width={1.3} />
      <Drawn d="M12.5,24 V16 H26 V31" color={color} style={at(-0.8)} width={1.3} />
      <Drawn d="M19.25,16 V6 H43 V21" color={color} style={at(-1.2)} width={1.3} />
      {[8, 17, 26, 38, 48].map((x) => (
        <Dot key={x} x={x} y={31} r={2} fill={SOFT} />
      ))}
    </>
  ),

  /** A PCA: o eixo roda até à direção em que os dados variam mais, e os pontos caem nele. */
  projection: ({ color, at }) => {
    const points = [
      [9, 24],
      [15, 26],
      [17, 18],
      [23, 21],
      [28, 15],
      [33, 18],
      [37, 11],
      [44, 13],
      [47, 7],
    ];
    // A reta do eixo (de 8,27 a 49,7) e o pé da perpendicular de cada ponto.
    const [ax, ay, bx, by] = [8, 27, 49, 7];
    const foot = ([x = 0, y = 0]: number[]) => {
      const t = ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2);
      return [ax + t * (bx - ax), ay + t * (by - ay)];
    };
    return (
      <>
        {points.map((point) => {
          const [fx = 0, fy = 0] = foot(point);
          return (
            <line
              key={`${String(point[0])}-${String(point[1])}`}
              x1={point[0]}
              y1={point[1]}
              x2={fx}
              y2={fy}
              stroke={color}
              strokeOpacity={0.55}
              strokeWidth={0.8}
              className="glyph-light"
              style={at(-0.4)}
            />
          );
        })}
        {points.map(([x = 0, y = 0]) => (
          <Dot key={`p${String(x)}-${String(y)}`} x={x} y={y} r={1.7} />
        ))}
        <line
          x1={ax}
          y1={ay}
          x2={bx}
          y2={by}
          stroke={color}
          strokeWidth={1.6}
          strokeLinecap="round"
          className="glyph-fit"
          style={at()}
        />
      </>
    );
  },

  /** A rede neuronal: o sinal passa de camada em camada. */
  network: ({ color, at }) => {
    const layers = [
      [9, 18, 27].map((y) => [8, y] as const),
      [6, 14, 22, 30].map((y) => [28, y] as const),
      [13, 23].map((y) => [48, y] as const),
    ];
    const edges: [readonly [number, number], readonly [number, number]][] = [];
    for (let layer = 0; layer < layers.length - 1; layer++) {
      for (const from of layers[layer] ?? []) {
        for (const to of layers[layer + 1] ?? []) {
          edges.push([from, to]);
        }
      }
    }
    return (
      <>
        {edges.map(([[x1, y1], [x2, y2]]) => (
          <line
            key={`${String(x1)}-${String(y1)}-${String(x2)}-${String(y2)}`}
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            stroke="#3a3a3a"
            strokeWidth={0.7}
          />
        ))}
        {edges
          .filter((_, index) => index % 3 === 0)
          .map(([[x1, y1], [x2, y2]], index) => (
            <path
              key={`luz-${String(x1)}-${String(y1)}-${String(x2)}-${String(y2)}`}
              d={`M${String(x1)},${String(y1)} L${String(x2)},${String(y2)}`}
              pathLength={1}
              fill="none"
              stroke={color}
              strokeWidth={1.6}
              strokeLinecap="round"
              className="glyph-flow"
              style={at(-0.37 * index)}
            />
          ))}
        {layers.flat().map(([x, y]) => (
          <circle
            key={`${String(x)}-${String(y)}`}
            cx={x}
            cy={y}
            r={2.6}
            fill={CARD}
            stroke={x === 8 ? INK : color}
            strokeWidth={1.3}
          />
        ))}
      </>
    );
  },

  /** Um modelo de uma família ainda sem desenho próprio. */
  dots: ({ color, at }) => (
    <>
      {[16, 28, 40].map((x, index) => (
        <circle
          key={x}
          cx={x}
          cy={18}
          r={3}
          fill={color}
          className="glyph-blink"
          style={at(-1.2 * index)}
        />
      ))}
    </>
  ),
};
