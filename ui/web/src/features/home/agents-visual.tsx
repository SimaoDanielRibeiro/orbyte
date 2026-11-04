/**
 * O desenho dos Agentes na entrada: uma equipa procura por ti.
 *
 * Um sistema orbital, como o nome: o coordenador no centro e os agentes de cada família na
 * órbita, cada um na cor da sua família (a mesma em toda a interface). Em ciclo, o coordenador
 * dá a vez a um agente de cada vez (a luz vai e volta), a nota dele aparece, e no fim o melhor
 * acende-se. Uns bytes andam pela órbita, e o agente de IA é um satélite numa órbita mais larga.
 *
 * As animações são do próprio SVG (SMIL), todas com o mesmo relógio. Com "reduzir movimento",
 * mostra-se o fim, parado: as notas todas e o vencedor aceso.
 */
import { useReducedMotion } from "motion/react";
import { useId } from "react";

import type { ModelFamily } from "@/api/generated";
import { GlowFilter } from "@/components/charts/chart";
import { AI_AGENT_COLOR } from "@/features/agents/agent-colors";
import { familyColor } from "@/lib/families";

const CYCLE = 9;
const RESET = 8.85;
/** O coordenador, a órbita dos agentes e a do agente de IA. */
const CENTER = { x: 260, y: 134 } as const;
const ORBIT = { rx: 188, ry: 84 } as const;
const OUTER = { rx: 240, ry: 110 } as const;

/** Onde fica o nome de cada agente, em relação ao seu nó. */
type Side = "left" | "right";

interface Agent {
  family: ModelFamily;
  name: string;
  /** O ângulo na órbita, em graus (0 à direita, a crescer para baixo). */
  angle: number;
  side: Side;
  /** A nota dele (um exemplo: a exatidão na validação cruzada). */
  score: string;
}

/** Os agentes pela ordem em que o coordenador lhes dá a vez (à volta da órbita). */
const AGENTS: readonly Agent[] = [
  { family: "linear", name: "Linear", angle: 198, side: "left", score: "78.4%" },
  { family: "tree", name: "Trees", angle: 270, side: "right", score: "78.6%" },
  { family: "ensemble", name: "Ensembles", angle: 342, side: "right", score: "81.6%" },
  { family: "kernel", name: "Kernel", angle: 54, side: "right", score: "80.2%" },
  { family: "neural", name: "Neural", angle: 126, side: "left", score: "79.5%" },
];
/** O vencedor: o de melhor nota. */
const WINNER = 2;

/** Quando começa a vez de cada agente, e quanto dura. */
const FIRST_TURN = 0.3;
const TURN = 1.4;
const CROWN = FIRST_TURN + AGENTS.length * TURN;

function at(agent: Agent): { x: number; y: number } {
  const radians = (agent.angle * Math.PI) / 180;
  return {
    x: CENTER.x + ORBIT.rx * Math.cos(radians),
    y: CENTER.y + ORBIT.ry * Math.sin(radians),
  };
}

function ellipse({ rx, ry }: { rx: number; ry: number }): string {
  const left = CENTER.x - rx;
  const right = CENTER.x + rx;
  return `M${String(left)} ${String(CENTER.y)} A${String(rx)} ${String(ry)} 0 1 1 ${String(right)} ${String(CENTER.y)} A${String(rx)} ${String(ry)} 0 1 1 ${String(left)} ${String(CENTER.y)}`;
}

function times(...seconds: readonly number[]): string {
  return seconds.map((second) => String(Math.round((second / CYCLE) * 10000) / 10000)).join(";");
}

/** As props comuns de uma animação que se repete com o ciclo. */
function loop(keyTimes: string) {
  return { keyTimes, dur: `${String(CYCLE)}s`, repeatCount: "indefinite" } as const;
}

export function AgentsVisual() {
  const reduced = useReducedMotion() ?? false;
  const id = useId();
  const glow = `${id}-glow`;
  const orbit = ellipse(ORBIT);
  const outer = ellipse(OUTER);
  const crown = times(0, CROWN, CROWN + 0.3, RESET, CYCLE);
  const winner = AGENTS[WINNER];
  const winnerAt = winner === undefined ? CENTER : at(winner);
  return (
    <svg
      viewBox="0 0 520 260"
      role="img"
      aria-label="With the agents: the coordinator gives each model family's agent a turn, each one reports its score, and the best one wins."
      className="mx-auto block h-auto max-h-80 w-full font-sans tabular-nums"
    >
      <defs>
        <GlowFilter id={glow} blur={4} />
      </defs>

      {/* As órbitas, finas, e os raios do coordenador a cada agente. */}
      <path d={outer} fill="none" stroke="#232323" strokeDasharray="1 7" strokeLinecap="round" />
      <path d={orbit} fill="none" stroke="#2c2c2c" strokeDasharray="3 6" />
      {AGENTS.map((agent) => {
        const point = at(agent);
        return (
          <path
            key={`spoke-${agent.family}`}
            d={`M${String(CENTER.x)} ${String(CENTER.y)} L${String(point.x)} ${String(point.y)}`}
            stroke="#232323"
          />
        );
      })}

      {/* Os bytes que andam pela órbita (por baixo dos agentes). */}
      {(["var(--data-2)", "var(--data-1)", "#d4d4d4"] as const).map((color, index) => (
        <rect
          key={color}
          x={-2}
          y={-2}
          width={4}
          height={4}
          rx={1}
          fill={color}
          transform={
            reduced
              ? `translate(${String(CENTER.x + ORBIT.rx * Math.cos(1 + index * 2.1))} ${String(CENTER.y + ORBIT.ry * Math.sin(1 + index * 2.1))})`
              : undefined
          }
        >
          {!reduced && (
            <animateMotion
              path={orbit}
              dur="27s"
              begin={`-${String(index * 9)}s`}
              repeatCount="indefinite"
            />
          )}
        </rect>
      ))}

      {/* O vencedor: o raio até ele acende-se no fim. */}
      <path
        d={`M${String(CENTER.x)} ${String(CENTER.y)} L${String(winnerAt.x)} ${String(winnerAt.y)}`}
        stroke="#fafafa"
        strokeOpacity={0.6}
        opacity={reduced ? 1 : 0}
      >
        {!reduced && <animate attributeName="opacity" values="0;0;1;1;0" {...loop(crown)} />}
      </path>

      {AGENTS.map((agent, index) => {
        const point = at(agent);
        const color = familyColor(agent.family);
        const start = FIRST_TURN + index * TURN;
        const request = `M${String(CENTER.x)} ${String(CENTER.y)} L${String(point.x)} ${String(point.y)}`;
        const reply = `M${String(point.x)} ${String(point.y)} L${String(CENTER.x)} ${String(CENTER.y)}`;
        const labelX = agent.side === "left" ? point.x - 16 : point.x + 16;
        const anchor = agent.side === "left" ? "end" : "start";
        return (
          <g key={agent.family}>
            {/* A vez dele: a luz vai do coordenador até ele, ele acende-se, e a resposta volta. */}
            {!reduced && (
              <>
                <path
                  d={request}
                  stroke="#f5f5f5"
                  strokeWidth={2.2}
                  strokeLinecap="round"
                  pathLength={100}
                  strokeDasharray="10 100"
                  strokeDashoffset={10}
                >
                  <animate
                    attributeName="stroke-dashoffset"
                    values="10;10;-100;-100"
                    {...loop(times(0, start, start + 0.55, CYCLE))}
                  />
                </path>
                <path
                  d={reply}
                  stroke={color}
                  strokeWidth={2.2}
                  strokeLinecap="round"
                  pathLength={100}
                  strokeDasharray="10 100"
                  strokeDashoffset={10}
                  filter={`url(#${glow})`}
                >
                  <animate
                    attributeName="stroke-dashoffset"
                    values="10;10;-100;-100"
                    {...loop(times(0, start + 0.65, start + 1.2, CYCLE))}
                  />
                </path>
                <circle
                  cx={point.x}
                  cy={point.y}
                  r={11}
                  fill="none"
                  stroke={color}
                  strokeWidth={1.5}
                  opacity={0}
                >
                  <animate
                    attributeName="opacity"
                    values="0;0;0.9;0;0"
                    {...loop(times(0, start + 0.45, start + 0.6, start + 1.05, CYCLE))}
                  />
                  <animate
                    attributeName="r"
                    values="11;11;17;19;19"
                    {...loop(times(0, start + 0.45, start + 0.6, start + 1.05, CYCLE))}
                  />
                </circle>
              </>
            )}

            {/* O agente: o nó na cor da família, o nome e a nota (que aparece quando responde). */}
            <circle
              cx={point.x}
              cy={point.y}
              r={9}
              fill="#161616"
              stroke={color}
              strokeWidth={1.5}
            />
            <circle cx={point.x} cy={point.y} r={3} fill={color} />
            <text
              x={labelX}
              y={point.y}
              dy="0.34em"
              textAnchor={anchor}
              fontSize={11}
              fill="#bdbdbd"
            >
              {agent.name}
            </text>
            <text
              x={point.x}
              y={point.y - 17}
              textAnchor="middle"
              fontSize={10.5}
              fill={index === WINNER ? "#fafafa" : "#8a8a8a"}
              opacity={reduced ? 1 : 0}
            >
              {!reduced && (
                <animate
                  attributeName="opacity"
                  values="0;0;1;1;0"
                  {...loop(times(0, start + 1.1, start + 1.3, RESET, CYCLE))}
                />
              )}
              {agent.score}
            </text>
          </g>
        );
      })}

      {/* O vencedor, no fim: um anel claro à volta dele. */}
      <circle
        cx={winnerAt.x}
        cy={winnerAt.y}
        r={15}
        fill="none"
        stroke="#fafafa"
        strokeWidth={1.25}
        opacity={reduced ? 1 : 0}
        filter={`url(#${glow})`}
      >
        {!reduced && <animate attributeName="opacity" values="0;0;1;1;0" {...loop(crown)} />}
      </circle>

      {/* O coordenador, no centro, com as ondas de quem chama. */}
      {!reduced &&
        [0, 1.2].map((begin) => (
          <circle
            key={begin}
            cx={CENTER.x}
            cy={CENTER.y}
            r={20}
            fill="none"
            stroke="#8a8a8a"
            strokeOpacity={0.35}
            opacity={0}
          >
            <animate
              attributeName="r"
              values="20;52"
              dur="2.4s"
              begin={`${String(begin)}s`}
              repeatCount="indefinite"
            />
            <animate
              attributeName="opacity"
              values="0.9;0"
              dur="2.4s"
              begin={`${String(begin)}s`}
              repeatCount="indefinite"
            />
          </circle>
        ))}
      <circle cx={CENTER.x} cy={CENTER.y} r={20} fill="#1b1b1b" stroke="#4a4a4a" />
      <circle cx={CENTER.x} cy={CENTER.y} r={5} fill="#fafafa" filter={`url(#${glow})`} />
      <text x={CENTER.x} y={CENTER.y + 36} textAnchor="middle" fontSize={11} fill="#d4d4d4">
        Coordinator
      </text>

      {/* O agente de IA: um satélite tracejado, numa órbita mais larga. */}
      <g
        transform={
          reduced
            ? `translate(${String(CENTER.x + OUTER.rx * 0.766)} ${String(CENTER.y - OUTER.ry * 0.643)})`
            : undefined
        }
      >
        {!reduced && <animateMotion path={outer} dur="40s" begin="-12s" repeatCount="indefinite" />}
        <circle
          r={7}
          fill="#161616"
          stroke={AI_AGENT_COLOR}
          strokeWidth={1.25}
          strokeDasharray="2.5 2"
        />
        <circle r={2.2} fill={AI_AGENT_COLOR} />
        <text x={12} y={0} dy="0.34em" fontSize={10.5} fill={AI_AGENT_COLOR}>
          AI
        </text>
      </g>
    </svg>
  );
}
