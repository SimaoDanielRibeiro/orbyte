/**
 * O painel de um nó do fluxograma, aberto ao lado: o que esse agente está a fazer, o que já fez e
 * como chegou ao que concluiu.
 *
 * - Um agente de uma família: o que mede agora (e o fold), os números do seu melhor candidato,
 *   os modelos que pode experimentar, cada candidato que mediu (com a nota em cada fold) e o que
 *   disse na conversa.
 * - O agente de dados: o plano (as decisões e as razões) e o que disse.
 * - O coordenador: quantos candidatos já mediu, como os repartiu pelos agentes e o que disse.
 * - O líder e o teste: o melhor candidato, com todos os hiperparâmetros e a nota no teste.
 *
 * O painel não é modal: o fluxograma continua à vista e a mexer, e clicar noutro nó troca o
 * painel em vez de o fechar. Tudo o que mostra vem do que a procura já devolve (os candidatos, a
 * conversa e o plano): o painel só os junta por agente.
 */
import { Link } from "@tanstack/react-router";
import { Star } from "lucide-react";
import type { ReactNode } from "react";

import type {
  AgentMessage,
  AutoMlPlan,
  Candidate,
  MetricResult,
  ModelAgentCard,
  ModelFamily,
  ObjectiveOption,
  SearchPhase,
  WorkInProgress,
} from "@/api/generated";
import { useLabel } from "@/api/vocabulary";
import { StatStrip } from "@/components/shell/blocks";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { formatCount, formatMetric, formatNumber, formatValue } from "@/lib/format";

import { COORDINATOR_COLOR, DATA_AGENT_COLOR, familyColor } from "./agent-colors";
import type { FlowNode } from "./flowchart";
import { PlanView } from "./plan-view";
import { Conversation } from "./search-parts";
import { bestCandidate, changesText, objectiveValue } from "./values";

export interface AgentDetailProps {
  node: FlowNode | null;
  onClose: () => void;
  agents: readonly ModelAgentCard[];
  candidates: readonly Candidate[];
  messages: readonly AgentMessage[];
  objective: ObjectiveOption | null;
  phase: SearchPhase | null;
  working: WorkInProgress | null;
  /** O melhor até agora (ou o vencedor, no fim). */
  leader: Candidate | null;
  budget: number | null;
  plan?: AutoMlPlan;
  /** No fim: a ordem do backend, do melhor para o pior. */
  ranking?: readonly number[];
  test?: MetricResult | null;
  /** No fim: o run do vencedor, para o abrir. */
  runId?: string | null;
}

export function AgentDetail(props: AgentDetailProps) {
  const { node, onClose } = props;
  const label = useLabel();
  const families = props.agents.map((agent) => agent.family);
  const heading = headingOf(node, props, label);
  return (
    <Sheet
      modal={false}
      open={node !== null}
      onOpenChange={(open) => {
        if (!open) {
          onClose();
        }
      }}
    >
      <SheetContent
        side="right"
        className="w-full gap-0 overflow-y-auto sm:max-w-2xl"
        onInteractOutside={(event) => {
          // Clicar noutro nó troca o painel; só um clique fora do fluxograma o fecha.
          if (event.target instanceof Element && event.target.closest("[data-flow-node]")) {
            event.preventDefault();
          }
        }}
      >
        {node !== null && (
          <>
            <SheetHeader className="gap-2 border-b p-6">
              <SheetTitle className="flex items-center gap-2.5 pr-8 text-xl font-medium text-strong">
                <span
                  aria-hidden="true"
                  className="size-2.5 shrink-0 rounded-full"
                  style={{ background: heading.color, boxShadow: `0 0 10px ${heading.color}` }}
                />
                {heading.title}
              </SheetTitle>
              <SheetDescription className="text-sm leading-relaxed text-soft">
                {heading.description}
              </SheetDescription>
            </SheetHeader>
            <div className="flex flex-col gap-8 p-6">
              {node === "data" ? (
                <DataDetail {...props} families={families} />
              ) : node === "coordinator" ? (
                <CoordinatorDetail {...props} families={families} />
              ) : node === "winner" || node === "test" ? (
                <LeaderDetail {...props} families={families} />
              ) : (
                <FamilyDetail {...props} family={node} families={families} />
              )}
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  );
}

function headingOf(
  node: FlowNode | null,
  props: AgentDetailProps,
  label: ReturnType<typeof useLabel>,
): { title: string; description: string; color: string } {
  const families = props.agents.map((agent) => agent.family);
  switch (node) {
    case null:
      return { title: "", description: "", color: "transparent" };
    case "data":
      return {
        title: label("AgentRole", "data"),
        description: "O plano para os dados e o que disse na conversa.",
        color: DATA_AGENT_COLOR,
      };
    case "coordinator":
      return {
        title: label("AgentRole", "coordinator"),
        description: "Os candidatos que repartiu pelos agentes e o que disse na conversa.",
        color: COORDINATOR_COLOR,
      };
    case "winner":
    case "test":
      return {
        title: props.phase === "done" || props.phase === "final" ? "O vencedor" : "O líder",
        description: "O melhor candidato na validação cruzada, com todos os números.",
        color: props.leader === null ? "#fafafa" : familyColor(families, props.leader.family),
      };
    default: {
      const agent = props.agents.find((one) => one.family === node);
      return {
        title: label("ModelFamily", node),
        description: agent?.name ?? "Um agente de uma família de modelos.",
        color: familyColor(families, node),
      };
    }
  }
}

type WithFamilies = AgentDetailProps & { families: readonly ModelFamily[] };

/** Um agente de uma família: agora, o melhor, os modelos, os candidatos e a conversa. */
function FamilyDetail(props: WithFamilies & { family: ModelFamily }) {
  const { family, families, candidates, objective, working, phase, leader, ranking } = props;
  const label = useLabel();
  const agent = props.agents.find((one) => one.family === family);
  const color = familyColor(families, family);
  const own = candidates.filter((candidate) => candidate.family === family);
  const best = bestCandidate(own, objective);
  const searching = phase === "screening" || phase === "tuning";
  const busy = searching && working?.family === family ? working : null;
  const seconds = own.reduce((total, candidate) => total + candidate.seconds, 0);
  const place = best === null || ranking === undefined ? 0 : ranking.indexOf(best.number) + 1;
  const gap =
    best?.mean == null || leader?.mean == null || leader.number === best.number
      ? null
      : best.mean - leader.mean;

  return (
    <>
      <Block title="Agora">
        {busy !== null ? (
          <div className="flex flex-col gap-2">
            <p className="text-sm text-strong">
              A medir {busy.model_display_name}, fold {formatCount(busy.fold)} de{" "}
              {formatCount(busy.folds)}
            </p>
            <p className="font-mono text-xs text-dim">{changesText(busy)}</p>
            <span className="h-1 overflow-hidden rounded bg-[#2a2a2a]">
              <span
                className="block h-full rounded transition-[width]"
                style={{
                  width: `${String((busy.fold / Math.max(1, busy.folds)) * 100)}%`,
                  background: color,
                }}
              />
            </span>
          </div>
        ) : (
          <p className="text-sm text-soft">
            {phase === null
              ? "A procura ainda não começou: estes são os modelos que vai experimentar."
              : own.length === 0
                ? phase === "done"
                  ? "Este agente não chegou a medir nenhum candidato."
                  : "À espera da sua vez: o coordenador ainda não lhe deu nenhum candidato."
                : phase === "done"
                  ? "Terminou. Os números em baixo são os finais."
                  : "Está à espera do próximo candidato do coordenador."}
          </p>
        )}
      </Block>

      {best !== null && (
        <StatStrip
          compact
          items={[
            {
              key: "melhor",
              label: `O melhor (${objective?.label.toLowerCase() ?? "valor"})`,
              value: objectiveValue(best.mean, objective),
              detail: `± ${objectiveValue(best.std, objective)} entre folds`,
            },
            {
              key: "treino",
              label: "O mesmo, no treino",
              value: objectiveValue(best.train_mean, objective),
            },
            {
              key: "medidos",
              label: "Candidatos medidos",
              value: formatCount(own.length),
              detail: `${formatNumber(seconds, 2)} s a medir`,
            },
            place > 0
              ? {
                  key: "lugar",
                  label: "Lugar no quadro",
                  value: `${formatCount(place)}.º`,
                  detail: `de ${formatCount(ranking?.length ?? 0)} candidatos`,
                }
              : {
                  key: "lider",
                  label: "Para o líder",
                  value: gap === null ? "É o líder" : signed(gap, objective),
                  detail: gap === null ? undefined : "a diferença do seu melhor",
                  tone: gap === null ? "positive" : "neutral",
                },
          ]}
        />
      )}

      {agent !== undefined && agent.models.length > 0 && (
        <Block title="Os modelos que pode experimentar">
          <ul className="flex flex-col divide-y divide-[#242424]">
            {agent.models.map((model) => (
              <li key={model.name} className="flex flex-col gap-0.5 py-2.5 first:pt-0">
                <span className="text-sm text-strong">{model.display_name}</span>
                {model.searches.length === 0 ? (
                  <span className="font-mono text-xs text-dim">só os valores por omissão</span>
                ) : (
                  <ul className="mt-1 flex flex-col gap-1">
                    {model.searches.map((search) => (
                      <li key={search} className="font-mono text-xs leading-relaxed text-dim">
                        {search}
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ul>
        </Block>
      )}

      {phase !== null && (
        <>
          <Block title="Os candidatos, pela ordem em que os mediu">
            {own.length === 0 ? (
              <p className="text-sm text-soft">Ainda nenhum.</p>
            ) : (
              <CandidateList
                candidates={own}
                objective={objective}
                color={color}
                best={best}
                leader={leader}
                final={phase === "done"}
                label={label}
              />
            )}
          </Block>

          <Block title="O que disse">
            <Conversation
              messages={props.messages.filter((message) => message.family === family)}
              families={families}
            />
          </Block>
        </>
      )}
    </>
  );
}

/** O agente de dados: o plano e o que disse. */
function DataDetail(props: WithFamilies) {
  return (
    <>
      {props.plan === undefined ? (
        <p className="text-sm text-soft">O plano ainda não chegou.</p>
      ) : (
        <PlanView plan={props.plan} />
      )}
      <Block title="O que disse">
        <Conversation
          messages={props.messages.filter((message) => message.role === "data")}
          families={props.families}
        />
      </Block>
    </>
  );
}

/** O coordenador: o progresso, a repartição pelos agentes e o que disse. */
function CoordinatorDetail(props: WithFamilies) {
  const label = useLabel();
  const { candidates, families, budget } = props;
  const counts = props.agents.map((agent) => ({
    family: agent.family,
    count: candidates.filter((candidate) => candidate.family === agent.family).length,
  }));
  const most = Math.max(1, ...counts.map((entry) => entry.count));
  const screening = candidates.filter((candidate) => candidate.phase === "screening").length;
  const failed = candidates.filter((candidate) => candidate.error !== null).length;
  return (
    <>
      <StatStrip
        compact
        items={[
          {
            key: "medidos",
            label: "Candidatos medidos",
            value: formatCount(candidates.length),
            detail: budget === null ? undefined : `de ${formatCount(budget)} no orçamento`,
          },
          {
            key: "triagem",
            label: label("SearchPhase", "screening"),
            value: formatCount(screening),
          },
          {
            key: "afinacao",
            label: label("SearchPhase", "tuning"),
            value: formatCount(candidates.length - screening),
          },
          {
            key: "falhados",
            label: "Falharam",
            value: formatCount(failed),
            tone: failed === 0 ? "positive" : "negative",
          },
        ]}
      />
      <Block title="Como repartiu os candidatos">
        <ul className="flex flex-col gap-2.5">
          {counts.map(({ family, count }) => (
            <li
              key={family}
              className="grid grid-cols-[10rem_minmax(0,1fr)_2rem] items-center gap-3"
            >
              <span className="truncate text-sm text-soft">{label("ModelFamily", family)}</span>
              <span className="h-1.5 rounded bg-[#232323]">
                <span
                  className="block h-full rounded"
                  style={{
                    width: `${String((count / most) * 100)}%`,
                    background: familyColor(families, family),
                    boxShadow: `0 0 8px ${familyColor(families, family)}`,
                  }}
                />
              </span>
              <span className="text-right font-mono text-xs text-dim tabular-nums">
                {formatCount(count)}
              </span>
            </li>
          ))}
        </ul>
      </Block>
      <Block title="O que disse">
        <Conversation
          messages={props.messages.filter((message) => message.role === "coordinator")}
          families={families}
        />
      </Block>
    </>
  );
}

/** O líder (o vencedor, no fim): os números, os folds, os hiperparâmetros e o teste. */
function LeaderDetail(props: WithFamilies) {
  const label = useLabel();
  const { leader, objective, families, test, runId } = props;
  if (leader === null) {
    return (
      <p className="text-sm text-soft">
        Ainda não há líder: o primeiro candidato medido passa a sê-lo.
      </p>
    );
  }
  const color = familyColor(families, leader.family);
  return (
    <>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-2xl text-strong">{leader.model_display_name}</p>
          <p className="mt-1 font-mono text-xs text-dim">
            {label("ModelFamily", leader.family)}, candidato n.º {formatCount(leader.number)},{" "}
            {label("SearchPhase", leader.phase).toLowerCase()}
          </p>
        </div>
        {runId != null && (
          <Button asChild>
            <Link to="/treinar/$runId" params={{ runId }}>
              Abrir o vencedor
            </Link>
          </Button>
        )}
      </div>
      <StatStrip
        compact
        items={[
          {
            key: "validacao",
            label: "Validação cruzada",
            value: objectiveValue(leader.mean, objective),
            detail: `± ${objectiveValue(leader.std, objective)} entre folds`,
          },
          {
            key: "treino",
            label: "No treino",
            value: objectiveValue(leader.train_mean, objective),
          },
          {
            key: "teste",
            label: "No teste, uma só vez",
            value: test == null ? "No fim" : formatMetric(test),
          },
          {
            key: "tempo",
            label: "A medir",
            value: `${formatNumber(leader.seconds, 2)} s`,
          },
        ]}
      />
      <Block title="Cada fold da validação cruzada">
        <FoldDots candidate={leader} domain={domainOf([leader])} color={color} />
      </Block>
      <Block title="Os hiperparâmetros">
        <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-6 gap-y-2 text-sm">
          {Object.entries(leader.hyperparameters).map(([name, value]) => (
            <div key={name} className="contents">
              <dt className="truncate font-mono text-xs leading-5 text-dim">
                {name}
                {name in leader.changes && <span className="ml-1.5 text-soft">(mudou)</span>}
              </dt>
              <dd className="text-right text-strong tabular-nums">{formatValue(value)}</dd>
            </div>
          ))}
        </dl>
      </Block>
    </>
  );
}

/** Os candidatos de um agente, cada um com a nota, o treino e os folds. */
function CandidateList({
  candidates,
  objective,
  color,
  best,
  leader,
  final,
  label,
}: {
  candidates: readonly Candidate[];
  objective: ObjectiveOption | null;
  color: string;
  best: Candidate | null;
  leader: Candidate | null;
  final: boolean;
  label: ReturnType<typeof useLabel>;
}) {
  const domain = domainOf(candidates);
  return (
    <ol className="flex flex-col divide-y divide-[#242424]">
      {candidates.map((candidate) => {
        const isBest = best?.number === candidate.number;
        const leads = leader?.number === candidate.number;
        return (
          <li
            key={candidate.number}
            className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-2 py-3 first:pt-0"
          >
            <div className="min-w-0">
              <p className="flex items-center gap-1.5 text-sm text-strong">
                {isBest && (
                  <Star aria-label="O melhor deste agente" className="size-3.5 fill-current" />
                )}
                <span className="truncate">{candidate.model_display_name}</span>
                {leads && (
                  <span className="rounded-full border border-[#3a3a3a] px-2 text-[11px] text-soft">
                    {final ? "vencedor" : "líder"}
                  </span>
                )}
              </p>
              <p className="mt-0.5 truncate font-mono text-xs text-dim">
                n.º {formatCount(candidate.number)},{" "}
                {label("SearchPhase", candidate.phase).toLowerCase()}, {changesText(candidate)}
              </p>
            </div>
            <div className="text-right">
              {candidate.error === null ? (
                <>
                  <p className="text-sm text-strong tabular-nums">
                    {objectiveValue(candidate.mean, objective)}
                    <span className="ml-1 text-xs text-dim">
                      ± {objectiveValue(candidate.std, objective)}
                    </span>
                  </p>
                  <p className="font-mono text-[11px] text-dim">
                    treino {objectiveValue(candidate.train_mean, objective)},{" "}
                    {formatNumber(candidate.seconds, 2)} s
                  </p>
                </>
              ) : (
                <p className="text-sm text-negative">falhou</p>
              )}
            </div>
            {candidate.error === null ? (
              <FoldDots candidate={candidate} domain={domain} color={color} />
            ) : (
              <p className="col-span-2 text-xs text-negative">{candidate.error}</p>
            )}
          </li>
        );
      })}
    </ol>
  );
}

/**
 * A nota de cada fold, num ponto, numa linha que vai do pior ao melhor fold destes candidatos; o
 * traço branco é a média. Pontos espalhados querem dizer um modelo instável entre folds.
 */
function FoldDots({
  candidate,
  domain,
  color,
}: {
  candidate: Candidate;
  domain: [number, number];
  color: string;
}) {
  const [low, high] = domain;
  const span = high - low || 1;
  // Em percentagem da largura (com uma margem), para os pontos ficarem redondos a qualquer largura.
  const left = (value: number) => `${String(2 + ((value - low) / span) * 96)}%`;
  const folds = candidate.folds.filter((value): value is number => value !== null);
  return (
    <div
      role="img"
      aria-label={`As notas nos ${String(folds.length)} folds: ${folds.map((value) => formatNumber(value, 3)).join(", ")}.`}
      className="relative col-span-2 h-4"
    >
      <span aria-hidden="true" className="absolute inset-x-0 top-1/2 h-px bg-[#2e2e2e]" />
      {folds.map((value, index) => (
        <span
          key={index}
          aria-hidden="true"
          className="absolute top-1/2 size-2 -translate-x-1/2 -translate-y-1/2 rounded-full"
          style={{ left: left(value), background: color, boxShadow: `0 0 6px ${color}` }}
        />
      ))}
      {candidate.mean !== null && (
        <span
          aria-hidden="true"
          className="absolute top-0 h-full w-[1.5px] -translate-x-1/2 bg-white"
          style={{ left: left(candidate.mean) }}
        />
      )}
    </div>
  );
}

function Block({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="flex flex-col gap-3">
      <h3 className="text-[0.9375rem] font-normal text-soft">{title}</h3>
      {children}
    </section>
  );
}

/** Do pior ao melhor fold destes candidatos (a escala dos pontos). */
function domainOf(candidates: readonly Candidate[]): [number, number] {
  const values = candidates.flatMap((candidate) =>
    candidate.folds.filter((value): value is number => value !== null),
  );
  if (values.length === 0) {
    return [0, 1];
  }
  const low = Math.min(...values);
  const high = Math.max(...values);
  return low === high ? [low - 0.01, high + 0.01] : [low, high];
}

/** Uma diferença com o sinal à frente, no formato da métrica. */
function signed(value: number, objective: ObjectiveOption | null): string {
  const text = objectiveValue(Math.abs(value), objective);
  return value > 0 ? `+${text}` : value < 0 ? `−${text}` : text;
}
