/**
 * O painel de um nó do fluxograma, aberto ao lado: o que esse agente está a fazer, o que já fez e
 * como chegou ao que concluiu.
 *
 * É largo e arrumado em três andares (ADR 0037):
 *
 * 1. **o cabeçalho**, sempre à vista: a cor e o nome do agente, o que faz agora numa frase, e uma
 *    fila com todos os agentes, para saltar de um para outro sem voltar ao fluxograma (que o
 *    painel tapa em parte);
 * 2. **os separadores**, para não ficar tudo numa coluna comprida:
 *    - um agente de uma família: o resumo (os números do seu melhor e os modelos que
 *      experimenta), os candidatos que mediu (com a nota em cada fold) e o que disse;
 *    - o agente de dados: o plano (as decisões e as razões) e o que disse;
 *    - o coordenador: o resumo (quantos candidatos mediu e como os repartiu) e o que disse;
 *    - o líder e o teste não têm separadores: o melhor candidato, os folds e os hiperparâmetros;
 * 3. **o conteúdo do separador**, que rola por baixo do cabeçalho.
 *
 * O painel não é modal: o fluxograma continua a mexer, e clicar noutro nó (ou noutro agente da
 * fila) troca o painel em vez de o fechar. Mudar de agente mantém o separador, quando o outro
 * também o tem: compara-se os candidatos de dois agentes num clique. Tudo o que mostra vem do que
 * a procura já devolve (os candidatos, a conversa e o plano): o painel só os junta por agente.
 */
import { Link } from "@tanstack/react-router";
import { cn } from "cn";
import { Star } from "lucide-react";
import { ToggleGroup } from "radix-ui";
import { useState, type ReactNode } from "react";

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
import { TopicText } from "@/components/markdown/topic-text";
import { Stat, type StatItem } from "@/components/shell/blocks";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { withAlpha } from "@/lib/colors";
import { formatCount, formatMetric, formatNumber, formatQuantity, formatValue } from "@/lib/format";

import { AI_AGENT_COLOR, COORDINATOR_COLOR, DATA_AGENT_COLOR, familyColor } from "./agent-colors";
import type { FlowNode } from "./flowchart";
import { PlanView } from "./plan-view";
import { Conversation } from "./search-parts";
import { bestCandidate, changesText, objectiveValue } from "./values";

export interface AgentDetailProps {
  node: FlowNode | null;
  onClose: () => void;
  /** Abrir outro nó (a fila de agentes do cabeçalho). */
  onSelect: (node: FlowNode) => void;
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

type DetailTab = "summary" | "plan" | "candidates" | "conversation";

interface TabInfo {
  value: DetailTab;
  title: string;
  /** Quantos há (os candidatos, as mensagens): um número pequeno ao lado do nome. */
  count?: number;
}

export function AgentDetail(props: AgentDetailProps) {
  const { node, onClose, onSelect } = props;
  const label = useLabel();
  const [chosen, setChosen] = useState<DetailTab>("summary");
  const heading = headingOf(node, props, label);
  const tabs = node === null ? [] : tabsOf(node, props);
  // O separador escolhido fica ao mudar de agente, se o outro também o tiver.
  const tab = tabs.find((one) => one.value === chosen)?.value ?? tabs[0]?.value ?? "summary";

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
        className="gap-0 overflow-hidden data-[side=right]:w-full data-[side=right]:sm:max-w-[46rem]"
        onInteractOutside={(event) => {
          // Clicar noutro nó troca o painel; só um clique fora do fluxograma o fecha.
          if (event.target instanceof Element && event.target.closest("[data-flow-node]")) {
            event.preventDefault();
          }
        }}
      >
        {node !== null && (
          <Tabs
            value={tab}
            onValueChange={(value) => {
              setChosen(value as DetailTab);
            }}
            className="flex min-h-0 flex-1 flex-col gap-0"
          >
            <SheetHeader className="shrink-0 gap-5 border-b px-7 pt-6 pb-5">
              <div className="flex flex-col gap-2 pr-10">
                <SheetTitle className="text-[1.375rem] leading-tight font-medium tracking-[-0.01em] text-strong">
                  {heading.title}
                </SheetTitle>
                {/* O que o agente faz, numa frase: os números do painel já o dizem à vista. */}
                <SheetDescription className="sr-only">{heading.description}</SheetDescription>
              </div>
              <NodeSwitcher {...props} node={node} onSelect={onSelect} label={label} />
              {tabs.length > 1 && (
                <TabsList aria-label="O que ver deste agente">
                  {tabs.map((one) => (
                    <TabsTrigger key={one.value} value={one.value}>
                      {one.title}
                      {one.count !== undefined && (
                        <span className="font-mono text-xs font-normal text-dim tabular-nums">
                          {formatCount(one.count)}
                        </span>
                      )}
                    </TabsTrigger>
                  ))}
                </TabsList>
              )}
            </SheetHeader>
            <div className="@container/panel min-h-0 flex-1 overflow-y-auto px-7 pt-7 pb-10">
              {node === "data" ? (
                <DataDetail {...props} />
              ) : node === "coordinator" ? (
                <CoordinatorDetail {...props} />
              ) : node === "winner" || node === "test" ? (
                <LeaderDetail {...props} />
              ) : node === "ai" ? (
                <AiDetail {...props} />
              ) : (
                <FamilyDetail {...props} family={node} />
              )}
            </div>
          </Tabs>
        )}
      </SheetContent>
    </Sheet>
  );
}

/** Os separadores de cada nó (o líder e o teste não têm: é uma página só). */
function tabsOf(node: FlowNode, props: AgentDetailProps): TabInfo[] {
  const said = (filter: (message: AgentMessage) => boolean) => props.messages.filter(filter).length;
  switch (node) {
    case "winner":
    case "test":
      return [];
    case "data":
      return [
        { value: "plan", title: "O plano" },
        {
          value: "conversation",
          title: "O que disse",
          count: said((message) => message.role === "data"),
        },
      ];
    case "coordinator":
      return [
        { value: "summary", title: "Resumo" },
        {
          value: "conversation",
          title: "O que disse",
          count: said((message) => message.role === "coordinator"),
        },
      ];
    case "ai":
      return props.phase === null
        ? [{ value: "summary", title: "Resumo" }]
        : [
            { value: "summary", title: "Resumo" },
            {
              value: "candidates",
              title: "Propostas",
              count: props.candidates.filter((candidate) => candidate.ai).length,
            },
            {
              value: "conversation",
              title: "O que pensou e disse",
              count: said((message) => message.role === "ai"),
            },
          ];
    default:
      // Antes de a procura começar, um agente só tem os modelos que vai experimentar.
      return props.phase === null
        ? [{ value: "summary", title: "Resumo" }]
        : [
            { value: "summary", title: "Resumo" },
            {
              value: "candidates",
              title: "Candidatos",
              count: props.candidates.filter(
                (candidate) => candidate.family === node && !candidate.ai,
              ).length,
            },
            {
              value: "conversation",
              title: "O que disse",
              count: said((message) => message.family === node),
            },
          ];
  }
}

function headingOf(
  node: FlowNode | null,
  props: AgentDetailProps,
  label: ReturnType<typeof useLabel>,
): { title: string; description: string; color: string } {
  const finished = props.phase === "done" || props.phase === "final";
  switch (node) {
    case null:
      return { title: "", description: "", color: "transparent" };
    case "data":
      return {
        title: label("AgentRole", "data"),
        description:
          props.plan === undefined
            ? "Prepara os dados antes de os outros agentes começarem."
            : `Separou ${formatCount(props.plan.train_rows)} linhas para treinar e guardou ${formatCount(props.plan.test_rows)} para o teste do vencedor.`,
        color: DATA_AGENT_COLOR,
      };
    case "coordinator":
      return {
        title: label("AgentRole", "coordinator"),
        description:
          props.phase === null
            ? "Vai repartir os candidatos pelos agentes e comparar o que cada um mede."
            : props.budget === null
              ? `Mediu ${formatQuantity(props.candidates.length, "candidato", "candidatos")}.`
              : `Mediu ${formatCount(props.candidates.length)} de ${formatCount(props.budget)} candidatos do orçamento.`,
        color: COORDINATOR_COLOR,
      };
    case "winner":
    case "test":
      return {
        title: finished ? "O vencedor" : "O líder",
        description:
          props.leader === null
            ? "Ainda não há líder: o primeiro candidato medido passa a sê-lo."
            : `${props.leader.model_display_name}, o melhor na validação cruzada${finished ? "" : " até agora"}.`,
        color: props.leader === null ? "#fafafa" : familyColor(props.leader.family),
      };
    case "ai":
      return {
        title: label("AgentRole", "ai"),
        description: aiStatus(props),
        color: AI_AGENT_COLOR,
      };
    default:
      return {
        title: label("ModelFamily", node),
        description: familyStatus(node, props),
        color: familyColor(node),
      };
  }
}

/** O que o agente de IA está a fazer, numa frase. */
function aiStatus(props: AgentDetailProps): string {
  const { phase, working, candidates } = props;
  const own = candidates.filter((candidate) => candidate.ai).length;
  if (phase === null) {
    return "Entra na afinação: lê os resultados e pensa no próximo candidato.";
  }
  if (working?.ai === true) {
    return `A medir a proposta dele, ${working.model_display_name}.`;
  }
  return own === 0
    ? "Ainda não propôs nenhum candidato medido."
    : `Propôs ${formatQuantity(own, "candidato medido", "candidatos medidos")}.`;
}

/** O que um agente de uma família está a fazer, numa frase. */
function familyStatus(family: ModelFamily, props: AgentDetailProps): string {
  const { phase, working, candidates } = props;
  const own = candidates.filter((candidate) => candidate.family === family && !candidate.ai).length;
  const searching = phase === "screening" || phase === "tuning";
  if (phase === null) {
    return "Ainda não começou: em baixo estão os modelos que vai experimentar.";
  }
  if (searching && working?.family === family && !working.ai) {
    return `A medir ${working.model_display_name}, fold ${formatCount(working.fold)} de ${formatCount(working.folds)}.`;
  }
  if (own === 0) {
    return phase === "done"
      ? "Não chegou a medir nenhum candidato."
      : "À espera da sua vez: o coordenador ainda não lhe deu nenhum candidato.";
  }
  return phase === "done"
    ? `Terminou: mediu ${formatQuantity(own, "candidato", "candidatos")}.`
    : `Mediu ${formatQuantity(own, "candidato", "candidatos")}; espera pelo próximo.`;
}

/**
 * Todos os agentes, pela ordem do fluxograma, cada um com a sua cor: o painel tapa parte do
 * fluxograma, e daqui salta-se para qualquer um. O teste abre o painel do vencedor.
 */
function NodeSwitcher({
  node,
  agents,
  phase,
  leader,
  plan,
  onSelect,
  label,
}: AgentDetailProps & {
  node: FlowNode;
  label: ReturnType<typeof useLabel>;
}) {
  const finished = phase === "done" || phase === "final";
  const current: FlowNode = node === "test" ? "winner" : node;
  const options: { key: FlowNode; title: string; color: string }[] = [
    { key: "data", title: label("AgentRole", "data"), color: DATA_AGENT_COLOR },
    { key: "coordinator", title: label("AgentRole", "coordinator"), color: COORDINATOR_COLOR },
    ...agents.map((agent) => ({
      key: agent.family,
      title: label("ModelFamily", agent.family),
      color: familyColor(agent.family),
    })),
    ...(plan?.ai_agent === true
      ? [{ key: "ai" as const, title: label("AgentRole", "ai"), color: AI_AGENT_COLOR }]
      : []),
    {
      key: "winner",
      title: finished ? "Vencedor" : "Líder",
      color: leader === null ? "#fafafa" : familyColor(leader.family),
    },
  ];
  return (
    <ToggleGroup.Root
      type="single"
      value={current}
      onValueChange={(value) => {
        if (value !== "") {
          onSelect(value as FlowNode);
        }
      }}
      aria-label="Ver outro agente"
      // No telemóvel, uma fila que rola para o lado; com espaço, os agentes partem em linhas.
      className="-mx-7 flex [scrollbar-width:none] gap-1.5 overflow-x-auto px-7 sm:mx-0 sm:flex-wrap sm:px-0"
    >
      {options.map((option) => (
        <ToggleGroup.Item
          key={option.key}
          value={option.key}
          // O contorno tem a cor do agente: fraco nos outros, inteiro no aberto.
          style={{
            borderColor: option.key === current ? option.color : withAlpha(option.color, 0.4),
          }}
          className={cn(
            "flex h-7 shrink-0 items-center rounded-full border px-3 text-[0.8125rem] whitespace-nowrap text-soft transition-colors outline-none hover:text-strong focus-visible:ring-2 focus-visible:ring-ring",
            "data-[state=on]:bg-surface-high data-[state=on]:text-strong",
          )}
        >
          {option.title}
        </ToggleGroup.Item>
      ))}
    </ToggleGroup.Root>
  );
}

/** Um agente de uma família: o resumo, os candidatos e o que disse. */
function FamilyDetail(props: AgentDetailProps & { family: ModelFamily }) {
  const { family, candidates, objective, working, phase, leader, ranking } = props;
  const label = useLabel();
  const agent = props.agents.find((one) => one.family === family);
  const color = familyColor(family);
  // Os candidatos que o agente de IA propôs nesta família são dele, não do agente da família.
  const own = candidates.filter((candidate) => candidate.family === family && !candidate.ai);
  const best = bestCandidate(own, objective);
  const searching = phase === "screening" || phase === "tuning";
  const busy = searching && working?.family === family && !working.ai ? working : null;
  const seconds = own.reduce((total, candidate) => total + candidate.seconds, 0);
  const place = best === null || ranking === undefined ? 0 : ranking.indexOf(best.number) + 1;
  const gap =
    best?.mean == null || leader?.mean == null || leader.number === best.number
      ? null
      : best.mean - leader.mean;

  return (
    <>
      <TabsContent value="summary" className="flex flex-col gap-10">
        {busy !== null && (
          <div
            className="flex flex-col gap-3 rounded-xl border bg-ground p-4"
            style={{ borderColor: color }}
          >
            <p className="text-sm text-strong">A medir {busy.model_display_name}</p>
            <p className="font-mono text-xs text-dim">{changesText(busy)}</p>
            <div className="flex items-center gap-3">
              <span className="h-1 flex-1 overflow-hidden rounded bg-[#2a2a2a]">
                <span
                  className="block h-full rounded transition-[width]"
                  style={{
                    width: `${String((busy.fold / Math.max(1, busy.folds)) * 100)}%`,
                    background: color,
                  }}
                />
              </span>
              <span className="font-mono text-xs text-dim tabular-nums">
                fold {formatCount(busy.fold)} de {formatCount(busy.folds)}
              </span>
            </div>
          </div>
        )}

        {best !== null && (
          <StatRow
            items={[
              {
                key: "melhor",
                label: `O melhor (${objective?.label.toLowerCase() ?? "valor"})`,
                value: objectiveValue(best.mean, objective),
                detail: `± ${objectiveValue(best.std, objective)} entre folds`,
              },
              {
                key: "treino",
                label: "No treino",
                value: objectiveValue(best.train_mean, objective),
              },
              {
                key: "medidos",
                label: "Candidatos",
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
          <Block title="Os modelos que experimenta">
            <ul className="grid gap-3 @xl/panel:grid-cols-2">
              {agent.models.map((model) => {
                const tried = own.filter((candidate) => candidate.model === model.name);
                const top = bestCandidate(tried, objective);
                return (
                  <li
                    key={model.name}
                    className="flex flex-col gap-3 rounded-xl border border-[#2a2a2a] bg-ground p-4"
                    style={{ borderColor: withAlpha(color, 0.55) }}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <span className="text-[0.9375rem] text-strong">{model.display_name}</span>
                      {tried.length > 0 && (
                        <span className="shrink-0 text-right font-mono text-xs text-dim tabular-nums">
                          {formatCount(tried.length)}×
                          {top !== null && (
                            <span className="ml-1.5 text-soft">
                              {objectiveValue(top.mean, objective)}
                            </span>
                          )}
                        </span>
                      )}
                    </div>
                    {model.searches.length === 0 ? (
                      <p className="font-mono text-xs text-dim">Só os valores por omissão.</p>
                    ) : (
                      <ul className="flex flex-col gap-1.5">
                        {model.searches.map((search) => (
                          <li key={search} className="font-mono text-xs leading-relaxed text-dim">
                            {search}
                          </li>
                        ))}
                      </ul>
                    )}
                  </li>
                );
              })}
            </ul>
          </Block>
        )}
      </TabsContent>

      {phase !== null && (
        <>
          <TabsContent value="candidates">
            {own.length === 0 ? (
              <p className="text-sm text-soft">Ainda não mediu nenhum candidato.</p>
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
          </TabsContent>
          <TabsContent value="conversation">
            <Conversation
              messages={props.messages.filter((message) => message.family === family)}
            />
          </TabsContent>
        </>
      )}
    </>
  );
}

/** O agente de dados: o plano e o que disse. */
/** O agente de IA: o que é, as propostas medidas (com a razão de cada uma) e o que pensou. */
function AiDetail(props: AgentDetailProps) {
  const { candidates, objective, working, phase, leader, ranking } = props;
  const label = useLabel();
  const own = candidates.filter((candidate) => candidate.ai);
  const best = bestCandidate(own, objective);
  const place = best === null || ranking === undefined ? 0 : ranking.indexOf(best.number) + 1;
  const busy = working?.ai === true ? working : null;

  return (
    <>
      <TabsContent value="summary" className="flex flex-col gap-10">
        {busy !== null && (
          <div
            className="flex flex-col gap-2 rounded-xl border border-dashed bg-ground p-4"
            style={{ borderColor: AI_AGENT_COLOR }}
          >
            <p className="text-sm text-strong">A medir a proposta: {busy.model_display_name}</p>
            <p className="font-mono text-xs text-dim">
              {changesText(busy)} · fold {formatCount(busy.fold)} de {formatCount(busy.folds)}
            </p>
          </div>
        )}
        {best !== null && (
          <StatRow
            items={[
              {
                key: "melhor",
                label: `A melhor proposta (${objective?.label.toLowerCase() ?? "valor"})`,
                value: objectiveValue(best.mean, objective),
                detail: best.model_display_name,
              },
              {
                key: "propostas",
                label: "Propostas medidas",
                value: formatCount(own.length),
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
                    label: "É o líder?",
                    value: leader?.number === best.number ? "Sim" : "Não",
                  },
            ]}
          />
        )}
        <Block
          title={`Como pensa${props.plan?.ai_model == null ? "" : ` (${props.plan.ai_model})`}`}
        >
          <TopicText topic="ai_agent" className="max-w-[65ch]" />
        </Block>
      </TabsContent>

      {phase !== null && (
        <>
          <TabsContent value="candidates" className="flex flex-col gap-8">
            {own.length === 0 ? (
              <p className="text-sm text-soft">Ainda não tem nenhuma proposta medida.</p>
            ) : (
              <>
                <CandidateList
                  candidates={own}
                  objective={objective}
                  color={AI_AGENT_COLOR}
                  best={best}
                  leader={leader}
                  final={phase === "done"}
                  label={label}
                />
                <Block title="Porque propôs cada uma">
                  <ol className="flex flex-col gap-3">
                    {own.map((candidate) => (
                      <li key={candidate.number} className="text-sm leading-relaxed">
                        <span className="font-mono text-xs text-dim tabular-nums">
                          n.º {formatCount(candidate.number)} · {candidate.model_display_name}
                        </span>
                        <p className="text-soft">{candidate.reason ?? "Não disse porquê."}</p>
                      </li>
                    ))}
                  </ol>
                </Block>
              </>
            )}
          </TabsContent>
          <TabsContent value="conversation">
            <Conversation messages={props.messages.filter((message) => message.role === "ai")} />
          </TabsContent>
        </>
      )}
    </>
  );
}

function DataDetail(props: AgentDetailProps) {
  return (
    <>
      <TabsContent value="plan">
        {props.plan === undefined ? (
          <p className="text-sm text-soft">O plano ainda não chegou.</p>
        ) : (
          <PlanView plan={props.plan} />
        )}
      </TabsContent>
      <TabsContent value="conversation">
        <Conversation messages={props.messages.filter((message) => message.role === "data")} />
      </TabsContent>
    </>
  );
}

/** O coordenador: o progresso, a repartição pelos agentes e o que disse. */
function CoordinatorDetail(props: AgentDetailProps) {
  const label = useLabel();
  const { candidates, budget } = props;
  const counts = props.agents.map((agent) => ({
    family: agent.family,
    count: candidates.filter((candidate) => candidate.family === agent.family).length,
  }));
  const most = Math.max(1, ...counts.map((entry) => entry.count));
  const screening = candidates.filter((candidate) => candidate.phase === "screening").length;
  const failed = candidates.filter((candidate) => candidate.error !== null).length;
  return (
    <>
      <TabsContent value="summary" className="flex flex-col gap-10">
        <StatRow
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
          <ul className="flex flex-col gap-3">
            {counts.map(({ family, count }) => (
              <li
                key={family}
                className="grid grid-cols-[minmax(0,15rem)_minmax(0,1fr)_2.5rem] items-center gap-4"
              >
                <span className="truncate text-sm text-soft">{label("ModelFamily", family)}</span>
                <span className="h-2 rounded-full bg-[#232323]">
                  <span
                    className="block h-full rounded-full"
                    style={{
                      width: `${String((count / most) * 100)}%`,
                      background: familyColor(family),
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
      </TabsContent>
      <TabsContent value="conversation">
        <Conversation
          messages={props.messages.filter((message) => message.role === "coordinator")}
        />
      </TabsContent>
    </>
  );
}

/** O líder (o vencedor, no fim): os números, os folds, os hiperparâmetros e o teste. */
function LeaderDetail(props: AgentDetailProps) {
  const label = useLabel();
  const { leader, objective, test, runId } = props;
  if (leader === null) {
    return (
      <p className="text-sm text-soft">
        Quando o coordenador medir o primeiro candidato, ele aparece aqui.
      </p>
    );
  }
  const color = familyColor(leader.family);
  const changed = Object.keys(leader.changes);
  return (
    <div className="flex flex-col gap-10">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="text-2xl text-strong">{leader.model_display_name}</p>
          <p
            className="mt-3 inline-flex h-7 items-center rounded-full border px-3 text-[0.8125rem] text-soft"
            style={{ borderColor: color }}
          >
            {label("ModelFamily", leader.family)}
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
      <StatRow
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
            label: "No teste",
            value: test == null ? "No fim" : formatMetric(test),
            detail: "uma só vez, no fim",
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
        <p className="font-mono text-xs text-dim">
          Cada ponto é a nota num fold; o traço branco é a média.
        </p>
      </Block>
      <Block title="Os hiperparâmetros">
        <dl className="grid gap-x-10 gap-y-2.5 @xl/panel:grid-cols-2">
          {Object.entries(leader.hyperparameters).map(([name, value]) => (
            <div
              key={name}
              className="flex items-baseline justify-between gap-4 border-b border-[#242424] pb-2.5"
            >
              <dt className="truncate font-mono text-xs text-dim">
                {name}
                {changed.includes(name) && <span className="ml-1.5 text-soft">(mudou)</span>}
              </dt>
              <dd className="text-sm text-strong tabular-nums">{formatValue(value)}</dd>
            </div>
          ))}
        </dl>
      </Block>
    </div>
  );
}

/** Os números do painel, numa fila: quatro lado a lado, ou dois a dois se o painel for estreito. */
function StatRow({ items }: { items: readonly StatItem[] }) {
  return (
    <div className="grid grid-cols-2 gap-y-6 @xl/panel:grid-cols-4">
      {items.map(({ key, ...stat }) => (
        <Stat
          key={key}
          {...stat}
          className="border-[#262626] pr-5 even:border-l even:pl-5 @xl/panel:border-l @xl/panel:pl-5 @xl/panel:first:border-l-0 @xl/panel:first:pl-0"
        />
      ))}
    </div>
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
    <ol className="flex flex-col gap-2">
      {candidates.map((candidate) => {
        const isBest = best?.number === candidate.number;
        const leads = leader?.number === candidate.number;
        return (
          <li
            key={candidate.number}
            className={cn(
              "grid grid-cols-[2.75rem_minmax(0,1fr)_auto] gap-x-4 gap-y-3 rounded-xl border px-4 py-3.5",
              isBest ? "border-[#363636] bg-surface-high" : "border-[#242424]",
            )}
          >
            <span className="pt-0.5 font-mono text-xs text-dim tabular-nums">
              n.º {formatCount(candidate.number)}
            </span>
            <div className="min-w-0">
              <p className="flex items-center gap-1.5 text-sm text-strong">
                {isBest && (
                  <Star
                    aria-label="O melhor deste agente"
                    className="size-3.5 shrink-0 fill-current"
                    style={{ color }}
                  />
                )}
                <span className="truncate">{candidate.model_display_name}</span>
                {leads && (
                  <span className="shrink-0 rounded-full border border-[#3a3a3a] px-2 text-[12px] text-soft">
                    {final ? "vencedor" : "líder"}
                  </span>
                )}
              </p>
              <p className="mt-1 font-mono text-xs leading-relaxed text-dim">
                {label("SearchPhase", candidate.phase)}: {changesText(candidate)}
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
                  <p className="mt-1 font-mono text-[12px] text-dim">
                    treino {objectiveValue(candidate.train_mean, objective)} ·{" "}
                    {formatNumber(candidate.seconds, 2)} s
                  </p>
                </>
              ) : (
                <p className="text-sm text-negative">falhou</p>
              )}
            </div>
            {candidate.error === null ? (
              <div className="col-span-2 col-start-2">
                <FoldDots candidate={candidate} domain={domain} color={color} />
              </div>
            ) : (
              <p className="col-span-2 col-start-2 text-xs text-negative">{candidate.error}</p>
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
      className="relative h-4"
    >
      <span aria-hidden="true" className="absolute inset-x-0 top-1/2 h-px bg-[#2e2e2e]" />
      {folds.map((value, index) => (
        <span
          key={index}
          aria-hidden="true"
          className="absolute top-1/2 size-2 -translate-x-1/2 -translate-y-1/2 rounded-full"
          style={{ left: left(value), background: color }}
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
    <section className="flex flex-col gap-4">
      <h3 className="text-[0.9375rem] font-medium tracking-normal text-foreground">{title}</h3>
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
