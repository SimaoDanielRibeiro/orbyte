/**
 * Uma procura dos agentes, ao vivo e no fim.
 *
 * Ao vivo: as fases por ordem, o fluxograma com o agente que trabalha agora (a amarelo), o
 * melhor até agora, os candidatos que vão sendo medidos e a conversa dos agentes, com os botões
 * para pausar, retomar e parar. No fim: o vencedor e a sua nota no teste, a procura num gráfico,
 * o melhor de cada agente, o quadro completo e a ponte para o modo manual (treinar qualquer
 * candidato como um run normal).
 */
import { useMutation } from "@tanstack/react-query";
import { Link, useNavigate, useParams } from "@tanstack/react-router";
import { ArrowRight, ChevronDown, ChevronLeft, LoaderCircle } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";

import type { AutoMlResult, Candidate } from "@/api/generated";
import {
  automlCancelMutation,
  automlPauseMutation,
  automlResumeMutation,
  automlTrainCandidateMutation,
} from "@/api/generated/@tanstack/react-query.gen";
import { useLabel } from "@/api/vocabulary";
import { BarChart } from "@/components/charts/bar-chart";
import { Explain } from "@/components/explain/explain";
import { TopicText } from "@/components/markdown/topic-text";
import { NoticeList } from "@/components/notices/notice-list";
import { ErrorNote, NeedsStep, NextStep, Stat, StatStrip } from "@/components/shell/blocks";
import { Page, PageHeader } from "@/components/shell/page";
import { SCREENS } from "@/components/shell/screens";
import { WhyButton } from "@/components/shell/why-button";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { PageSkeleton } from "@/features/data/require-dataset";
import { familyColor } from "@/lib/families";
import { formatCount, formatMetric, formatNumber } from "@/lib/format";
import { useSession } from "@/state/session";

import { AgentDetail } from "./agent-detail";
import { ExchangeHistory } from "./exchange-history";
import { Flowchart, type FlowNode } from "./flowchart";
import { PlanView } from "./plan-view";
import { SearchChart } from "./search-chart";
import { Board, CandidateFeed, Conversation, PhaseSteps } from "./search-parts";
import { useAgentSearch } from "./use-agent-search";
import { changesText, objectiveValue } from "./values";

export function SearchScreen() {
  const { searchId } = useParams({ from: "/agentes/$searchId" });
  return (
    <Page>
      <Search key={searchId} searchId={searchId} />
    </Page>
  );
}

function Search({ searchId }: { searchId: string }) {
  const label = useLabel();
  const { live, status, detached, plan, result } = useAgentSearch(searchId);
  const [selected, setSelected] = useState<FlowNode | null>(null);
  // O fluxograma ao vivo, ou o histórico das trocas (com a mensagem à vista, para a conversa).
  const [view, setView] = useState<"live" | "history">("live");
  const [moment, setMoment] = useState<number | null>(null);
  const openSearch = useSession((state) => state.openSearch);
  useEffect(() => {
    openSearch(searchId);
  }, [searchId, openSearch]);

  if (detached && plan.error !== null && result.error !== null) {
    return (
      <>
        <PageHeader title={SCREENS.agents.title} description={SCREENS.agents.description} />
        <NeedsStep
          message="Esta procura já não está na memória da API: ela recomeçou, ou a procura saiu para dar lugar a outras. O run do vencedor continua no ecrã Experiências."
          to="/agentes"
          action="Fazer outro pedido aos agentes"
        />
      </>
    );
  }
  const data = result.data;
  const searchPlan = data?.plan ?? plan.data;
  const phase = data !== undefined ? "done" : live.phase;
  const candidates = data?.candidates ?? live.candidates;
  const messages = data?.messages ?? live.messages;
  const objective = searchPlan?.objective ?? live.objective;
  const winner =
    data?.winner === null || data === undefined
      ? null
      : (data.candidates.find((candidate) => candidate.number === data.winner) ?? null);
  const running = status === "pending" || status === "running" || status === "paused";
  const working = data === undefined ? live.working : null;
  const leader = winner ?? live.leader;
  return (
    <>
      <Link
        to="/agentes"
        className="-ml-1 inline-flex items-center gap-1 text-sm text-soft hover:text-strong"
      >
        <ChevronLeft aria-hidden="true" className="size-4" />
        Agentes
      </Link>
      <header className="mt-4 flex flex-col gap-5 @4xl/main:flex-row @4xl/main:items-end @4xl/main:justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h1 className="text-[2rem] leading-tight tracking-[-0.025em]">
              {searchPlan === undefined
                ? "A procura dos agentes"
                : `Prever ${searchPlan.target} em ${searchPlan.dataset_name}`}
            </h1>
            <WhyButton
              title="A procura dos agentes"
              description={SCREENS.agents.description}
              topics={[
                "automl",
                "data_agent",
                "model_selection",
                "screening_and_tuning",
                "random_search",
                "winners_curse",
                "train_candidate",
              ]}
            />
          </div>
          {searchPlan !== undefined && (
            <p className="mt-2 font-mono text-xs text-dim">
              {label("TaskType", searchPlan.task_type).toLowerCase()}, a otimizar{" "}
              {searchPlan.objective.label.toLowerCase()}
            </p>
          )}
        </div>
        {running && data === undefined && (
          <Controls searchId={searchId} status={status} measured={candidates.length} live={live} />
        )}
        {data?.run_id != null && <WinnerProceed runId={data.run_id} />}
      </header>
      {status === "failed" && (
        <ErrorNote
          className="mt-8"
          error={new Error(live.state?.error ?? "A procura falhou sem dizer porquê.")}
        />
      )}

      <div className="mt-8 flex flex-wrap items-center justify-between gap-x-8 gap-y-4">
        {view === "history" ? (
          <p className="text-sm text-soft">
            A procura repetida troca a troca: quem fala a quem, e o que diz.
          </p>
        ) : (
          <PhaseSteps phase={phase} />
        )}
        <div className="flex items-center gap-1">
          <ToggleGroup
            type="single"
            variant="outline"
            size="sm"
            aria-label="O que o fluxograma mostra"
            value={view}
            onValueChange={(value) => {
              if (value === "live" || value === "history") {
                setView(value);
                setMoment(null);
              }
            }}
          >
            <ToggleGroupItem value="live" className="px-3">
              {data === undefined ? "Ao vivo" : "No fim"}
            </ToggleGroupItem>
            <ToggleGroupItem value="history" className="px-3" disabled={messages.length === 0}>
              Histórico das trocas
            </ToggleGroupItem>
          </ToggleGroup>
          <Explain topic="exchange_history" />
        </div>
      </div>

      <div className="mt-6 grid gap-x-12 gap-y-10 @8xl/main:grid-cols-[minmax(0,1fr)_26rem]">
        <section
          aria-label={view === "history" ? "O histórico das trocas" : "Os agentes a trabalhar"}
          className="flex min-w-0 flex-col gap-3"
        >
          <div className="rounded-2xl bg-[radial-gradient(circle,#262626_1px,transparent_1.2px)] bg-size-[22px_22px] p-4">
            {searchPlan === undefined ? (
              <PageSkeleton />
            ) : view === "history" ? (
              <ExchangeHistory
                plan={searchPlan}
                messages={messages}
                candidates={candidates}
                objective={objective}
                test={data?.test ?? null}
                finished={data !== undefined}
                onMoment={setMoment}
                selected={selected}
                onSelect={setSelected}
              />
            ) : (
              <Flowchart
                agents={searchPlan.agents}
                phase={phase}
                working={working}
                candidates={candidates}
                objective={objective}
                leader={leader}
                budget={searchPlan.settings.budget ?? null}
                trainRows={searchPlan.train_rows}
                test={data?.test ?? null}
                selected={selected}
                onSelect={setSelected}
              />
            )}
          </div>
          {view === "live" && (
            <p className="mono-label">
              Clica num agente para veres o que está a fazer e o que já fez.
            </p>
          )}
        </section>
        <section className="flex min-w-0 flex-col gap-4 @8xl/main:border-l @8xl/main:pl-8">
          <h2 className="text-[0.9375rem] font-normal tracking-normal text-soft">
            A conversa dos agentes
          </h2>
          <Conversation
            messages={messages}
            active={view === "history" ? moment : null}
            className="max-h-[36rem]"
          />
        </section>
      </div>

      {data === undefined ? (
        <div className="mt-12 grid gap-x-12 gap-y-10 @5xl/main:grid-cols-[minmax(0,24rem)_minmax(0,1fr)]">
          <Leader candidate={live.leader} objective={objective} />
          <Part title="Os candidatos medidos">
            <CandidateFeed candidates={candidates} objective={objective} leader={live.leader} />
          </Part>
        </div>
      ) : (
        <div className="mt-16 flex flex-col gap-16">
          <Finished result={data} winner={winner} />
        </div>
      )}

      {searchPlan !== undefined && (
        <Collapsible className="group/plano mt-16">
          <CollapsibleTrigger asChild>
            <Button variant="outline">
              Ver o plano da procura
              <ChevronDown
                aria-hidden="true"
                className="transition-transform group-data-[state=open]/plano:rotate-180"
              />
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-8">
            <PlanView plan={searchPlan} />
          </CollapsibleContent>
        </Collapsible>
      )}
      {data === undefined && (status === "finished" || detached) && <PageSkeleton />}
      {data?.run_id != null && (
        <NextStep text="A procura acabou. O vencedor é um run como os outros: abre-o para o avaliar e explicar.">
          <WinnerProceed runId={data.run_id} />
        </NextStep>
      )}

      {searchPlan !== undefined && (
        <AgentDetail
          node={selected}
          onClose={() => {
            setSelected(null);
          }}
          onSelect={setSelected}
          agents={searchPlan.agents}
          candidates={candidates}
          messages={messages}
          objective={objective}
          phase={phase}
          working={working}
          leader={leader}
          budget={searchPlan.settings.budget ?? null}
          plan={searchPlan}
          ranking={data?.ranking}
          test={data?.test ?? null}
          runId={data?.run_id ?? null}
        />
      )}
    </>
  );
}

interface ControlsProps {
  searchId: string;
  status: string | null;
  measured: number;
  live: ReturnType<typeof useAgentSearch>["live"];
}

/** O estado da procura e os botões para a pausar, retomar e parar. */
function Controls({ searchId, status, measured, live }: ControlsProps) {
  const label = useLabel();
  const path = { search_id: searchId };
  const pause = useMutation(automlPauseMutation());
  const resume = useMutation(automlResumeMutation());
  const cancel = useMutation(automlCancelMutation());
  const planned = live.state?.planned_steps ?? null;
  const paused = status === "paused";
  return (
    <section
      aria-label="A procura a decorrer"
      className="flex w-full flex-col gap-3 @4xl/main:w-96"
    >
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div role="status" className="flex flex-wrap items-center gap-3">
          <span aria-hidden="true" className="relative flex size-2.5">
            {!paused && (
              <span className="absolute inline-flex size-full animate-ping rounded-full bg-positive opacity-60 motion-reduce:animate-none" />
            )}
            <span
              className={
                paused
                  ? "relative inline-flex size-2.5 rounded-full bg-warning"
                  : "relative inline-flex size-2.5 rounded-full bg-positive shadow-[0_0_10px_var(--positive)]"
              }
            />
          </span>
          <span className="font-medium text-strong">
            {live.state === null ? "A ligar" : label("JobStatus", live.state.status)}
          </span>
          <span className="font-mono text-xs text-dim tabular-nums">
            {formatCount(measured)}
            {planned === null ? "" : ` de ${formatCount(planned)}`} candidatos medidos
          </span>
        </div>
        <div className="flex gap-2">
          {paused ? (
            <Button
              onClick={() => {
                resume.mutate({ path });
              }}
              disabled={resume.isPending}
            >
              Retomar
            </Button>
          ) : (
            <Button
              variant="outline"
              onClick={() => {
                pause.mutate({ path });
              }}
              disabled={pause.isPending || live.state === null}
            >
              Pausar
            </Button>
          )}
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                onClick={() => {
                  cancel.mutate({ path });
                }}
                disabled={cancel.isPending || live.state === null}
              >
                Parar
              </Button>
            </TooltipTrigger>
            <TooltipContent className="max-w-64">
              A procura pára no fim do candidato atual; o vencedor é o melhor até aí.
            </TooltipContent>
          </Tooltip>
        </div>
      </div>
      {planned !== null && (
        <Progress
          value={Math.min(100, (measured / planned) * 100)}
          aria-label="O progresso da procura"
        />
      )}
      <ErrorNote error={pause.error ?? resume.error ?? cancel.error} />
    </section>
  );
}

function Leader({
  candidate,
  objective,
}: {
  candidate: Candidate | null;
  objective: ReturnType<typeof useAgentSearch>["live"]["objective"];
}) {
  const label = useLabel();
  if (candidate === null) {
    return (
      <Part title="O melhor até agora">
        <p className="text-sm text-soft">Ainda nenhum candidato medido.</p>
      </Part>
    );
  }
  return (
    <Part title="O melhor até agora">
      <div>
        <p className="text-xl text-strong">{candidate.model_display_name}</p>
        <p className="mt-1 font-mono text-xs text-dim">
          {label("ModelFamily", candidate.family)}, {changesText(candidate)}
        </p>
      </div>
      <Stat
        label={`${objective?.label ?? "Valor"} na validação cruzada`}
        value={objectiveValue(candidate.mean, objective)}
        detail={`± ${objectiveValue(candidate.std, objective)} entre folds`}
      />
    </Part>
  );
}

/** A procura terminada: o vencedor, a procura num gráfico, o melhor de cada agente e o quadro. */
function Finished({ result, winner }: { result: AutoMlResult; winner: Candidate | null }) {
  const label = useLabel();
  const objective = result.plan.objective;
  const families = result.plan.agents.map((agent) => agent.family);
  const baseline =
    result.baseline === null
      ? null
      : (result.candidates.find((candidate) => candidate.number === result.baseline) ?? null);
  const best = families
    .map((family) => {
      const own = result.ranking
        .map((number) => result.candidates.find((candidate) => candidate.number === number))
        .find((candidate) => candidate?.family === family);
      return own === undefined ? null : { family, candidate: own };
    })
    .filter(
      (entry): entry is { family: (typeof families)[number]; candidate: Candidate } =>
        entry !== null,
    );

  return (
    <>
      {result.notices.length > 0 && <NoticeList notices={result.notices} />}
      <div className="grid gap-x-12 gap-y-10 @5xl/main:grid-cols-2">
        {winner !== null && (
          <Part title="O vencedor">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <p className="text-2xl text-strong">{winner.model_display_name}</p>
                <p className="mt-1 font-mono text-xs text-dim">
                  {label("ModelFamily", winner.family)}, {changesText(winner)}
                </p>
              </div>
              {result.run_id !== null && (
                <Button asChild variant="outline">
                  <Link to="/treinar/$runId" params={{ runId: result.run_id }}>
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
                  value: objectiveValue(winner.mean, objective),
                  detail: `± ${objectiveValue(winner.std, objective)} entre folds`,
                },
                {
                  key: "teste",
                  label: "No teste, uma só vez",
                  value: result.test === null ? "—" : formatMetric(result.test),
                },
                ...(baseline === null
                  ? []
                  : [
                      {
                        key: "referencia",
                        label: "A referência",
                        value: objectiveValue(baseline.mean, objective),
                      },
                    ]),
                {
                  key: "candidatos",
                  label: "Candidatos",
                  value: formatCount(result.candidates.length),
                  detail: `em ${formatNumber(result.duration_seconds, 1)} s`,
                },
              ]}
            />
          </Part>
        )}
        {best.length > 0 && (
          <Part title="O melhor de cada agente">
            <BarChart
              items={best.map(({ family, candidate }) => ({
                key: family,
                label: label("ModelFamily", family),
                value: candidate.mean ?? 0,
                color: familyColor(family),
              }))}
              format={(value) => objectiveValue(value, objective)}
              valueLabel={objective.label}
              labelWidth={190}
              description="O melhor candidato de cada agente, na validação cruzada."
              caption={
                baseline === null
                  ? undefined
                  : `A referência (${baseline.model_display_name} com os valores por omissão) tem ${objectiveValue(baseline.mean, objective)}.`
              }
            />
          </Part>
        )}
      </div>
      <Part title="A procura">
        <SearchChart
          candidates={result.candidates}
          objective={objective}
          families={families}
          winner={result.winner}
          description="Cada candidato pela ordem em que foi medido, com a média e o desvio na validação cruzada."
          caption={<TopicText topic="screening_and_tuning" />}
        />
      </Part>
      <Part title="O quadro dos candidatos">
        <Board result={result} />
        <TrainCandidate result={result} />
      </Part>
    </>
  );
}

/** Prosseguir para o vencedor, no "Treinar e avaliar" (o mesmo aspeto do `Proceed`). */
function WinnerProceed({ runId }: { runId: string }) {
  return (
    <Button asChild variant="proceed" size="proceed">
      <Link to="/treinar/$runId" params={{ runId }}>
        Abrir o vencedor
        <ArrowRight aria-hidden="true" />
      </Link>
    </Button>
  );
}

/** A ponte para o modo manual: treinar um candidato do quadro como um run normal. */
function TrainCandidate({ result }: { result: AutoMlResult }) {
  const navigate = useNavigate();
  const openRun = useSession((state) => state.openRun);
  const [chosen, setChosen] = useState<number | null>(result.ranking[0] ?? null);
  const train = useMutation(automlTrainCandidateMutation());
  const objective = result.plan.objective;
  if (result.ranking.length === 0) {
    return null;
  }
  const candidate = result.candidates.find((one) => one.number === chosen);
  // Fica por baixo do quadro, alinhada à direita: é uma ação sobre ele, não uma secção à parte.
  return (
    <div className="flex flex-col items-end gap-2 pt-4">
      <div className="flex flex-wrap items-center justify-end gap-3">
        <span className="flex items-center gap-1 text-sm text-soft">
          Treinar um candidato à mão
          <Explain topic="train_candidate" />
        </span>
        <Select
          value={chosen === null ? undefined : String(chosen)}
          onValueChange={(value) => {
            setChosen(Number(value));
          }}
        >
          <SelectTrigger className="min-w-72" aria-label="O candidato a treinar">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {result.ranking.map((number, index) => {
              const one = result.candidates.find((item) => item.number === number);
              return (
                <SelectItem key={number} value={String(number)}>
                  {`${formatCount(index + 1)}.º ${one?.model_display_name ?? ""}, ${objectiveValue(one?.mean ?? null, objective)}`}
                </SelectItem>
              );
            })}
          </SelectContent>
        </Select>
        <Button
          variant="outline"
          disabled={chosen === null || train.isPending}
          onClick={() => {
            if (chosen === null) {
              return;
            }
            train.mutate(
              { path: { search_id: result.search_id, number: chosen } },
              {
                onSuccess: (job) => {
                  openRun(job.run_id, candidate?.model_display_name ?? null);
                  void navigate({ to: "/treinar/$runId", params: { runId: job.run_id } });
                },
              },
            );
          }}
        >
          {train.isPending && <LoaderCircle aria-hidden="true" className="animate-spin" />}
          Treinar no modo manual
        </Button>
      </div>
      <ErrorNote error={train.error} />
    </div>
  );
}

interface PartProps {
  title: string;
  /** O tema que explica a parte (no "?" ao lado do nome). */
  topic?: string;
  children: ReactNode;
}

/**
 * Uma parte da procura, numa secção aberta (sem caixa nem filete: o espaço separa-a das outras).
 * Sem tema, o título fica filho direto da secção: quem o procura (um leitor de ecrã, um teste)
 * encontra o conteúdo no mesmo sítio.
 */
function Part({ title, topic, children }: PartProps) {
  const heading = (
    <h2 className="text-[0.9375rem] font-medium tracking-normal text-foreground">{title}</h2>
  );
  return (
    <section className="flex min-w-0 flex-col gap-4">
      {topic === undefined ? (
        heading
      ) : (
        <div className="flex items-center gap-1">
          {heading}
          <Explain topic={topic} />
        </div>
      )}
      {children}
    </section>
  );
}
