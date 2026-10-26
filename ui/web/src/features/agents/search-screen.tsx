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
import { ChevronLeft, LoaderCircle } from "lucide-react";
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
import { TopicText } from "@/components/markdown/topic-text";
import { NoticeList } from "@/components/notices/notice-list";
import { ErrorNote, Facts, NeedsStep } from "@/components/shell/blocks";
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
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { PageSkeleton } from "@/features/data/require-dataset";
import { categoryColor } from "@/lib/colors";
import { formatCount, formatMetric, formatNumber } from "@/lib/format";
import { useSession } from "@/state/session";

import { Flowchart } from "./flowchart";
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

  return (
    <>
      <Link
        to="/agentes"
        className="-ml-1 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
      >
        <ChevronLeft aria-hidden="true" className="size-4" />
        Agentes
      </Link>
      <header className="mt-4 flex flex-col gap-4 @3xl/main:flex-row @3xl/main:items-start @3xl/main:justify-between">
        <div className="max-w-[68ch]">
          <h1 className="text-4xl leading-tight">
            {searchPlan === undefined
              ? "A procura dos agentes"
              : `Prever ${searchPlan.target} em ${searchPlan.dataset_name}`}
          </h1>
          {searchPlan !== undefined && (
            <p className="mt-3 font-serif text-reading text-muted-foreground">
              {label("TaskType", searchPlan.task_type)}, a otimizar{" "}
              {searchPlan.objective.label.toLowerCase()}.
            </p>
          )}
        </div>
        <WhyButton
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
      </header>

      {running && data === undefined && (
        <Controls searchId={searchId} status={status} measured={candidates.length} live={live} />
      )}
      {status === "failed" && (
        <ErrorNote
          className="mt-8"
          error={new Error(live.state?.error ?? "A procura falhou sem dizer porquê.")}
        />
      )}

      <section aria-label="Os agentes a trabalhar" className="mt-10 flex flex-col gap-6">
        <PhaseSteps phase={phase} />
        {searchPlan === undefined ? (
          <PageSkeleton />
        ) : (
          <Flowchart
            agents={searchPlan.agents}
            phase={phase}
            working={data === undefined ? live.working : null}
            candidates={candidates}
            objective={objective}
            leader={winner ?? live.leader}
            budget={searchPlan.settings.budget ?? null}
            trainRows={searchPlan.train_rows}
            test={data?.test ?? null}
          />
        )}
      </section>

      <div className="mt-12 grid gap-12 @5xl/main:grid-cols-[minmax(0,1fr)_minmax(0,26rem)]">
        <div className="flex min-w-0 flex-col gap-12">
          {data === undefined ? (
            <>
              <Leader candidate={live.leader} objective={objective} />
              <Part title="Os candidatos medidos">
                <CandidateFeed candidates={candidates} objective={objective} leader={live.leader} />
              </Part>
            </>
          ) : (
            <Finished result={data} winner={winner} />
          )}
        </div>
        <Part title="A conversa dos agentes">
          <Conversation messages={messages} className="max-h-[36rem]" />
        </Part>
      </div>

      {searchPlan !== undefined && (
        <Collapsible className="mt-14 border-t pt-6">
          <CollapsibleTrigger className="text-sm text-muted-foreground underline-offset-4 hover:text-foreground hover:underline">
            Ver o plano da procura
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-6">
            <PlanView plan={searchPlan} />
          </CollapsibleContent>
        </Collapsible>
      )}
      {data === undefined && (status === "finished" || detached) && <PageSkeleton />}
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
    <section aria-label="A procura a decorrer" className="mt-8 flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div role="status" className="flex flex-wrap items-center gap-3">
          <span aria-hidden="true" className="relative flex size-2.5">
            {!paused && (
              <span className="absolute inline-flex size-full animate-ping rounded-full bg-now opacity-60 motion-reduce:animate-none" />
            )}
            <span className="relative inline-flex size-2.5 rounded-full bg-now" />
          </span>
          <span className="font-medium">
            {live.state === null ? "A ligar" : label("JobStatus", live.state.status)}
          </span>
          <span className="text-sm text-muted-foreground tabular-nums">
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
    return null;
  }
  return (
    <Part title="O melhor até agora">
      <p className="text-xl">{candidate.model_display_name}</p>
      <p className="font-serif text-sm text-muted-foreground">
        {label("ModelFamily", candidate.family)}, {changesText(candidate)}
      </p>
      <Facts
        size="sm"
        className="mt-3"
        facts={[
          {
            label: `${objective?.label ?? "Valor"} na validação cruzada`,
            value: `${objectiveValue(candidate.mean, objective)} ± ${objectiveValue(candidate.std, objective)}`,
          },
        ]}
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
      <NoticeList notices={result.notices} />
      {winner !== null && (
        <Part title="O vencedor">
          <p className="text-2xl">{winner.model_display_name}</p>
          <p className="font-serif text-sm text-muted-foreground">
            {label("ModelFamily", winner.family)}, {changesText(winner)}
          </p>
          <Facts
            className="mt-4"
            facts={[
              {
                label: "Validação cruzada",
                value: objectiveValue(winner.mean, objective),
              },
              { label: "Desvio entre folds", value: `± ${objectiveValue(winner.std, objective)}` },
              {
                label: "No teste, uma só vez",
                value: result.test === null ? "—" : formatMetric(result.test),
              },
              ...(baseline === null
                ? []
                : [{ label: "A referência", value: objectiveValue(baseline.mean, objective) }]),
            ]}
          />
          <p className="mt-3 text-xs text-muted-foreground tabular-nums">
            {formatCount(result.candidates.length)} candidatos em{" "}
            {formatNumber(result.duration_seconds, 1)} s, com o treino do vencedor.
          </p>
          {result.run_id !== null && (
            <div className="mt-4">
              <Button asChild>
                <Link to="/treinar/$runId" params={{ runId: result.run_id }}>
                  Abrir o vencedor
                </Link>
              </Button>
            </div>
          )}
        </Part>
      )}
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
      {best.length > 0 && (
        <Part title="O melhor de cada agente">
          <BarChart
            items={best.map(({ family, candidate }) => ({
              key: family,
              label: label("ModelFamily", family),
              value: candidate.mean ?? 0,
              color: categoryColor(families.indexOf(family)),
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
      <Part title="O quadro dos candidatos">
        <Board result={result} />
      </Part>
      <TrainCandidate result={result} />
    </>
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
  return (
    <Part title="Treinar um candidato à mão">
      <TopicText topic="train_candidate" />
      <div className="mt-2 flex flex-wrap items-end gap-3">
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
    </Part>
  );
}

function Part({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="flex min-w-0 flex-col gap-3">
      <h2 className="text-xl">{title}</h2>
      {children}
    </section>
  );
}
