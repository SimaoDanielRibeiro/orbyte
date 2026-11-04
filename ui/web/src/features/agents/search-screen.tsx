/**
 * Uma procura dos agentes, ao vivo e no fim.
 *
 * Ao vivo: as fases por ordem, o fluxograma com o agente que trabalha agora (a amarelo), o
 * melhor até agora, os candidatos que vão sendo medidos e a conversa dos agentes, com os botões
 * para pausar, retomar e parar. No fim: o vencedor e a sua nota no teste, a procura num gráfico,
 * o melhor de cada agente, o quadro completo e a ponte para o modo manual (treinar qualquer
 * candidato como um run normal). A conversa abre-se em ecrã inteiro, como um chat
 * (`ConversationRoom`), para se ler como a procura aconteceu.
 */
import { useMutation } from "@tanstack/react-query";
import { Link, useNavigate, useParams } from "@tanstack/react-router";
import { ChevronDown, LoaderCircle, Maximize2, Plus } from "lucide-react";
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
import { ErrorNote, NeedsStep, Stat, StatStrip } from "@/components/shell/blocks";
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
import { formatCount, formatMetric, formatNumber, formatOrdinal } from "@/lib/format";
import { useSession } from "@/state/session";

import { AgentDetail } from "./agent-detail";
import { ConversationRoom } from "./conversation-room";
import { ExchangeHistory } from "./exchange-history";
import { Flowchart, type FlowNode } from "./flowchart";
import { PlanView } from "./plan-view";
import { SearchChart } from "./search-chart";
import { Board, CandidateFeed, Conversation, PhaseSteps } from "./search-parts";
import { useAgentSearch } from "./use-agent-search";
import { changesText, objectiveValue } from "./values";

export function SearchScreen() {
  const { searchId } = useParams({ from: "/agents/$searchId" });
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
  // A conversa em ecrã inteiro, como um chat.
  const [room, setRoom] = useState(false);
  const openSearch = useSession((state) => state.openSearch);
  const lost = detached && plan.error !== null && result.error !== null;
  useEffect(() => {
    // A procura aberta é a que os Agentes mostram ao voltar; uma que a API já não tem esquece-se
    // (senão os Agentes voltavam sempre a este aviso, em vez de deixarem fazer outro pedido).
    openSearch(lost ? null : searchId);
  }, [searchId, lost, openSearch]);

  if (lost) {
    return (
      <>
        <PageHeader title={SCREENS.agents.title} description={SCREENS.agents.description} />
        <NeedsStep
          message="This search is no longer in the API's memory: the API restarted, or the search made room for newer ones. The winner's run is still in History."
          to="/agents"
          action="Make a new request to the agents"
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
  const thought = data === undefined ? live.thought : null;
  const leader = winner ?? live.leader;
  const title =
    searchPlan === undefined
      ? "The agents' search"
      : `Predict ${searchPlan.target} in ${searchPlan.dataset_name}`;
  return (
    <>
      <header className="flex flex-col gap-5 @4xl/main:flex-row @4xl/main:items-end @4xl/main:justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h1 className="text-title">{title}</h1>
            <WhyButton
              title="The agents' search"
              description={
                searchPlan === undefined
                  ? SCREENS.agents.description
                  : `${label("TaskType", searchPlan.task_type)}, optimizing ${searchPlan.objective.label.toLowerCase()}. ${SCREENS.agents.description}`
              }
              topics={[
                "automl",
                "data_agent",
                "model_selection",
                "screening_and_tuning",
                "random_search",
                ...(searchPlan?.ai_agent === true ? ["ai_agent"] : []),
                "winners_curse",
                "train_candidate",
              ]}
            />
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-3 @4xl/main:shrink-0 @4xl/main:flex-nowrap">
          {running && data === undefined && (
            <Controls
              searchId={searchId}
              status={status}
              measured={candidates.length}
              live={live}
            />
          )}
          {!running && <NewSearch />}
          {data?.run_id != null && <WinnerProceed runId={data.run_id} />}
        </div>
      </header>
      {status === "failed" && (
        <ErrorNote
          className="mt-8"
          error={new Error(live.state?.error ?? "The search failed without saying why.")}
        />
      )}

      <div className="mt-8 flex flex-wrap items-center justify-between gap-x-8 gap-y-4">
        {/* No histórico, as fases não dizem nada (o momento é o da troca escolhida). */}
        {view === "history" ? <div /> : <PhaseSteps phase={phase} />}
        <div className="flex items-center gap-1">
          <ToggleGroup
            type="single"
            variant="outline"
            size="sm"
            aria-label="What the flowchart shows"
            value={view}
            onValueChange={(value) => {
              if (value === "live" || value === "history") {
                setView(value);
                setMoment(null);
              }
            }}
          >
            <ToggleGroupItem value="live" className="px-3">
              {data === undefined ? "Live" : "At the end"}
            </ToggleGroupItem>
            <ToggleGroupItem value="history" className="px-3" disabled={messages.length === 0}>
              Exchange history
            </ToggleGroupItem>
          </ToggleGroup>
          {/* O "?" explica a vista aberta (ADR 0038). */}
          {view === "history" ? (
            <Explain topic="exchange_history" align="end" />
          ) : (
            <Explain title="The agents at work" align="end">
              The lit box is the agent working right now. Click an agent to see what it is doing and
              what it has done.
            </Explain>
          )}
        </div>
      </div>

      <div className="mt-6 grid gap-x-12 gap-y-10 @8xl/main:grid-cols-[minmax(0,1fr)_26rem]">
        <section
          aria-label={view === "history" ? "The exchange history" : "The agents at work"}
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
                ai={searchPlan.ai_agent ? { model: searchPlan.ai_model } : null}
                thought={thought}
              />
            )}
          </div>
        </section>
        <section className="flex min-w-0 flex-col gap-4 @8xl/main:border-l @8xl/main:pl-8">
          <div className="flex items-center justify-between gap-3">
            <h2 className="text-base font-normal tracking-normal text-soft">
              The agents&apos; conversation
            </h2>
            <Button
              variant="outline"
              size="sm"
              disabled={messages.length === 0}
              onClick={() => {
                setRoom(true);
              }}
            >
              <Maximize2 aria-hidden="true" />
              Expand
            </Button>
          </div>
          <Conversation
            messages={messages}
            active={view === "history" ? moment : null}
            thought={view === "history" ? null : thought}
            className="max-h-[36rem]"
          />
        </section>
      </div>
      <ConversationRoom
        open={room}
        onOpenChange={setRoom}
        title={title}
        messages={messages}
        candidates={candidates}
        objective={objective}
        thought={thought}
        live={data === undefined && running}
        test={data?.test ?? null}
      />

      {data === undefined ? (
        <div className="mt-12 grid gap-x-12 gap-y-10 @5xl/main:grid-cols-[minmax(0,24rem)_minmax(0,1fr)]">
          <Leader candidate={live.leader} objective={objective} />
          <Part title="The candidates measured">
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
              See the search plan
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

/**
 * O estado da procura e os botões para a pausar, retomar e parar. Parar é parar tudo (ADR 0051):
 * a procura acaba no fim do candidato atual, sem treinar um vencedor, e o ecrã volta logo ao
 * pedido, para uma procura nova.
 */
function Controls({ searchId, status, measured, live }: ControlsProps) {
  const label = useLabel();
  const navigate = useNavigate();
  const openSearch = useSession((state) => state.openSearch);
  const path = { search_id: searchId };
  const pause = useMutation(automlPauseMutation());
  const resume = useMutation(automlResumeMutation());
  const cancel = useMutation({
    ...automlCancelMutation(),
    onSuccess: () => {
      openSearch(null);
      void navigate({ to: "/agents" });
    },
  });
  const planned = live.state?.planned_steps ?? null;
  const paused = status === "paused";
  return (
    <section
      aria-label="The search in progress"
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
            {live.state === null ? "Connecting" : label("JobStatus", live.state.status)}
          </span>
          <span className="font-mono text-xs text-dim tabular-nums">
            {formatCount(measured)}
            {planned === null ? "" : ` of ${formatCount(planned)}`} candidates measured
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
              Resume
            </Button>
          ) : (
            <Button
              variant="outline"
              onClick={() => {
                pause.mutate({ path });
              }}
              disabled={pause.isPending || live.state === null}
            >
              Pause
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
                Stop
              </Button>
            </TooltipTrigger>
            <TooltipContent className="max-w-64">
              Stops everything: nothing more is measured, no winner is trained, and you go back to a
              new search.
            </TooltipContent>
          </Tooltip>
        </div>
      </div>
      {planned !== null && (
        <Progress
          value={Math.min(100, (measured / planned) * 100)}
          aria-label="The search's progress"
        />
      )}
      <ErrorNote error={pause.error ?? resume.error ?? cancel.error} />
    </section>
  );
}

/**
 * Esquece a procura aberta e volta ao pedido aos agentes. Sem isto, os Agentes mostram sempre esta
 * procura (sai-se para o Treinar e volta-se a ela como estava); o vencedor fica no Histórico.
 */
function NewSearch() {
  const navigate = useNavigate();
  const openSearch = useSession((state) => state.openSearch);
  return (
    <Button
      variant="outline"
      size="proceed"
      onClick={() => {
        openSearch(null);
        void navigate({ to: "/agents" });
      }}
    >
      <Plus aria-hidden="true" />
      New search
    </Button>
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
      <Part title="The best so far">
        <p className="text-sm text-soft">No candidate measured yet.</p>
      </Part>
    );
  }
  return (
    <Part title="The best so far">
      <div>
        <p className="text-xl text-strong">{candidate.model_display_name}</p>
        <p className="mt-1 font-mono text-xs text-dim">
          {label("ModelFamily", candidate.family)}, {changesText(candidate)}
        </p>
      </div>
      <Stat
        label={`${objective?.label ?? "Value"} in cross-validation`}
        value={objectiveValue(candidate.mean, objective)}
        detail={`± ${objectiveValue(candidate.std, objective)} across folds`}
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
          <Part title="The winner">
            {/* Abrir o vencedor fica só no cabeçalho, como o passo seguinte de todos os ecrãs. */}
            <div>
              <p className="text-2xl text-strong">{winner.model_display_name}</p>
              <p className="mt-1 font-mono text-xs text-dim">
                {label("ModelFamily", winner.family)}, {changesText(winner)}
              </p>
            </div>
            <StatStrip
              compact
              items={[
                {
                  key: "validation",
                  label: "Cross-validation",
                  value: objectiveValue(winner.mean, objective),
                  detail: `± ${objectiveValue(winner.std, objective)} across folds`,
                },
                {
                  key: "test",
                  label: "On the test, once",
                  value: result.test === null ? "—" : formatMetric(result.test),
                },
                ...(baseline === null
                  ? []
                  : [
                      {
                        key: "baseline",
                        label: "The baseline",
                        value: objectiveValue(baseline.mean, objective),
                      },
                    ]),
                {
                  key: "candidates",
                  label: "Candidates",
                  value: formatCount(result.candidates.length),
                  detail: `in ${formatNumber(result.duration_seconds, 1)} s`,
                },
              ]}
            />
          </Part>
        )}
        {best.length > 0 && (
          <Part title="Each agent's best">
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
              description="Each agent's best candidate, in cross-validation."
              caption={
                baseline === null
                  ? undefined
                  : `The baseline (${baseline.model_display_name} with its default values) has ${objectiveValue(baseline.mean, objective)}.`
              }
            />
          </Part>
        )}
      </div>
      <Part title="The search">
        <SearchChart
          candidates={result.candidates}
          objective={objective}
          families={families}
          winner={result.winner}
          description="Each candidate in the order it was measured, with its mean and deviation in cross-validation."
          caption={<TopicText topic="screening_and_tuning" />}
        />
      </Part>
      <Part title="The candidate board">
        <Board result={result} />
        <TrainCandidate result={result} />
      </Part>
    </>
  );
}

/** Prosseguir para o vencedor, no "Train & evaluate" (o mesmo aspeto do `Proceed`). */
function WinnerProceed({ runId }: { runId: string }) {
  return (
    <Button asChild variant="proceed" size="proceed">
      <Link to="/train/$runId" params={{ runId }}>
        Open the winner
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
          Train a candidate by hand
          <Explain topic="train_candidate" />
        </span>
        <Select
          value={chosen === null ? undefined : String(chosen)}
          onValueChange={(value) => {
            setChosen(Number(value));
          }}
        >
          <SelectTrigger className="min-w-72" aria-label="The candidate to train">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {result.ranking.map((number, index) => {
              const one = result.candidates.find((item) => item.number === number);
              return (
                <SelectItem key={number} value={String(number)}>
                  {`${formatOrdinal(index + 1)} ${one?.model_display_name ?? ""}, ${objectiveValue(one?.mean ?? null, objective)}`}
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
                  void navigate({ to: "/train/$runId", params: { runId: job.run_id } });
                },
              },
            );
          }}
        >
          {train.isPending && <LoaderCircle aria-hidden="true" className="animate-spin" />}
          Train in manual mode
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
    <h2 className="text-base font-medium tracking-normal text-foreground">{title}</h2>
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
