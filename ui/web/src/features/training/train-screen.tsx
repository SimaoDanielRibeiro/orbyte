/**
 * O ecrã Treinar e avaliar de um run: o treino ao vivo e, quando acaba, o treino para rever,
 * a avaliação e o que o modelo aprendeu.
 *
 * Ao vivo, o que chega pelo WebSocket: o estado (com o ponto verde a pulsar, que marca o agora),
 * o passo, a loss e a fronteira no último snapshot, com os botões para pausar, retomar e parar.
 * No fim, o resultado completo, em cartões: o campo passo a passo (a rede, se for uma), as
 * métricas no treino e no teste (cada uma abre a sua gaveta), os gráficos da tarefa e o que
 * aprendeu.
 *
 * No cabeçalho ficam só o título e dois botões (guardar e prosseguir), para o nome do modelo não
 * se partir em três linhas; os dados do run são o primeiro número da faixa, não uma linha por
 * baixo do título, e "Experimentar outro modelo" fica no fim, junto do passo seguinte (ADR 0038).
 * "Guardar" leva ao ecrã Guardar, com este run: lá vê-se o ficheiro antes de o gravar, e guardar
 * também guarda o run no histórico (ADR 0039).
 */
import { useMutation } from "@tanstack/react-query";
import { Link, useParams } from "@tanstack/react-router";
import { ChevronLeft, Package } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";

import { ApiRequestError } from "@/api/errors";
import type { ExperimentResult } from "@/api/generated";
import {
  experimentsCancelMutation,
  experimentsPauseMutation,
  experimentsResumeMutation,
} from "@/api/generated/@tanstack/react-query.gen";
import { useLabel } from "@/api/vocabulary";
import { FieldView } from "@/components/charts/field";
import { FieldPlayer } from "@/components/charts/field-player";
import { TopicText } from "@/components/markdown/topic-text";
import { NoticeList } from "@/components/notices/notice-list";
import { ErrorNote, NeedsStep, StatStrip } from "@/components/shell/blocks";
import { Page, PageHeader } from "@/components/shell/page";
import { SCREENS } from "@/components/shell/screens";
import { WhyButton } from "@/components/shell/why-button";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { PageSkeleton } from "@/features/data/require-dataset";
import { EmbeddingView } from "@/features/models/embedding-view";
import { LearnedViewPanel } from "@/features/models/learned-view";
import { formatCount, formatNumber, formatStatistic } from "@/lib/format";
import { useSession } from "@/state/session";

import { Evaluation } from "./evaluation";
import { LossCharts } from "./loss-chart";
import { NetworkPanel } from "./network-panel";
import { useTrainingRun, type LiveTraining } from "./use-training-run";

const FIELD_HEIGHT = 360;

export function TrainScreen() {
  const { runId } = useParams({ from: "/train/$runId" });
  return (
    <Page>
      <TrainRun key={runId} runId={runId} />
    </Page>
  );
}

function TrainRun({ runId }: { runId: string }) {
  const { live, status, detached, result, fromHistory } = useTrainingRun(runId);
  const runTitle = useSession((state) => (state.runId === runId ? state.runTitle : null));
  const openRun = useSession((state) => state.openRun);
  const data = result.data;

  // O run aberto passa a ser o último (o ecrã Explicar e a barra lateral voltam a ele).
  useEffect(() => {
    openRun(runId);
  }, [runId, openRun]);

  if (detached && result.error !== null) {
    const missing =
      result.error instanceof ApiRequestError && result.error.code === "run_not_found";
    return (
      <>
        <PageHeader title={SCREENS.train.title} description={SCREENS.train.description} />
        {missing ? (
          <NeedsStep
            message="This run is neither in the API's memory nor in the history: maybe it was deleted. The recorded runs are in History."
            to="/history"
            action="Open the history"
          />
        ) : (
          <ErrorNote error={result.error} className="mt-10" />
        )}
      </>
    );
  }
  const title = data?.model_display_name ?? runTitle ?? "The training";
  const live_ = status === "pending" || status === "running" || status === "paused";

  return (
    <>
      <Link
        to={data === undefined ? "/models" : "/models/$name"}
        params={data === undefined ? undefined : { name: data.model_name }}
        className="-ml-1 inline-flex items-center gap-1 text-sm text-soft hover:text-strong"
      >
        <ChevronLeft aria-hidden="true" className="size-4" />
        {data === undefined ? "Models" : data.model_display_name}
      </Link>
      {/* O título fica numa linha sempre que cabe: os botões só lhe ficam ao lado se houver
          espaço para os dois, e senão passam para baixo (sem o espremer em três linhas). */}
      <header className="mt-4 flex flex-wrap items-end justify-between gap-x-8 gap-y-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h1 className="text-title">{title}</h1>
            <WhyButton
              title={SCREENS.train.title}
              description={SCREENS.train.description}
              topics={[
                "loss",
                "snapshots",
                "decision_boundary",
                "train_vs_test",
                "cross_validation",
                "learning_curve",
              ]}
            />
          </div>
        </div>
        {data !== undefined && (
          <div className="flex flex-wrap items-center gap-2 @3xl/main:ml-auto">
            <Button asChild variant="outline" size="proceed">
              <Link to="/save/$runId" params={{ runId }}>
                <Package aria-hidden="true" />
                Save
              </Link>
            </Button>
            <ExplainProceed runId={runId} />
          </div>
        )}
      </header>
      {live_ && data === undefined && <Live live={live} runId={runId} />}
      {status === "failed" && (
        <ErrorNote
          className="mt-10"
          error={new Error(live.state?.error ?? "The training failed without saying why.")}
        />
      )}
      {data === undefined ? (
        (status === "finished" || detached) && <PageSkeleton />
      ) : (
        <Finished result={data} fromHistory={fromHistory} />
      )}
    </>
  );
}

/** Prosseguir para o Explicar deste run (o mesmo aspeto do `Proceed`, com o id do run). */
function ExplainProceed({ runId }: { runId: string }) {
  return (
    <Button asChild variant="proceed" size="proceed">
      <Link to="/explain/$runId" params={{ runId }}>
        Explain the predictions
      </Link>
    </Button>
  );
}

/** O treino a decorrer. */
function Live({ live, runId }: { live: LiveTraining; runId: string }) {
  const label = useLabel();
  const path = { run_id: runId };
  const pause = useMutation(experimentsPauseMutation());
  const resume = useMutation(experimentsResumeMutation());
  const cancel = useMutation(experimentsCancelMutation());
  const state = live.state;
  const steps = state?.progress.steps ?? 0;
  const planned = state?.planned_steps ?? null;
  const loss = state?.progress.last_loss ?? null;
  const paused = state?.status === "paused";

  return (
    <section aria-label="The training in progress" className="mt-8 flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-4 py-1">
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
            {state === null ? "Connecting" : label("JobStatus", state.status)}
          </span>
          <span className="font-mono text-xs text-dim tabular-nums">
            Step {formatCount(steps)}
            {planned === null ? "" : ` of ${formatCount(planned)} at most`}
            {loss === null ? "" : `, loss ${formatStatistic(loss)}`}
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
              disabled={pause.isPending || state === null}
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
                disabled={cancel.isPending || state === null}
              >
                Stop
              </Button>
            </TooltipTrigger>
            <TooltipContent className="max-w-64">
              The training stops at the end of the current step; the model keeps what it has learned
              and is still evaluated.
            </TooltipContent>
          </Tooltip>
        </div>
      </div>
      {planned !== null && (
        <Progress
          value={Math.min(100, (steps / planned) * 100)}
          aria-label="The training's progress"
        />
      )}
      <ErrorNote error={pause.error ?? resume.error ?? cancel.error} />
      <div className="grid gap-x-12 gap-y-10 @5xl/main:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
        <div className="min-w-0">
          {live.preview === null ? (
            <div
              className="flex items-center justify-center rounded-xl border border-dashed text-sm text-dim"
              style={{ height: FIELD_HEIGHT }}
            >
              Waiting for the first snapshot of the training.
            </div>
          ) : (
            <FieldView
              animation={live.preview}
              height={FIELD_HEIGHT}
              description="What the model answers at each point of the plane, in the latest snapshot."
              caption={live.preview.how_to_read}
            />
          )}
        </div>
        <div className="min-w-0">
          <LossCharts steps={live.steps} />
        </div>
      </div>
    </section>
  );
}

/** O run terminado: o treino para rever, a avaliação e o que aprendeu. */
function Finished({ result, fromHistory }: { result: ExperimentResult; fromHistory: boolean }) {
  const label = useLabel();
  const [step, setStep] = useState<number | undefined>(undefined);
  const steps = result.history.at(-1)?.step ?? 0;
  return (
    <>
      <StatStrip
        className="mt-8"
        items={[
          {
            key: "data",
            label: "Data",
            value: result.dataset_name,
            // A tarefa, e de onde veio o run quando não é desta sessão.
            detail: `${label("TaskType", result.task_type).toLowerCase()}${fromHistory ? ", from the history" : ""}`,
          },
          {
            key: "training",
            label: "Training",
            value: steps > 0 ? `${formatCount(steps)} steps` : "In one go",
          },
          {
            key: "duration",
            label: "Duration",
            value: `${formatNumber(result.duration_seconds, 2)} s`,
          },
          {
            key: "run",
            label: "Run",
            value: <span className="font-mono text-lg font-normal">{result.run_id}</span>,
            detail: `seed ${String(result.config.seed ?? "—")}`,
          },
        ]}
      />
      {result.training_notices.length > 0 && (
        <div className="mt-6">
          <NoticeList notices={result.training_notices} />
        </div>
      )}

      <Part id="training" title="The training">
        <div className="grid gap-x-12 gap-y-10 @5xl/main:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
          <div className="min-w-0">
            {result.animation !== null ? (
              <FieldPlayer
                animation={result.animation}
                autoPlay={result.animation.frames.length > 1}
                height={FIELD_HEIGHT}
                onStepChange={setStep}
                description="What the model answers at each point of the plane, step by step."
                caption={result.animation.how_to_read}
              />
            ) : result.embedding !== null ? (
              <EmbeddingView
                embedding={result.embedding}
                height={FIELD_HEIGHT}
                description="The training points on the two new coordinates."
                caption={<TopicText topic="embedding" />}
              />
            ) : (
              <p className="text-sm text-soft">
                There&apos;s no way to draw the model: the data has no 2D projection (choose one in
                Prepare), or the run opened from the history without the step-by-step training.
              </p>
            )}
          </div>
          <div className="min-w-0">
            {result.history.length > 0 ? (
              <LossCharts steps={result.history} />
            ) : (
              <p className="text-sm text-soft">
                This model trains in one go: there are no steps to review.
              </p>
            )}
          </div>
        </div>
      </Part>

      {result.network !== null && (
        <Part id="network" title="The network's layers">
          <div>
            <NetworkPanel network={result.network} step={step} />
          </div>
        </Part>
      )}

      <Part id="evaluation" title="The evaluation">
        <Evaluation result={result} />
      </Part>

      <Part id="learned" title="What it learned">
        <LearnedViewPanel learned={result.learned} network={null} />
      </Part>
    </>
  );
}

function Part({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-titulo`} className="mt-20 scroll-mt-6">
      <h2 id={`${id}-titulo`} className="mb-6 text-xl">
        {title}
      </h2>
      {children}
    </section>
  );
}
