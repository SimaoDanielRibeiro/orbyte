/**
 * O ecrã Treinar e avaliar de um run: o treino ao vivo e, quando acaba, o treino para rever,
 * a avaliação e o que o modelo aprendeu.
 *
 * Ao vivo, o que chega pelo WebSocket: o estado (com o ponto amarelo, que marca o agora), o
 * passo, a loss e a fronteira no último snapshot, com os botões para pausar, retomar e parar.
 * No fim, o resultado completo: o campo passo a passo (a rede, se for uma), as métricas no
 * treino e no teste (cada uma abre a sua gaveta), os gráficos da tarefa e o que aprendeu.
 */
import { useMutation } from "@tanstack/react-query";
import { Link, useParams } from "@tanstack/react-router";
import { ChevronLeft } from "lucide-react";
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
import { ErrorNote, Facts, NeedsStep } from "@/components/shell/blocks";
import { Page, PageHeader } from "@/components/shell/page";
import { SCREENS } from "@/components/shell/screens";
import { WhyButton } from "@/components/shell/why-button";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { PageSkeleton } from "@/features/data/require-dataset";
import { SaveRunButton } from "@/features/history/save-run";
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
  const { runId } = useParams({ from: "/treinar/$runId" });
  return (
    <Page>
      <TrainRun key={runId} runId={runId} />
    </Page>
  );
}

function TrainRun({ runId }: { runId: string }) {
  const label = useLabel();
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
            message="Este run não está na memória da API nem no histórico: talvez tenha sido apagado. Os runs registados estão no ecrã Experiências."
            to="/experiencias"
            action="Abrir as experiências"
          />
        ) : (
          <ErrorNote error={result.error} className="mt-10" />
        )}
      </>
    );
  }
  const title = data?.model_display_name ?? runTitle ?? "O treino";
  const live_ = status === "pending" || status === "running" || status === "paused";

  return (
    <>
      <Link
        to={data === undefined ? "/modelos" : "/modelos/$name"}
        params={data === undefined ? undefined : { name: data.model_name }}
        className="-ml-1 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
      >
        <ChevronLeft aria-hidden="true" className="size-4" />
        {data === undefined ? "Modelos" : data.model_display_name}
      </Link>
      <header className="mt-4 flex flex-col gap-4 @3xl/main:flex-row @3xl/main:items-start @3xl/main:justify-between">
        <div className="max-w-[68ch]">
          <h1 className="text-4xl leading-tight">{title}</h1>
          {data !== undefined && (
            <p className="mt-3 font-serif text-reading text-muted-foreground">
              Em {data.dataset_name}, {label("TaskType", data.task_type).toLowerCase()}.
            </p>
          )}
        </div>
        <div className="flex shrink-0 flex-wrap items-start gap-2">
          {data !== undefined && <SaveRunButton runId={runId} />}
          <WhyButton
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
      </header>
      {fromHistory && data !== undefined && (
        <p className="mt-4 text-sm text-muted-foreground">Aberto do histórico.</p>
      )}
      {live_ && data === undefined && <Live live={live} runId={runId} />}
      {status === "failed" && (
        <ErrorNote
          className="mt-10"
          error={new Error(live.state?.error ?? "O treino falhou sem dizer porquê.")}
        />
      )}
      {data === undefined ? (
        (status === "finished" || detached) && <PageSkeleton />
      ) : (
        <Finished result={data} />
      )}
    </>
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
    <section aria-label="O treino a decorrer" className="mt-10 flex flex-col gap-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div role="status" className="flex flex-wrap items-center gap-3">
          <span aria-hidden="true" className="relative flex size-2.5">
            {!paused && (
              <span className="absolute inline-flex size-full animate-ping rounded-full bg-now opacity-60 motion-reduce:animate-none" />
            )}
            <span className="relative inline-flex size-2.5 rounded-full bg-now" />
          </span>
          <span className="font-medium">
            {state === null ? "A ligar" : label("JobStatus", state.status)}
          </span>
          <span className="text-sm text-muted-foreground tabular-nums">
            Passo {formatCount(steps)}
            {planned === null ? "" : ` de ${formatCount(planned)}, no máximo`}
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
              Retomar
            </Button>
          ) : (
            <Button
              variant="outline"
              onClick={() => {
                pause.mutate({ path });
              }}
              disabled={pause.isPending || state === null}
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
                disabled={cancel.isPending || state === null}
              >
                Parar
              </Button>
            </TooltipTrigger>
            <TooltipContent className="max-w-64">
              O treino pára no fim do passo atual; o modelo fica com o que já aprendeu e é avaliado
              na mesma.
            </TooltipContent>
          </Tooltip>
        </div>
      </div>
      {planned !== null && (
        <Progress
          value={Math.min(100, (steps / planned) * 100)}
          aria-label="O progresso do treino"
        />
      )}
      <ErrorNote error={pause.error ?? resume.error ?? cancel.error} />
      <div className="grid gap-10 @5xl/main:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
        {live.preview === null ? (
          <div
            className="flex items-center justify-center rounded-md border border-dashed font-serif text-sm text-muted-foreground"
            style={{ height: FIELD_HEIGHT }}
          >
            À espera da primeira fotografia do treino.
          </div>
        ) : (
          <FieldView
            animation={live.preview}
            height={FIELD_HEIGHT}
            description="O que o modelo responde em cada ponto do plano, no último snapshot."
            caption={live.preview.how_to_read}
          />
        )}
        <LossCharts steps={live.steps} />
      </div>
    </section>
  );
}

/** O run terminado: o treino para rever, a avaliação e o que aprendeu. */
function Finished({ result }: { result: ExperimentResult }) {
  const [step, setStep] = useState<number | undefined>(undefined);
  const steps = result.history.at(-1)?.step ?? 0;
  return (
    <>
      <Facts
        className="mt-6"
        size="sm"
        facts={[
          {
            label: "Treino",
            value: steps > 0 ? `${formatCount(steps)} passos` : "De uma vez",
          },
          { label: "Duração", value: `${formatNumber(result.duration_seconds, 2)} s` },
          { label: "Seed", value: result.config.seed ?? "—" },
          { label: "Run", value: result.run_id },
        ]}
      />
      <NoticeList notices={result.training_notices} className="mt-6" />

      <Part id="treino" title="O treino">
        <div className="grid gap-10 @5xl/main:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
          <div className="min-w-0">
            {result.animation !== null ? (
              <FieldPlayer
                animation={result.animation}
                autoPlay={result.animation.frames.length > 1}
                height={FIELD_HEIGHT}
                onStepChange={setStep}
                description="O que o modelo responde em cada ponto do plano, passo a passo."
                caption={result.animation.how_to_read}
              />
            ) : result.embedding !== null ? (
              <EmbeddingView
                embedding={result.embedding}
                height={FIELD_HEIGHT}
                description="Os pontos de treino nas duas coordenadas novas."
                caption={<TopicText topic="embedding" />}
              />
            ) : (
              <p className="font-serif text-reading text-muted-foreground">
                Não há como desenhar o modelo: os dados não têm uma projeção 2D (escolhe uma no ecrã
                Preparar), ou o run abriu do histórico sem o treino passo a passo.
              </p>
            )}
          </div>
          {result.history.length > 0 ? (
            <LossCharts steps={result.history} />
          ) : (
            <p className="font-serif text-reading text-muted-foreground">
              Este modelo treina de uma vez: não há passos para rever.
            </p>
          )}
        </div>
      </Part>

      {result.network !== null && (
        <Part id="rede" title="As camadas da rede">
          <NetworkPanel network={result.network} step={step} />
        </Part>
      )}

      <Part id="avaliacao" title="A avaliação">
        <Evaluation result={result} />
      </Part>

      <Part id="aprendeu" title="O que aprendeu">
        <LearnedViewPanel learned={result.learned} network={null} />
      </Part>

      <div className="mt-16 flex flex-wrap items-center gap-4 border-t pt-6">
        <p className="font-serif text-reading text-muted-foreground">E agora?</p>
        <Button asChild>
          <Link to="/explicar/$runId" params={{ runId: result.run_id }}>
            Explicar as previsões
          </Link>
        </Button>
        <Button asChild variant="outline">
          <Link to="/modelos">Experimentar outro modelo</Link>
        </Button>
      </div>
    </>
  );
}

function Part({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-titulo`} className="mt-14 scroll-mt-6">
      <h2 id={`${id}-titulo`} className="mb-6 text-2xl">
        {title}
      </h2>
      {children}
    </section>
  );
}
