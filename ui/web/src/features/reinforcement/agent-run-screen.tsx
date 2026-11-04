/**
 * O treino de um agente de reinforcement learning: ao vivo e, quando acaba, o que aprendeu.
 *
 * Ao vivo, pelo WebSocket: o estado (com o ponto amarelo, que marca o agora), o episódio, a
 * recompensa de cada episódio e a política no último snapshot, com Pausar, Retomar e Parar.
 * No fim: as métricas no treino (a explorar) e no teste (com a política aprendida), a
 * recompensa e a parte dos episódios que chega ao objetivo (dois gráficos: escalas diferentes
 * nunca partilham eixos), a grelha com o que o agente sabia em cada snapshot e a tabela Q.
 */
import { useMutation } from "@tanstack/react-query";
import { Link, useParams } from "@tanstack/react-router";
import { ChevronLeft } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";

import { ApiRequestError } from "@/api/errors";
import type {
  EvaluationReport,
  Partition,
  PolicyFrame,
  ReinforcementResult,
} from "@/api/generated";
import {
  reinforcementCancelMutation,
  reinforcementPauseMutation,
  reinforcementResumeMutation,
} from "@/api/generated/@tanstack/react-query.gen";
import { useLabel } from "@/api/vocabulary";
import { GridWorld } from "@/components/charts/grid-world";
import { LineChart } from "@/components/charts/line-chart";
import { Explain } from "@/components/explain/explain";
import { OptionSlider } from "@/components/forms/option-slider";
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
import { SaveRunButton } from "@/features/history/save-run";
import { CATEGORICAL } from "@/lib/colors";
import {
  formatCount,
  formatMetric,
  formatNumber,
  formatPercent,
  formatStatistic,
} from "@/lib/format";
import { useSession } from "@/state/session";

import { useAgentRun, type LiveAgent } from "./use-agent-run";

const PARTITIONS: readonly Partition[] = ["train", "validation", "test"];

export function AgentRunScreen() {
  const { runId } = useParams({ from: "/rl/$runId" });
  return (
    <Page>
      <AgentRun key={runId} runId={runId} />
    </Page>
  );
}

function AgentRun({ runId }: { runId: string }) {
  const { live, status, detached, result, fromHistory } = useAgentRun(runId);
  const openAgentRun = useSession((state) => state.openAgentRun);
  const data = result.data;

  // O treino aberto passa a ser o último (o ecrã do RL volta a ele).
  useEffect(() => {
    openAgentRun(runId);
  }, [runId, openAgentRun]);

  if (detached && result.error !== null) {
    const missing =
      result.error instanceof ApiRequestError && result.error.code === "run_not_found";
    return (
      <>
        <PageHeader
          title={SCREENS.reinforcement.title}
          description={SCREENS.reinforcement.description}
        />
        {missing ? (
          <NeedsStep
            message="This training is neither in the API's memory nor in the history: maybe it was deleted. The recorded runs are in History."
            to="/history"
            action="Open the history"
          />
        ) : (
          <ErrorNote error={result.error} className="mt-10" />
        )}
      </>
    );
  }
  const inProgress = status === "pending" || status === "running" || status === "paused";

  return (
    <>
      <Link
        to="/rl"
        className="-ml-1 inline-flex items-center gap-1 text-sm text-soft hover:text-strong"
      >
        <ChevronLeft aria-hidden="true" className="size-4" />
        {SCREENS.reinforcement.title}
      </Link>
      <header className="mt-4 flex flex-col gap-4 @3xl/main:flex-row @3xl/main:items-end @3xl/main:justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h1 className="text-title">{data?.agent_display_name ?? "The agent's training"}</h1>
            <WhyButton
              title={SCREENS.reinforcement.title}
              description={SCREENS.reinforcement.description}
              topics={[
                "reinforcement_learning",
                "q_values",
                "temporal_difference",
                "exploration",
                "discount",
                "rl_evaluation",
              ]}
            />
          </div>
        </div>
        {data !== undefined && (
          <div className="flex shrink-0 flex-wrap items-center gap-2">
            <SaveRunButton runId={runId} />
            <Button asChild variant="outline">
              <Link to="/history">See it in History</Link>
            </Button>
            <Button asChild>
              <Link to="/rl">Train another agent</Link>
            </Button>
          </div>
        )}
      </header>
      {inProgress && data === undefined && <Live live={live} runId={runId} />}
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

/** O treino a decorrer. */
function Live({ live, runId }: { live: LiveAgent; runId: string }) {
  const label = useLabel();
  const path = { run_id: runId };
  const pause = useMutation(reinforcementPauseMutation());
  const resume = useMutation(reinforcementResumeMutation());
  const cancel = useMutation(reinforcementCancelMutation());
  const state = live.state;
  const episodes = state?.progress.steps ?? 0;
  const planned = state?.planned_steps ?? null;
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
            Episode {formatCount(episodes)}
            {planned === null ? "" : ` of ${formatCount(planned)}`}
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
              The training stops at the end of the current episode; the agent keeps what it has
              learned and is still evaluated.
            </TooltipContent>
          </Tooltip>
        </div>
      </div>
      {planned !== null && (
        <Progress
          value={Math.min(100, (episodes / planned) * 100)}
          aria-label="The training's progress"
        />
      )}
      <ErrorNote error={pause.error ?? resume.error ?? cancel.error} />
      <div className="grid gap-x-12 gap-y-10 @5xl/main:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
        <div className="min-w-0">
          {live.preview === null || live.layout === null ? (
            <div className="flex h-72 items-center justify-center rounded-xl border border-dashed text-sm text-dim">
              Waiting for the first snapshot of the training.
            </div>
          ) : (
            <GridWorld
              layout={live.layout}
              values={live.preview.values}
              actions={live.preview.actions}
              path={live.preview.path}
              description={`What the agent knew at episode ${String(live.preview.episode)}: the value of each cell, the policy and the path.`}
              caption={`At episode ${formatCount(live.preview.episode)}.`}
            />
          )}
        </div>
        <div className="min-w-0">
          <RewardDots episodes={live.episodes} />
        </div>
      </div>
    </section>
  );
}

/** A recompensa de cada episódio que já correu (ao vivo: só os pontos). */
function RewardDots({ episodes }: { episodes: LiveAgent["episodes"] }) {
  if (episodes.length === 0) {
    return null;
  }
  return (
    <LineChart
      series={[]}
      dots={[
        {
          key: "reward",
          label: "Episode reward",
          color: CATEGORICAL[0],
          radius: 2.5,
          opacity: 0.6,
          points: episodes.map((record) => ({
            x: record.step,
            y: record.metrics.reward ?? 0,
          })),
        },
      ]}
      height={300}
      xLabel="Episode"
      yLabel="Reward"
      formatY={formatStatistic}
      description="The reward of each training episode, so far."
    />
  );
}

/**
 * O treino terminado: as métricas, a recompensa e o que o agente aprendeu. O ambiente é o
 * primeiro número da faixa, não uma linha por baixo do título (ADR 0038).
 */
function Finished({ result, fromHistory }: { result: ReinforcementResult; fromHistory: boolean }) {
  const frames = result.trace?.frames ?? [];
  const episodes = frames.map((frame) => frame.episode);
  const [episode, setEpisode] = useState(episodes.at(-1) ?? result.learned.episode);
  const frame: PolicyFrame =
    frames.length > 1
      ? (frames.find((one) => one.episode === episode) ?? result.learned)
      : result.learned;
  const steps = result.history.at(-1)?.step ?? 0;

  return (
    <>
      <StatStrip
        className="mt-8"
        items={[
          {
            key: "environment",
            label: "Environment",
            value: result.environment_name,
            detail: fromHistory ? "from the history" : undefined,
          },
          { key: "training", label: "Training", value: `${formatCount(steps)} episodes` },
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
      {result.notices.length > 0 && (
        <div className="mt-6">
          <NoticeList notices={result.notices} />
        </div>
      )}

      <Part id="metrics" title="The metrics" topic="rl_evaluation">
        <div>
          <ReportsTable reports={result.reports} main={result.main_partition} />
        </div>
      </Part>

      <Part id="reward" title="The reward over the training">
        <div className="grid gap-x-12 gap-y-10 *:min-w-0 @5xl/main:grid-cols-2">
          <LineChart
            series={[
              {
                key: "mean",
                label: `Mean of the last ${String(result.rewards.window)} episodes`,
                color: CATEGORICAL[0],
                width: 2.5,
                points: result.rewards.episodes.map((x, i) => ({
                  x,
                  y: result.rewards.average[i] ?? null,
                })),
              },
            ]}
            dots={[
              {
                key: "episodes",
                label: "Episode reward",
                color: CATEGORICAL[0],
                radius: 2,
                opacity: 0.35,
                below: true,
                points: result.rewards.episodes.map((x, i) => ({
                  x,
                  y: result.rewards.rewards[i] ?? 0,
                })),
              },
            ]}
            height={300}
            xLabel="Episode"
            yLabel="Reward"
            formatY={formatStatistic}
            formatX={(value) => formatNumber(value, 0)}
            description="The reward of each training episode and the moving average."
            caption={<TopicText topic="reward_curve" />}
          />
          <LineChart
            series={[
              {
                key: "success",
                label: "Reached the goal",
                color: CATEGORICAL[2],
                points: result.rewards.episodes.map((x, i) => ({
                  x,
                  y: result.rewards.success[i] ?? null,
                })),
              },
            ]}
            height={300}
            yDomain={[0, 1]}
            xLabel="Episode"
            yLabel="Reached the goal"
            formatY={(value) => formatPercent(value, 0)}
            formatX={(value) => formatNumber(value, 0)}
            description="The share of the latest episodes in which the agent reached the goal."
            caption={<TopicText topic="success_curve" />}
          />
        </div>
      </Part>

      <Part id="learned" title="What the agent learned">
        <div className="grid gap-x-12 gap-y-10 @5xl/main:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <div className="flex min-w-0 flex-col gap-5">
            {frames.length > 1 ? (
              <OptionSlider
                label="Episode"
                values={episodes}
                value={episode}
                onChange={setEpisode}
                format={(value) => formatCount(value)}
              />
            ) : (
              result.trace === null && (
                <p className="text-sm text-soft">
                  This run opened from the history without the intermediate episodes (only saved
                  runs keep them): you see what the agent knew at the end.
                </p>
              )
            )}
            <GridWorld
              layout={result.layout}
              values={frame.values}
              actions={frame.actions}
              path={frame.path}
              description={`What the agent knew at episode ${String(frame.episode)}: the value of each cell, the policy and the path from the start.`}
              caption={<TopicText topic="policy_map" />}
            />
          </div>
          <QTable result={result} frame={frame} />
        </div>
      </Part>
    </>
  );
}

/** A tabela Q: uma grelha por ação, duas a duas, na mesma escala da grelha dos valores. */
function QTable({ result, frame }: { result: ReinforcementResult; frame: PolicyFrame }) {
  return (
    <section aria-labelledby="tabela-q-titulo" className="flex min-w-0 flex-col gap-4">
      <div className="flex items-center gap-1">
        <h3 id="tabela-q-titulo" className="text-base font-normal tracking-normal text-soft">
          The Q table, action by action
        </h3>
        <Explain topic="q_table" />
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-6">
        {result.layout.action_labels.map((name, action) => (
          <div key={name} className="flex min-w-0 flex-col gap-1.5">
            <p className="text-xs font-medium text-dim capitalize">{name}</p>
            <GridWorld
              layout={result.layout}
              values={frame.q_values.map((q, cell) =>
                frame.actions[cell] === null ? null : (q[action] ?? null),
              )}
              valueLabel={`Q of ${name}`}
              maxCell={36}
              legend={false}
              description={`The Q of action ${name} in each cell, at episode ${String(frame.episode)}.`}
            />
          </div>
        ))}
      </div>
    </section>
  );
}

/** As métricas em cada parte (o treino, a explorar; o teste, com a política aprendida). */
function ReportsTable({
  reports,
  main,
}: {
  reports: Partial<Record<string, EvaluationReport>>;
  main: Partition;
}) {
  const label = useLabel();
  const partitions = PARTITIONS.filter((partition) => reports[partition] !== undefined);
  const mainReport = reports[main];
  if (mainReport === undefined) {
    return null;
  }
  return (
    <div className="relative max-w-2xl overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b">
            <th scope="col" className="px-3 py-2.5 text-left text-xs font-medium text-dim">
              Metric
            </th>
            {partitions.map((partition) => (
              <th
                key={partition}
                scope="col"
                className={
                  partition === main
                    ? "px-3 py-2.5 text-right text-xs font-medium text-soft"
                    : "px-3 py-2.5 text-right text-xs font-medium text-dim"
                }
              >
                {label("Partition", partition)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {mainReport.metrics.map((metric) => (
            <tr key={metric.name} className="border-b border-border/60 last:border-0">
              <th scope="row" className="px-3 py-2.5 text-left font-normal">
                <Tooltip>
                  <TooltipTrigger className="cursor-help text-left underline decoration-input decoration-dotted underline-offset-4">
                    {metric.display_name}
                  </TooltipTrigger>
                  <TooltipContent className="max-w-72">{metric.what}</TooltipContent>
                </Tooltip>
              </th>
              {partitions.map((partition) => {
                const found = reports[partition]?.metrics.find((one) => one.name === metric.name);
                return (
                  <td
                    key={partition}
                    className={
                      partition === main
                        ? "px-3 py-2.5 text-right font-medium text-strong tabular-nums"
                        : "px-3 py-2.5 text-right text-soft tabular-nums"
                    }
                  >
                    {found === undefined ? "—" : formatMetric(found)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface PartProps {
  id: string;
  title: string;
  /** O tema que explica a parte (no "?" ao lado do título). */
  topic?: string;
  children: ReactNode;
}

function Part({ id, title, topic, children }: PartProps) {
  return (
    <section id={id} aria-labelledby={`${id}-titulo`} className="mt-10 scroll-mt-6">
      <div className="mb-4 flex items-center gap-1.5">
        <h2 id={`${id}-titulo`} className="text-xl">
          {title}
        </h2>
        {topic !== undefined && <Explain topic={topic} />}
      </div>
      {children}
    </section>
  );
}
