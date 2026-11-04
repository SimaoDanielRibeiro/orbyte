/**
 * O ecrã Agentes: o pedido aos agentes, arrumado em três partes (ADR 0049):
 *
 * 1. **o pedido**, à esquerda: o que prever, a métrica, o tamanho da procura e as opções;
 * 2. **os modelos**, à direita: todos os que se podem medir, por família, e cada um pode sair
 *    da procura (`model-picker.tsx`);
 * 3. por baixo, a toda a largura, **quem vai trabalhar** (o fluxograma dos agentes, que segue
 *    a escolha) e **o plano** que o agente de dados já fez para o pedido.
 *
 * "Ask the agents" fica no cabeçalho, à direita, como o passo seguinte de todos os ecrãs
 * (ADR 0044): começa a procura e abre a página dela.
 */
import { keepPreviousData, useMutation, useQuery } from "@tanstack/react-query";
import { Link, useNavigate } from "@tanstack/react-router";
import { LoaderCircle, Sparkles } from "lucide-react";
import { useState, type ReactNode } from "react";

import type { AutoMlPlan, AutoMlRequest, DatasetSummary, TaskType } from "@/api/generated";
import {
  automlOptionsOptions,
  automlStartMutation,
  datasetsSuggestTaskOptions,
  llmStatusOptions,
} from "@/api/generated/@tanstack/react-query.gen";
import { automlPlan } from "@/api/generated/sdk.gen";
import { useLabel, useVocabulary } from "@/api/vocabulary";
import { schemaDefaults, type Values } from "@/components/forms/schema";
import { SchemaForm } from "@/components/forms/schema-form";
import { Explain } from "@/components/explain/explain";
import { NoticeList } from "@/components/notices/notice-list";
import { ErrorNote, Panel } from "@/components/shell/blocks";
import { Page, PageHeader } from "@/components/shell/page";
import { SCREENS } from "@/components/shell/screens";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { RequireDataset } from "@/features/data/require-dataset";
import { useOpenDataset } from "@/features/data/use-open-dataset";
import { useDebounced } from "@/hooks/use-debounced";
import { isValidSeed, MAX_SEED } from "@/lib/seed";
import { useSession } from "@/state/session";

import { AI_AGENT_COLOR } from "./agent-colors";
import { AgentDetail } from "./agent-detail";
import { Flowchart, type FlowNode } from "./flowchart";
import { ModelPicker } from "./model-picker";
import { PlanView } from "./plan-view";

/**
 * Só se vê sem uma procura aberta: com uma, a rota passa logo para ela (até "Nova procura").
 */
export function AgentsScreen() {
  const { datasetId, query, lost } = useOpenDataset();
  // Com o dataset aberto, é o pedido que desenha o cabeçalho (com o botão de começar).
  const open = datasetId !== null && !lost && query.data !== undefined;
  return (
    <Page>
      {!open && <AgentsHeader />}
      <RequireDataset>{(dataset) => <Setup key={dataset.id} dataset={dataset} />}</RequireDataset>
    </Page>
  );
}

function AgentsHeader({ children }: { children?: ReactNode }) {
  return (
    <PageHeader
      title={SCREENS.agents.title}
      description={SCREENS.agents.description}
      topics={[
        "automl",
        "automatic_or_manual",
        "data_agent",
        "model_selection",
        "screening_and_tuning",
        "random_search",
        "ai_agent",
        "winners_curse",
      ]}
    >
      {children}
    </PageHeader>
  );
}

/** Os modelos escolhidos numa tarefa (mudar de tarefa volta a escolher todos). */
interface Picked {
  task: TaskType;
  models: string[];
}

function Setup({ dataset }: { dataset: DatasetSummary }) {
  const label = useLabel();
  const vocabulary = useVocabulary();
  const navigate = useNavigate();
  const preparation = useSession((state) => state.preparation);
  const openSearch = useSession((state) => state.openSearch);
  const columns = dataset.columns.map((column) => column.name);
  const [target, setTarget] = useState(
    (preparation?.datasetId === dataset.id ? preparation.config.target : null) ??
      dataset.suggested_target ??
      columns.at(-1) ??
      "",
  );
  const [chosenTask, setChosenTask] = useState<TaskType | null>(null);
  const [objective, setObjective] = useState<string | null>(null);
  const [picked, setPicked] = useState<Picked | null>(null);
  const [settings, setSettings] = useState<Values | null>(null);
  const [seed, setSeed] = useState(42);
  const [own, setOwn] = useState(false);
  const [ai, setAi] = useState(false);
  // Se o Ollama está pronto: pergunta-se outra vez de vez em quando, para o interruptor ligar
  // sozinho quando se abre o Ollama com o ecrã aberto.
  const llm = useQuery({ ...llmStatusOptions(), refetchInterval: 10_000 });
  const aiReady = llm.data?.ready === true;

  const suggestion = useQuery({
    ...datasetsSuggestTaskOptions({ path: { dataset_id: dataset.id }, query: { target } }),
    placeholderData: keepPreviousData,
  });
  // As tarefas em que os agentes trabalham (uma regra do backend, no vocabulário).
  const tasks = (suggestion.data?.allowed ?? []).filter(
    (task) => vocabulary.tasks.find((info) => info.task_type === task)?.agents === true,
  );
  const task =
    chosenTask !== null && tasks.includes(chosenTask)
      ? chosenTask
      : suggestion.data?.task_type !== undefined &&
          suggestion.data.task_type !== null &&
          tasks.includes(suggestion.data.task_type)
        ? suggestion.data.task_type
        : (tasks[0] ?? null);
  // A métrica escolhida muda os modelos que podem entrar (sem probabilidades, não há ROC AUC).
  const options = useQuery({
    ...automlOptionsOptions({
      query: { task_type: task ?? "binary_classification", objective: objective ?? undefined },
    }),
    enabled: task !== null,
    staleTime: "static",
    // Ao mudar de métrica, a lista fica à vista até chegar a nova (não pisca).
    placeholderData: keepPreviousData,
  });
  const available = (options.data?.agents ?? []).flatMap((agent) =>
    agent.models.map((model) => model.name),
  );
  const models = picked !== null && picked.task === task ? picked.models : available;
  const ownAllowed = preparation?.datasetId === dataset.id && preparation.config.target === target;
  const validSeed = isValidSeed(seed);

  // As opções de outra tarefa (ainda à vista enquanto chegam as novas) não fazem um pedido: os
  // modelos delas não são desta tarefa.
  const request: AutoMlRequest | null =
    task === null || options.data?.task_type !== task || !validSeed || models.length === 0
      ? null
      : {
          dataset: dataset.source,
          target,
          task_type: task,
          objective,
          models,
          preparation: own && ownAllowed ? preparation.config : null,
          settings: settings ?? options.data.settings,
          seed,
          ai_agent: ai && aiReady,
        };
  const settled = useDebounced(request === null ? null : JSON.stringify(request), 300);
  const plan = useQuery({
    queryKey: ["automl-plan", settled],
    queryFn: async ({ signal }) => {
      const { data } = await automlPlan({
        body: JSON.parse(settled ?? "{}") as AutoMlRequest,
        signal,
        throwOnError: true,
      });
      return data;
    },
    enabled: settled !== null,
    placeholderData: keepPreviousData,
    staleTime: "static",
  });
  const start = useMutation({
    ...automlStartMutation(),
    onSuccess: (job) => {
      openSearch(job.run_id);
      void navigate({ to: "/agents/$searchId", params: { searchId: job.run_id } });
    },
  });

  const unsupported =
    suggestion.data !== undefined && (suggestion.data.task_type === null || tasks.length === 0);

  const preview = plan.data !== undefined && !unsupported ? plan.data : null;

  return (
    <>
      <AgentsHeader>
        {!unsupported && (
          <Button
            variant="proceed"
            size="proceed"
            disabled={request === null || plan.data === undefined || start.isPending}
            onClick={() => {
              if (request !== null) {
                start.mutate({ body: request });
              }
            }}
          >
            {start.isPending && <LoaderCircle aria-hidden="true" className="animate-spin" />}
            Ask the agents
          </Button>
        )}
      </AgentsHeader>
      <ErrorNote error={start.error ?? plan.error ?? options.error} className="mt-6" />

      <div className="mt-12 grid gap-x-14 gap-y-14 @5xl/main:grid-cols-[minmax(0,25rem)_minmax(0,1fr)]">
        <section aria-labelledby="agentes-pedido" className="flex flex-col gap-6 self-start">
          <h2 id="agentes-pedido" className="text-xl">
            The request
          </h2>
          <Field label="Target column" htmlFor="agentes-alvo" help={suggestion.data?.reason}>
            <Select
              value={target}
              onValueChange={(value) => {
                setTarget(value);
                setChosenTask(null);
                setObjective(null);
                setOwn(false);
              }}
            >
              <SelectTrigger id="agentes-alvo" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {columns.map((column) => (
                  <SelectItem key={column} value={column}>
                    {column}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
          {unsupported ? (
            <div className="flex flex-col items-start gap-3">
              <p className="text-sm text-soft">{suggestion.data.reason}</p>
              <p className="text-sm text-soft">
                {options.data?.reason ??
                  "The agents don't work with this target: try another column, or the manual mode."}
              </p>
              <Button asChild variant="outline">
                <Link to="/prepare">Go to the manual mode</Link>
              </Button>
            </div>
          ) : (
            task !== null && (
              <>
                {tasks.length > 1 && (
                  <Field label="Task" htmlFor="agentes-tarefa">
                    <Select
                      value={task}
                      onValueChange={(value) => {
                        setChosenTask(value as TaskType);
                        setObjective(null);
                      }}
                    >
                      <SelectTrigger id="agentes-tarefa" className="w-full">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {tasks.map((one) => (
                          <SelectItem key={one} value={one}>
                            {label("TaskType", one)}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </Field>
                )}
                {options.data === undefined ? (
                  <Skeleton className="h-40 w-full" />
                ) : (
                  <>
                    <Field
                      label="The metric to optimize"
                      htmlFor="agentes-metrica"
                      help="The suggested one comes from the data agent (the reason is in its decisions)."
                    >
                      <Select
                        value={objective ?? plan.data?.objective.name ?? ""}
                        onValueChange={setObjective}
                      >
                        <SelectTrigger id="agentes-metrica" className="w-full">
                          <SelectValue placeholder="Choosing" />
                        </SelectTrigger>
                        <SelectContent>
                          {options.data.objectives.map((one) => (
                            <SelectItem key={one.name} value={one.name}>
                              {one.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </Field>
                    <section className="flex flex-col gap-3 border-t border-[#202020] pt-6">
                      <h3 className="text-base font-medium text-strong">The size of the search</h3>
                      <SchemaForm
                        schema={options.data.settings_schema}
                        values={
                          settings ?? {
                            ...schemaDefaults(options.data.settings_schema),
                            ...options.data.settings,
                          }
                        }
                        onChange={setSettings}
                        idPrefix="agentes-procura"
                      />
                    </section>
                    <section className="flex flex-col gap-5 border-t border-[#202020] pt-6">
                      <Field
                        label="Seed"
                        htmlFor="agentes-seed"
                        help="The same seed gives the same split, the same folds and the same draws in the search."
                      >
                        <Input
                          id="agentes-seed"
                          name="seed"
                          autoComplete="off"
                          type="number"
                          inputMode="numeric"
                          min={0}
                          max={MAX_SEED}
                          step={1}
                          value={Number.isNaN(seed) ? "" : seed}
                          onChange={(event) => {
                            setSeed(event.target.valueAsNumber);
                          }}
                          aria-invalid={!validSeed}
                          className="w-40 tabular-nums"
                        />
                      </Field>
                      {ownAllowed && (
                        <div className="flex items-center justify-between gap-4">
                          <div className="flex items-center gap-1">
                            <Label htmlFor="agentes-preparacao" className="text-sm text-soft">
                              Use my preparation
                            </Label>
                            <Explain title="Use my preparation">
                              The data agent adopts your cleaning, split and transformations.
                            </Explain>
                          </div>
                          <Switch id="agentes-preparacao" checked={own} onCheckedChange={setOwn} />
                        </div>
                      )}
                      <div className="flex flex-col gap-2">
                        <div className="flex items-center justify-between gap-4">
                          <div className="flex items-center gap-1">
                            <Sparkles
                              aria-hidden="true"
                              className="mr-1 size-4"
                              style={{ color: AI_AGENT_COLOR }}
                            />
                            <Label htmlFor="agentes-ia" className="text-sm text-soft">
                              Call the AI agent
                              {llm.data !== undefined && (
                                <span className="ml-1.5 font-mono text-xs text-dim">
                                  {llm.data.model}
                                </span>
                              )}
                            </Label>
                            <Explain topic="ai_agent" />
                          </div>
                          <Switch
                            id="agentes-ia"
                            checked={ai && aiReady}
                            disabled={!aiReady}
                            onCheckedChange={setAi}
                          />
                        </div>
                        {llm.data !== undefined && !aiReady && (
                          <NoticeList notices={llm.data.notices} />
                        )}
                      </div>
                    </section>
                  </>
                )}
              </>
            )
          )}
        </section>

        {!unsupported && task !== null && (
          <section aria-labelledby="agentes-modelos" className="min-w-0">
            {options.data === undefined ? (
              <Skeleton className="h-96 w-full rounded-2xl" />
            ) : (
              <ModelPicker
                titleId="agentes-modelos"
                agents={options.data.agents}
                unavailable={options.data.skipped}
                chosen={models}
                onChange={(chosen) => {
                  setPicked({ task, models: chosen });
                }}
              />
            )}
          </section>
        )}
      </div>

      {preview !== null && (
        <div className="mt-20 flex flex-col gap-20">
          <WhoWorks plan={preview} />
          <Panel title="The data agent's plan" topic="data_agent" as="h2">
            <PlanView plan={preview} models={false} />
          </Panel>
        </div>
      )}
    </>
  );
}

/** O fluxograma dos agentes que vão trabalhar (ainda parados); cada um abre o seu painel. */
function WhoWorks({ plan }: { plan: AutoMlPlan }) {
  const [selected, setSelected] = useState<FlowNode | null>(null);
  return (
    <>
      <Panel title="Who will work" as="h2" explain="Click an agent to see the models it will try.">
        <div className="rounded-2xl bg-[radial-gradient(circle,#262626_1px,transparent_1.2px)] bg-size-[22px_22px] p-4">
          <Flowchart
            agents={plan.agents}
            phase={null}
            working={null}
            candidates={[]}
            objective={plan.objective}
            leader={null}
            budget={plan.settings.budget ?? null}
            trainRows={plan.train_rows}
            test={null}
            selected={selected}
            onSelect={setSelected}
            ai={plan.ai_agent ? { model: plan.ai_model } : null}
          />
        </div>
      </Panel>
      <AgentDetail
        node={selected}
        onClose={() => {
          setSelected(null);
        }}
        onSelect={setSelected}
        agents={plan.agents}
        candidates={[]}
        messages={[]}
        objective={plan.objective}
        phase={null}
        working={null}
        leader={null}
        budget={plan.settings.budget ?? null}
        plan={plan}
      />
    </>
  );
}

interface FieldProps {
  label: string;
  htmlFor?: string;
  /** A explicação do controlo: fica no "?" ao lado do nome. */
  help?: string;
  children: ReactNode;
}

function Field({ label, htmlFor, help, children }: FieldProps) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex min-h-6 items-center gap-1">
        {htmlFor === undefined ? (
          <span className="text-sm text-soft">{label}</span>
        ) : (
          <Label htmlFor={htmlFor} className="text-sm text-soft">
            {label}
          </Label>
        )}
        {help !== undefined && help !== "" && <Explain title={label}>{help}</Explain>}
      </div>
      {children}
    </div>
  );
}
