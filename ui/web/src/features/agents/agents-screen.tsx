/**
 * O ecrã Agentes: o pedido aos agentes (o que prever, a métrica, os agentes chamados e o tamanho
 * da procura) e, ao lado, o plano que o agente de dados já fez para ele, com o fluxograma dos
 * agentes que vão trabalhar. "Pedir aos agentes" começa a procura e abre a página dela.
 */
import { keepPreviousData, useMutation, useQuery } from "@tanstack/react-query";
import { Link, useNavigate } from "@tanstack/react-router";
import { ArrowRight, LoaderCircle } from "lucide-react";
import { useState, type ReactNode } from "react";

import type {
  AutoMlPlan,
  AutoMlRequest,
  DatasetSummary,
  ModelFamily,
  TaskType,
} from "@/api/generated";
import {
  automlOptionsOptions,
  automlStartMutation,
  datasetsSuggestTaskOptions,
} from "@/api/generated/@tanstack/react-query.gen";
import { automlPlan } from "@/api/generated/sdk.gen";
import { useLabel, useVocabulary } from "@/api/vocabulary";
import { ChipGroup } from "@/components/forms/chip-group";
import { schemaDefaults, type Values } from "@/components/forms/schema";
import { SchemaForm } from "@/components/forms/schema-form";
import { Explain } from "@/components/explain/explain";
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
import { useDebounced } from "@/hooks/use-debounced";
import { isValidSeed, MAX_SEED } from "@/lib/seed";
import { useSession } from "@/state/session";

import { AgentDetail } from "./agent-detail";
import { Flowchart, type FlowNode } from "./flowchart";
import { PlanView } from "./plan-view";

export function AgentsScreen() {
  const searchId = useSession((state) => state.searchId);
  return (
    <Page>
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
          "winners_curse",
        ]}
      >
        {searchId !== null && (
          <Button asChild variant="outline">
            <Link to="/agentes/$searchId" params={{ searchId }}>
              Ver a última procura
            </Link>
          </Button>
        )}
      </PageHeader>
      <RequireDataset>{(dataset) => <Setup key={dataset.id} dataset={dataset} />}</RequireDataset>
    </Page>
  );
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
  const [families, setFamilies] = useState<ModelFamily[] | null>(null);
  const [settings, setSettings] = useState<Values | null>(null);
  const [seed, setSeed] = useState(42);
  const [own, setOwn] = useState(false);

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
  const options = useQuery({
    ...automlOptionsOptions({ query: { task_type: task ?? "binary_classification" } }),
    enabled: task !== null,
    staleTime: "static",
  });
  const ownAllowed = preparation?.datasetId === dataset.id && preparation.config.target === target;
  const validSeed = isValidSeed(seed);

  const request: AutoMlRequest | null =
    task === null || options.data === undefined || !validSeed
      ? null
      : {
          dataset: dataset.source,
          target,
          task_type: task,
          objective,
          families: families ?? options.data.agents.map((agent) => agent.family),
          preparation: own && ownAllowed ? preparation.config : null,
          settings: settings ?? options.data.settings,
          seed,
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
      void navigate({ to: "/agentes/$searchId", params: { searchId: job.run_id } });
    },
  });

  const unsupported =
    suggestion.data !== undefined && (suggestion.data.task_type === null || tasks.length === 0);

  const preview = plan.data !== undefined && !unsupported ? plan.data : null;

  return (
    // O fluxograma em cima, a toda a largura; por baixo, o pedido e o plano em duas colunas iguais.
    <div className="mt-8 flex flex-col gap-12">
      {preview !== null && <WhoWorks plan={preview} />}
      <div className="grid gap-x-12 gap-y-10 @5xl/main:grid-cols-2">
        <div className="flex flex-col gap-6 self-start">
          <Field
            label="O que queres prever (a coluna-alvo)"
            htmlFor="agentes-alvo"
            help={suggestion.data?.reason}
          >
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
                  "Os agentes não trabalham com este alvo: experimenta outra coluna, ou o modo manual."}
              </p>
              <Button asChild variant="outline">
                <Link to="/preparar">Ir para o modo manual</Link>
              </Button>
            </div>
          ) : (
            task !== null && (
              <>
                {tasks.length > 1 && (
                  <Field label="Tarefa" htmlFor="agentes-tarefa">
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
                      label="A métrica a otimizar"
                      htmlFor="agentes-metrica"
                      help="A sugerida é a do agente de dados (a razão está nas decisões dele)."
                    >
                      <Select
                        value={objective ?? plan.data?.objective.name ?? ""}
                        onValueChange={setObjective}
                      >
                        <SelectTrigger id="agentes-metrica" className="w-full">
                          <SelectValue placeholder="A escolher" />
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
                    <Field label="Os agentes chamados (um por família de modelos)">
                      <ChipGroup
                        label="Os agentes chamados"
                        options={options.data.agents.map((agent) => ({
                          value: agent.family,
                          label: label("ModelFamily", agent.family),
                        }))}
                        value={families ?? options.data.agents.map((agent) => agent.family)}
                        onChange={(chosen) => {
                          setFamilies(chosen as ModelFamily[]);
                        }}
                      />
                    </Field>
                    <section className="flex flex-col gap-3">
                      <h2 className="text-sm font-normal text-soft">O tamanho da procura</h2>
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
                    <Field
                      label="Seed"
                      htmlFor="agentes-seed"
                      help="A mesma seed dá a mesma divisão, os mesmos folds e os mesmos sorteios da procura."
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
                            Usar a minha preparação
                          </Label>
                          <Explain title="Usar a minha preparação">
                            O agente de dados adota a tua limpeza, divisão e transformações.
                          </Explain>
                        </div>
                        <Switch id="agentes-preparacao" checked={own} onCheckedChange={setOwn} />
                      </div>
                    )}
                    <div className="flex flex-col items-start gap-3">
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
                        {start.isPending && (
                          <LoaderCircle aria-hidden="true" className="animate-spin" />
                        )}
                        Pedir aos agentes
                        {!start.isPending && <ArrowRight aria-hidden="true" />}
                      </Button>
                      <ErrorNote error={start.error ?? plan.error ?? options.error} />
                    </div>
                  </>
                )}
              </>
            )
          )}
        </div>
        <div className="min-w-0">
          {preview !== null && (
            <Panel title="O plano do agente de dados" topic="data_agent" as="h2">
              <PlanView plan={preview} />
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}

/** O fluxograma dos agentes que vão trabalhar (ainda parados); cada um abre o seu painel. */
function WhoWorks({ plan }: { plan: AutoMlPlan }) {
  const [selected, setSelected] = useState<FlowNode | null>(null);
  return (
    <>
      <Panel title="Quem vai trabalhar" as="h2">
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
          />
        </div>
        <p className="mono-label">Clica num agente para veres os modelos que vai experimentar.</p>
      </Panel>
      <AgentDetail
        node={selected}
        onClose={() => {
          setSelected(null);
        }}
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
