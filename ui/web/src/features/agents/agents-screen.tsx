/**
 * O ecrã Agentes: o pedido aos agentes (o que prever, a métrica, os agentes chamados e o tamanho
 * da procura) e, ao lado, o plano que o agente de dados já fez para ele, com o fluxograma dos
 * agentes que vão trabalhar. "Pedir aos agentes" começa a procura e abre a página dela.
 */
import { keepPreviousData, useMutation, useQuery } from "@tanstack/react-query";
import { Link, useNavigate } from "@tanstack/react-router";
import { LoaderCircle } from "lucide-react";
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
import { ErrorNote, Section } from "@/components/shell/blocks";
import { Page, PageHeader } from "@/components/shell/page";
import { SCREENS } from "@/components/shell/screens";
import { WhyButton } from "@/components/shell/why-button";
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

import { Flowchart } from "./flowchart";
import { PlanView } from "./plan-view";

export function AgentsScreen() {
  const searchId = useSession((state) => state.searchId);
  return (
    <Page>
      <PageHeader title={SCREENS.agents.title} description={SCREENS.agents.description}>
        <WhyButton
          topics={[
            "automl",
            "automatic_or_manual",
            "data_agent",
            "model_selection",
            "screening_and_tuning",
            "random_search",
            "winners_curse",
          ]}
        />
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

  return (
    <div className="mt-10 grid gap-14 @6xl/main:grid-cols-[minmax(0,24rem)_minmax(0,1fr)]">
      <div className="flex flex-col gap-6">
        <Field label="O que queres prever (a coluna-alvo)" htmlFor="agentes-alvo">
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
        {suggestion.data !== undefined && (
          <p className="font-serif text-reading">{suggestion.data.reason}</p>
        )}
        {unsupported ? (
          <div className="flex flex-col items-start gap-3 border-t pt-5">
            <p className="font-serif text-reading text-muted-foreground">
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
                  <section className="flex flex-col gap-3 border-t pt-5">
                    <h2 className="text-lg">O tamanho da procura</h2>
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
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <Label htmlFor="agentes-preparacao" className="text-sm font-medium">
                          Usar a minha preparação
                        </Label>
                        <p className="mt-1 font-serif text-sm text-muted-foreground">
                          O agente de dados adota a tua limpeza, divisão e transformações.
                        </p>
                      </div>
                      <Switch id="agentes-preparacao" checked={own} onCheckedChange={setOwn} />
                    </div>
                  )}
                  <div className="flex flex-col items-start gap-3 border-t pt-5">
                    <Button
                      size="lg"
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
        {plan.data !== undefined && !unsupported && <PlanPreview plan={plan.data} />}
      </div>
    </div>
  );
}

/** O plano do pedido, com o fluxograma dos agentes que vão trabalhar (ainda parados). */
function PlanPreview({ plan }: { plan: AutoMlPlan }) {
  return (
    <div className="flex flex-col gap-10">
      <Section title="Quem vai trabalhar">
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
        />
      </Section>
      <Section title="O plano">
        <PlanView plan={plan} />
      </Section>
    </div>
  );
}

interface FieldProps {
  label: string;
  htmlFor?: string;
  help?: string;
  children: ReactNode;
}

function Field({ label, htmlFor, help, children }: FieldProps) {
  return (
    <div className="flex flex-col gap-2">
      {htmlFor === undefined ? (
        <span className="text-sm font-medium">{label}</span>
      ) : (
        <Label htmlFor={htmlFor} className="text-sm font-medium">
          {label}
        </Label>
      )}
      {children}
      {help !== undefined && (
        <p className="font-serif text-sm leading-relaxed text-muted-foreground">{help}</p>
      )}
    </div>
  );
}
