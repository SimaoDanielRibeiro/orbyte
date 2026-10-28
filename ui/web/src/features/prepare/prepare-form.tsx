/**
 * Os controlos do ecrã Preparar, pela ordem em que a preparação corre: o alvo e a tarefa, a
 * limpeza, a divisão e as transformações.
 *
 * Cada controlo muda as escolhas (`PrepareChoices`); o que não se toca segue as sugestões do
 * backend. Os nomes e as explicações das operações de limpeza vêm do backend, e as explicações
 * ficam no "?" de cada controlo.
 */
import { useQuery } from "@tanstack/react-query";
import { cn } from "cn";
import { ChevronDown } from "lucide-react";
import type { ReactNode } from "react";

import type {
  CleaningSuggestion,
  DatasetSummary,
  ImputationStrategy,
  OperationSummary,
  ScalingMethod,
  SplitStrategy,
  TaskOptions,
  TaskSuggestion,
  TaskType,
} from "@/api/generated";
import { datasetsCleaningOperationsOptions } from "@/api/generated/@tanstack/react-query.gen";
import { useLabel } from "@/api/vocabulary";
import { Explain } from "@/components/explain/explain";
import { ChipGroup } from "@/components/forms/chip-group";
import { NoticeList } from "@/components/notices/notice-list";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
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
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { formatPercent } from "@/lib/format";

import {
  effectiveCleaning,
  effectiveStrategy,
  droppedSuggestions,
  suggestionKey,
  toggleSuggestions,
  withTarget,
  type CleaningChoices,
  type OutlierMethod,
  type PrepareChoices,
  type TransformChoices,
} from "./choices";

/** O valor de "nenhum" nas listas (o Radix não aceita um valor vazio). */
const NONE = "__nenhum__";
const MAX_SEED = 2 ** 32 - 1;

interface PrepareFormProps {
  dataset: DatasetSummary;
  choices: PrepareChoices;
  onChange: (choices: PrepareChoices) => void;
  suggestion: TaskSuggestion | undefined;
  cleaning: readonly CleaningSuggestion[] | undefined;
  options: TaskOptions | undefined;
  taskType: TaskType | null;
}

export function PrepareForm({
  dataset,
  choices,
  onChange,
  suggestion,
  cleaning,
  options,
  taskType,
}: PrepareFormProps) {
  return (
    <div className="@container/form flex flex-col gap-4">
      <Step number={1} title="Alvo e tarefa">
        <TaskControls
          dataset={dataset}
          choices={choices}
          onChange={onChange}
          suggestion={suggestion}
          options={options}
          taskType={taskType}
        />
      </Step>
      <Step number={2} title="Limpeza">
        {cleaning === undefined ? (
          <ControlsSkeleton />
        ) : (
          <CleaningControls
            dataset={dataset}
            choices={choices}
            onChange={onChange}
            suggestions={cleaning}
          />
        )}
      </Step>
      <Step number={3} title="Divisão">
        {options === undefined ? (
          <ControlsSkeleton />
        ) : (
          <SplitControls
            dataset={dataset}
            choices={choices}
            onChange={onChange}
            options={options}
          />
        )}
      </Step>
      <Step number={4} title="Transformações">
        <TransformControls
          dataset={dataset}
          choices={choices}
          onChange={onChange}
          dropped={cleaning === undefined ? [] : effectiveCleaning(choices, cleaning).dropped}
        />
      </Step>
    </div>
  );
}

/** Um passo da preparação, num cartão: são uma sequência, por isso vão numerados. */
function Step({ number, title, children }: { number: number; title: string; children: ReactNode }) {
  return (
    <section
      aria-labelledby={`passo-${String(number)}`}
      className="flex flex-col gap-5 border-t pt-5"
    >
      <h2 id={`passo-${String(number)}`} className="flex items-center gap-3 text-base">
        <span
          aria-hidden="true"
          className="flex size-6 items-center justify-center rounded-full border border-[#333] bg-surface-high font-mono text-xs text-soft tabular-nums"
        >
          {number}
        </span>
        {title}
      </h2>
      {children}
    </section>
  );
}

interface FieldProps {
  label: string;
  htmlFor?: string;
  /** A explicação do controlo: fica no "?" ao lado do nome, não à vista. */
  help?: ReactNode;
  /** O valor escrito à direita do nome (o de um controlo deslizante). */
  readout?: string;
  children: ReactNode;
  className?: string;
}

function Field({ label, htmlFor, help, readout, children, className }: FieldProps) {
  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <div className="flex min-h-6 items-center justify-between gap-3">
        <div className="flex items-center gap-1">
          {htmlFor === undefined ? (
            <span className="text-sm text-soft">{label}</span>
          ) : (
            <Label id={`${htmlFor}-rotulo`} htmlFor={htmlFor} className="text-sm text-soft">
              {label}
            </Label>
          )}
          {help !== undefined && help !== null && help !== "" && (
            <Explain title={label}>{help}</Explain>
          )}
        </div>
        {readout !== undefined && (
          <span className="font-mono text-sm text-strong tabular-nums">{readout}</span>
        )}
      </div>
      {children}
    </div>
  );
}

function ControlsSkeleton() {
  return (
    <div className="flex flex-col gap-3" role="status" aria-busy="true" aria-label="A carregar">
      <Skeleton className="h-4 w-40" />
      <Skeleton className="h-9 w-full" />
    </div>
  );
}

interface ControlsProps {
  dataset: DatasetSummary;
  choices: PrepareChoices;
  onChange: (choices: PrepareChoices) => void;
}

function TaskControls({
  dataset,
  choices,
  onChange,
  suggestion,
  options,
  taskType,
}: ControlsProps & {
  suggestion: TaskSuggestion | undefined;
  options: TaskOptions | undefined;
  taskType: TaskType | null;
}) {
  const label = useLabel();
  return (
    <>
      <Field label="Coluna-alvo: o que queres prever?" htmlFor="preparar-alvo">
        <Select
          value={choices.target ?? NONE}
          onValueChange={(value) => {
            onChange(withTarget(choices, value === NONE ? null : value));
          }}
        >
          <SelectTrigger id="preparar-alvo" className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={NONE}>Sem alvo (aprendizagem não supervisionada)</SelectItem>
            {dataset.columns.map((column) => (
              <SelectItem key={column.name} value={column.name}>
                {column.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Field>
      {suggestion === undefined ? (
        <ControlsSkeleton />
      ) : (
        <>
          {(taskType === null || suggestion.allowed.length === 0) && (
            <p className="text-sm text-soft">{suggestion.reason}</p>
          )}
          <NoticeList notices={suggestion.notices} />
          {taskType !== null && suggestion.allowed.length > 0 && (
            <Field
              label="Tipo de tarefa"
              htmlFor="preparar-tarefa"
              help={
                <>
                  <p>{suggestion.reason}</p>
                  <p className="mt-2">
                    A sugerida vem primeiro; as outras também fazem sentido para este alvo.
                  </p>
                  {options?.reference_note !== undefined && options.reference_note !== null && (
                    <p className="mt-2">{options.reference_note}</p>
                  )}
                </>
              }
            >
              <Select
                value={taskType}
                onValueChange={(value) => {
                  onChange({ ...choices, taskType: value as TaskType });
                }}
              >
                <SelectTrigger id="preparar-tarefa" className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {suggestion.allowed.map((task) => (
                    <SelectItem key={task} value={task}>
                      {label("TaskType", task)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
          )}
        </>
      )}
    </>
  );
}

function CleaningControls({
  dataset,
  choices,
  onChange,
  suggestions,
}: ControlsProps & { suggestions: readonly CleaningSuggestion[] }) {
  const operations = useQuery({ ...datasetsCleaningOperationsOptions(), staleTime: "static" });
  const byName = new Map<string, OperationSummary>(
    (operations.data ?? []).map((operation) => [operation.name, operation]),
  );
  const cleaning = effectiveCleaning(choices, suggestions);
  const set = (next: Partial<CleaningChoices>) => {
    onChange({ ...choices, cleaning: { ...cleaning, ...next } });
  };
  const toggles = toggleSuggestions(suggestions);
  const reasons = droppedSuggestions(suggestions);
  const candidates = dataset.columns.filter((column) => column.name !== choices.target);
  const remaining = dataset.columns.filter((column) => !cleaning.dropped.includes(column.name));
  const numeric = remaining.filter(
    (column) => column.kind === "numeric" && column.name !== choices.target,
  );
  const dropColumns = byName.get("drop_columns");
  const dropMissing = byName.get("drop_missing_rows");
  const outliers = byName.get("treat_outliers");

  return (
    <>
      {toggles.map((suggestion) => {
        const key = suggestionKey(suggestion);
        const id = `limpeza-${key.replace(/[^a-z0-9]+/gi, "-")}`;
        const operation = byName.get(suggestion.step.operation);
        return (
          <div key={key} className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-1">
              <Label htmlFor={id} className="text-sm">
                {operation?.title ?? suggestion.step.operation}
              </Label>
              <Explain title={operation?.title ?? suggestion.step.operation}>
                {suggestion.reason}
              </Explain>
            </div>
            <Switch
              id={id}
              checked={cleaning.enabled.includes(key)}
              onCheckedChange={(on) => {
                set({
                  enabled: on
                    ? [...cleaning.enabled, key]
                    : cleaning.enabled.filter((one) => one !== key),
                });
              }}
            />
          </div>
        );
      })}
      {!suggestions.some((suggestion) => suggestion.step.operation === "drop_duplicates") && (
        <p className="mono-label">Não há linhas duplicadas.</p>
      )}
      <Field
        label={dropColumns?.title ?? "Remover colunas"}
        help={
          cleaning.dropped.some((column) => reasons.has(column)) ? (
            <ul className="flex flex-col gap-1">
              {cleaning.dropped
                .filter((column) => reasons.has(column))
                .map((column) => (
                  <li key={column}>{reasons.get(column)}</li>
                ))}
            </ul>
          ) : (
            dropColumns?.description
          )
        }
      >
        <ChipGroup
          label={dropColumns?.title ?? "Remover colunas"}
          options={candidates.map((column) => ({ value: column.name, label: column.name }))}
          value={cleaning.dropped}
          onChange={(dropped) => {
            set({
              dropped,
              dropMissing: cleaning.dropMissing.filter((column) => !dropped.includes(column)),
              outliers: cleaning.outliers.filter((column) => !dropped.includes(column)),
            });
          }}
        />
      </Field>
      <Collapsible
        defaultOpen={cleaning.dropMissing.length > 0 || cleaning.outliers.length > 0}
        className="flex flex-col gap-5"
      >
        <CollapsibleTrigger className="group flex items-center gap-2 self-start rounded-md text-sm text-soft outline-none hover:text-strong focus-visible:ring-2 focus-visible:ring-ring">
          <ChevronDown
            aria-hidden="true"
            className="size-4 transition-transform group-data-[state=open]:rotate-180"
          />
          Mais operações de limpeza
        </CollapsibleTrigger>
        <CollapsibleContent className="flex flex-col gap-6">
          <Field
            label={dropMissing?.title ?? "Remover linhas com valores em falta"}
            help={dropMissing?.description}
          >
            <ChipGroup
              label={dropMissing?.title ?? "Remover linhas com valores em falta"}
              options={remaining.map((column) => ({ value: column.name, label: column.name }))}
              value={cleaning.dropMissing}
              onChange={(dropMissing) => {
                set({ dropMissing });
              }}
            />
          </Field>
          <Field label={outliers?.title ?? "Tratar valores extremos"} help={outliers?.description}>
            <ChipGroup
              label={outliers?.title ?? "Tratar valores extremos"}
              options={numeric.map((column) => ({ value: column.name, label: column.name }))}
              value={cleaning.outliers}
              onChange={(chosen) => {
                set({ outliers: chosen });
              }}
            />
          </Field>
          {cleaning.outliers.length > 0 && (
            <Field label="Como tratar os extremos">
              <ToggleGroup
                type="single"
                variant="outline"
                size="sm"
                value={cleaning.outlierMethod}
                onValueChange={(method) => {
                  if (method !== "") {
                    set({ outlierMethod: method as OutlierMethod });
                  }
                }}
                aria-label="Como tratar os extremos"
                className="flex-wrap justify-start"
              >
                <ToggleGroupItem value="clip" className="px-3">
                  Limitar aos limites
                </ToggleGroupItem>
                <ToggleGroupItem value="remove" className="px-3">
                  Remover as linhas
                </ToggleGroupItem>
              </ToggleGroup>
            </Field>
          )}
        </CollapsibleContent>
      </Collapsible>
    </>
  );
}

function SplitControls({
  dataset,
  choices,
  onChange,
  options,
}: ControlsProps & { options: TaskOptions }) {
  const label = useLabel();
  const strategy = effectiveStrategy(choices, options);
  const times = dataset.columns.filter(
    (column) => column.kind === "datetime" || column.kind === "numeric",
  );
  const validSeed = Number.isInteger(choices.seed) && choices.seed >= 0 && choices.seed <= MAX_SEED;
  return (
    <>
      <Field label="Estratégia">
        <ToggleGroup
          type="single"
          variant="outline"
          size="sm"
          value={strategy ?? ""}
          onValueChange={(chosen) => {
            if (chosen !== "") {
              onChange({ ...choices, strategy: chosen as SplitStrategy });
            }
          }}
          aria-label="Estratégia da divisão"
          className="flex-wrap justify-start"
        >
          {options.split_strategies.map((one) => (
            <ToggleGroupItem key={one} value={one} className="px-3">
              {label("SplitStrategy", one)}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </Field>
      <div className="grid gap-6 @lg/form:grid-cols-2">
        <Field label="Teste" htmlFor="divisao-teste" readout={formatPercent(choices.testSize, 0)}>
          <Slider
            id="divisao-teste"
            aria-labelledby="divisao-teste-rotulo"
            valueText={formatPercent(choices.testSize, 0)}
            min={0.05}
            max={0.5}
            step={0.05}
            value={[choices.testSize]}
            onValueChange={([value]) => {
              if (value !== undefined) {
                onChange({ ...choices, testSize: value });
              }
            }}
          />
        </Field>
        <Field
          label="Validação"
          htmlFor="divisao-validacao"
          readout={formatPercent(choices.validationSize, 0)}
        >
          <Slider
            id="divisao-validacao"
            aria-labelledby="divisao-validacao-rotulo"
            valueText={formatPercent(choices.validationSize, 0)}
            min={0}
            max={0.3}
            step={0.05}
            value={[choices.validationSize]}
            onValueChange={([value]) => {
              if (value !== undefined) {
                onChange({ ...choices, validationSize: value });
              }
            }}
          />
        </Field>
      </div>
      <Field label="Seed" htmlFor="divisao-seed" help="A mesma seed dá sempre a mesma divisão.">
        <Input
          id="divisao-seed"
          name="seed"
          autoComplete="off"
          type="number"
          inputMode="numeric"
          min={0}
          max={MAX_SEED}
          step={1}
          value={Number.isNaN(choices.seed) ? "" : choices.seed}
          onChange={(event) => {
            onChange({ ...choices, seed: event.target.valueAsNumber });
          }}
          aria-invalid={!validSeed}
          className="w-40 tabular-nums"
        />
      </Field>
      {strategy === "time" && (
        <Field label="Coluna do tempo" htmlFor="divisao-tempo">
          <Select
            value={choices.timeColumn ?? NONE}
            onValueChange={(value) => {
              onChange({ ...choices, timeColumn: value === NONE ? null : value });
            }}
          >
            <SelectTrigger id="divisao-tempo" className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={NONE}>A ordem das linhas</SelectItem>
              {times.map((column) => (
                <SelectItem key={column.name} value={column.name}>
                  {column.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
      )}
    </>
  );
}

function TransformControls({
  dataset,
  choices,
  onChange,
  dropped,
}: ControlsProps & { dropped: readonly string[] }) {
  const label = useLabel();
  const { transforms } = choices;
  const set = (next: Partial<TransformChoices>) => {
    onChange({ ...choices, transforms: { ...transforms, ...next } });
  };
  const numeric = dataset.columns.filter(
    (column) =>
      column.kind === "numeric" && column.name !== choices.target && !dropped.includes(column.name),
  );
  return (
    <>
      <div className="grid gap-6 @lg/form:grid-cols-2">
        <Field label="Nulos nas colunas numéricas" htmlFor="transformar-numericos">
          <EnumSelect
            id="transformar-numericos"
            value={transforms.numericImputation}
            options={(["median", "mean"] as const).map((value) => ({
              value,
              label: label("ImputationStrategy", value),
            }))}
            none="Não preencher"
            onChange={(value) => {
              set({ numericImputation: value as ImputationStrategy | null });
            }}
          />
        </Field>
        <Field label="Nulos nas colunas categóricas" htmlFor="transformar-categoricos">
          <EnumSelect
            id="transformar-categoricos"
            value={transforms.categoricalImputation}
            options={[
              { value: "most_frequent", label: label("ImputationStrategy", "most_frequent") },
            ]}
            none="Não preencher"
            onChange={(value) => {
              set({ categoricalImputation: value as ImputationStrategy | null });
            }}
          />
        </Field>
      </div>
      <Field label="Escala" htmlFor="transformar-escala">
        <EnumSelect
          id="transformar-escala"
          value={transforms.scaling}
          options={(["standard", "minmax"] as const).map((value) => ({
            value,
            label: label("ScalingMethod", value),
          }))}
          none="Sem escala"
          onChange={(value) => {
            set({ scaling: value as ScalingMethod | null });
          }}
        />
      </Field>
      <Field
        label="Grau das features polinomiais"
        help="1 é sem features polinomiais; 2 junta os quadrados e os produtos de cada par (x², x·y)."
      >
        <ToggleGroup
          type="single"
          variant="outline"
          size="sm"
          value={String(transforms.polynomialDegree)}
          onValueChange={(value) => {
            if (value !== "") {
              set({ polynomialDegree: Number(value) });
            }
          }}
          aria-label="Grau das features polinomiais"
          className="justify-start"
        >
          {[1, 2, 3].map((degree) => (
            <ToggleGroupItem key={degree} value={String(degree)} className="w-10 tabular-nums">
              {degree}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </Field>
      <div className="flex items-center justify-between gap-4">
        <Label htmlFor="transformar-datas" className="text-sm text-soft">
          Extrair o ano, o mês, o dia e o dia da semana das datas
        </Label>
        <Switch
          id="transformar-datas"
          checked={transforms.dateFeatures}
          onCheckedChange={(dateFeatures) => {
            set({ dateFeatures });
          }}
        />
      </div>
      <Field label="Dividir em intervalos (com os quantis do treino)">
        <ChipGroup
          label="Colunas a dividir em intervalos"
          options={numeric.map((column) => ({ value: column.name, label: column.name }))}
          value={transforms.binned.filter((column) => !dropped.includes(column))}
          onChange={(binned) => {
            set({ binned });
          }}
        />
      </Field>
      {transforms.binned.length > 0 && (
        <Field
          label="Número de intervalos"
          htmlFor="transformar-intervalos"
          readout={String(transforms.nBins)}
        >
          <Slider
            id="transformar-intervalos"
            aria-labelledby="transformar-intervalos-rotulo"
            min={2}
            max={20}
            step={1}
            value={[transforms.nBins]}
            onValueChange={([value]) => {
              if (value !== undefined) {
                set({ nBins: value });
              }
            }}
          />
        </Field>
      )}
    </>
  );
}

interface EnumSelectProps {
  id: string;
  value: string | null;
  options: readonly { value: string; label: string }[];
  /** O nome da opção "nenhuma" (`null`). */
  none: string;
  onChange: (value: string | null) => void;
}

function EnumSelect({ id, value, options, none, onChange }: EnumSelectProps) {
  return (
    <Select
      value={value ?? NONE}
      onValueChange={(chosen) => {
        onChange(chosen === NONE ? null : chosen);
      }}
    >
      <SelectTrigger id={id} className="w-full">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {options.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
          </SelectItem>
        ))}
        <SelectItem value={NONE}>{none}</SelectItem>
      </SelectContent>
    </Select>
  );
}
