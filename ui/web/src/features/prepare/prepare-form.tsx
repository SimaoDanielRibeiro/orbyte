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
import { TopicText } from "@/components/markdown/topic-text";
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
    // Os passos separados por um filete, sem números (a ordem é a da página; ADR 0038).
    <div className="@container/form flex flex-col divide-y">
      <Step number={1} title="Target and task">
        <TaskControls
          dataset={dataset}
          choices={choices}
          onChange={onChange}
          suggestion={suggestion}
          options={options}
          taskType={taskType}
        />
      </Step>
      <Step number={2} title="Cleaning">
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
      <Step number={3} title="Split">
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
      <Step number={4} title="Transformations">
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

/** Um passo da preparação: o título e os controlos (a ordem dos passos é a da página). */
function Step({ number, title, children }: { number: number; title: string; children: ReactNode }) {
  return (
    <section
      aria-labelledby={`passo-${String(number)}`}
      className="flex flex-col gap-5 py-8 first:pt-0 last:pb-0"
    >
      <h2 id={`passo-${String(number)}`} className="text-lg">
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
    <div className="flex flex-col gap-3" role="status" aria-busy="true" aria-label="Loading">
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
      <Field label="Target column" htmlFor="preparar-alvo" help={<TopicText topic="task" />}>
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
            <SelectItem value={NONE}>No target (unsupervised learning)</SelectItem>
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
              label="Task type"
              htmlFor="preparar-tarefa"
              help={
                <>
                  <p>{suggestion.reason}</p>
                  <p className="mt-2">
                    The suggested one comes first; the others also make sense for this target.
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
        <p className="text-sm text-dim">There are no duplicate rows.</p>
      )}
      <Field
        label={dropColumns?.title ?? "Drop columns"}
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
          label={dropColumns?.title ?? "Drop columns"}
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
          More cleaning operations
        </CollapsibleTrigger>
        <CollapsibleContent className="flex flex-col gap-6">
          <Field
            label={dropMissing?.title ?? "Drop rows with missing values"}
            help={dropMissing?.description}
          >
            <ChipGroup
              label={dropMissing?.title ?? "Drop rows with missing values"}
              options={remaining.map((column) => ({ value: column.name, label: column.name }))}
              value={cleaning.dropMissing}
              onChange={(dropMissing) => {
                set({ dropMissing });
              }}
            />
          </Field>
          <Field label={outliers?.title ?? "Treat outliers"} help={outliers?.description}>
            <ChipGroup
              label={outliers?.title ?? "Treat outliers"}
              options={numeric.map((column) => ({ value: column.name, label: column.name }))}
              value={cleaning.outliers}
              onChange={(chosen) => {
                set({ outliers: chosen });
              }}
            />
          </Field>
          {cleaning.outliers.length > 0 && (
            <Field label="How to treat the outliers">
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
                aria-label="How to treat the outliers"
                className="flex-wrap justify-start"
              >
                <ToggleGroupItem value="clip" className="px-3">
                  Clip to the limits
                </ToggleGroupItem>
                <ToggleGroupItem value="remove" className="px-3">
                  Drop the rows
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
      <Field label="Strategy">
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
          aria-label="Split strategy"
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
        <Field label="Test" htmlFor="divisao-teste" readout={formatPercent(choices.testSize, 0)}>
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
          label="Validation"
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
      <Field label="Seed" htmlFor="divisao-seed" help="The same seed always gives the same split.">
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
        <Field label="Time column" htmlFor="divisao-tempo">
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
              <SelectItem value={NONE}>The order of the rows</SelectItem>
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
        <Field label="Nulls in numeric columns" htmlFor="transformar-numericos">
          <EnumSelect
            id="transformar-numericos"
            value={transforms.numericImputation}
            options={(["median", "mean"] as const).map((value) => ({
              value,
              label: label("ImputationStrategy", value),
            }))}
            none="Don't fill"
            onChange={(value) => {
              set({ numericImputation: value as ImputationStrategy | null });
            }}
          />
        </Field>
        <Field label="Nulls in categorical columns" htmlFor="transformar-categoricos">
          <EnumSelect
            id="transformar-categoricos"
            value={transforms.categoricalImputation}
            options={[
              { value: "most_frequent", label: label("ImputationStrategy", "most_frequent") },
            ]}
            none="Don't fill"
            onChange={(value) => {
              set({ categoricalImputation: value as ImputationStrategy | null });
            }}
          />
        </Field>
      </div>
      <Field label="Scaling" htmlFor="transformar-escala">
        <EnumSelect
          id="transformar-escala"
          value={transforms.scaling}
          options={(["standard", "minmax"] as const).map((value) => ({
            value,
            label: label("ScalingMethod", value),
          }))}
          none="No scaling"
          onChange={(value) => {
            set({ scaling: value as ScalingMethod | null });
          }}
        />
      </Field>
      <Field
        label="Polynomial features degree"
        help="1 means no polynomial features; 2 adds the squares and the products of each pair (x², x·y)."
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
          aria-label="Polynomial features degree"
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
        <div className="flex items-center gap-1">
          <Label htmlFor="transformar-datas" className="text-sm text-soft">
            Extract the date parts
          </Label>
          <Explain title="Extract the date parts">
            The year, month, day and weekday of each date, in new columns.
          </Explain>
        </div>
        <Switch
          id="transformar-datas"
          checked={transforms.dateFeatures}
          onCheckedChange={(dateFeatures) => {
            set({ dateFeatures });
          }}
        />
      </div>
      <Field
        label="Bin into intervals"
        help="Each chosen column becomes intervals, with the limits taken from the training quantiles."
      >
        <ChipGroup
          label="Columns to bin into intervals"
          options={numeric.map((column) => ({ value: column.name, label: column.name }))}
          value={transforms.binned.filter((column) => !dropped.includes(column))}
          onChange={(binned) => {
            set({ binned });
          }}
        />
      </Field>
      {transforms.binned.length > 0 && (
        <Field
          label="Number of intervals"
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
