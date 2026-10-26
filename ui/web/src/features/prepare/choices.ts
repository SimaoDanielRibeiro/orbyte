/**
 * As escolhas do ecrã Preparar e a configuração que sai delas.
 *
 * O ecrã guarda o que o utilizador escolheu (`PrepareChoices`), não a configuração: assim, o que
 * ele não tocou segue sempre as sugestões do backend. Um `null` quer dizer "o que o backend
 * sugere": a tarefa sugerida para o alvo, as limpezas sugeridas, a primeira divisão possível. Ao
 * trocar de alvo, a tarefa e a limpeza voltam às sugestões do alvo novo.
 *
 * A configuração (`PreparationConfig`) monta-se com as escolhas e as sugestões, e é ela que vai
 * para a API e fica na sessão para o treino.
 */
import type {
  CleaningStep,
  CleaningSuggestion,
  ImputationStrategy,
  PreparationConfig,
  ScalingMethod,
  SplitStrategy,
  TaskOptions,
  TaskSuggestion,
  TaskType,
} from "@/api/generated";

export type OutlierMethod = "clip" | "remove";

/** O que se escolheu na limpeza. */
export interface CleaningChoices {
  /** As sugestões ligadas (remover duplicados, converter tipos), pela chave de cada uma. */
  enabled: string[];
  /** As colunas a remover. */
  dropped: string[];
  /** As colunas em que uma linha com um valor em falta é removida. */
  dropMissing: string[];
  /** As colunas numéricas em que se tratam os valores extremos. */
  outliers: string[];
  outlierMethod: OutlierMethod;
}

export interface TransformChoices {
  numericImputation: ImputationStrategy | null;
  categoricalImputation: ImputationStrategy | null;
  scaling: ScalingMethod | null;
  dateFeatures: boolean;
  polynomialDegree: number;
  binned: string[];
  nBins: number;
}

export interface PrepareChoices {
  target: string | null;
  /** `null`: a tarefa sugerida para o alvo. */
  taskType: TaskType | null;
  /** `null`: as limpezas sugeridas para o alvo. */
  cleaning: CleaningChoices | null;
  /** `null`: a primeira divisão possível para a tarefa. */
  strategy: SplitStrategy | null;
  testSize: number;
  validationSize: number;
  /** A coluna do tempo, na divisão temporal (`null`: a ordem das linhas). */
  timeColumn: string | null;
  seed: number;
  transforms: TransformChoices;
}

export const DEFAULT_SEED = 42;

const DEFAULT_TRANSFORMS: TransformChoices = {
  numericImputation: "median",
  categoricalImputation: "most_frequent",
  scaling: "standard",
  dateFeatures: true,
  polynomialDegree: 1,
  binned: [],
  nBins: 5,
};

/** As escolhas de um dataset que se abre pela primeira vez: tudo como o backend sugere. */
export function initialChoices(suggestedTarget: string | null): PrepareChoices {
  return {
    target: suggestedTarget,
    taskType: null,
    cleaning: null,
    strategy: null,
    testSize: 0.2,
    validationSize: 0,
    timeColumn: null,
    seed: DEFAULT_SEED,
    transforms: DEFAULT_TRANSFORMS,
  };
}

/** Trocar de alvo: a tarefa e a limpeza voltam às sugestões do alvo novo. */
export function withTarget(choices: PrepareChoices, target: string | null): PrepareChoices {
  return {
    ...choices,
    target,
    taskType: null,
    cleaning: null,
    transforms: {
      ...choices.transforms,
      binned: choices.transforms.binned.filter((column) => column !== target),
    },
  };
}

/** A chave de uma sugestão que se liga e desliga. */
export function suggestionKey(suggestion: CleaningSuggestion): string {
  return `${suggestion.step.operation}:${suggestion.column ?? ""}`;
}

/** As sugestões que são interruptores: todas menos as colunas a remover (essas têm a sua lista). */
export function toggleSuggestions(
  suggestions: readonly CleaningSuggestion[],
): CleaningSuggestion[] {
  return suggestions.filter((suggestion) => suggestion.step.operation !== "drop_columns");
}

/** As colunas que o backend sugere remover, e porquê. */
export function droppedSuggestions(
  suggestions: readonly CleaningSuggestion[],
): Map<string, string> {
  const reasons = new Map<string, string>();
  for (const suggestion of suggestions) {
    if (suggestion.step.operation === "drop_columns") {
      for (const column of columnsOf(suggestion.step)) {
        reasons.set(column, suggestion.reason);
      }
    }
  }
  return reasons;
}

/** A limpeza sugerida: todas as sugestões ligadas. */
export function suggestedCleaning(
  suggestions: readonly CleaningSuggestion[],
  target: string | null,
): CleaningChoices {
  return {
    enabled: toggleSuggestions(suggestions).map(suggestionKey),
    dropped: [...droppedSuggestions(suggestions).keys()].filter((column) => column !== target),
    dropMissing: [],
    outliers: [],
    outlierMethod: "clip",
  };
}

/** Os passos da limpeza, pela ordem em que correm (a mesma da interface simples). */
export function cleaningSteps(
  choices: CleaningChoices,
  suggestions: readonly CleaningSuggestion[],
): CleaningStep[] {
  const steps = toggleSuggestions(suggestions)
    .filter((suggestion) => choices.enabled.includes(suggestionKey(suggestion)))
    .map((suggestion) => suggestion.step);
  if (choices.dropped.length > 0) {
    steps.push({ operation: "drop_columns", params: { columns: choices.dropped } });
  }
  if (choices.dropMissing.length > 0) {
    steps.push({ operation: "drop_missing_rows", params: { columns: choices.dropMissing } });
  }
  for (const column of choices.outliers) {
    steps.push({ operation: "treat_outliers", params: { column, method: choices.outlierMethod } });
  }
  return steps;
}

/** A tarefa que vale: a escolhida, se o alvo a permitir; senão, a sugerida. */
export function effectiveTask(
  choices: PrepareChoices,
  suggestion: TaskSuggestion,
): TaskType | null {
  return choices.taskType !== null && suggestion.allowed.includes(choices.taskType)
    ? choices.taskType
    : suggestion.task_type;
}

/** A divisão que vale: a escolhida, se a tarefa a permitir; senão, a primeira possível. */
export function effectiveStrategy(
  choices: PrepareChoices,
  options: TaskOptions,
): SplitStrategy | null {
  return choices.strategy !== null && options.split_strategies.includes(choices.strategy)
    ? choices.strategy
    : (options.split_strategies[0] ?? null);
}

/** A limpeza que vale: a escolhida, ou a sugerida. */
export function effectiveCleaning(
  choices: PrepareChoices,
  suggestions: readonly CleaningSuggestion[],
): CleaningChoices {
  return choices.cleaning ?? suggestedCleaning(suggestions, choices.target);
}

/**
 * A configuração da preparação. `null` se o alvo não der nenhuma tarefa (uma coluna constante,
 * por exemplo: a sugestão diz porquê).
 */
export function buildConfig(
  choices: PrepareChoices,
  suggestion: TaskSuggestion,
  suggestions: readonly CleaningSuggestion[],
  options: TaskOptions,
): PreparationConfig | null {
  const taskType = effectiveTask(choices, suggestion);
  const strategy = effectiveStrategy(choices, options);
  if (taskType === null || strategy === null) {
    return null;
  }
  const cleaning = effectiveCleaning(choices, suggestions);
  const removed = new Set([
    ...cleaning.dropped,
    ...(choices.target === null ? [] : [choices.target]),
  ]);
  const { transforms } = choices;
  return {
    target: choices.target,
    task_type: taskType,
    cleaning: cleaningSteps(cleaning, suggestions),
    split: {
      strategy,
      // Os controlos andam de 5 em 5%: arredondar evita um 0,30000000000000004.
      test_size: round(choices.testSize),
      validation_size: round(choices.validationSize),
      time_column: strategy === "time" ? choices.timeColumn : null,
    },
    transforms: {
      numeric_imputation: transforms.numericImputation,
      categorical_imputation: transforms.categoricalImputation,
      scaling: transforms.scaling,
      date_features: transforms.dateFeatures,
      polynomial_degree: transforms.polynomialDegree,
      binned_columns: transforms.binned.filter((column) => !removed.has(column)),
      n_bins: transforms.nBins,
    },
  };
}

function columnsOf(step: CleaningStep): string[] {
  const columns = step.params?.columns;
  return Array.isArray(columns) ? columns.filter((one) => typeof one === "string") : [];
}

function round(value: number): number {
  return Math.round(value * 100) / 100;
}
