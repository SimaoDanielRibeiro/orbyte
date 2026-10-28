import { describe, expect, it } from "vitest";

import type { CleaningSuggestion, TaskOptions, TaskSuggestion } from "@/api/generated";

import {
  buildConfig,
  cleaningSteps,
  initialChoices,
  suggestedCleaning,
  withTarget,
  type PrepareChoices,
} from "./choices";

const SUGGESTIONS: CleaningSuggestion[] = [
  {
    step: { operation: "drop_duplicates", params: {} },
    reason: "Há 3 linhas duplicadas.",
    column: null,
  },
  {
    step: { operation: "convert_type", params: { column: "Price", to: "numeric" } },
    reason: "Price tem números guardados como texto.",
    column: "Price",
  },
  {
    step: { operation: "drop_columns", params: { columns: ["PassengerId"] } },
    reason: "PassengerId é um identificador.",
    column: "PassengerId",
  },
  {
    step: { operation: "drop_columns", params: { columns: ["Name"] } },
    reason: "Name é texto livre.",
    column: "Name",
  },
];

const SUGGESTION: TaskSuggestion = {
  target: "Survived",
  task_type: "binary_classification",
  allowed: ["binary_classification", "clustering"],
  reason: "O alvo Survived só tem dois valores.",
  n_classes: 2,
  notices: [],
};

const OPTIONS: TaskOptions = {
  reference: null,
  reference_note: null,
  split_strategies: ["stratified", "random", "time"],
};

describe("as escolhas do ecrã Preparar", () => {
  it("sem nada escolhido, seguem as sugestões do backend", () => {
    const config = buildConfig(initialChoices("Survived"), SUGGESTION, SUGGESTIONS, OPTIONS);

    expect(config?.task_type).toBe("binary_classification");
    expect(config?.split?.strategy).toBe("stratified");
    expect(config?.cleaning).toEqual([
      { operation: "drop_duplicates", params: {} },
      { operation: "convert_type", params: { column: "Price", to: "numeric" } },
      { operation: "drop_columns", params: { columns: ["PassengerId", "Name"] } },
    ]);
  });

  it("monta a limpeza pela ordem em que os passos correm", () => {
    const steps = cleaningSteps(
      {
        enabled: ["convert_type:Price"],
        dropped: ["Name"],
        dropMissing: ["Age"],
        outliers: ["Fare", "Age"],
        outlierMethod: "remove",
      },
      SUGGESTIONS,
    );

    expect(steps.map((step) => step.operation)).toEqual([
      "convert_type",
      "drop_columns",
      "drop_missing_rows",
      "treat_outliers",
      "treat_outliers",
    ]);
    expect(steps.at(-1)?.params).toEqual({ column: "Age", method: "remove" });
  });

  it("uma tarefa ou uma divisão que deixou de ser possível volta à sugerida", () => {
    const choices: PrepareChoices = {
      ...initialChoices("Survived"),
      taskType: "regression",
      strategy: "time",
    };

    const config = buildConfig(choices, SUGGESTION, SUGGESTIONS, {
      ...OPTIONS,
      split_strategies: ["random"],
    });

    expect(config?.task_type).toBe("binary_classification");
    expect(config?.split?.strategy).toBe("random");
    expect(config?.split?.time_column).toBeNull();
  });

  it("trocar de alvo volta às sugestões e tira o alvo dos intervalos", () => {
    const choices: PrepareChoices = {
      ...initialChoices("Survived"),
      taskType: "clustering",
      cleaning: suggestedCleaning([], "Survived"),
      transforms: { ...initialChoices(null).transforms, binned: ["Age", "Fare"] },
    };

    const changed = withTarget(choices, "Fare");

    expect(changed.taskType).toBeNull();
    expect(changed.cleaning).toBeNull();
    expect(changed.transforms.binned).toEqual(["Age"]);
  });

  it("não remove o alvo nem divide em intervalos uma coluna removida", () => {
    const choices: PrepareChoices = {
      ...initialChoices("Name"),
      transforms: { ...initialChoices(null).transforms, binned: ["Name", "Age", "PassengerId"] },
    };

    const config = buildConfig(choices, SUGGESTION, SUGGESTIONS, OPTIONS);

    expect(config?.cleaning).toContainEqual({
      operation: "drop_columns",
      params: { columns: ["PassengerId"] },
    });
    expect(config?.transforms?.binned_columns).toEqual(["Age"]);
  });

  it("arredonda as partes da divisão", () => {
    const config = buildConfig(
      { ...initialChoices("Survived"), testSize: 0.30000000000000004, validationSize: 0.1 },
      SUGGESTION,
      SUGGESTIONS,
      OPTIONS,
    );

    expect(config?.split?.test_size).toBe(0.3);
    expect(config?.split?.validation_size).toBe(0.1);
  });

  it("sem tarefa possível, não há configuração", () => {
    const config = buildConfig(
      initialChoices("Constante"),
      { ...SUGGESTION, task_type: null, allowed: [] },
      [],
      OPTIONS,
    );

    expect(config).toBeNull();
  });
});
