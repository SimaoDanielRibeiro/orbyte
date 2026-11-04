/**
 * O vocabulário: o nome a mostrar de cada valor de cada enum, e o que muda com cada tarefa.
 *
 * Pede-se uma vez, ao arrancar (o carregamento da rota raiz espera por ele), e fica na cache
 * para sempre. Um enum mostra-se sempre com `label`, nunca pelo valor cru: um teste do lado do
 * Python garante que cada enum que a API devolve tem um nome para cada valor (ADR 0026).
 */
import { queryOptions, useSuspenseQuery } from "@tanstack/react-query";
import { useCallback } from "react";

import type {
  AgentRole,
  CellKind,
  ChoiceGroup,
  ColumnKind,
  Implementation,
  ImputationStrategy,
  JobStatus,
  LaboratoryDemo,
  MessageKind,
  ModelFamily,
  NoticeLevel,
  PackageFileKind,
  PackageOrigin,
  Paradigm,
  ParticipantKind,
  Partition,
  Problem,
  ScalingMethod,
  SearchPhase,
  SplitStrategy,
  Stage,
  TaskInfo,
  TaskType,
  Vocabulary,
} from "./generated";
import { explanationsVocabularyOptions } from "./generated/@tanstack/react-query.gen";

/** Cada enum do vocabulário, pelo nome do tipo (a chave de `labels`). */
export interface VocabularyKinds {
  AgentRole: AgentRole;
  CellKind: CellKind;
  ChoiceGroup: ChoiceGroup;
  ColumnKind: ColumnKind;
  Implementation: Implementation;
  ImputationStrategy: ImputationStrategy;
  JobStatus: JobStatus;
  LaboratoryDemo: LaboratoryDemo;
  MessageKind: MessageKind;
  ModelFamily: ModelFamily;
  NoticeLevel: NoticeLevel;
  PackageFileKind: PackageFileKind;
  PackageOrigin: PackageOrigin;
  Paradigm: Paradigm;
  ParticipantKind: ParticipantKind;
  Partition: Partition;
  Problem: Problem;
  ScalingMethod: ScalingMethod;
  SearchPhase: SearchPhase;
  SplitStrategy: SplitStrategy;
  Stage: Stage;
  TaskType: TaskType;
}

export type VocabularyKind = keyof VocabularyKinds;

export const vocabularyQuery = queryOptions({
  ...explanationsVocabularyOptions(),
  // O vocabulário só muda com o backend: nunca fica velho, nem quando se invalida a cache.
  staleTime: "static",
  gcTime: Infinity,
});

/**
 * O nome a mostrar de `value`, um valor do enum `kind`.
 *
 * Se faltar (não devia: o contrato garante-o), mostra o valor cru em vez de nada.
 */
export function labelOf<K extends VocabularyKind>(
  vocabulary: Vocabulary,
  kind: K,
  value: VocabularyKinds[K],
): string {
  return vocabulary.labels[kind]?.[value] ?? value;
}

export function useVocabulary(): Vocabulary {
  return useSuspenseQuery(vocabularyQuery).data;
}

/** A função `label(kind, value)`, já ligada ao vocabulário. */
export function useLabel(): <K extends VocabularyKind>(
  kind: K,
  value: VocabularyKinds[K],
) => string {
  const vocabulary = useVocabulary();
  return useCallback((kind, value) => labelOf(vocabulary, kind, value), [vocabulary]);
}

/** O que muda com uma tarefa: o paradigma, a métrica principal, se os agentes trabalham nela… */
export function useTaskInfo(taskType: TaskType): TaskInfo | undefined {
  return useVocabulary().tasks.find((task) => task.task_type === taskType);
}
