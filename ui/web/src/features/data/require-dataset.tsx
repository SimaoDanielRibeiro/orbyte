/**
 * Os ecrãs que precisam de um dataset aberto (Explorar, Preparar…) começam aqui: sem dataset,
 * dizem onde se escolhe; com um id que a API já não tem (recomeçou), dizem que é preciso
 * carregá-lo outra vez.
 */
import type { ReactNode } from "react";

import type { DatasetSummary } from "@/api/generated";
import { ErrorNote, NeedsStep } from "@/components/shell/blocks";
import { Skeleton } from "@/components/ui/skeleton";

import { useOpenDataset } from "./use-open-dataset";

interface RequireDatasetProps {
  children: (dataset: DatasetSummary) => ReactNode;
}

export function RequireDataset({ children }: RequireDatasetProps) {
  const { datasetId, query, lost } = useOpenDataset();
  if (datasetId === null) {
    return (
      <NeedsStep
        message="Ainda não há dados abertos. Escolhe um dataset de exemplo, carrega um ficheiro ou gera dados sintéticos."
        to="/dados"
        action="Escolher os dados"
      />
    );
  }
  if (lost) {
    return <LostDataset />;
  }
  if (query.error !== null) {
    return <ErrorNote error={query.error} className="mt-10" />;
  }
  if (query.data === undefined) {
    return <PageSkeleton />;
  }
  return children(query.data);
}

/** O dataset aberto perdeu-se: a API recomeçou. */
export function LostDataset() {
  return (
    <NeedsStep
      message="O dataset que estava aberto já não está na API: ela recomeçou, e os datasets carregados vivem na memória dela. Carrega-o outra vez."
      to="/dados"
      action="Carregar os dados"
    />
  );
}

/** O lugar de um ecrã enquanto os dados não chegam. */
export function PageSkeleton() {
  return (
    <div
      className="mt-10 flex flex-col gap-4"
      role="status"
      aria-busy="true"
      aria-label="A carregar"
    >
      <Skeleton className="h-6 w-64" />
      <Skeleton className="h-4 w-full max-w-xl" />
      <Skeleton className="h-48 w-full" />
    </div>
  );
}
