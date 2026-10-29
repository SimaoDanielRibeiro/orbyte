/**
 * O ecrã Dados: de onde vêm os dados e o que se leu deles.
 *
 * Com um dataset aberto, ele vem primeiro (é com ele que se trabalha): os números em cartões, os
 * avisos da validação, as primeiras linhas e as colunas, e o passo seguinte da interface aberta.
 * As fontes ficam por baixo, para abrir outro. Ao chegar um dataset novo, a página volta ao topo
 * para o mostrar. As descrições ficam nos "?" (ADR 0032).
 */
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useReducedMotion } from "motion/react";
import { useEffect, useRef } from "react";

import type { DatasetSummary } from "@/api/generated";
import {
  datasetsLoadedOptions,
  datasetsPreviewOptions,
  datasetsSummaryQueryKey,
} from "@/api/generated/@tanstack/react-query.gen";
import { useLabel } from "@/api/vocabulary";
import { TablePreviewView } from "@/components/data/table-preview";
import { Explain } from "@/components/explain/explain";
import { NoticeList } from "@/components/notices/notice-list";
import { ErrorNote, NextStep, Panel, Proceed, Section, StatStrip } from "@/components/shell/blocks";
import { Page, PageHeader } from "@/components/shell/page";
import { SCREENS } from "@/components/shell/screens";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { formatCount } from "@/lib/format";
import { useSession } from "@/state/session";

import { FileSource, SampleSource, SyntheticSource } from "./sources";
import { useOpenDataset } from "./use-open-dataset";

export function DataScreen() {
  const { query, lost } = useOpenDataset();
  const dataset = query.data;
  return (
    <Page>
      <PageHeader
        title={SCREENS.data.title}
        description={SCREENS.data.description}
        topics={["input", "validation", "upload", "automatic_or_manual"]}
      >
        {dataset?.validation.is_usable === true && (
          <Proceed to="/explorar">Explorar este dataset</Proceed>
        )}
      </PageHeader>
      {lost && (
        <p className="mt-8 max-w-2xl text-sm text-soft">
          O dataset que estava aberto já não está na API: ela recomeçou, e os datasets carregados
          vivem na memória dela. Carrega-o outra vez aqui em baixo.
        </p>
      )}
      {dataset !== undefined && <OpenDataset dataset={dataset} />}
      <Section
        title={dataset === undefined ? "Escolhe os dados" : "Abrir outros dados"}
        topic="input"
        className="mt-14"
      >
        {/* As fontes umas por baixo das outras, sem separadores: os exemplos e, logo a seguir, o
            ficheiro, numa zona grande; os sintéticos no fim (ADR 0037). */}
        <div className="flex flex-col gap-14">
          <Panel title="Os exemplos">
            <SampleSource open={dataset?.source} />
          </Panel>
          <Panel title="Um ficheiro teu" topic="upload">
            <FileSource />
          </Panel>
          <Panel title="Dados sintéticos">
            <SyntheticSource />
          </Panel>
        </div>
      </Section>
    </Page>
  );
}

/** O dataset aberto: os números, os avisos da validação, as primeiras linhas e as colunas. */
function OpenDataset({ dataset }: { dataset: DatasetSummary }) {
  const heading = useRef<HTMLHeadingElement>(null);
  const reduced = useReducedMotion();
  const shown = useRef(dataset.id);
  const { validation } = dataset;
  const warnings = validation.notices.filter((notice) => notice.level !== "info").length;

  // Um dataset novo: a página volta ao topo, e o título recebe o foco (quem usa um leitor de
  // ecrã ouve o que abriu).
  useEffect(() => {
    if (shown.current === dataset.id) {
      return;
    }
    shown.current = dataset.id;
    heading.current?.scrollIntoView({ behavior: reduced === true ? "auto" : "smooth" });
    heading.current?.focus({ preventScroll: true });
  }, [dataset.id, reduced]);

  return (
    <section className="mt-10" aria-labelledby="dataset-aberto">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex min-w-0 items-center gap-1.5">
          <h2
            id="dataset-aberto"
            ref={heading}
            tabIndex={-1}
            className="scroll-mt-16 text-2xl outline-none"
          >
            {dataset.name}
          </h2>
          {dataset.description !== "" && (
            <Explain title={dataset.name}>{dataset.description}</Explain>
          )}
        </div>
        <LoadedSwitcher current={dataset} />
      </div>

      <StatStrip
        className="mt-6"
        items={[
          { key: "linhas", label: "Linhas", value: formatCount(dataset.n_rows) },
          { key: "colunas", label: "Colunas", value: formatCount(dataset.n_columns) },
          {
            key: "avisos",
            label: "Avisos da validação",
            value: formatCount(warnings),
            detail: !validation.is_usable
              ? "os dados não podem seguir"
              : warnings === 0
                ? "passaram sem avisos"
                : "vê-os em baixo",
            tone: !validation.is_usable ? "negative" : warnings === 0 ? "positive" : "neutral",
          },
        ]}
      />

      {validation.notices.length > 0 && (
        <Panel title="A validação" topic="validation" className="mt-10">
          <NoticeList notices={validation.notices} />
          {!validation.is_usable && (
            <p role="alert" className="text-sm text-destructive">
              Estes dados não podem seguir para as próximas etapas: corrige os erros acima e carrega
              o ficheiro outra vez.
            </p>
          )}
        </Panel>
      )}

      {validation.is_usable && (
        <Panel className="mt-12">
          <Tabs defaultValue="preview">
            <TabsList aria-label="O que se leu">
              <TabsTrigger value="preview">Primeiras linhas</TabsTrigger>
              <TabsTrigger value="columns">Colunas e tipos</TabsTrigger>
            </TabsList>
            <TabsContent value="preview" className="mt-5">
              <Preview datasetId={dataset.id} />
            </TabsContent>
            <TabsContent value="columns" className="mt-5">
              <Columns dataset={dataset} />
            </TabsContent>
          </Tabs>
        </Panel>
      )}

      {validation.is_usable && (
        <NextStep
          className="mt-16"
          text="Os dados estão prontos. A seguir, explora-os: o perfil de cada coluna, as correlações e os avisos."
        >
          <Proceed to="/explorar">Explorar este dataset</Proceed>
        </NextStep>
      )}
    </section>
  );
}

function Preview({ datasetId }: { datasetId: string }) {
  const preview = useQuery(datasetsPreviewOptions({ path: { dataset_id: datasetId } }));
  if (preview.error !== null) {
    return <ErrorNote error={preview.error} />;
  }
  if (preview.data === undefined) {
    return <Skeleton className="h-72 w-full" />;
  }
  return <TablePreviewView preview={preview.data} label="As primeiras linhas do dataset" />;
}

/** As colunas: o tipo que o perfil inferiu (o porquê fica no "?" de cada uma) e o do pandas. */
function Columns({ dataset }: { dataset: DatasetSummary }) {
  const label = useLabel();
  return (
    <div className="overflow-hidden rounded-xl border">
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            <TableHead className="px-4">Coluna</TableHead>
            <TableHead className="px-4">Tipo inferido</TableHead>
            <TableHead className="px-4">Tipo no pandas</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {dataset.columns.map((column) => (
            <TableRow key={column.name}>
              <TableCell className="px-4 font-medium text-strong">{column.name}</TableCell>
              <TableCell className="px-4">
                <span className="inline-flex items-center gap-1">
                  {label("ColumnKind", column.kind)}
                  <Explain
                    title={`Porquê «${label("ColumnKind", column.kind)}»`}
                    label={`Porque é que ${column.name} é ${label("ColumnKind", column.kind).toLowerCase()}`}
                  >
                    {column.kind_reason}
                  </Explain>
                </span>
              </TableCell>
              <TableCell className="px-4 font-mono text-xs text-dim">{column.dtype}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

/** Com mais de um dataset carregado, troca-se aqui entre eles (sem os carregar outra vez). */
function LoadedSwitcher({ current }: { current: DatasetSummary }) {
  const queryClient = useQueryClient();
  const openDataset = useSession((state) => state.openDataset);
  const loaded = useQuery(datasetsLoadedOptions());
  const datasets = loaded.data ?? [];
  if (datasets.length < 2) {
    return null;
  }
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor="datasets-carregados" className="mono-label">
        Datasets desta sessão
      </label>
      <Select
        value={current.id}
        onValueChange={(id) => {
          const chosen = datasets.find((dataset) => dataset.id === id);
          if (chosen !== undefined) {
            queryClient.setQueryData(datasetsSummaryQueryKey({ path: { dataset_id: id } }), chosen);
            openDataset(id);
          }
        }}
      >
        <SelectTrigger id="datasets-carregados" className="min-w-56">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {datasets.map((dataset) => (
            <SelectItem key={dataset.id} value={dataset.id}>
              {dataset.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
