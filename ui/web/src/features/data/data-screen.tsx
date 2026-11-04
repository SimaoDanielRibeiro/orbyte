/**
 * O ecrã Dados: de onde vêm os dados e o que se leu deles.
 *
 * Com um dataset aberto, ele vem primeiro (é com ele que se trabalha): os números em cartões, os
 * avisos da validação, as primeiras linhas e as colunas; o passo seguinte fica no cabeçalho.
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
import { ErrorNote, Panel, Proceed, Section, StatStrip } from "@/components/shell/blocks";
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
          <Proceed to="/explore">Explore this dataset</Proceed>
        )}
      </PageHeader>
      {lost && (
        <p className="mt-8 max-w-2xl text-sm text-soft">
          The dataset that was open is no longer in the API: it restarted, and loaded datasets live
          in its memory. Load it again below.
        </p>
      )}
      {dataset !== undefined && <OpenDataset dataset={dataset} />}
      <Section
        title={dataset === undefined ? "Choose the data" : "Open other data"}
        topic="input"
        className="mt-14"
      >
        {/* As fontes umas por baixo das outras, sem separadores: os exemplos e, logo a seguir, o
            ficheiro, numa zona grande; os sintéticos no fim (ADR 0037). */}
        <div className="flex flex-col gap-14">
          <Panel title="The samples">
            <SampleSource open={dataset?.source} />
          </Panel>
          <Panel title="A file of yours" topic="upload">
            <FileSource />
          </Panel>
          <Panel title="Synthetic data">
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
          { key: "rows", label: "Rows", value: formatCount(dataset.n_rows) },
          { key: "columns", label: "Columns", value: formatCount(dataset.n_columns) },
          {
            key: "warnings",
            label: "Validation warnings",
            value: formatCount(warnings),
            // Só se diz alguma coisa quando os dados não podem seguir: o número diz o resto.
            detail: validation.is_usable ? undefined : "the data can't go on",
            tone: !validation.is_usable ? "negative" : warnings === 0 ? "positive" : "neutral",
          },
        ]}
      />

      {validation.notices.length > 0 && (
        <Panel title="The validation" topic="validation" className="mt-10">
          <NoticeList notices={validation.notices} />
          {!validation.is_usable && (
            <p role="alert" className="text-sm text-destructive">
              This data can&apos;t go on to the next steps: fix the errors above and load the file
              again.
            </p>
          )}
        </Panel>
      )}

      {validation.is_usable && (
        <Panel className="mt-12">
          <Tabs defaultValue="preview">
            <TabsList aria-label="What was read">
              <TabsTrigger value="preview">First rows</TabsTrigger>
              <TabsTrigger value="columns">Columns and types</TabsTrigger>
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
  return <TablePreviewView preview={preview.data} label="The dataset's first rows" />;
}

/** As colunas: o tipo que o perfil inferiu (o porquê fica no "?" de cada uma) e o do pandas. */
function Columns({ dataset }: { dataset: DatasetSummary }) {
  const label = useLabel();
  return (
    <div className="overflow-hidden rounded-xl border">
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            <TableHead className="px-4">Column</TableHead>
            <TableHead className="px-4">Inferred type</TableHead>
            <TableHead className="px-4">pandas type</TableHead>
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
                    title={`Why “${label("ColumnKind", column.kind)}”`}
                    label={`Why ${column.name} is ${label("ColumnKind", column.kind).toLowerCase()}`}
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
      <label htmlFor="datasets-carregados" className="small-label">
        Datasets in this session
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
