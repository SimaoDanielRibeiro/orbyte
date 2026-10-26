/**
 * O ecrã Dados: de onde vêm os dados e o que se leu deles.
 *
 * Com um dataset aberto, ele vem primeiro (é com ele que se trabalha), e as fontes ficam por
 * baixo, para abrir outro. Ao chegar um dataset novo, a página volta ao topo para o mostrar.
 */
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "@tanstack/react-router";
import { useReducedMotion } from "motion/react";
import { useEffect, useRef } from "react";

import type { DatasetSummary } from "@/api/generated";
import {
  datasetsLoadedOptions,
  datasetsPreviewOptions,
  datasetsSummaryQueryKey,
} from "@/api/generated/@tanstack/react-query.gen";
import { useTopic } from "@/api/topics";
import { useLabel } from "@/api/vocabulary";
import { TablePreviewView } from "@/components/data/table-preview";
import { Prose } from "@/components/markdown/prose";
import { NoticeList } from "@/components/notices/notice-list";
import { ErrorNote, Facts, Section } from "@/components/shell/blocks";
import { Page, PageHeader } from "@/components/shell/page";
import { SCREENS } from "@/components/shell/screens";
import { WhyButton } from "@/components/shell/why-button";
import { Button } from "@/components/ui/button";
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
import { useSession, type WorkPath } from "@/state/session";

import { FileSource, SampleSource, SyntheticSource } from "./sources";
import { useOpenDataset } from "./use-open-dataset";

export function DataScreen() {
  const { query, lost } = useOpenDataset();
  const dataset = query.data;
  return (
    <Page>
      <PageHeader title={SCREENS.data.title} description={SCREENS.data.description}>
        <WhyButton topics={["input", "validation", "upload", "automatic_or_manual"]} />
      </PageHeader>
      {lost && (
        <p className="mt-10 max-w-[68ch] border-t pt-6 font-serif text-reading text-muted-foreground">
          O dataset que estava aberto já não está na API: ela recomeçou, e os datasets carregados
          vivem na memória dela. Carrega-o outra vez aqui em baixo.
        </p>
      )}
      {dataset !== undefined && <OpenDataset dataset={dataset} />}
      <Section
        title={dataset === undefined ? "De onde vêm os dados" : "Abrir outros dados"}
        className="mt-14"
      >
        <Tabs defaultValue="samples">
          <TabsList variant="line" aria-label="Fontes de dados">
            <TabsTrigger value="samples">Exemplos</TabsTrigger>
            <TabsTrigger value="file">Ficheiro</TabsTrigger>
            <TabsTrigger value="synthetic">Sintéticos</TabsTrigger>
          </TabsList>
          <TabsContent value="samples" className="mt-6 text-base">
            <SampleSource open={dataset?.source} />
          </TabsContent>
          <TabsContent value="file" className="mt-6 text-base">
            <FileSource />
          </TabsContent>
          <TabsContent value="synthetic" className="mt-6 text-base">
            <SyntheticSource />
          </TabsContent>
        </Tabs>
      </Section>
    </Page>
  );
}

/** O dataset aberto: o resumo, os avisos da validação, as primeiras linhas e as colunas. */
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
    <>
      <section className="mt-12" aria-labelledby="dataset-aberto">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div className="max-w-[68ch]">
            <h2
              id="dataset-aberto"
              ref={heading}
              tabIndex={-1}
              className="scroll-mt-16 text-3xl outline-none"
            >
              {dataset.name}
            </h2>
            {dataset.description !== "" && (
              <p className="mt-2 font-serif text-reading text-muted-foreground">
                {dataset.description}
              </p>
            )}
          </div>
          <LoadedSwitcher current={dataset} />
        </div>
        <Facts
          className="mt-6"
          facts={[
            { label: "Linhas", value: formatCount(dataset.n_rows) },
            { label: "Colunas", value: formatCount(dataset.n_columns) },
            { label: "Avisos da validação", value: formatCount(warnings) },
          ]}
        />
        <NoticeList notices={validation.notices} className="mt-6" />
        {!validation.is_usable ? (
          <p role="alert" className="mt-4 max-w-[68ch] font-serif text-reading text-destructive">
            Estes dados não podem seguir para as próximas etapas: corrige os erros acima e carrega o
            ficheiro outra vez.
          </p>
        ) : (
          validation.notices.length === 0 && (
            <p className="mt-4 font-serif text-reading text-muted-foreground">
              Os dados passaram a validação sem avisos.
            </p>
          )
        )}
        {validation.is_usable && (
          <Tabs defaultValue="preview" className="mt-8">
            <TabsList variant="line" aria-label="O que se leu">
              <TabsTrigger value="preview">Primeiras linhas</TabsTrigger>
              <TabsTrigger value="columns">Colunas e tipos</TabsTrigger>
            </TabsList>
            <TabsContent value="preview" className="mt-4">
              <Preview datasetId={dataset.id} />
            </TabsContent>
            <TabsContent value="columns" className="mt-4">
              <Columns dataset={dataset} />
            </TabsContent>
          </Tabs>
        )}
      </section>
      {validation.is_usable && <NextStep />}
    </>
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

function Columns({ dataset }: { dataset: DatasetSummary }) {
  const label = useLabel();
  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="px-3">Coluna</TableHead>
            <TableHead className="px-3">Tipo inferido</TableHead>
            <TableHead className="px-3">Porquê</TableHead>
            <TableHead className="px-3">Tipo no pandas</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {dataset.columns.map((column) => (
            <TableRow key={column.name} className="hover:bg-transparent">
              <TableCell className="px-3 font-medium">{column.name}</TableCell>
              <TableCell className="px-3">{label("ColumnKind", column.kind)}</TableCell>
              <TableCell className="min-w-[18rem] px-3 font-serif whitespace-normal text-muted-foreground">
                {column.kind_reason}
              </TableCell>
              <TableCell className="px-3 text-muted-foreground">{column.dtype}</TableCell>
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
      <label htmlFor="datasets-carregados" className="text-xs text-muted-foreground">
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

const NEXT_STEPS: readonly {
  path: WorkPath;
  title: string;
  text: string;
  to: "/explorar" | "/agentes";
  action: string;
}[] = [
  {
    path: "manual",
    title: "À mão",
    text: "Exploras, preparas e escolhes o modelo tu, e vês o treino por dentro.",
    to: "/explorar",
    action: "Explorar este dataset",
  },
  {
    path: "agents",
    title: "Com os agentes",
    text: "Dizes o que queres prever, e os agentes procuram o melhor modelo.",
    to: "/agentes",
    action: "Pedir aos agentes",
  },
];

/** Os dois caminhos depois dos dados; o escolhido no início vem primeiro. */
function NextStep() {
  const topic = useTopic("automatic_or_manual");
  const path = useSession((state) => state.path);
  const choosePath = useSession((state) => state.choosePath);
  const navigate = useNavigate();
  const steps = [...NEXT_STEPS].sort((a, b) => Number(b.path === path) - Number(a.path === path));
  return (
    <Section title={topic?.title ?? "E agora?"} className="mt-14">
      {topic !== undefined && <Prose text={topic.text} />}
      <div className="mt-6 grid gap-10 @3xl/main:grid-cols-2">
        {steps.map((step, index) => (
          <div key={step.path} className="flex flex-col items-start gap-3 border-t pt-5">
            <h3 className="text-lg font-medium">{step.title}</h3>
            <p className="max-w-[52ch] font-serif text-reading text-muted-foreground">
              {step.text}
            </p>
            <Button
              asChild
              variant={index === 0 ? "default" : "outline"}
              onClick={(event) => {
                event.preventDefault();
                choosePath(step.path);
                void navigate({ to: step.to });
              }}
            >
              <Link to={step.to}>{step.action}</Link>
            </Button>
          </div>
        ))}
      </div>
    </Section>
  );
}
