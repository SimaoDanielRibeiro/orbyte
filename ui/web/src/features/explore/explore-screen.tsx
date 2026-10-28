/**
 * O ecrã Explorar: o perfil do dataset aberto, para decidir o que limpar e como transformar.
 *
 * Em cima, os números em cartões e os tipos das colunas (uma faixa com uma marca por coluna,
 * pela ordem da tabela); depois, os avisos; em baixo, quatro separadores: as colunas uma a uma,
 * as correlações, os valores em falta e o alvo. O "como se lê" de cada gráfico fica no "?".
 */
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { cn } from "cn";
import { useState } from "react";

import type { ColumnKind, DatasetProfile, DatasetSummary } from "@/api/generated";
import {
  datasetsProfileOptions,
  datasetsSuggestTaskOptions,
} from "@/api/generated/@tanstack/react-query.gen";
import { useLabel, useVocabulary } from "@/api/vocabulary";
import { BarChart } from "@/components/charts/bar-chart";
import { Swatch } from "@/components/charts/chart";
import { DotPlot } from "@/components/charts/dot-plot";
import { Heatmap } from "@/components/charts/heatmap";
import { MissingMapView } from "@/components/charts/missing-map";
import { TopicText } from "@/components/markdown/topic-text";
import { NoticeList } from "@/components/notices/notice-list";
import { ErrorNote, Facts, NextStep, Panel, Proceed, StatStrip } from "@/components/shell/blocks";
import { Page, PageHeader } from "@/components/shell/page";
import { SCREENS } from "@/components/shell/screens";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { PageSkeleton, RequireDataset } from "@/features/data/require-dataset";
import { classColors, classOrder } from "@/lib/classes";
import { categoryColor, diverging, withAlpha } from "@/lib/colors";
import {
  formatCount,
  formatNumber,
  formatPercent,
  formatQuantity,
  formatStatistic,
} from "@/lib/format";
import { useSession } from "@/state/session";

import { ColumnDetail } from "./column-detail";

/** O valor do "sem alvo" na lista (o Radix não aceita um valor vazio). */
const NO_TARGET = "__sem_alvo__";

type Tab = "columns" | "correlations" | "missing" | "target";

export function ExploreScreen() {
  return (
    <Page>
      <PageHeader
        title={SCREENS.explore.title}
        description={SCREENS.explore.description}
        topics={["profiling", "correlation_matrix", "missing_map", "class_balance", "group_means"]}
      />
      <RequireDataset>{(dataset) => <Explore key={dataset.id} dataset={dataset} />}</RequireDataset>
    </Page>
  );
}

/**
 * O passo seguinte depende da interface: à mão, preparar os dados; com os agentes, fazer-lhes o
 * pedido (o Explorar é um dos primeiros passos, comum às duas; ADR 0036).
 */
const NEXT = {
  manual: {
    to: "/preparar",
    action: "Preparar os dados",
    text: "Já conheces os dados. A seguir, escolhe o alvo e prepara-os para o modelo: limpar, transformar e dividir.",
  },
  agents: {
    to: "/agentes",
    action: "Pedir aos agentes",
    text: "Já conheces os dados. A seguir, diz aos agentes o que queres prever, e eles experimentam os modelos por ti.",
  },
} as const;

function Explore({ dataset }: { dataset: DatasetSummary }) {
  const preparation = useSession((state) => state.preparation);
  const next = NEXT[useSession((state) => state.path)];
  const [target, setTarget] = useState<string | null>(
    preparation?.datasetId === dataset.id
      ? (preparation.config.target ?? null)
      : dataset.suggested_target,
  );
  const [tab, setTab] = useState<Tab>("columns");
  const [column, setColumn] = useState<string>(dataset.columns[0]?.name ?? "");
  const profile = useQuery({
    ...datasetsProfileOptions({
      path: { dataset_id: dataset.id },
      query: target === null ? undefined : { target },
    }),
    // Ao trocar de alvo, o perfil anterior fica à vista até chegar o novo.
    placeholderData: keepPreviousData,
  });

  return (
    <>
      <div className="mt-8 flex flex-wrap items-end justify-between gap-x-8 gap-y-5">
        <div className="flex flex-wrap items-end gap-x-8 gap-y-4">
          <p className="pb-2.5 text-[0.9375rem] text-soft">
            A explorar <span className="font-medium text-strong">{dataset.name}</span>
          </p>
          <TargetPicker dataset={dataset} target={target} onChange={setTarget} />
        </div>
        <Proceed to={next.to}>{next.action}</Proceed>
      </div>
      <ErrorNote error={profile.error} className="mt-6" />
      {profile.data === undefined ? (
        profile.error === null && <PageSkeleton />
      ) : (
        <div
          className={cn(profile.isPlaceholderData && "stale")}
          aria-busy={profile.isPlaceholderData}
        >
          <Overview
            profile={profile.data}
            onColumn={(name) => {
              setColumn(name);
              setTab("columns");
            }}
          />
          <Notices profile={profile.data} />
          <Tabs
            value={tab}
            onValueChange={(value) => {
              setTab(value as Tab);
            }}
            className="mt-16"
          >
            <TabsList
              aria-label="O perfil por partes"
              className="flex-wrap group-data-horizontal/tabs:h-auto"
            >
              <TabsTrigger value="columns">Colunas</TabsTrigger>
              <TabsTrigger value="correlations">Correlações</TabsTrigger>
              <TabsTrigger value="missing">Valores em falta</TabsTrigger>
              <TabsTrigger value="target">Alvo</TabsTrigger>
            </TabsList>
            <TabsContent value="columns" className="mt-4">
              <Columns profile={profile.data} selected={column} onSelect={setColumn} />
            </TabsContent>
            <TabsContent value="correlations" className="mt-4">
              <Correlations profile={profile.data} />
            </TabsContent>
            <TabsContent value="missing" className="mt-4">
              <Missing profile={profile.data} />
            </TabsContent>
            <TabsContent value="target" className="mt-4">
              <Target dataset={dataset} profile={profile.data} />
            </TabsContent>
          </Tabs>
          <NextStep text={next.text}>
            <Proceed to={next.to}>{next.action}</Proceed>
          </NextStep>
        </div>
      )}
    </>
  );
}

interface TargetPickerProps {
  dataset: DatasetSummary;
  target: string | null;
  onChange: (target: string | null) => void;
}

/** A coluna-alvo: muda o separador Alvo (as classes e as médias por classe). */
function TargetPicker({ dataset, target, onChange }: TargetPickerProps) {
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor="explorar-alvo" className="mono-label">
        Coluna-alvo
      </label>
      <Select
        value={target ?? NO_TARGET}
        onValueChange={(value) => {
          onChange(value === NO_TARGET ? null : value);
        }}
      >
        <SelectTrigger id="explorar-alvo" className="min-w-56">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={NO_TARGET}>Sem alvo</SelectItem>
          {dataset.columns.map((column) => (
            <SelectItem key={column.name} value={column.name}>
              {column.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

/** Os números do dataset e a faixa dos tipos das colunas. */
function Overview({
  profile,
  onColumn,
}: {
  profile: DatasetProfile;
  onColumn: (name: string) => void;
}) {
  const cells = profile.n_rows * profile.n_columns;
  const stats = [
    { label: "Linhas", value: formatCount(profile.n_rows) },
    { label: "Colunas", value: formatCount(profile.n_columns) },
    {
      label: "Células em falta",
      value: formatPercent(cells === 0 ? 0 : profile.n_missing_cells / cells),
    },
    { label: "Linhas duplicadas", value: formatCount(profile.n_duplicate_rows) },
    { label: "Memória", value: `${formatNumber(profile.memory_bytes / 1024 ** 2, 1)} MB` },
  ];
  return (
    <section className="mt-6" aria-label="Resumo">
      <StatStrip items={stats.map((stat) => ({ key: stat.label, ...stat }))} />
      <KindStrip profile={profile} onColumn={onColumn} />
    </section>
  );
}

/**
 * Uma marca por coluna, pela ordem da tabela, com a cor do tipo inferido: vê-se de relance de
 * que é feito o dataset. Cada marca abre a coluna em pormenor.
 */
function KindStrip({
  profile,
  onColumn,
}: {
  profile: DatasetProfile;
  onColumn: (name: string) => void;
}) {
  const vocabulary = useVocabulary();
  const label = useLabel();
  // As cores seguem a ordem dos tipos no vocabulário: o mesmo tipo tem sempre a mesma cor.
  const kinds = Object.keys(vocabulary.labels.ColumnKind ?? {}) as ColumnKind[];
  const color = (kind: ColumnKind) => categoryColor(kinds.indexOf(kind));
  const counts = kinds
    .map((kind) => ({ kind, count: profile.columns.filter((one) => one.kind === kind).length }))
    .filter((entry) => entry.count > 0);

  return (
    <Panel title="As colunas pelo tipo inferido" className="mt-12">
      <div className="flex h-7 gap-1" role="list" aria-label="As colunas pelo tipo inferido">
        {profile.columns.map((column) => (
          <Tooltip key={column.name}>
            <TooltipTrigger asChild>
              <button
                type="button"
                role="listitem"
                aria-label={`${column.name}: ${label("ColumnKind", column.kind)}`}
                onClick={() => {
                  onColumn(column.name);
                }}
                className="h-full min-w-1.5 flex-1 rounded-[4px] outline-none hover:opacity-80 focus-visible:ring-3 focus-visible:ring-ring/50"
                style={{
                  background: color(column.kind),
                  boxShadow: `0 0 10px ${withAlpha(color(column.kind), 0.35)}`,
                }}
              />
            </TooltipTrigger>
            <TooltipContent>
              {column.name}: {label("ColumnKind", column.kind).toLowerCase()}
            </TooltipContent>
          </Tooltip>
        ))}
      </div>
      <ul className="flex flex-wrap gap-x-6 gap-y-1.5 text-xs text-soft">
        {counts.map(({ kind, count }) => (
          <li key={kind} className="flex items-center gap-2">
            <Swatch color={color(kind)} kind="dot" />
            {label("ColumnKind", kind)}
            <span className="font-mono text-dim tabular-nums">{formatCount(count)}</span>
          </li>
        ))}
      </ul>
    </Panel>
  );
}

/** Quantos avisos se mostram antes de "Mostrar todos". */
const FIRST_NOTICES = 5;

/** Os avisos do dataset e depois os de cada coluna, pela ordem das colunas. */
function Notices({ profile }: { profile: DatasetProfile }) {
  const [all, setAll] = useState(false);
  const notices = [...profile.notices, ...profile.columns.flatMap((column) => column.notices)];
  const hidden = notices.length - FIRST_NOTICES;
  return (
    <Panel
      as="h2"
      title="Avisos"
      explain={
        notices.length === 0
          ? "O perfil não encontrou nada que mereça atenção."
          : `${formatQuantity(notices.length, "aviso", "avisos")} sobre os dados, do dataset e de cada coluna.`
      }
      actions={
        <span className="font-mono text-xs text-dim tabular-nums">
          {formatCount(notices.length)}
        </span>
      }
      className="mt-12"
    >
      {notices.length === 0 ? (
        <p className="text-sm text-soft">Nada que mereça atenção.</p>
      ) : (
        <NoticeList
          notices={all || hidden <= 1 ? notices : notices.slice(0, FIRST_NOTICES)}
          className="block gap-x-12 @5xl/main:columns-2 [&>li]:mb-3 [&>li]:break-inside-avoid"
        />
      )}
      {hidden > 1 && (
        <Button
          variant="ghost"
          size="sm"
          className="-ml-3 self-start"
          aria-expanded={all}
          onClick={() => {
            setAll(!all);
          }}
        >
          {all ? "Mostrar só os primeiros" : `Mostrar os outros ${formatCount(hidden)} avisos`}
        </Button>
      )}
    </Panel>
  );
}

interface ColumnsProps {
  profile: DatasetProfile;
  selected: string;
  onSelect: (name: string) => void;
}

/** A tabela das colunas e, ao lado, a escolhida em pormenor. */
function Columns({ profile, selected, onSelect }: ColumnsProps) {
  const label = useLabel();
  const column = profile.columns.find((one) => one.name === selected) ?? profile.columns[0] ?? null;
  return (
    <div className="grid gap-x-12 gap-y-10 @5xl/main:grid-cols-[minmax(0,28rem)_minmax(0,1fr)]">
      <div className="relative self-start overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left">
              <th scope="col" className="px-3 py-2.5 font-mono text-xs font-normal text-dim">
                Coluna
              </th>
              <th scope="col" className="px-3 py-2.5 font-mono text-xs font-normal text-dim">
                Tipo
              </th>
              <th
                scope="col"
                className="px-3 py-2.5 text-right font-mono text-xs font-normal text-dim"
              >
                Em falta
              </th>
              <th
                scope="col"
                className="px-3 py-2.5 text-right font-mono text-xs font-normal text-dim"
              >
                Distintos
              </th>
            </tr>
          </thead>
          <tbody>
            {profile.columns.map((one) => {
              const isSelected = one.name === column?.name;
              return (
                <tr
                  key={one.name}
                  className={cn(
                    "relative transition-colors hover:bg-surface-high/50",
                    isSelected && "bg-surface-high hover:bg-surface-high",
                  )}
                >
                  <td className="rounded-l-lg px-3 py-2">
                    <button
                      type="button"
                      aria-pressed={isSelected}
                      onClick={() => {
                        onSelect(one.name);
                      }}
                      className={cn(
                        "text-left outline-none after:absolute after:inset-0 after:rounded-lg focus-visible:after:ring-2 focus-visible:after:ring-ring",
                        isSelected ? "font-medium text-strong" : "text-foreground",
                      )}
                    >
                      {one.name}
                    </button>
                  </td>
                  <td className="px-3 py-2 text-soft">{label("ColumnKind", one.kind)}</td>
                  <td className="px-3 py-2 text-right tabular-nums">
                    {one.n_missing === 0 ? (
                      <span className="text-dim">0</span>
                    ) : (
                      formatPercent(one.n_missing / one.n_rows)
                    )}
                  </td>
                  <td className="rounded-r-lg px-3 py-2 text-right tabular-nums">
                    {formatCount(one.n_unique)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {column !== null && (
        <div className="min-w-0">
          <ColumnDetail key={column.name} column={column} />
        </div>
      )}
    </div>
  );
}

function Correlations({ profile }: { profile: DatasetProfile }) {
  const matrix = profile.correlations;
  if (matrix === null) {
    return (
      <p className="text-sm text-soft">
        São precisas pelo menos duas colunas numéricas para medir correlações.
      </p>
    );
  }
  const labelWidth = 140;
  return (
    <Panel title="Correlação de Pearson">
      <div style={{ maxWidth: labelWidth + matrix.columns.length * 72 }}>
        <Heatmap
          rows={matrix.columns}
          columns={matrix.columns}
          values={matrix.values}
          color={diverging}
          format={(value) => formatNumber(value, 2)}
          description="A correlação de Pearson entre cada par de colunas numéricas."
          caption={<TopicText topic="correlation_matrix" />}
          labelWidth={labelWidth}
          cellHeight={Math.min(40, Math.max(28, 360 / matrix.columns.length))}
        />
      </div>
    </Panel>
  );
}

function Missing({ profile }: { profile: DatasetProfile }) {
  if (profile.n_missing_cells === 0) {
    return <p className="text-sm text-soft">Não há valores em falta.</p>;
  }
  const withMissing = profile.columns
    .filter((column) => column.n_missing > 0)
    .sort((a, b) => b.n_missing / b.n_rows - a.n_missing / a.n_rows);
  const complete = profile.columns.length - withMissing.length;
  return (
    <div className="grid gap-x-12 gap-y-10 @5xl/main:grid-cols-2">
      <Panel title="A parte em falta de cada coluna">
        <BarChart
          items={withMissing.map((column) => ({
            key: column.name,
            label: column.name,
            value: column.n_missing / column.n_rows,
          }))}
          format={(value) => formatPercent(value)}
          valueLabel="dos valores em falta"
          max={1}
          description="A percentagem de valores em falta em cada coluna que tem nulos."
          caption={
            complete > 0
              ? `${complete === 1 ? "A outra coluna não tem" : `As outras ${formatCount(complete)} colunas não têm`} valores em falta.`
              : undefined
          }
        />
      </Panel>
      <Panel title="Onde estão os nulos">
        <MissingMapView
          map={profile.missing_map}
          description="O mapa dos valores em falta: as colunas na horizontal e as linhas na vertical."
          caption={
            <>
              <p className="mb-1">
                {formatQuantity(
                  profile.missing_map.rows.length,
                  "linha mostrada",
                  "linhas mostradas",
                )}{" "}
                de {formatCount(profile.n_rows)}.
              </p>
              <TopicText topic="missing_map" />
            </>
          }
        />
      </Panel>
    </div>
  );
}

function Target({ dataset, profile }: { dataset: DatasetSummary; profile: DatasetProfile }) {
  const target = profile.target;
  const suggestion = useQuery({
    ...datasetsSuggestTaskOptions({
      path: { dataset_id: dataset.id },
      query: target === null ? undefined : { target },
    }),
    enabled: target !== null,
  });
  if (target === null) {
    return (
      <p className="text-sm text-soft">
        Escolhe uma coluna-alvo, no topo do ecrã, para ver as classes.
      </p>
    );
  }
  const column = profile.columns.find((one) => one.name === target);
  return (
    <div className="flex flex-col gap-4">
      {suggestion.data !== undefined && (
        <p className="flex items-center gap-2 text-sm text-soft">
          <span aria-hidden="true" className="size-1.5 shrink-0 rounded-full bg-strong" />
          {suggestion.data.reason}
        </p>
      )}
      {profile.class_balance === null ? (
        column !== undefined && (
          <div>
            <ColumnDetail column={column} />
          </div>
        )
      ) : (
        <Classes profile={profile} />
      )}
    </div>
  );
}

/** O equilíbrio das classes e a média de cada feature numérica em cada classe. */
function Classes({ profile }: { profile: DatasetProfile }) {
  const balance = profile.class_balance;
  if (balance === null) {
    return null;
  }
  const color = classColors(classOrder(balance.classes.map((value) => value.value)));
  const ratios = new Map(balance.classes.map((value) => [value.value, value.ratio]));
  const largest = balance.classes[0]?.count ?? 0;
  const smallest = balance.classes.at(-1)?.count ?? 0;
  const means = profile.group_means;

  return (
    <div className="grid gap-x-12 gap-y-10 @5xl/main:grid-cols-2">
      <Panel title={`As classes de ${balance.target}`}>
        <Facts
          size="sm"
          facts={[
            { label: "Classes", value: formatCount(balance.classes.length) },
            { label: "Equilíbrio", value: formatPercent(largest === 0 ? 0 : smallest / largest) },
          ]}
        />
        <BarChart
          items={balance.classes.map((value) => ({
            key: value.value,
            label: value.value,
            value: value.count,
            color: color(value.value),
          }))}
          format={(count, item) =>
            `${formatCount(count)} (${formatPercent(ratios.get(item.key) ?? 0)})`
          }
          valueLabel="linhas"
          description={`Quantas linhas tem cada classe de ${balance.target}.`}
          caption={<TopicText topic="class_balance" />}
        />
      </Panel>
      {means !== null && (
        <Panel title="A média de cada feature por classe">
          <div>
            <DotPlot
              series={means.groups.map((group) => ({
                key: group,
                label: group,
                color: color(group),
              }))}
              rows={means.columns.map((name, index) => {
                const summary = profile.columns.find((one) => one.name === name)?.numeric;
                return {
                  key: name,
                  label: name,
                  min: summary?.minimum ?? 0,
                  max: summary?.maximum ?? 1,
                  values: means.means.map((row) => row[index] ?? null),
                };
              })}
              format={formatStatistic}
              description={`A média de cada feature numérica em cada classe de ${means.target}, entre o mínimo e o máximo da feature.`}
              caption={<TopicText topic="group_means" />}
            />
          </div>
          <details className="text-sm">
            <summary className="cursor-pointer text-soft hover:text-strong">Ver os números</summary>
            <div className="relative mt-3 overflow-x-auto rounded-xl border">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th
                      scope="col"
                      className="px-4 py-2.5 text-left font-mono text-xs font-normal text-dim"
                    >
                      Classe
                    </th>
                    {means.columns.map((name) => (
                      <th
                        key={name}
                        scope="col"
                        className="px-4 py-2.5 text-right font-mono text-xs font-normal text-dim"
                      >
                        {name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {means.groups.map((group, row) => (
                    <tr key={group} className="border-b border-border/60 last:border-0">
                      <th scope="row" className="px-4 py-2 text-left font-medium text-strong">
                        {group}
                      </th>
                      {means.columns.map((name, index) => {
                        const value = means.means[row]?.[index] ?? null;
                        return (
                          <td key={name} className="px-4 py-2 text-right tabular-nums">
                            {value === null ? "—" : formatStatistic(value)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </Panel>
      )}
    </div>
  );
}
