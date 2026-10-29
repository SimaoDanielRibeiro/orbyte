/**
 * O ecrã Explorar: o perfil do dataset aberto, para decidir o que limpar e como transformar.
 *
 * O perfil lê-se por partes, todas à vista e pela ordem em que se percorrem (ADR 0037):
 *
 * 1. o resumo: os números do dataset e os tipos das colunas (uma faixa com uma marca por coluna,
 *    pela ordem da tabela);
 * 2. os avisos, do dataset e de cada coluna;
 * 3. as colunas, uma a uma;
 * 4. as correlações;
 * 5. os valores em falta;
 * 6. o alvo.
 *
 * Por cima das partes há um índice que fica preso ao topo ao descer a página: diz em que parte se
 * está (o destaque desliza para ela, como na barra lateral) e salta para qualquer outra. Não há
 * números nem subtítulos: só os títulos (ADR 0038). O "como se lê" de cada gráfico fica no "?".
 */
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { cn } from "cn";
import { motion, useReducedMotion } from "motion/react";
import { useEffect, useState, type ReactNode } from "react";

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
import { Explain } from "@/components/explain/explain";
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

/** As partes do perfil, pela ordem em que se leem: o índice e as secções saem daqui. */
const PARTS = [
  { id: "resumo", title: "Resumo", topic: "profiling" },
  { id: "avisos", title: "Avisos" },
  { id: "colunas", title: "Colunas" },
  { id: "correlacoes", title: "Correlações", topic: "correlation_matrix" },
  { id: "em-falta", title: "Valores em falta", topic: "missing_map" },
  { id: "alvo", title: "Alvo", topic: "class_balance" },
] as const;

type PartId = (typeof PARTS)[number]["id"];

const PART_IDS: readonly PartId[] = PARTS.map((part) => part.id);

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
  },
  agents: {
    to: "/agentes",
    action: "Pedir aos agentes",
  },
} as const;

function Explore({ dataset }: { dataset: DatasetSummary }) {
  const preparation = useSession((state) => state.preparation);
  const next = NEXT[useSession((state) => state.path)];
  const reduced = useReducedMotion() ?? false;
  const [target, setTarget] = useState<string | null>(
    preparation?.datasetId === dataset.id
      ? (preparation.config.target ?? null)
      : dataset.suggested_target,
  );
  const [column, setColumn] = useState<string>(dataset.columns[0]?.name ?? "");
  const profile = useQuery({
    ...datasetsProfileOptions({
      path: { dataset_id: dataset.id },
      query: target === null ? undefined : { target },
    }),
    // Ao trocar de alvo, o perfil anterior fica à vista até chegar o novo.
    placeholderData: keepPreviousData,
  });
  const suggestion = useQuery({
    ...datasetsSuggestTaskOptions({
      path: { dataset_id: dataset.id },
      query: target === null ? undefined : { target },
    }),
    enabled: target !== null,
  });

  function go(id: PartId) {
    const part = document.getElementById(id);
    part?.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
    // Quem usa o teclado continua na parte para onde saltou.
    part?.querySelector<HTMLElement>("h2")?.focus({ preventScroll: true });
  }

  const notices =
    profile.data === undefined
      ? 0
      : profile.data.notices.length +
        profile.data.columns.reduce((total, one) => total + one.notices.length, 0);

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
        <>
          <PartNav onGo={go} />
          <div
            className={cn(profile.isPlaceholderData && "stale")}
            aria-busy={profile.isPlaceholderData}
          >
            <Part id="resumo">
              <Overview
                profile={profile.data}
                onColumn={(name) => {
                  setColumn(name);
                  go("colunas");
                }}
              />
            </Part>
            <Part
              id="avisos"
              explain={
                notices === 0
                  ? "O perfil não encontrou nada que mereça atenção."
                  : `${formatQuantity(notices, "aviso", "avisos")} sobre os dados, do dataset e de cada coluna.`
              }
            >
              <Notices profile={profile.data} />
            </Part>
            <Part id="colunas">
              <Columns profile={profile.data} selected={column} onSelect={setColumn} />
            </Part>
            <Part id="correlacoes">
              <Correlations profile={profile.data} />
            </Part>
            <Part id="em-falta">
              <Missing profile={profile.data} />
            </Part>
            {/* A tarefa que o alvo pede, e porquê, fica no "?" da parte. */}
            <Part id="alvo" explain={suggestion.data?.reason}>
              <Target profile={profile.data} />
            </Part>
          </div>
          <NextStep>
            <Proceed to={next.to}>{next.action}</Proceed>
          </NextStep>
        </>
      )}
    </>
  );
}

/**
 * A parte em que se está: a última cujo título já passou por baixo do índice (ou a última de
 * todas, no fim da página). Lê-se ao descer a página, uma vez por fotograma.
 */
function useActivePart(): PartId {
  const [active, setActive] = useState<PartId>("resumo");
  useEffect(() => {
    let frame = 0;
    const update = () => {
      frame = 0;
      let current: PartId = "resumo";
      for (const id of PART_IDS) {
        const top = document.getElementById(id)?.getBoundingClientRect().top;
        if (top !== undefined && top <= 170) {
          current = id;
        }
      }
      const root = document.documentElement;
      if (window.innerHeight + window.scrollY >= root.scrollHeight - 4 && root.scrollTop > 0) {
        current = PART_IDS.at(-1) ?? current;
      }
      setActive(current);
    };
    const schedule = () => {
      if (frame === 0) {
        frame = requestAnimationFrame(update);
      }
    };
    schedule();
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
    };
  }, []);
  return active;
}

/**
 * O índice das partes, preso ao topo ao descer: só o nome de cada uma. O destaque da parte em que
 * se está desliza de uma para a outra.
 */
function PartNav({ onGo }: { onGo: (id: PartId) => void }) {
  const active = useActivePart();
  return (
    <nav
      aria-label="As partes do perfil"
      className="sticky top-14 z-20 -mx-4 mt-8 bg-ground/90 px-4 py-3 shadow-[0_10px_18px_-14px_rgb(0_0_0/0.9)] backdrop-blur-md md:top-0 md:-mx-[2cm] md:px-[2cm]"
    >
      <ol className="flex [scrollbar-width:none] gap-1 overflow-x-auto">
        {PARTS.map((part) => {
          const current = part.id === active;
          return (
            <li key={part.id} className="relative shrink-0">
              {current && (
                <motion.span
                  aria-hidden="true"
                  layoutId="explorar-parte"
                  transition={{ type: "spring", bounce: 0.2, visualDuration: 0.38 }}
                  className="absolute inset-0 rounded-[10px] border border-[#3a3a3a] bg-[#262626] shadow-[inset_0_1px_0_rgb(255_255_255/0.06)]"
                />
              )}
              <a
                href={`#${part.id}`}
                aria-current={current ? "location" : undefined}
                onClick={(event) => {
                  event.preventDefault();
                  onGo(part.id);
                }}
                className={cn(
                  "relative flex h-9 items-center rounded-[10px] px-3.5 text-sm whitespace-nowrap transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  current ? "text-strong" : "text-soft hover:text-strong",
                )}
              >
                {part.title}
              </a>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

/** Uma parte do perfil: o título (com o "?") e o conteúdo. */
function Part({
  id,
  explain,
  children,
}: {
  id: PartId;
  /** Uma explicação montada aqui, quando a parte não tem tema no backend. */
  explain?: ReactNode;
  children: ReactNode;
}) {
  const part = PARTS[PART_IDS.indexOf(id)];
  const topic = part !== undefined && "topic" in part ? part.topic : undefined;
  return (
    <section
      id={id}
      aria-labelledby={`${id}-titulo`}
      className="scroll-mt-32 pt-16 first:pt-10 md:scroll-mt-20"
    >
      <div className="mb-6 flex items-center gap-1.5">
        <h2 id={`${id}-titulo`} tabIndex={-1} className="text-xl outline-none">
          {part?.title}
        </h2>
        <Explain topic={topic} title={topic === undefined ? part?.title : undefined}>
          {explain}
        </Explain>
      </div>
      {children}
    </section>
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
    <>
      <StatStrip items={stats.map((stat) => ({ key: stat.label, ...stat }))} />
      <KindStrip profile={profile} onColumn={onColumn} />
    </>
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
    <div className="flex flex-col gap-4">
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
    </div>
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

function Target({ profile }: { profile: DatasetProfile }) {
  const target = profile.target;
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
