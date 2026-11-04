/**
 * Uma coluna em pormenor: o tipo (o porquê fica no "?"), as estatísticas, a distribuição e os
 * avisos.
 */
import type { ColumnProfile } from "@/api/generated";
import { useLabel } from "@/api/vocabulary";
import { BarChart, Histogram } from "@/components/charts/bar-chart";
import { Explain } from "@/components/explain/explain";
import { NoticeList } from "@/components/notices/notice-list";
import { Facts } from "@/components/shell/blocks";
import { CATEGORICAL } from "@/lib/colors";
import {
  formatCount,
  formatNumber,
  formatPercent,
  formatQuantity,
  formatStatistic,
  formatValue,
} from "@/lib/format";

export function ColumnDetail({ column }: { column: ColumnProfile }) {
  const label = useLabel();
  const { numeric } = column;
  const missing = column.n_rows === 0 ? 0 : column.n_missing / column.n_rows;

  return (
    <div className="flex min-w-0 flex-col gap-6">
      <div>
        <h3 className="text-xl">{column.name}</h3>
        <p className="mt-1 flex items-center gap-1 text-sm text-soft">
          {label("ColumnKind", column.kind)}
          <Explain title={`Why “${label("ColumnKind", column.kind)}”`}>
            {column.kind_reason}
          </Explain>
        </p>
      </div>
      <Facts
        size="sm"
        facts={[
          {
            label: "Missing",
            value: `${formatCount(column.n_missing)} (${formatPercent(missing)})`,
          },
          { label: "Distinct values", value: formatCount(column.n_unique) },
          { label: "pandas type", value: column.dtype },
          ...(column.date_range === null
            ? []
            : [{ label: "Dates", value: `${column.date_range[0]} to ${column.date_range[1]}` }]),
        ]}
      />
      {numeric !== null && (
        <Facts
          size="sm"
          facts={[
            { label: "Mean", value: formatStatistic(numeric.mean) },
            { label: "Standard deviation", value: formatStatistic(numeric.std) },
            { label: "Minimum", value: formatStatistic(numeric.minimum) },
            { label: "1st quartile", value: formatStatistic(numeric.q1) },
            { label: "Median", value: formatStatistic(numeric.median) },
            { label: "3rd quartile", value: formatStatistic(numeric.q3) },
            { label: "Maximum", value: formatStatistic(numeric.maximum) },
            ...(numeric.skewness === null
              ? []
              : [{ label: "Skewness", value: formatNumber(numeric.skewness) }]),
          ]}
        />
      )}
      <Distribution column={column} />
      <NoticeList notices={column.notices} />
    </div>
  );
}

function Distribution({ column }: { column: ColumnProfile }) {
  // Um identificador tem um valor diferente em cada linha: não há distribuição para desenhar
  // (o aviso da coluna diz porquê).
  if (column.kind === "identifier") {
    return null;
  }
  if (column.histogram !== null) {
    return (
      <Histogram
        edges={column.histogram.edges}
        series={[{ label: column.name, color: CATEGORICAL[0], counts: column.histogram.counts }]}
        formatEdge={(value) => formatValue(value)}
        description={`The distribution of ${column.name}: how many rows fall in each interval.`}
        xLabel={column.name}
      />
    );
  }
  if (column.top_values.length === 0) {
    return null;
  }
  const ratios = new Map(column.top_values.map((value) => [value.value, value.ratio]));
  return (
    <BarChart
      items={column.top_values.map((value) => ({
        key: value.value,
        label: value.value,
        value: value.count,
      }))}
      format={(count, item) =>
        `${formatCount(count)} (${formatPercent(ratios.get(item.key) ?? 0)})`
      }
      valueLabel="rows"
      description={`The most frequent values of ${column.name} and how many rows each one has.`}
      caption={
        column.n_unique > column.top_values.length
          ? `The ${formatCount(column.top_values.length)} most frequent values, out of ${formatQuantity(column.n_unique, "distinct value", "distinct values")}.`
          : undefined
      }
      labelWidth={150}
    />
  );
}
