/**
 * Uma coluna em pormenor: o tipo e porquê, as estatísticas, a distribuição e os avisos.
 */
import type { ColumnProfile } from "@/api/generated";
import { useLabel } from "@/api/vocabulary";
import { BarChart, Histogram } from "@/components/charts/bar-chart";
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
        <h3 className="text-2xl">{column.name}</h3>
        <p className="mt-1 max-w-[62ch] font-serif text-reading text-muted-foreground">
          {label("ColumnKind", column.kind)}: {column.kind_reason}
        </p>
      </div>
      <Facts
        size="sm"
        facts={[
          {
            label: "Em falta",
            value: `${formatCount(column.n_missing)} (${formatPercent(missing)})`,
          },
          { label: "Valores distintos", value: formatCount(column.n_unique) },
          { label: "Tipo no pandas", value: column.dtype },
          ...(column.date_range === null
            ? []
            : [{ label: "Datas", value: `${column.date_range[0]} a ${column.date_range[1]}` }]),
        ]}
      />
      {numeric !== null && (
        <Facts
          size="sm"
          facts={[
            { label: "Média", value: formatStatistic(numeric.mean) },
            { label: "Desvio padrão", value: formatStatistic(numeric.std) },
            { label: "Mínimo", value: formatStatistic(numeric.minimum) },
            { label: "1.º quartil", value: formatStatistic(numeric.q1) },
            { label: "Mediana", value: formatStatistic(numeric.median) },
            { label: "3.º quartil", value: formatStatistic(numeric.q3) },
            { label: "Máximo", value: formatStatistic(numeric.maximum) },
            ...(numeric.skewness === null
              ? []
              : [{ label: "Assimetria", value: formatNumber(numeric.skewness) }]),
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
        description={`A distribuição de ${column.name}: quantas linhas caem em cada intervalo.`}
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
      valueLabel="linhas"
      description={`Os valores mais frequentes de ${column.name} e quantas linhas tem cada um.`}
      caption={
        column.n_unique > column.top_values.length
          ? `Os ${formatCount(column.top_values.length)} valores mais frequentes, de ${formatQuantity(column.n_unique, "valor distinto", "valores distintos")}.`
          : undefined
      }
      labelWidth={150}
    />
  );
}
