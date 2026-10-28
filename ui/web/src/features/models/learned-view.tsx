/**
 * O que um modelo aprendeu, pronto a ver: os pesos de um modelo linear, a árvore, a importância
 * das features, as tabelas (os centros dos grupos, as médias de cada classe), as propriedades e
 * a rede neuronal.
 *
 * Serve a página de cada modelo (a demonstração) e o fim de um treino: os dois mandam o mesmo
 * `LearnedView` do backend, que já decidiu o que cada modelo tem para mostrar.
 */
import type { ReactNode } from "react";

import type { LearnedView, LinearWeights, Matrix, NetworkTrace } from "@/api/generated";
import { BarChart } from "@/components/charts/bar-chart";
import { Heatmap } from "@/components/charts/heatmap";
import { NetworkView } from "@/components/charts/network-view";
import { TreeDiagram } from "@/components/charts/tree-diagram";
import { TopicText } from "@/components/markdown/topic-text";
import { Facts } from "@/components/shell/blocks";
import { diverging } from "@/lib/colors";
import {
  formatList,
  formatNumber,
  formatPercent,
  formatStatistic,
  formatValue,
} from "@/lib/format";

/** Quantas features se mostram na importância (as mais importantes). */
const TOP_FEATURES = 20;

interface LearnedViewProps {
  learned: LearnedView;
  network: NetworkTrace | null;
  /** O passo do treino da rede a mostrar (o mesmo do campo, quando se revê o treino). */
  step?: number;
}

export function LearnedViewPanel({ learned, network, step }: LearnedViewProps) {
  const empty =
    learned.weights === null &&
    learned.importances.length === 0 &&
    learned.tree.length === 0 &&
    learned.tree_rules.length === 0 &&
    learned.tables.length === 0 &&
    learned.properties.length === 0 &&
    learned.notes.length === 0 &&
    network === null;
  if (empty) {
    return null;
  }
  return (
    <div className="flex flex-col gap-10">
      {learned.properties.length > 0 && (
        <Facts
          size="sm"
          facts={learned.properties.map((property) => ({
            label: property.label,
            value: formatValue(property.value),
          }))}
        />
      )}
      {learned.notes.length > 0 && (
        <div className="flex max-w-[68ch] flex-col gap-2 font-serif text-reading">
          {learned.notes.map((note) => (
            <p key={note}>{note}</p>
          ))}
        </div>
      )}
      {network !== null && (
        <Block title="A rede">
          <NetworkView
            network={network}
            step={step}
            description="As camadas da rede neuronal, com a ativação média de cada neurónio."
            caption={<TopicText topic="network_diagram" />}
          />
        </Block>
      )}
      {learned.weights !== null && <Weights weights={learned.weights} />}
      {learned.importances.length > 0 && (
        <Block title="A importância de cada feature">
          <BarChart
            items={learned.importances.slice(0, TOP_FEATURES).map(([feature, value]) => ({
              key: feature,
              label: feature,
              value,
            }))}
            format={(value) => formatPercent(value)}
            valueLabel="da importância"
            description="A importância de cada feature, pela redução das impurezas nas árvores."
            caption={<TopicText topic="impurity_importance" />}
          />
        </Block>
      )}
      {learned.tree.length > 0 ? (
        <Block title="A árvore">
          <TreeDiagram
            nodes={learned.tree}
            description="A árvore de decisão: as perguntas de cada nó e a previsão de cada folha."
            caption={<TopicText topic="tree_diagram" />}
          />
        </Block>
      ) : (
        learned.tree_rules.length > 0 && (
          <Block title="As regras da árvore">
            <pre className="relative overflow-x-auto rounded-md border bg-card p-3 text-xs leading-relaxed">
              {learned.tree_rules.join("\n")}
            </pre>
          </Block>
        )
      )}
      {learned.tables.map((table) => (
        <Table key={table.title} table={table} />
      ))}
    </div>
  );
}

function Block({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="flex min-w-0 flex-col gap-3">
      <h3 className="text-lg">{title}</h3>
      {children}
    </section>
  );
}

/** Os pesos de um modelo linear: barras com uma saída, um mapa de cores com várias. */
function Weights({ weights }: { weights: LinearWeights }) {
  const intercepts =
    weights.outputs.length === 1
      ? `A ordenada (b) é ${formatNumber(weights.intercepts[0] ?? 0, 3)}.`
      : `As ordenadas (b) são ${formatList(
          weights.outputs.map(
            (output, index) => `${formatNumber(weights.intercepts[index] ?? 0, 3)} em ${output}`,
          ),
        )}.`;
  const caption = (
    <>
      <p className="mb-1">{weights.how_to_read}</p>
      <p>{intercepts}</p>
    </>
  );
  if (weights.outputs.length === 1) {
    const row = weights.weights[0] ?? [];
    return (
      <Block title="Os pesos">
        <BarChart
          items={weights.features.map((feature, index) => ({
            key: feature,
            label: feature,
            value: row[index] ?? 0,
          }))}
          format={(value) => formatNumber(value, 3)}
          valueLabel="peso"
          description="O peso de cada feature no modelo linear."
          caption={caption}
        />
      </Block>
    );
  }
  const strongest = Math.max(1e-9, ...weights.weights.flat().map(Math.abs));
  return (
    <Block title="Os pesos">
      <div style={{ maxWidth: 140 + weights.outputs.length * 88 }}>
        <Heatmap
          rows={weights.features}
          columns={weights.outputs}
          values={weights.features.map((_, feature) =>
            weights.outputs.map((__, output) => weights.weights[output]?.[feature] ?? null),
          )}
          color={(value) => diverging(value / strongest)}
          format={(value) => formatNumber(value, 2)}
          rowTitle="Feature"
          columnTitle="Classe"
          description="O peso de cada feature para cada classe, no modelo linear."
          caption={caption}
          labelWidth={140}
          cellHeight={30}
        />
      </div>
    </Block>
  );
}

/** Uma tabela do que o modelo guardou (os centros, as médias de cada classe). */
function Table({ table }: { table: Matrix }) {
  const strongest = Math.max(1e-9, ...table.values.flat().map(Math.abs));
  return (
    <Block title={table.title}>
      <div style={{ maxWidth: 140 + table.columns.length * 88 }}>
        <Heatmap
          rows={table.rows}
          columns={table.columns}
          values={table.values}
          color={(value) => diverging(value / strongest)}
          format={formatStatistic}
          description={table.title}
          labelWidth={140}
          cellHeight={30}
        />
      </div>
    </Block>
  );
}
