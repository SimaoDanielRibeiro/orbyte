/**
 * As camadas de uma rede neuronal ao longo do treino: o desenho da rede no passo que se está a
 * ver, o tamanho do gradiente de cada camada (em escala logarítmica) e a parte de neurónios
 * parados de cada camada.
 */
import type { NetworkTrace } from "@/api/generated";
import { LineChart } from "@/components/charts/line-chart";
import { NetworkView } from "@/components/charts/network-view";
import { TopicText } from "@/components/markdown/topic-text";
import { categoryColor } from "@/lib/colors";
import { formatPercent } from "@/lib/format";

interface NetworkPanelProps {
  network: NetworkTrace;
  /** O passo que se está a ver no campo. */
  step?: number;
}

export function NetworkPanel({ network, step }: NetworkPanelProps) {
  const layers = network.layers;
  return (
    <div className="flex flex-col gap-10">
      <NetworkView
        network={network}
        step={step}
        description="As camadas da rede neuronal, com a ativação média de cada neurónio no passo à vista."
        caption={<TopicText topic="network_diagram" />}
      />
      <div className="grid gap-10 @5xl/main:grid-cols-2">
        <LineChart
          series={layers.map((layer, index) => ({
            key: layer.name,
            label: layer.name,
            color: categoryColor(index),
            points: network.frames.map((frame) => ({
              x: frame.step,
              y: frame.layers[index]?.gradient ?? null,
            })),
          }))}
          logY
          height={240}
          xLabel="Passo do treino"
          yLabel="Gradiente"
          description="O tamanho do gradiente dos pesos de cada camada, passo a passo."
          caption={<TopicText topic="network_gradients" />}
        />
        <LineChart
          series={layers
            .map((layer, index) => ({ layer, index }))
            .filter(({ layer }) => layer.hidden)
            .map(({ layer, index }) => ({
              key: layer.name,
              label: layer.name,
              color: categoryColor(index),
              points: network.frames.map((frame) => ({
                x: frame.step,
                y: frame.layers[index]?.stuck ?? null,
              })),
            }))}
          height={240}
          yDomain={[0, 1]}
          xLabel="Passo do treino"
          yLabel="Neurónios parados"
          formatY={(value) => formatPercent(value, 0)}
          description="A parte dos neurónios parados em cada camada escondida, passo a passo."
          caption={<TopicText topic="network_stuck" />}
        />
      </div>
    </div>
  );
}
