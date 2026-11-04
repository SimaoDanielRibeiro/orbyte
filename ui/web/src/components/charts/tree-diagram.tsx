/**
 * Uma árvore de decisão desenhada de cima para baixo: cada nó com a pergunta que faz, cada
 * folha com a previsão e a parte da classe que a decide. O "sim" segue para a esquerda.
 *
 * A borda de uma folha tem a cor da classe que ela prevê (a mesma das classes nos outros
 * gráficos). Uma árvore larga demais rola na horizontal, sem encolher o texto.
 */
import { hierarchy, tree, type HierarchyPointNode } from "d3";
import type { ReactNode } from "react";

import type { TreeNode } from "@/api/generated";
import { Explain } from "@/components/explain/explain";
import { categoryColor } from "@/lib/colors";
import { formatCount, formatPercent, formatStatistic } from "@/lib/format";

interface TreeDiagramProps {
  nodes: readonly TreeNode[];
  description: string;
  caption?: ReactNode;
}

const NODE_WIDTH = 116;
const NODE_HEIGHT = 46;
const GAP_X = 12;
const GAP_Y = 34;

interface Branch {
  node: TreeNode;
  children: Branch[];
}

export function TreeDiagram({ nodes, description, caption }: TreeDiagramProps) {
  const byId = new Map(nodes.map((node) => [node.node, node]));
  const build = (id: number): Branch | null => {
    const node = byId.get(id);
    if (node === undefined) {
      return null;
    }
    const children = [node.left, node.right]
      .map((child) => (child === null ? null : build(child)))
      .filter((child): child is Branch => child !== null);
    return { node, children };
  };
  const root = nodes[0] === undefined ? null : build(nodes[0].node);
  if (root === null) {
    return null;
  }
  const layout = tree<Branch>().nodeSize([NODE_WIDTH + GAP_X, NODE_HEIGHT + GAP_Y])(
    hierarchy(root, (branch) => branch.children),
  );
  const points = layout.descendants();
  const minX = Math.min(...points.map((point) => point.x));
  const maxX = Math.max(...points.map((point) => point.x));
  const maxY = Math.max(...points.map((point) => point.y));
  const width = maxX - minX + NODE_WIDTH + 8;
  const height = maxY + NODE_HEIGHT + 8;
  const shift = (point: HierarchyPointNode<Branch>) => point.x - minX + NODE_WIDTH / 2 + 4;

  return (
    <figure className="flex min-w-0 flex-col gap-3">
      {caption !== undefined && caption !== null && caption !== false && (
        <div className="flex min-h-6 justify-end">
          <Explain title="How to read it" label="How to read this chart" align="end">
            {caption}
          </Explain>
        </div>
      )}
      <div className="relative overflow-x-auto" tabIndex={0} role="region" aria-label={description}>
        <svg width={width} height={height} role="img" aria-label={description} className="block">
          {layout.links().map((link) => {
            const x1 = shift(link.source);
            const y1 = link.source.y + NODE_HEIGHT + 4;
            const x2 = shift(link.target);
            const y2 = link.target.y + 4;
            const yes = link.source.data.node.left === link.target.data.node.node;
            return (
              <g
                key={`${String(link.source.data.node.node)}-${String(link.target.data.node.node)}`}
              >
                <path
                  d={`M${String(x1)},${String(y1)} C${String(x1)},${String((y1 + y2) / 2)} ${String(x2)},${String((y1 + y2) / 2)} ${String(x2)},${String(y2)}`}
                  fill="none"
                  stroke="var(--input)"
                />
                <text
                  x={(x1 + x2) / 2 + (yes ? -6 : 6)}
                  y={(y1 + y2) / 2}
                  textAnchor={yes ? "end" : "start"}
                  className="fill-muted-foreground text-[11px]"
                >
                  {yes ? "yes" : "no"}
                </text>
              </g>
            );
          })}
          {points.map((point) => (
            <NodeBox
              key={point.data.node.node}
              x={shift(point)}
              y={point.y + 4}
              node={point.data.node}
            />
          ))}
        </svg>
      </div>
    </figure>
  );
}

function NodeBox({ x, y, node }: { x: number; y: number; node: TreeNode }) {
  const leaf = node.feature === null;
  const left = x - NODE_WIDTH / 2;
  const accent = leaf && node.code !== null ? categoryColor(node.code) : "var(--input)";
  return (
    <g>
      <rect
        x={left}
        y={y}
        width={NODE_WIDTH}
        height={NODE_HEIGHT}
        rx={6}
        fill={leaf ? "var(--card)" : "var(--background)"}
        stroke={accent}
        strokeWidth={leaf ? 1.5 : 1}
      />
      <text x={x} y={y + 18} textAnchor="middle" className="fill-foreground text-[12px]">
        {leaf
          ? `→ ${node.prediction}`
          : `${truncate(node.feature ?? "", 9)} ≤ ${formatStatistic(node.threshold ?? 0)}`}
      </text>
      <text
        x={x}
        y={y + 34}
        textAnchor="middle"
        className="fill-muted-foreground text-[11px] tabular-nums"
      >
        {node.hidden_leaves > 0
          ? `${formatCount(node.hidden_leaves)} more leaves below`
          : leaf && node.share !== null
            ? `${formatPercent(node.share, 0)} of ${formatCount(node.samples)}`
            : `${formatCount(node.samples)} points`}
      </text>
    </g>
  );
}

function truncate(text: string, length: number): string {
  return text.length > length ? `${text.slice(0, length - 1)}…` : text;
}
