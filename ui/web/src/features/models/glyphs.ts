/**
 * Que desenho tem cada modelo na lista do ecrã Modelos (`model-glyph.tsx`): escolhe-se pelo que o
 * backend já diz de cada modelo (a família e as tarefas), não pelo nome. Um modelo novo de uma
 * família conhecida tem desenho sem mudar nada aqui.
 */
import type { ModelCard } from "@/api/generated";

export type GlyphKind =
  | "line-fit"
  | "separator"
  | "neighbors"
  | "outlier"
  | "bells"
  | "ellipses"
  | "tree"
  | "forest"
  | "isolation"
  | "margin"
  | "tube"
  | "enclosure"
  | "centroids"
  | "density"
  | "dendrogram"
  | "projection"
  | "network"
  | "dots";

/** O desenho de um modelo: pela família e, dentro dela, pela tarefa. */
export function glyphOf(model: Pick<ModelCard, "family" | "task_types">): GlyphKind {
  const does = (task: ModelCard["task_types"][number]) => model.task_types.includes(task);
  switch (model.family) {
    case "linear":
      return does("regression") ? "line-fit" : "separator";
    case "instance_based":
      return does("anomaly_detection") ? "outlier" : "neighbors";
    case "probabilistic":
      return does("clustering") ? "ellipses" : "bells";
    case "tree":
      return "tree";
    case "ensemble":
      return does("anomaly_detection") ? "isolation" : "forest";
    case "kernel":
      return does("anomaly_detection") ? "enclosure" : does("regression") ? "tube" : "margin";
    case "centroid":
      return "centroids";
    case "density":
      return "density";
    case "hierarchical":
      return "dendrogram";
    case "projection":
      return "projection";
    case "neural":
      return "network";
    default:
      return "dots";
  }
}
