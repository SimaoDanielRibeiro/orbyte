/**
 * A cor de cada família de modelos: a mesma no ecrã Modelos, no fluxograma dos agentes, na
 * conversa, nos gráficos da procura e na entrada (ADR 0037).
 *
 * A cor segue a família, nunca a posição: antes, cada agente tinha a cor da sua posição no plano,
 * e as árvores mudavam de cor entre uma procura de classificação e uma de regressão (ou ao tirar
 * um agente). Agora uma família tem sempre a mesma cor.
 *
 * São onze famílias e oito cores (as da paleta dos dados, `colors.ts`), por isso três cores
 * repetem-se, mas nunca no mesmo sítio. As famílias dividem-se pelo paradigma dos seus modelos:
 *
 * - as sete dos modelos supervisionados (as dos agentes) têm sete cores diferentes;
 * - as famílias com modelos não supervisionados são oito, também todas diferentes entre si: as
 *   quatro que só têm não supervisionados repetem as cores de famílias que só têm supervisionados
 *   (os centróides o azul dos lineares, os hierárquicos o amarelo das árvores, as projeções o roxo
 *   das redes) ou ficam com a oitava (a densidade, o vermelho).
 *
 * No ecrã Modelos, os dois paradigmas são duas secções: as cores repetidas nunca ficam juntas, e
 * cada família tem o nome escrito ao lado da cor.
 *
 * As duas sequências passam no validador do skill de visualização de dados (sobre o cartão,
 * `#191919`), na ordem do catálogo: as cores vizinhas distinguem-se com a visão normal e com
 * daltonismo (a mais próxima, rosa e ciano nas não supervisionadas, fica no limite aceite com o
 * nome escrito ao lado). A faixa de luminosidade é a mesma exceção da paleta (`colors.ts`).
 */
import type { ModelFamily } from "@/api/generated";

import { CATEGORICAL, OTHER } from "./colors";

const [BLUE, YELLOW, PINK, GREEN, PURPLE, ORANGE, CYAN, RED] = CATEGORICAL;

export const FAMILY_COLORS: Readonly<Record<ModelFamily, string>> = {
  linear: BLUE,
  instance_based: ORANGE,
  probabilistic: CYAN,
  tree: YELLOW,
  ensemble: PINK,
  kernel: GREEN,
  centroid: BLUE,
  density: RED,
  hierarchical: YELLOW,
  projection: PURPLE,
  neural: PURPLE,
};

/** A cor de uma família (cinzenta, se o backend trouxer uma família que ainda não tem cor). */
export function familyColor(family: ModelFamily): string {
  return (FAMILY_COLORS as Partial<Record<string, string>>)[family] ?? OTHER;
}
