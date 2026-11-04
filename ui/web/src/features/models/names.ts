/** O valor do filtro da lista dos modelos para ver todos (`/models?task=all`). */
export const ALL_TASKS = "all";

/** O `layoutId` do nome de um modelo: o mesmo na lista e no título da página dele. */
export function nameLayoutId(name: string): string {
  return `modelo-${name}`;
}
