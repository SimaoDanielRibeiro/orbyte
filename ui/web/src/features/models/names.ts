/** O valor do filtro da lista dos modelos para ver todos (`/modelos?tarefa=todas`). */
export const ALL_TASKS = "todas";

/** O `layoutId` do nome de um modelo: o mesmo na lista e no título da página dele. */
export function nameLayoutId(name: string): string {
  return `modelo-${name}`;
}
