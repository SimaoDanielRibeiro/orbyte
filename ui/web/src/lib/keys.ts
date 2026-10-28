/** Se a interface corre num Mac (as teclas escrevem-se com ⌘ em vez de Ctrl). */
const IS_MAC = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.userAgent);

/** O espaço que não parte (U+00A0): as duas teclas de um atalho nunca ficam em linhas diferentes. */
const NO_BREAK = String.fromCharCode(0xa0);

/** Uma tecla com o modificador deste sistema: `shortcut("K")` dá "Ctrl K" ou "⌘ K". */
export function shortcut(key: string): string {
  return `${IS_MAC ? "⌘" : "Ctrl"}${NO_BREAK}${key}`;
}
