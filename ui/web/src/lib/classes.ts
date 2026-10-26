/**
 * A cor de cada classe de um alvo.
 *
 * A cor segue a classe, nunca a posição numa lista ordenada pelo tamanho (uma regra do skill de
 * visualização de dados): a classe "1" do Titanic tem a mesma cor no equilíbrio das classes, na
 * projeção 2D e no campo de decisão. Por isso as cores dão-se pela ordem das classes do backend
 * (`class_labels`, a ordem em que o modelo as numera). Onde ele não a manda (o perfil ordena as
 * classes pela frequência), usa-se a mesma regra do `class_labels` do backend: numérica se os
 * valores forem todos números, senão a ordem do texto.
 */
import { categoryColor, OTHER } from "./colors";

/** As classes pela ordem do backend. */
export function classOrder(values: readonly string[]): string[] {
  const numeric = values.every((value) => value.trim() !== "" && Number.isFinite(Number(value)));
  return numeric ? [...values].sort((a, b) => Number(a) - Number(b)) : [...values].sort();
}

/** A função que dá a cor de cada classe, dada a ordem das classes (uma desconhecida é cinzenta). */
export function classColors(ordered: readonly string[]): (label: string) => string {
  const positions = new Map(ordered.map((label, index) => [label, index]));
  return (label) => {
    const position = positions.get(label);
    return position === undefined ? OTHER : categoryColor(position);
  };
}
