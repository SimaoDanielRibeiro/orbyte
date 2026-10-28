import { describe, expect, it } from "vitest";

import { classColors, classOrder } from "./classes";
import {
  betweenClasses,
  CATEGORICAL,
  channels,
  DARK_INK,
  inkOn,
  LIGHT_INK,
  OTHER,
  sequential,
  withAlpha,
} from "./colors";

describe("as cores", () => {
  it("lê os canais de uma cor da paleta e de uma cor de uma escala do d3", () => {
    expect(channels("#6996d4")).toEqual([105, 150, 212]);
    // As escalas devolvem `rgb(…)`: o campo de decisão pinta-se com estas.
    expect(sequential(1)).toMatch(/^rgb\(/);
    expect(channels(sequential(1))).toEqual(channels("#dbe6f8"));
    expect(channels(betweenClasses(0))).toEqual(channels(CATEGORICAL[0]));
    expect(channels(betweenClasses(1))).toEqual(channels(CATEGORICAL[1]));
  });

  it("junta a transparência a qualquer formato", () => {
    expect(withAlpha("#6996d4", 0.5)).toBe("rgba(105, 150, 212, 0.5)");
    expect(withAlpha("rgb(1, 2, 3)", 0.25)).toBe("rgba(1, 2, 3, 0.25)");
  });

  it("escreve a branco nas cores escuras e a preto nas claras", () => {
    expect(inkOn("#33507f")).toBe(LIGHT_INK);
    expect(inkOn("#dbe6f8")).toBe(DARK_INK);
  });
});

describe("as cores das classes", () => {
  it("ordena como o class_labels do backend: números pelo valor, texto pela ordem do texto", () => {
    expect(classOrder(["10", "2", "1"])).toEqual(["1", "2", "10"]);
    expect(classOrder(["virginica", "setosa", "versicolor"])).toEqual([
      "setosa",
      "versicolor",
      "virginica",
    ]);
  });

  it("a cor segue a classe, não a posição numa lista ordenada pelo tamanho", () => {
    const color = classColors(classOrder(["1", "0"]));

    expect(color("0")).toBe(CATEGORICAL[0]);
    expect(color("1")).toBe(CATEGORICAL[1]);
    expect(color("desconhecida")).toBe(OTHER);
  });

  it("depois das oito cores da paleta, a categoria fica cinzenta", () => {
    const labels = Array.from({ length: 9 }, (_, index) => `c${String(index)}`);

    expect(classColors(labels)("c8")).toBe(OTHER);
  });
});
