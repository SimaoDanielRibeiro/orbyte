import { describe, expect, it } from "vitest";

import { leftMargin, slantedOverhang } from "./margins";

describe("leftMargin", () => {
  it("fica no mínimo com números curtos", () => {
    expect(leftMargin(["0", "0,5", "1"], false)).toBe(48);
  });

  it("cresce com o número mais comprido, e mais ainda com o nome do eixo", () => {
    const plain = leftMargin(["12 000 000 000"], false);
    const named = leftMargin(["12 000 000 000"], true);
    expect(plain).toBeGreaterThan(14 * 7);
    expect(named - plain).toBe(22);
  });
});

describe("slantedOverhang", () => {
  it("cresce com o rótulo, e menos quanto mais ele sobe", () => {
    const flat = slantedOverhang("Embarked", 0);
    const slanted = slantedOverhang("Embarked", 40);
    expect(flat).toBeGreaterThan(8 * 7);
    expect(slanted).toBeLessThan(flat);
    expect(slanted).toBeGreaterThan(flat * 0.7);
  });

  it("nunca fica abaixo do mínimo", () => {
    expect(slantedOverhang("", 40)).toBe(8);
  });
});
